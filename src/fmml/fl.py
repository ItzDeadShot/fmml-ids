"""Flower client and server for federated training of one dataset's detector."""
import os
import shutil
import tempfile
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from pathlib import Path

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from flwr.simulation import run_simulation

from . import config
from .data import load_client
from .metrics import mcc


_MODELS = {}     # per actor process: one model per dataset, reused by every client call it serves


def _shared_model(dataset: str, parameters):
    """The process's model for the dataset, loaded with the given weights and a reset optimizer.

    Reusing one model (instead of building one per call) avoids TensorFlow's memory growth from
    repeatedly created models and training functions in a long-lived simulation actor. Zeroing the
    optimizer's iteration count and moment estimates makes every call start like a fresh Adam.
    """
    import tensorflow as tf
    from .models import build_model
    if dataset not in _MODELS:
        _MODELS[dataset] = build_model(dataset)
    model = _MODELS[dataset]
    model.set_weights(parameters)
    for v in model.optimizer.variables:
        if not v.path.endswith("learning_rate"):
            v.assign(tf.zeros_like(v))
    return model


class FLClient(NumPyClient):
    """Trains the shared model on one client's local data (already scaled with the client's scaler).
    state_dir (SCAFFOLD): where the client keeps its control variate between rounds."""

    def __init__(self, dataset: str, client_dir: Path, cid: int, seed: int, state_dir: Path = None):
        self.dataset, self.cid, self.seed, self.state_dir = dataset, cid, seed, state_dir
        self.X_tr, self.y_tr, self.X_va, self.y_va = load_client(client_dir, cid)

    def fit(self, parameters, cfg):
        import tensorflow as tf
        rnd = int(cfg["round"])
        fit_seed = self.seed * 100_000 + rnd * 100 + self.cid
        tf.keras.utils.set_random_seed(fit_seed)
        start = time.perf_counter()
        model = _shared_model(self.dataset, parameters)
        batch_size = config.MODELS[self.dataset]["batch_size"]
        mu = float(cfg.get("proximal_mu", 0.0))
        extra = {}
        if "control" in cfg:                              # SCAFFOLD: the server's control variate
            control = _scaffold(model).fit(self.X_tr, self.y_tr, np.frombuffer(cfg["control"], dtype="float32"),
                                           self.state_dir / f"control_{self.cid}.npy", int(cfg["local_epochs"]),
                                           batch_size, fit_seed)
            extra["control"] = control.tobytes()
        elif mu > 0:
            _proximal(model).fit(self.X_tr, self.y_tr, mu, int(cfg["local_epochs"]), batch_size, fit_seed)
        else:
            model.fit(self.X_tr, self.y_tr, epochs=int(cfg["local_epochs"]), batch_size=batch_size, verbose=0)
        weights = model.get_weights()
        return weights, len(self.y_tr), {"cid": self.cid, "fit_seconds": time.perf_counter() - start, **extra}

    def evaluate(self, parameters, cfg):
        """Validation loss, plus the confusion counts at threshold 0.5 (the server pools them into
        a validation MCC, which chooses the round)."""
        model = _shared_model(self.dataset, parameters)
        loss, acc = model.evaluate(self.X_va, self.y_va, batch_size=8192, verbose=0)
        pred = model.predict(self.X_va, batch_size=8192, verbose=0).ravel() > 0.5
        y = self.y_va.astype(bool)
        counts = {"tp": int(np.sum(pred & y)), "fp": int(np.sum(pred & ~y)), "tn": int(np.sum(~pred & ~y)),
                  "fn": int(np.sum(~pred & y))}
        return float(loss), len(self.y_va), {"cid": self.cid, "accuracy": float(acc), **counts}


_PROXIMAL = {}


def _proximal(model):
    if id(model) not in _PROXIMAL:
        _PROXIMAL[id(model)] = _ProximalTrainer(model)
    return _PROXIMAL[id(model)]


class _ProximalTrainer:
    """Local training with the FedProx proximal term mu/2 * ||w - w_global||^2; the step function
    is traced once per model, with the received global weights and mu held in variables. The step
    is XLA-compiled, as Keras compiles model.fit (the FedAvg path), so both strategies' local
    training runs the same way and their timings are comparable."""

    def __init__(self, model):
        import tensorflow as tf
        self.model = model
        self.anchors = [tf.Variable(v, trainable=False) for v in model.trainable_variables]
        self.mu = tf.Variable(0.0, trainable=False)
        bce = tf.keras.losses.BinaryCrossentropy()

        @tf.function(jit_compile=True, reduce_retracing=True)
        def step(xb, yb):
            with tf.GradientTape() as tape:
                loss = bce(yb, tf.squeeze(model(xb, training=True), -1))
                loss += 0.5 * self.mu * tf.add_n([tf.reduce_sum(tf.square(v - a))
                                                  for v, a in zip(model.trainable_variables, self.anchors)])
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        self.step = step

    def fit(self, X, y, mu, epochs, batch_size, seed):
        import tensorflow as tf
        for a, v in zip(self.anchors, self.model.trainable_variables):
            a.assign(v)                                   # the received global weights
        self.mu.assign(mu)
        rng = np.random.default_rng(seed)
        y = y.astype("float32")
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                batch = order[start:start + batch_size]
                self.step(tf.constant(X[batch]), tf.constant(y[batch]))


_SCAFFOLD = {}


def _scaffold(model):
    if id(model) not in _SCAFFOLD:
        _SCAFFOLD[id(model)] = _ScaffoldTrainer(model)
    return _SCAFFOLD[id(model)]


class _ScaffoldTrainer:
    """Local training for SCAFFOLD (Karimireddy et al. 2020): every gradient step is corrected by
    c - c_i, the server's control variate minus the client's, before the optimizer applies it.

    The client's new control variate is option I of the paper: its full-data gradient at the
    received global weights (one extra pass over its training data). Option II derives it from the
    weight change divided by the number of steps times the learning rate, which assumes plain SGD;
    option I does not depend on the optimizer, so local training keeps the same Adam as FedAvg and
    FedProx. Control variates cover the trainable variables, flattened in their order. The steps
    are XLA-compiled, as in the other strategies."""

    def __init__(self, model):
        import tensorflow as tf
        self.model = model
        tv = model.trainable_variables
        self.shapes = [tuple(v.shape) for v in tv]
        self.correction = [tf.Variable(tf.zeros_like(v), trainable=False) for v in tv]
        self.grad_sum = [tf.Variable(tf.zeros_like(v), trainable=False) for v in tv]
        bce = tf.keras.losses.BinaryCrossentropy()

        @tf.function(jit_compile=True, reduce_retracing=True)
        def step(xb, yb):
            with tf.GradientTape() as tape:
                loss = bce(yb, tf.squeeze(model(xb, training=True), -1))
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip([g + c for g, c in zip(grads, self.correction)],
                                                model.trainable_variables))

        @tf.function(jit_compile=True, reduce_retracing=True)
        def accumulate(xb, yb):                            # adds the batch's summed loss gradient
            with tf.GradientTape() as tape:
                loss = bce(yb, tf.squeeze(model(xb, training=True), -1))
            n = tf.cast(tf.shape(yb)[0], loss.dtype)
            for total, g in zip(self.grad_sum, tape.gradient(loss, model.trainable_variables)):
                total.assign_add(g * n)

        self.step, self.accumulate = step, accumulate

    def split(self, flat):
        cuts = np.cumsum([int(np.prod(s)) for s in self.shapes])[:-1]
        return [part.reshape(s) for part, s in zip(np.split(flat, cuts), self.shapes)]

    def fit(self, X, y, control, state_file: Path, epochs, batch_size, seed) -> np.ndarray:
        """Train with the correction; returns the client's new control variate (flat float32)."""
        import tensorflow as tf
        own = np.load(state_file) if state_file.exists() else np.zeros_like(control)
        y = y.astype("float32")
        for total in self.grad_sum:
            total.assign(tf.zeros_like(total))
        for start in range(0, len(y), 8192):               # new control variate, at the received weights
            self.accumulate(tf.constant(X[start:start + 8192]), tf.constant(y[start:start + 8192]))
        new = (np.concatenate([total.numpy().ravel() for total in self.grad_sum]) / len(y)).astype("float32")
        for corr, value in zip(self.correction, self.split(control - own)):
            corr.assign(value)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                batch = order[start:start + batch_size]
                self.step(tf.constant(X[batch]), tf.constant(y[batch]))
        np.save(state_file, new)
        return new


class _Recorder:
    """Keeps every round's aggregated global weights, the clients' aggregated validation loss,
    and statistics of each client's update (the reference for later attack/defence studies)."""

    def aggregate_fit(self, server_round, results, failures):
        # Clients report in the order they finish, which depends on scheduling; sorting by client id
        # makes the weighted average (a floating-point sum) independent of how many train at once.
        results = sorted(results, key=lambda r: int(r[1].metrics["cid"]))
        if failures:
            self.record["failures"].append({"round": server_round, "stage": "fit", "failures": len(failures)})
        previous = np.concatenate([w.ravel() for w in self.record["weights"][server_round - 1]])
        updates = {int(res.metrics["cid"]): np.concatenate([w.ravel() for w in parameters_to_ndarrays(res.parameters)])
                   - previous for _, res in results}
        mean_update = np.mean(list(updates.values()), axis=0)
        fit_seconds = {int(res.metrics["cid"]): float(res.metrics.get("fit_seconds", np.nan)) for _, res in results}
        for cid, u in sorted(updates.items()):
            cos = float(u @ mean_update / (np.linalg.norm(u) * np.linalg.norm(mean_update) + 1e-12))
            self.record["updates"].append({"round": server_round, "client": cid, "update_l2": float(np.linalg.norm(u)),
                                           "cosine_to_mean": cos, "fit_seconds": fit_seconds[cid]})
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is not None:
            self.record["weights"][server_round] = parameters_to_ndarrays(params)
        return params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        results = sorted(results, key=lambda r: int(r[1].metrics["cid"]))
        if failures:
            self.record["failures"].append({"round": server_round, "stage": "evaluate", "failures": len(failures)})
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self.record["val_loss"][server_round] = loss
        self.record["val_accuracy"][server_round] = metrics.get("accuracy")
        self.record["val_mcc"][server_round] = metrics.get("mcc")
        self.record["round_end"][server_round] = time.perf_counter()     # end of the round's fit + evaluate
        return loss, metrics


class RecordingFedAvg(_Recorder, FedAvg):
    def __init__(self, record, **kwargs):
        super().__init__(**kwargs)
        self.record = record


class RecordingFedProx(_Recorder, FedProx):
    def __init__(self, record, **kwargs):
        super().__init__(**kwargs)
        self.record = record


class RecordingScaffold(_Recorder, FedAvg):
    """SCAFFOLD's server: the global weights are averaged as in FedAvg; the control variate c, sent
    to the clients with every fit instruction, is the size-weighted mean of the clients' control
    variates (all clients take part every round, so this equals the paper's incremental update)."""

    def __init__(self, record, control_size, **kwargs):
        super().__init__(**kwargs)
        self.record = record
        self.control = np.zeros(control_size, dtype="float32")

    def configure_fit(self, server_round, parameters, client_manager):
        instructions = super().configure_fit(server_round, parameters, client_manager)
        for _, ins in instructions:
            ins.config = {**ins.config, "control": self.control.tobytes()}
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if results and not failures:
            n = np.array([res.num_examples for _, res in results], dtype="float64")
            controls = np.stack([np.frombuffer(res.metrics["control"], dtype="float32") for _, res in results])
            self.control = (n @ controls / n.sum()).astype("float32")
        return params, metrics


def run_federation(dataset: str, client_dir: Path, n_clients: int, strategy: str, rounds: int, local_epochs: int,
                   seed: int, proximal_mu: float = 0.0, num_gpus_per_client: float = 0.5):
    """Run the simulation; returns {"weights": {round: ndarrays}, "val_loss": {...}, "val_accuracy": {...},
    "val_mcc": {...}, "updates": [...], "failures": [...], "round_end": {round: time}, "start": time,
    "model_bytes": size of one copy of the weights, "control_bytes": size of one SCAFFOLD control variate
    (0 for the other strategies)}."""
    import tensorflow as tf
    from .models import build_model

    tf.keras.utils.set_random_seed(seed)
    model = build_model(dataset)
    initial = model.get_weights()
    control_size = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    record = {"weights": {0: initial}, "val_loss": {}, "val_accuracy": {}, "val_mcc": {}, "updates": [], "failures": [],
              "round_end": {}, "model_bytes": int(sum(w.nbytes for w in initial)),
              "control_bytes": 4 * control_size if strategy == "scaffold" else 0}
    # SCAFFOLD clients keep their control variates here between rounds (removed at the end).
    state_dir = Path(tempfile.mkdtemp(prefix="scaffold-", dir=client_dir)) if strategy == "scaffold" else None

    def pooled_validation(metrics):
        n = sum(k for k, _ in metrics)
        counts = {c: sum(m[c] for _, m in metrics) for c in ("tp", "fp", "tn", "fn")}
        return {"accuracy": sum(k * m["accuracy"] for k, m in metrics) / n, "mcc": mcc(**counts)}

    kwargs = dict(fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=n_clients, min_evaluate_clients=n_clients,
                  min_available_clients=n_clients, initial_parameters=ndarrays_to_parameters(initial),
                  on_fit_config_fn=lambda r: {"round": r, "local_epochs": local_epochs},
                  evaluate_metrics_aggregation_fn=pooled_validation)

    def server_fn(context: Context):
        if strategy == "fedprox":
            strat = RecordingFedProx(record, proximal_mu=proximal_mu, **kwargs)
        elif strategy == "scaffold":
            strat = RecordingScaffold(record, control_size, **kwargs)
        else:
            strat = RecordingFedAvg(record, **kwargs)
        return ServerAppComponents(strategy=strat, config=ServerConfig(num_rounds=rounds))

    def client_fn(context: Context):
        return FLClient(dataset, client_dir, int(context.node_config["partition-id"]), seed, state_dir).to_client()

    record["start"] = time.perf_counter()
    try:
        run_simulation(server_app=ServerApp(server_fn=server_fn), client_app=ClientApp(client_fn=client_fn),
                       num_supernodes=n_clients,
                       backend_config={"client_resources": {"num_cpus": 2, "num_gpus": num_gpus_per_client}})
    finally:
        if state_dir is not None:
            shutil.rmtree(state_dir, ignore_errors=True)
    return record
