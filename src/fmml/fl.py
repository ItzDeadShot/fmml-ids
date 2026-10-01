"""Flower client and server for federated training of one dataset's detector."""
import os
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
    """Trains the shared model on one client's local data (already scaled with the client's scaler)."""

    def __init__(self, dataset: str, client_dir: Path, cid: int, seed: int):
        self.dataset, self.cid, self.seed = dataset, cid, seed
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
        if mu > 0:
            _proximal(model).fit(self.X_tr, self.y_tr, mu, int(cfg["local_epochs"]), batch_size, fit_seed)
        else:
            model.fit(self.X_tr, self.y_tr, epochs=int(cfg["local_epochs"]), batch_size=batch_size, verbose=0)
        weights = model.get_weights()
        return weights, len(self.y_tr), {"cid": self.cid, "fit_seconds": time.perf_counter() - start}

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


def run_federation(dataset: str, client_dir: Path, n_clients: int, strategy: str, rounds: int, local_epochs: int,
                   seed: int, proximal_mu: float = 0.0, num_gpus_per_client: float = 0.5):
    """Run the simulation; returns {"weights": {round: ndarrays}, "val_loss": {...}, "val_accuracy": {...},
    "val_mcc": {...}, "updates": [...], "failures": [...], "round_end": {round: time}, "start": time,
    "model_bytes": size of one copy of the weights}."""
    import tensorflow as tf
    from .models import build_model

    tf.keras.utils.set_random_seed(seed)
    initial = build_model(dataset).get_weights()
    record = {"weights": {0: initial}, "val_loss": {}, "val_accuracy": {}, "val_mcc": {}, "updates": [], "failures": [],
              "round_end": {}, "model_bytes": int(sum(w.nbytes for w in initial))}

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
        else:
            strat = RecordingFedAvg(record, **kwargs)
        return ServerAppComponents(strategy=strat, config=ServerConfig(num_rounds=rounds))

    def client_fn(context: Context):
        return FLClient(dataset, client_dir, int(context.node_config["partition-id"]), seed).to_client()

    record["start"] = time.perf_counter()
    run_simulation(server_app=ServerApp(server_fn=server_fn), client_app=ClientApp(client_fn=client_fn),
                   num_supernodes=n_clients,
                   backend_config={"client_resources": {"num_cpus": 2, "num_gpus": num_gpus_per_client}})
    return record
