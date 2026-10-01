"""Federated training of one detector per dataset, with non-IID clients.

    python scripts/run_fl.py --datasets UNSW-NB15 --scaler local --partition dirichlet --alpha 0.5 --strategy fedavg --seed 0

For each dataset:
  1. Split the train and validation splits across --clients clients (IID, or skewed over attack
     types with Dirichlet(--alpha)); a client's flows keep their split, so its validation data is
     later in time than its training data. Each client fits a QuantileTransformer on its own
     training data; with --scaler local each client keeps its own, with --scaler federated they
     are merged into one common scaler from the clients' quantiles (fmml.data.merge_quantile_scalers).
  2. Train the dataset's model with Flower (FedAvg or FedProx) for --rounds rounds of
     --local-epochs local epochs each.
  3. Choose the round with the highest validation MCC, from the clients' pooled confusion counts
     (validation loss is not used: under the drift between the time-ordered splits it favours the
     barely trained, undecided early rounds).
  4. Evaluate that global model on the common test split as each client would see it (through
     the client's own scaler); report every client, the mean and the worst.

Writes results/<dataset>/<scaler>/<partition>/<strategy>/seed<N>/ (partition.json, history.csv,
test_metrics.csv, client_updates.csv, summary.json), with <strategy> fedavg or fedprox-mu<mu>, and
the chosen global model to runs/models/.
client_updates.csv holds each client's per-round update norm and cosine similarity to the mean
update: the benign reference for later attack and defence experiments.
With FMML_WANDB=1 each finished run is also logged to Weights & Biases (fmml.tracking).
"""
import argparse
import json
import os
import socket
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import pandas as pd

import fmml.architectures  # noqa: F401  (registers custom layers for saved models)
from fmml import config, tracking
from fmml.data import load_scaler, load_xy, prepare_clients
from fmml.fl import run_federation
from fmml.metrics import binary_metrics


def evaluate_on_test(model, weights, X_clients, y_test):
    """Global model on the test split as seen by each client (X_clients: the test split through
    each client's scaler)."""
    model.set_weights(weights)
    rows, cache = [], {}
    for k, X in enumerate(X_clients):
        if id(X) not in cache:                          # clients sharing one scaler share predictions
            cache[id(X)] = model.predict(X, verbose=0, batch_size=8192).ravel() > 0.5
        rows.append({"client": k, **binary_metrics(y_test, cache[id(X)].astype(int))})
    return pd.DataFrame(rows)


def run(dataset, args):
    import tensorflow as tf
    from fmml.models import build_model

    tag = "iid" if args.partition == "iid" else f"dirichlet-a{args.alpha:g}"
    strategy = args.strategy if args.strategy == "fedavg" else f"fedprox-mu{args.mu:g}"
    if args.refit:                                   # retrain with the main study's chosen number of rounds
        assert config.EXPERIMENT == "refit", "run --refit with FMML_EXPERIMENT=refit"
        reference = json.loads((config.MAIN_RESULTS_DIR / dataset / args.scaler / tag / strategy
                                / f"seed{args.seed}" / "summary.json").read_text())
        args.rounds = reference["best_round"]
    client_dir = config.RUNS_DIR / "clients" / dataset / args.scaler / tag / f"seed{args.seed}"
    out = config.RESULTS_DIR / dataset / args.scaler / tag / strategy / f"seed{args.seed}"
    if args.skip_existing and (out / "summary.json").exists():
        print(f"skip {out} (done)")
        return
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".running"                          # marks the run as in progress (see scripts/run_queue.py)
    if lock.exists() and _alive(int(lock.read_text() or 0)):
        print(f"skip {out} (running in process {lock.read_text()})")
        return
    lock.write_text(str(os.getpid()))
    try:
        _run(dataset, args, tag, strategy, client_dir, out)
    finally:
        lock.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return pid > 0
    except OSError:
        return False


def _run(dataset, args, tag, strategy, client_dir, out):
    import tensorflow as tf
    from fmml.models import build_model
    t0 = time.time()

    stats = prepare_clients(dataset, args.partition, args.alpha, args.clients, args.seed, client_dir, args.scaler,
                            refit=args.refit)
    (out / "partition.json").write_text(json.dumps(stats, indent=2))
    print(f"\n=== {dataset} | {args.scaler} scaler | {tag} | {strategy} | seed {args.seed}")
    print(pd.DataFrame(stats).drop(columns=["types"]).to_string(index=False))

    record = run_federation(dataset, client_dir, args.clients, args.strategy, args.rounds, args.local_epochs,
                            args.seed, proximal_mu=args.mu, num_gpus_per_client=args.gpus_per_client)
    # A client that failed (e.g. killed for memory) would silently drop out of the average: discard the run.
    if record["failures"] or len(record["val_mcc"]) < args.rounds:
        raise RuntimeError(f"{out}: client failures {record['failures']}, {len(record['val_mcc'])}/{args.rounds} rounds")

    # Choose the round on the clients' validation data only; the test split plays no part.
    # (Refit: the validation flows were trained on; the last round is the chosen number of rounds.)
    val = pd.Series(record["val_mcc"]).sort_index()
    best_round = args.rounds if args.refit else int(val.idxmax())
    X_test, y_test = load_xy(config.split_path(dataset, "test"))
    scalers = [load_scaler(client_dir, k) for k in range(args.clients)]
    X_clients = [scalers[0].transform(X_test)] * args.clients if args.scaler == "federated" \
        else [s.transform(X_test) for s in scalers]
    model = build_model(dataset)

    # Round timings: simulation wall clock (Ray start-up counted in round 1); clients train two at a time on one GPU.
    ends = pd.Series(record["round_end"]).sort_index()
    round_seconds = ends.diff().fillna(ends.iloc[0] - record["start"])
    fit_seconds = pd.DataFrame(record["updates"]).groupby("round").fit_seconds

    history = []
    for rnd in sorted(record["val_loss"]):
        t = evaluate_on_test(model, record["weights"][rnd], X_clients, y_test)
        history.append({"round": rnd, "val_loss": record["val_loss"][rnd], "val_accuracy": record["val_accuracy"][rnd],
                        "val_mcc": record["val_mcc"][rnd], "test_f1_mean": t.F1.mean(), "test_f1_worst": t.F1.min(),
                        "test_mcc_mean": t.MCC.mean(), "test_mcc_worst": t.MCC.min(), "round_seconds": round_seconds[rnd],
                        "client_fit_seconds_mean": fit_seconds.mean()[rnd], "client_fit_seconds_max": fit_seconds.max()[rnd]})
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame(record["updates"]).to_csv(out / "client_updates.csv", index=False)

    test = evaluate_on_test(model, record["weights"][best_round], X_clients, y_test)
    test.to_csv(out / "test_metrics.csv", index=False)
    model.set_weights(record["weights"][best_round])
    model_path = config.RUNS_DIR / "models" / dataset / args.scaler / tag / strategy / f"seed{args.seed}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    summary = {"dataset": dataset, "scaler": args.scaler, "partition": tag, "strategy": args.strategy, "seed": args.seed,
               "refit": args.refit, "host": socket.gethostname(),
               "clients": args.clients, "rounds": args.rounds, "local_epochs": args.local_epochs,
               "proximal_mu": args.mu if args.strategy == "fedprox" else None, "best_round": best_round,
               "val_mcc_best": float(val[best_round]), "val_loss_best": float(record["val_loss"][best_round]),
               **{f"test_{m}_mean": float(test[m].mean()) for m in ["Accuracy", "Precision", "Recall", "F1", "FPR", "BalancedAccuracy", "MCC"]},
               "test_F1_worst": float(test.F1.min()), "test_MCC_worst": float(test.MCC.min()),
               "simulation_seconds": float(ends.iloc[-1] - record["start"]),
               "round_seconds_median": float(round_seconds.iloc[1:].median()),            # round 1 includes start-up
               "client_fit_seconds_mean": float(pd.DataFrame(record["updates"]).fit_seconds.mean()),
               "model_bytes": record["model_bytes"],
               # every round each client downloads and uploads one copy of the weights
               "communication_bytes_total": 2 * record["model_bytes"] * args.clients * args.rounds,
               "minutes": round((time.time() - t0) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"best round {best_round} (val MCC {val[best_round]:.4f}) | test MCC mean {summary['test_MCC_mean']:.4f}, "
          f"worst client {summary['test_MCC_worst']:.4f} | F1 mean {summary['test_F1_mean']:.4f} | {summary['minutes']} min")
    tf.keras.backend.clear_session()
    tracking.maybe_sync(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    parser.add_argument("--scaler", choices=["local", "federated"], default="local",
                        help="each client's own QuantileTransformer, or one merged from the clients' quantiles")
    parser.add_argument("--partition", choices=["iid", "dirichlet"], default="dirichlet")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal coefficient")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus-per-client", type=float, default=0.5,
                        help="GPU share per simulated client (0.5 = two clients train at a time)")
    parser.add_argument("--skip-existing", action="store_true", help="skip runs whose summary.json exists")
    parser.add_argument("--refit", action="store_true",
                        help="(FMML_EXPERIMENT=refit) train on train+val for the main study's chosen number of rounds")
    args = parser.parse_args()
    for dataset in args.datasets:
        run(dataset, args)


if __name__ == "__main__":
    main()
