"""Decision-threshold calibration on validation data, applied to the test split once.

    python scripts/calibrate_threshold.py            # every finished run without calibrated.json
    python scripts/calibrate_threshold.py --force

The models were trained and chosen with the usual threshold 0.5. Under the drift between the
time-ordered splits the ranking of flows often stays good (ROC-AUC) while 0.5 becomes a poor
operating point, so the threshold is chosen afresh on validation data:

  federated runs   each client reports its confusion counts at every candidate threshold on its
                   own validation data (scored through its own scaler); the server adds them up
                   and takes the threshold with the highest pooled validation MCC. Only counts
                   leave the clients, as for the round selection.
  centralised      the threshold with the highest MCC on the validation split.

In the refit experiment (FMML_EXPERIMENT=refit) the validation data was trained on, so nothing is
chosen: each refit model uses the threshold chosen for the same run in the main study.

Candidates are 481 thresholds evenly spaced in log-odds from -12 to 12 (0.5 included). The test
split plays no part in the choice. Writes calibrated.json (threshold, validation MCC at 0.5 and at
the threshold) and calibrated_test_metrics.csv (per client, at the threshold) next to each run's
test_metrics.csv.
"""
import argparse
import json
import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer

import fmml.architectures  # noqa: F401  (registers custom layers for saved models)
from fmml import config
from fmml.data import load_client, load_scaler, load_split
from fmml.metrics import binary_metrics, mcc

THRESHOLDS = 1 / (1 + np.exp(-np.linspace(-12, 12, 481)))


def counts(scores, y):
    """Confusion counts (tp, fp, tn, fn) at every candidate threshold; arrays of len(THRESHOLDS)."""
    y = y.astype(bool)
    pos, neg = np.sort(scores[y]), np.sort(scores[~y])
    tp = len(pos) - np.searchsorted(pos, THRESHOLDS, side="right")      # score > threshold
    fp = len(neg) - np.searchsorted(neg, THRESHOLDS, side="right")
    return tp, fp, len(neg) - fp, len(pos) - tp


def choose(tp, fp, tn, fn):
    scores = np.array([mcc(*c) for c in zip(tp, fp, tn, fn)])
    k = int(np.argmax(scores))                        # ties: the lowest threshold
    at_half = int(np.argmin(np.abs(THRESHOLDS - 0.5)))
    return float(THRESHOLDS[k]), float(scores[k]), float(scores[at_half])


def reference_threshold(run):
    """Refit experiment: the threshold chosen for the same run in the main study."""
    main_run = config.MAIN_RESULTS_DIR / run.relative_to(config.RESULTS_DIR)
    return json.loads((main_run / "calibrated.json").read_text())["threshold"]


def predict(model, X):
    return model.predict(X, verbose=0, batch_size=8192).ravel()


def write(run, threshold, val_mcc, val_mcc_half, per_client):
    per_client.to_csv(run / "calibrated_test_metrics.csv", index=False)
    (run / "calibrated.json").write_text(json.dumps(
        {"threshold": threshold, "val_mcc_at_threshold": val_mcc, "val_mcc_at_0.5": val_mcc_half,
         "test_MCC_mean": float(per_client.MCC.mean()), "test_MCC_worst": float(per_client.MCC.min()),
         "test_FPR_mean": float(per_client.FPR.mean())}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for dataset in args.datasets:
        X_te, y_te, _ = load_split(config.split_path(dataset, "test"))

        central = sorted((config.RESULTS_DIR / "central" / dataset).glob("seed*/test_metrics.csv"))
        if central:
            X_tr = load_split(config.split_path(dataset, "train"))[0]
            X_va, y_va, _ = load_split(config.split_path(dataset, "val"))
        for f in central:
            run = f.parent
            if (run / "calibrated.json").exists() and not args.force:
                continue
            seed = int(run.name.removeprefix("seed"))
            scaler = QuantileTransformer(output_distribution="normal", random_state=seed).fit(X_tr)
            model = tf.keras.models.load_model(config.RUNS_DIR / "central" / dataset / f"seed{seed}.keras")
            if config.EXPERIMENT == "refit":
                scaler = QuantileTransformer(output_distribution="normal", random_state=seed).fit(np.concatenate([X_tr, X_va]))
                t, v, v_half = reference_threshold(run), float("nan"), float("nan")
            else:
                t, v, v_half = choose(*counts(predict(model, scaler.transform(X_va)), y_va))
            test = binary_metrics(y_te, (predict(model, scaler.transform(X_te)) > t).astype(int))
            write(run, t, v, v_half, pd.DataFrame([{"client": -1, **test}]))
            tf.keras.backend.clear_session()
            print(f"{run}: threshold {t:.4g}, val MCC {v_half:.4f} -> {v:.4f}, test MCC {test['MCC']:.4f}")

        for summary in sorted((config.RESULTS_DIR / dataset).glob("*/*/*/seed*/summary.json")):
            run = summary.parent
            if (run / "calibrated.json").exists() and not args.force:
                continue
            scaler, tag, strategy, seed = run.parents[2].name, run.parents[1].name, run.parent.name, run.name
            model = tf.keras.models.load_model(
                config.RUNS_DIR / "models" / dataset / scaler / tag / strategy / f"{seed}.keras")
            client_dir = config.RUNS_DIR / "clients" / dataset / scaler / tag / seed
            n_clients = len(list(client_dir.glob("scaler_*.pkl")))
            if config.EXPERIMENT == "refit":
                t, v, v_half = reference_threshold(run), float("nan"), float("nan")
            else:
                pooled = np.zeros((4, len(THRESHOLDS)), dtype=np.int64)
                for k in range(n_clients):               # each client: counts on its own validation data
                    _, _, X_va, y_va = load_client(client_dir, k)
                    pooled += np.array(counts(predict(model, X_va), y_va))
                t, v, v_half = choose(*pooled)
            rows, cache = [], {}
            for k in range(n_clients):
                sc = load_scaler(client_dir, 0 if scaler == "federated" else k)
                key = 0 if scaler == "federated" else k
                if key not in cache:
                    cache[key] = binary_metrics(y_te, (predict(model, sc.transform(X_te)) > t).astype(int))
                rows.append({"client": k, **cache[key]})
            per_client = pd.DataFrame(rows)
            write(run, t, v, v_half, per_client)
            tf.keras.backend.clear_session()
            print(f"{run}: threshold {t:.4g}, val MCC {v_half:.4f} -> {v:.4f}, "
                  f"test MCC {per_client.MCC.mean():.4f} (worst {per_client.MCC.min():.4f})")


if __name__ == "__main__":
    main()
