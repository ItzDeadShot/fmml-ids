"""Detection rate per attack type on the test split, for the centralised and federated models.

    python scripts/eval_types.py            # every finished run without a type_metrics.csv
    python scripts/eval_types.py --force    # recompute all

For a federated run, the chosen global model (runs/models/...) is applied to the test split
through each client's own scaler, as in run_fl.py; the rate of flows predicted as attacks is
reported per attack type and client ("Benign" gives the false-positive rate). For a centralised
run, the checkpoint in runs/central/ is applied through the scaler refit on the training split.

Writes type_metrics.csv (columns: client, attack, flows, predicted_attack) and auc.csv (per client:
ROC-AUC and average precision of the attack score, which do not depend on the 0.5 threshold, so
they separate how well a model ranks flows from where its operating point lies) next to each
run's test_metrics.csv; centralised runs have client -1.
"""
import argparse
import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer

import fmml.architectures  # noqa: F401  (registers custom layers for saved models)
from fmml import config
from fmml.data import load_scaler, load_split, shares_scaler


def rates(model, X, attack, client):
    """Returns (rate of flows predicted as attacks per attack type, threshold-free scores)."""
    score = model.predict(X, verbose=0, batch_size=8192).ravel()
    df = pd.DataFrame({"attack": attack, "pred": score > 0.5})
    out = df.groupby("attack").pred.agg(flows="size", predicted_attack="mean").reset_index()
    out.insert(0, "client", client)
    y = attack != "Benign"
    auc = pd.DataFrame([{"client": client, "ROC_AUC": roc_auc_score(y, score), "PR_AUC": average_precision_score(y, score)}])
    return out, auc


def write(results, run):
    pd.concat([r[0] for r in results]).to_csv(run / "type_metrics.csv", index=False)
    pd.concat([r[1] for r in results]).to_csv(run / "auc.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for dataset in args.datasets:
        X_te, _, att_te = load_split(config.split_path(dataset, "test"))

        central = sorted((config.RESULTS_DIR / "central" / dataset).glob("seed*/test_metrics.csv"))
        if central:
            X_tr = load_split(config.split_path(dataset, "train"))[0]
        for metrics in central:
            out = metrics.parent / "type_metrics.csv"
            if out.exists() and (out.parent / "auc.csv").exists() and not args.force:
                continue
            seed = int(metrics.parent.name.removeprefix("seed"))
            scaler = QuantileTransformer(output_distribution="normal", random_state=seed).fit(X_tr)
            model = tf.keras.models.load_model(config.RUNS_DIR / "central" / dataset / f"seed{seed}.keras")
            write([rates(model, scaler.transform(X_te), att_te, -1)], metrics.parent)
            tf.keras.backend.clear_session()
            print(out)

        for summary in sorted((config.RESULTS_DIR / dataset).glob("*/*/*/seed*/summary.json")):
            run = summary.parent
            out = run / "type_metrics.csv"
            if out.exists() and (run / "auc.csv").exists() and not args.force:
                continue
            scaler, tag, strategy, seed = run.parents[2].name, run.parents[1].name, run.parent.name, run.name
            model = tf.keras.models.load_model(
                config.RUNS_DIR / "models" / dataset / scaler / tag / strategy / f"{seed}.keras")
            client_dir = config.RUNS_DIR / "clients" / dataset / scaler / tag / seed
            n_clients = len(list(client_dir.glob("scaler_*.pkl")))
            if shares_scaler(scaler):                   # one common scaler: identical for every client
                first = rates(model, load_scaler(client_dir, 0).transform(X_te), att_te, 0)
                per_client = [first] + [(first[0].assign(client=k), first[1].assign(client=k))
                                        for k in range(1, n_clients)]
            else:
                per_client = [rates(model, load_scaler(client_dir, k).transform(X_te), att_te, k)
                              for k in range(n_clients)]
            write(per_client, run)
            tf.keras.backend.clear_session()
            print(out)


if __name__ == "__main__":
    main()
