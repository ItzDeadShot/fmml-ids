"""How close the federated (merged) quantile scaler is to one fit on the pooled training data.

    python scripts/check_scaler_merge.py            # FedAvg runs, time split, all datasets and partitions

The federated scaler (fmml.data.merge_quantile_scalers) is built from the clients' own fitted
QuantileTransformers, each estimated from a random subsample of at most 10,000 of the client's
training flows (sklearn's default). The pooled scaler, the non-deployable reference (and the
centralised baseline's scaler), is fit the same way on the train split, i.e. on 10,000 of all
clients' training flows. Neither is exact, so both are compared with the exact scaler: the
quantiles of the complete train split (no subsampling).

For every federated-scaler run this reports, on the common test split, the rank error
|F(x) - F_exact(x)| (uniform space, 0-1) of the merged and of the pooled scaler: the mean over
features and flows, the mean over features of the 99th percentile and of the largest error, and the
largest per-feature mean error; and the normal-space error the model actually sees. It checks that
the merge of the local-scaler runs' client scalers reproduces the saved federated scaler, and it
re-evaluates each federated-scaler global model on the test split through the pooled and the exact
scaler instead of the merged one it was trained with.

Writes results/mechanism/scaler_merge.csv (one row per run) and scaler_merge.txt (summary).
"""
import argparse
import json
import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

import fmml.architectures  # noqa: F401  (registers custom layers for saved models)
from fmml import config
from fmml.data import load_scaler, load_split, merge_quantile_scalers
from fmml.metrics import binary_metrics

PARTITIONS = ["iid", "dirichlet-a1", "dirichlet-a0.5", "dirichlet-a0.1"]


def score(model, X, y):
    p = model.predict(X, verbose=0, batch_size=8192).ravel()
    return binary_metrics(y, (p > 0.5).astype(int))["MCC"], roc_auc_score(y, p)


def errors(name, Z, Z_exact):
    """Rank and normal-space errors of one scaler's test encoding Z against the exact one."""
    rank = np.abs(norm.cdf(Z) - norm.cdf(Z_exact))
    return {f"{name}_rank_mean": rank.mean(), f"{name}_rank_p99": np.percentile(rank, 99, axis=0).mean(),
            f"{name}_rank_max": rank.max(axis=0).mean(), f"{name}_rank_worst_feature": rank.mean(axis=0).max(),
            f"{name}_normal_mean": np.abs(Z - Z_exact).mean()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    parser.add_argument("--strategy", default="fedavg")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    out_dir = config.RESULTS_DIR / "mechanism"
    out_dir.mkdir(exist_ok=True)
    if not args.report_only:
        measure(args, out_dir)
    report(out_dir)


def measure(args, out_dir):
    rows = []
    for dataset in args.datasets:
        X_te, y_te, _ = load_split(config.split_path(dataset, "test"))
        X_tr = load_split(config.split_path(dataset, "train"))[0]
        exact = QuantileTransformer(output_distribution="normal", subsample=None).fit(X_tr)
        Z_exact = exact.transform(X_te)
        features = pd.read_csv(config.split_path(dataset, "test"), nrows=0).columns.drop([config.LABEL_COL, config.ATTACK_COL])
        for tag in PARTITIONS:
            for run in sorted((config.RESULTS_DIR / dataset / "federated" / tag / args.strategy).glob("seed*")):
                seed = int(run.name.removeprefix("seed"))
                stats = json.loads((run / "partition.json").read_text())
                merged = load_scaler(config.RUNS_DIR / "clients" / dataset / "federated" / tag / run.name, 0)
                local_dir = config.RUNS_DIR / "clients" / dataset / "local" / tag / run.name
                remerged = merge_quantile_scalers([load_scaler(local_dir, k) for k in range(len(stats))],
                                                  [s["train"] for s in stats], seed)
                assert np.array_equal(remerged.quantiles_, merged.quantiles_), run
                pooled = QuantileTransformer(output_distribution="normal", random_state=seed).fit(X_tr)
                Z_merged, Z_pooled = merged.transform(X_te), pooled.transform(X_te)
                worst = np.abs(norm.cdf(Z_merged) - norm.cdf(Z_exact)).mean(axis=0).argmax()

                model = tf.keras.models.load_model(
                    config.RUNS_DIR / "models" / dataset / "federated" / tag / args.strategy / f"{run.name}.keras")
                mcc_merged, auc_merged = score(model, Z_merged, y_te)
                assert abs(mcc_merged - pd.read_csv(run / "test_metrics.csv").MCC.iloc[0]) < 1e-6, run
                mcc_pooled, auc_pooled = score(model, Z_pooled, y_te)
                mcc_exact, auc_exact = score(model, Z_exact, y_te)
                rows.append({"dataset": dataset, "partition": tag, "seed": seed,
                             **errors("merged", Z_merged, Z_exact), **errors("pooled", Z_pooled, Z_exact),
                             "merged_worst_feature_name": features[worst],
                             "MCC_merged": mcc_merged, "MCC_via_pooled": mcc_pooled, "MCC_via_exact": mcc_exact,
                             "ROC_AUC_merged": auc_merged, "ROC_AUC_via_pooled": auc_pooled, "ROC_AUC_via_exact": auc_exact})
                r = rows[-1]
                print(f"{dataset} {tag} {run.name}: rank error merged {r['merged_rank_mean']:.4f} pooled {r['pooled_rank_mean']:.4f} "
                      f"| MCC merged {mcc_merged:.3f} via pooled {mcc_pooled:.3f} via exact {mcc_exact:.3f}", flush=True)
                tf.keras.backend.clear_session()
    pd.DataFrame(rows).to_csv(out_dir / "scaler_merge.csv", index=False)


def report(out_dir):
    r = pd.read_csv(out_dir / "scaler_merge.csv")
    cols = ["merged_rank_mean", "pooled_rank_mean", "merged_rank_p99", "pooled_rank_p99", "merged_rank_max", "pooled_rank_max",
            "merged_normal_mean", "pooled_normal_mean"]
    d_pool, d_exact = r.MCC_via_pooled - r.MCC_merged, r.MCC_via_exact - r.MCC_merged
    lines = ["Errors against the exact (full train split) scaler on the test split, mean over seeds:",
             r.groupby(["dataset", "partition"])[cols].mean().round(4).to_string(), "",
             f"Merged scaler more accurate than pooled (mean rank error) in {(r.merged_rank_mean < r.pooled_rank_mean).sum()}/{len(r)} runs",
             "Feature with the largest mean rank error of the merged scaler (runs):",
             r.merged_worst_feature_name.value_counts().head(5).to_string(), "",
             "Federated-scaler global model on the test split through another scaler, mean over seeds:",
             r.groupby(["dataset", "partition"])[["MCC_merged", "MCC_via_pooled", "MCC_via_exact",
                                                  "ROC_AUC_merged", "ROC_AUC_via_exact"]].mean().round(3).to_string(), "",
             f"MCC via pooled minus merged: mean {d_pool.mean():+.4f}, largest |diff| {d_pool.abs().max():.4f}; "
             f"via exact minus merged: mean {d_exact.mean():+.4f}, largest |diff| {d_exact.abs().max():.4f}"]
    (out_dir / "scaler_merge.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
