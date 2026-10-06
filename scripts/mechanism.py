"""Why the local quantile scaler fails: per-client encoding shift against worst-client damage.

    python scripts/mechanism.py                     # FedAvg, time split, all datasets and partitions
    python scripts/mechanism.py --report-only       # only the statistics, from the CSVs below

A local QuantileTransformer encodes a feature value as its rank within the client's own training
flows. Under attack-type skew the clients' marginals differ, so the same test flow is encoded
differently by every client, while the global model is shared. For every client of every
local-scaler run this measures that difference on the common test split, as the shift in rank
(uniform space, 0-1) from the pooled scaler (the scaler fit on all clients' training flows, i.e.
the train split, as for the centralised baseline):

  shift_mean   mean over features and test flows of |F_k(x) - F_pool(x)|
  shift_max    mean over features of the largest such shift on the test split (a KS-type distance)

and relates it to the client's test MCC / ROC-AUC, to the damage against the federated-scaler run
on the same partition (federated MCC minus the client's MCC), and to the client's attack-type
composition (total variation distance from the pooled composition). It also re-evaluates each
local-scaler global model on the test split through the pooled and the federated scaler: if the
model does well through a common encoding, the collapse comes from the clients' inconsistent
encodings at inference, not from a model that failed to learn.

Writes results/mechanism/clients.csv (one row per client and run), runs.csv (one row per run),
by_partition.csv (runs.csv averaged over seeds per dataset and partition) and report.txt (the
statistics: within-run rank correlations, how often the worst client is the most shifted one,
pooled correlations, and the paired gains from a common encoding).
"""
import argparse
import json
import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

import fmml.architectures  # noqa: F401  (registers custom layers for saved models)
from fmml import config
from fmml.data import load_scaler, load_split
from fmml.metrics import binary_metrics

PARTITIONS = ["iid", "dirichlet-a1", "dirichlet-a0.5", "dirichlet-a0.1"]


def uniform(scaler, X):
    """Rank encoding in (0, 1): the quantile scaler's normal output mapped back through the normal CDF."""
    return norm.cdf(scaler.transform(X))


def score(model, X, y):
    p = model.predict(X, verbose=0, batch_size=8192).ravel()
    return binary_metrics(y, (p > 0.5).astype(int))["MCC"], roc_auc_score(y, p)


def composition_tv(types):
    """Total variation distance of each client's attack-type mix (benign included) from the pooled mix."""
    mix = pd.DataFrame(types).fillna(0)                          # clients x types
    p = mix.div(mix.sum(axis=1), axis=0)
    pooled = mix.sum() / mix.values.sum()
    return 0.5 * (p - pooled).abs().sum(axis=1).to_numpy()


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
    clients, runs = [], []
    for dataset in args.datasets:
        X_te, y_te, _ = load_split(config.split_path(dataset, "test"))
        X_tr = load_split(config.split_path(dataset, "train"))[0]
        for tag in PARTITIONS:
            for run in sorted((config.RESULTS_DIR / dataset / "local" / tag / args.strategy).glob("seed*")):
                seed = int(run.name.removeprefix("seed"))
                fed_run = config.RESULTS_DIR / dataset / "federated" / tag / args.strategy / run.name
                assert (run / "partition.json").read_text() == (fed_run / "partition.json").read_text(), run
                test = pd.read_csv(run / "test_metrics.csv")
                auc = pd.read_csv(run / "auc.csv")
                fed_mcc = pd.read_csv(fed_run / "test_metrics.csv").MCC.iloc[0]
                stats = json.loads((run / "partition.json").read_text())

                pooled = QuantileTransformer(output_distribution="normal", random_state=seed).fit(X_tr)
                fed_dir = config.RUNS_DIR / "clients" / dataset / "federated" / tag / run.name
                federated = load_scaler(fed_dir, 0)
                client_dir = config.RUNS_DIR / "clients" / dataset / "local" / tag / run.name
                model = tf.keras.models.load_model(
                    config.RUNS_DIR / "models" / dataset / "local" / tag / args.strategy / f"{run.name}.keras")

                U_pool = uniform(pooled, X_te)
                tv = composition_tv([s["types"] for s in stats])
                for k, s in enumerate(stats):
                    scaler = load_scaler(client_dir, k)
                    shift = np.abs(uniform(scaler, X_te) - U_pool)
                    mcc_check, _ = score(model, scaler.transform(X_te), y_te)
                    assert abs(mcc_check - test.MCC[k]) < 1e-6, (run, k, mcc_check, test.MCC[k])
                    clients.append({"dataset": dataset, "partition": tag, "seed": seed, "client": k,
                                    "train_flows": s["train"], "attack_share": s["attack_share"],
                                    "attack_types": s["attack_types"], "composition_tv": tv[k],
                                    "shift_mean": shift.mean(), "shift_max": shift.max(axis=0).mean(),
                                    "MCC": test.MCC[k], "ROC_AUC": auc.ROC_AUC[k], "MCC_federated_run": fed_mcc,
                                    "damage": fed_mcc - test.MCC[k]})
                mcc_pool, auc_pool = score(model, pooled.transform(X_te), y_te)
                mcc_fed, auc_fed = score(model, federated.transform(X_te), y_te)
                runs.append({"dataset": dataset, "partition": tag, "seed": seed,
                             "MCC_mean": test.MCC.mean(), "MCC_worst": test.MCC.min(),
                             "ROC_AUC_worst": auc.ROC_AUC.min(), "MCC_federated_run": fed_mcc,
                             "MCC_via_pooled": mcc_pool, "ROC_AUC_via_pooled": auc_pool,
                             "MCC_via_federated": mcc_fed, "ROC_AUC_via_federated": auc_fed})
                print(f"{dataset} {tag} {run.name}: worst {test.MCC.min():.3f} mean {test.MCC.mean():.3f} | "
                      f"same model via pooled {mcc_pool:.3f}, via federated {mcc_fed:.3f} | fed run {fed_mcc:.3f}", flush=True)
                tf.keras.backend.clear_session()
    pd.DataFrame(clients).to_csv(out_dir / "clients.csv", index=False)
    pd.DataFrame(runs).to_csv(out_dir / "runs.csv", index=False)


def report(out_dir):
    """Statistics over clients.csv and runs.csv; written to report.txt and by_partition.csv."""
    c, r = pd.read_csv(out_dir / "clients.csv"), pd.read_csv(out_dir / "runs.csv")
    lines = []

    def rho(a, b):
        return spearmanr(a, b)[0] if a.nunique() > 1 and b.nunique() > 1 else np.nan

    within = []
    for (dataset, tag, seed), g in c.groupby(["dataset", "partition", "seed"]):
        within.append({"dataset": dataset, "partition": tag, "seed": seed,
                       "rho_shift_MCC": rho(g.shift_mean, g.MCC), "rho_tv_MCC": rho(g.composition_tv, g.MCC),
                       "worst_is_most_shifted": g.MCC.idxmin() == g.shift_mean.idxmax(),
                       "worst_in_top2_shifted": g.MCC.idxmin() in g.shift_mean.nlargest(2).index,
                       "MCC_spread": g.MCC.max() - g.MCC.min()})
    w = pd.DataFrame(within)
    nid, cn, rn = w[w.partition != "iid"], c[c.partition != "iid"], r[r.partition != "iid"]
    lines += ["Within a run (same model and test flows; only the client's scaler differs), mean over seeds:",
              w.groupby(["partition"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string(), "",
              f"Non-IID runs: {len(nid)}; Spearman(shift_mean, MCC) mean {nid.rho_shift_MCC.mean():.3f}, "
              f"negative in {(nid.rho_shift_MCC < 0).sum()}; worst client = most shifted in {nid.worst_is_most_shifted.sum()} "
              f"(chance {len(nid) / 10:.0f}), among the 2 most shifted in {nid.worst_in_top2_shifted.sum()} (chance {len(nid) / 5:.0f})", ""]
    lines.append("Pooled over the clients of non-IID runs, Spearman with damage / MCC / ROC-AUC:")
    for v in ["shift_mean", "shift_max", "composition_tv", "attack_share", "train_flows"]:
        lines.append(f"  {v:15} {rho(cn[v], cn.damage):6.3f} {rho(cn[v], cn.MCC):6.3f} {rho(cn[v], cn.ROC_AUC):6.3f}")
    lines += ["", "Rank shift (shift_mean) by dataset and partition, mean over clients and seeds:",
              c.pivot_table("shift_mean", "partition", "dataset").round(3).to_string(), ""]

    by_part = r.groupby(["dataset", "partition"]).mean(numeric_only=True).drop(columns="seed")
    by_part.round(4).to_csv(out_dir / "by_partition.csv")
    gain, rest = rn.MCC_via_pooled - rn.MCC_worst, rn.MCC_federated_run - rn.MCC_via_pooled
    lines += ["The local-scaler global model through a common encoding (mean over seeds):",
              by_part[["MCC_worst", "MCC_mean", "MCC_via_pooled", "MCC_via_federated", "MCC_federated_run",
                       "ROC_AUC_worst", "ROC_AUC_via_pooled"]].round(3).to_string(), "",
              f"Non-IID runs: via pooled > worst client in {(gain > 0).sum()}/{len(rn)} (mean gain {gain.mean():.3f}, "
              f"median {gain.median():.3f}); via pooled > client mean in {(rn.MCC_via_pooled > rn.MCC_mean).sum()}/{len(rn)}; "
              f"federated-scaler run minus via pooled: mean {rest.mean():.3f}, median {rest.median():.3f}"]
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
