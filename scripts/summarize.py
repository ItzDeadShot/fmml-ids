"""Aggregate the runs over seeds: the centralised baseline and every federated configuration.

    python scripts/summarize.py

Writes
  results/summary.csv        one row per dataset x scaler x partition x strategy (FedProx at the mu
                             chosen for that dataset and scaler, results/<dataset>/<scaler>/fedprox_mu.json): mean and std over seeds of
                             the test metrics (for federated runs: the mean over clients, and the
                             worst client's F1 and MCC), the chosen round (centralised: epoch) and
                             the number of seeds;
  (ROC_AUC and PR_AUC come from scripts/eval_types.py, MCC_cal/MCC_cal_worst/FPR_cal/threshold from
  scripts/calibrate_threshold.py; empty until those have run)
  results/mu_selection.csv   the FedProx mu candidates (seed 0, Dirichlet 0.5), for reference;
  results/type_summary.csv   detection rate per attack type (mean over clients, then over seeds),
                             from scripts/eval_types.py, with the worst client's rate;
and prints the main table.
"""
import json

import pandas as pd

from fmml import config

METRICS = ["F1", "F1_worst", "MCC", "MCC_worst", "Precision", "Recall", "FPR", "BalancedAccuracy", "ROC_AUC",
           "PR_AUC", "best_round", "MCC_cal", "MCC_cal_worst", "FPR_cal", "threshold"]
# Timings (runs recorded before the timing instrumentation have only the total minutes). Federated:
# simulation wall clock, median round (rounds 2+), mean client local-training call, total bytes
# exchanged; centralised: time spent in the training epochs.
TIMING = ["minutes", "simulation_seconds", "round_seconds_median", "client_fit_seconds_mean",
          "communication_bytes_total", "train_seconds"]


def calibrated(run):
    """Test metrics at the validation-calibrated threshold (scripts/calibrate_threshold.py), if computed."""
    f = run / "calibrated.json"
    if not f.exists():
        return {"MCC_cal": None, "MCC_cal_worst": None, "FPR_cal": None, "threshold": None}
    c = json.loads(f.read_text())
    return {"MCC_cal": c["test_MCC_mean"], "MCC_cal_worst": c["test_MCC_worst"], "FPR_cal": c["test_FPR_mean"],
            "threshold": c["threshold"]}


def auc(run):
    """Mean over clients of the threshold-free scores from scripts/eval_types.py (if computed)."""
    f = run / "auc.csv"
    if not f.exists():
        return {"ROC_AUC": None, "PR_AUC": None}
    t = pd.read_csv(f)
    return {"ROC_AUC": t.ROC_AUC.mean(), "PR_AUC": t.PR_AUC.mean()}
BASIC = ["Precision", "Recall", "FPR", "BalancedAccuracy"]
PARTITIONS = ["central", "iid", "dirichlet-a1", "dirichlet-a0.5", "dirichlet-a0.1"]


def load_runs():
    rows = []
    for dataset in config.DATASETS:
        for f in sorted((config.RESULTS_DIR / "central" / dataset).glob("seed*/test_metrics.csv")):
            m = pd.read_csv(f).iloc[0]
            rows.append({"dataset": dataset, "scaler": "central", "partition": "central", "strategy": "central",
                         "seed": int(f.parent.name[4:]), **{k: m[k] for k in BASIC}, "F1": m.F1, "F1_worst": m.F1,
                         "MCC": m.MCC, "MCC_worst": m.MCC, **auc(f.parent), **calibrated(f.parent), "best_round": m.epoch,
                         "train_seconds": m.get("train_seconds")})
        for f in sorted((config.RESULTS_DIR / dataset).glob("*/*/*/seed*/summary.json")):
            s = json.loads(f.read_text())
            rows.append({"dataset": dataset, "scaler": f.parents[3].name, "partition": f.parents[2].name,
                         "strategy": f.parents[1].name, "seed": s["seed"], "F1": s["test_F1_mean"],
                         "F1_worst": s["test_F1_worst"], "MCC": s["test_MCC_mean"], "MCC_worst": s["test_MCC_worst"],
                         **{k: s[f"test_{k}_mean"] for k in BASIC}, **auc(f.parent), **calibrated(f.parent), "best_round": s["best_round"],
                         **{k: s.get(k) for k in TIMING if k != "train_seconds"}})
    return pd.DataFrame(rows)


def load_types():
    frames = []
    files = [f for d in list(config.DATASETS) + ["central"] for f in (config.RESULTS_DIR / d).glob("**/seed*/type_metrics.csv")]
    for f in files:
        run = f.parent
        if run.parents[1].name == "central":
            dataset, scaler, partition, strategy = run.parent.name, "central", "central", "central"
        else:
            dataset, scaler, partition, strategy = (run.parents[3].name, run.parents[2].name, run.parents[1].name,
                                                    run.parent.name)
        t = pd.read_csv(f)
        per_seed = t.groupby("attack").predicted_attack.agg(rate="mean", rate_worst=lambda r: r.min()).reset_index()
        frames.append(per_seed.assign(dataset=dataset, scaler=scaler, partition=partition, strategy=strategy, seed=run.name))
    return pd.concat(frames) if frames else pd.DataFrame()


def chosen_mu_only(df):
    """Drop FedProx runs at a mu other than the one chosen for their dataset and scaler."""
    keep = []
    for (dataset, scaler), group in df.groupby(["dataset", "scaler"]):
        f = config.RESULTS_DIR / dataset / scaler / "fedprox_mu.json"
        if not f.exists():                             # refit experiment: the main study's choice
            f = config.MAIN_RESULTS_DIR / dataset / scaler / "fedprox_mu.json"
        chosen = f"fedprox-mu{json.loads(f.read_text())['proximal_mu']:g}" if f.exists() else None
        keep.append(group[~group.strategy.str.startswith("fedprox") | (group.strategy == chosen)])
    return pd.concat(keep)


def order(df):
    df = df.copy()
    df["_d"] = df.dataset.map({d: i for i, d in enumerate(config.DATASETS)})
    df["_p"] = df.partition.map({p: i for i, p in enumerate(PARTITIONS)})
    return df.sort_values(["_d", "scaler", "_p", "strategy"]).drop(columns=["_d", "_p"])


def main():
    all_runs = load_runs()
    keys = ["dataset", "scaler", "partition", "strategy"]
    mu_runs = all_runs[all_runs.strategy.str.startswith("fedprox") & (all_runs.partition == "dirichlet-a0.5")
                       & (all_runs.seed == 0)]
    order(mu_runs)[keys + ["MCC", "MCC_worst", "FPR", "best_round"]].to_csv(
        config.RESULTS_DIR / "mu_selection.csv", index=False)
    runs = chosen_mu_only(all_runs)
    runs = runs.reindex(columns=list(runs.columns) + [c for c in TIMING if c not in runs.columns])
    runs[TIMING] = runs[TIMING].astype(float)
    agg = runs.groupby(keys)[METRICS + TIMING].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.join(runs.groupby(keys).seed.nunique().rename("seeds")).reset_index()
    agg = order(agg)
    agg.to_csv(config.RESULTS_DIR / "summary.csv", index=False)

    types = load_types()
    if len(types):
        types = chosen_mu_only(types)
        t = types.groupby(keys + ["attack"]).agg(rate=("rate", "mean"), rate_std=("rate", "std"),
                                                 rate_worst_client=("rate_worst", "mean"),
                                                 seeds=("seed", "nunique")).reset_index()
        order(t).to_csv(config.RESULTS_DIR / "type_summary.csv", index=False)

    def pm(c):
        text = agg[f"{c}_mean"].map("{:.4f}".format) + " ± " + agg[f"{c}_std"].fillna(0).map("{:.4f}".format)
        return text.where(agg[f"{c}_mean"].notna(), "")

    table = agg.assign(**{c: pm(c) for c in ["F1", "MCC", "MCC_worst", "BalancedAccuracy", "FPR", "ROC_AUC", "MCC_cal"]},
                       best_round=agg.best_round_mean.map(lambda r: "" if pd.isna(r) else f"{r:.1f}"))
    print(table[keys + ["MCC", "MCC_worst", "MCC_cal", "BalancedAccuracy", "FPR", "ROC_AUC", "F1", "best_round", "seeds"]]
          .rename(columns={"BalancedAccuracy": "BalAcc"}).to_string(index=False))
    timed = agg[agg[[f"{c}_mean" for c in TIMING[1:]]].notna().any(axis=1)]
    if len(timed):
        print("\nTimings (mean over seeds with timings)")
        print(timed[keys + [f"{c}_mean" for c in TIMING]].rename(columns=lambda c: c.removesuffix("_mean"))
              .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
