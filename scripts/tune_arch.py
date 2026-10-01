"""Architecture search for the v3 detectors (centralised, time-ordered splits, seed 0).

    python scripts/tune_arch.py --worker 0 --workers 2      # run trials i with i % 2 == 0
    python scripts/tune_arch.py --select                     # pick the best configuration per dataset

For every dataset and every family in fmml.architectures, a fixed, seeded random sample of
configurations (TRIALS per family) is trained on the training split, with features scaled by a
QuantileTransformer fit on it. Each trial trains up to MAX_EPOCHS (20 by default), evaluates the
validation MCC after every epoch, stops after 3 epochs without improvement and keeps the best
epoch. The choice uses the validation MCC only; the test metrics are recorded for the report,
never for choosing.

Run a single worker when the GPU would otherwise be shared: convolution and attention kernels
are autotuned when first compiled, and timing them next to another process can lock in far
slower ones.

Writes results/arch/<dataset>/trials_w<worker>.csv (one row per trial: configuration, parameter
count, epochs, seconds, validation MCC, test metrics); --select writes results/arch/<dataset>/
best.json (overall best) and best_per_family.csv. Trials already in the files are skipped, so an
interrupted search resumes.
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer

from fmml import config
from fmml.architectures import MAX_EPOCHS, SPACES, TRIALS, build, sample_configs
from fmml.data import load_xy
from fmml.metrics import binary_metrics

PATIENCE = 3
OUT = config.RESULTS_DIR / "arch"


def tasks():
    return [(d, fam, i, c) for d in config.DATASETS for fam in SPACES
            for i, c in enumerate(sample_configs(fam, TRIALS[fam]))]


def load(dataset):
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = (load_xy(config.split_path(dataset, s)) for s in ("train", "val", "test"))
    qt = QuantileTransformer(output_distribution="normal", random_state=0).fit(X_tr)
    f = lambda X: qt.transform(X).astype("float32")
    return (f(X_tr), y_tr.astype("float32")), (f(X_va), y_va), (f(X_te), y_te)


def run_trial(family, params, data, seed=0):
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build(family, params)
    # Explicit XLA: with jit_compile="auto", Keras sometimes leaves a model on the much slower path.
    model.compile(optimizer=tf.keras.optimizers.Adam(params["learning_rate"]), loss="binary_crossentropy",
                  jit_compile=True)
    predict = lambda X: model.predict(X, verbose=0, batch_size=8192).ravel()
    best, best_epoch, best_weights, history, seconds = -np.inf, 0, None, [], 0.0
    for epoch in range(1, MAX_EPOCHS.get(family, 20) + 1):
        start = time.perf_counter()
        model.fit(X_tr, y_tr, epochs=1, batch_size=params["batch_size"], verbose=0)
        seconds += time.perf_counter() - start
        val_mcc = binary_metrics(y_va, (predict(X_va) > 0.5).astype(int))["MCC"]
        history.append(round(val_mcc, 4))
        if val_mcc > best:
            best, best_epoch, best_weights = val_mcc, epoch, model.get_weights()
        elif epoch - best_epoch >= PATIENCE:
            break
    model.set_weights(best_weights)
    test = binary_metrics(y_te, (predict(X_te) > 0.5).astype(int))
    return {"val_mcc": best, "best_epoch": best_epoch, "epochs_run": len(history), "train_seconds": seconds,
            "seconds_per_epoch": seconds / len(history), "params_count": model.count_params(),
            "val_history": json.dumps(history), **{f"test_{k}": v for k, v in test.items()}}


def search(worker, workers):
    mine = [t for i, t in enumerate(tasks()) if i % workers == worker]
    data, loaded = None, None
    for dataset, family, trial, params in mine:
        out = OUT / dataset / f"trials_w{worker}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = [pd.read_csv(f) for f in (OUT / dataset).glob("trials_w*.csv")]         # any worker's trials
        if any(((d.family == family) & (d.trial == trial)).any() for d in done):
            continue
        if loaded != dataset:
            data, loaded = load(dataset), dataset
        row = {"dataset": dataset, "family": family, "trial": trial, "params": json.dumps(params),
               **run_trial(family, params, data)}
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"{dataset} {family} #{trial}: val MCC {row['val_mcc']:.4f} (epoch {row['best_epoch']}/{row['epochs_run']}, "
              f"{row['params_count']:,} params, {row['train_seconds']:.0f}s) | test MCC {row['test_MCC']:.4f}", flush=True)


def select():
    """Per dataset: the best configuration of every family (validation MCC). Across datasets: one
    family for all (the design used in the federated study), the one with the highest mean
    validation MCC over the datasets (ties: better mean rank); each dataset keeps that family's
    best configuration. Mean MCC rather than mean rank, because on a saturated dataset (UNSW-NB15,
    where families differ in the fourth decimal) ranks would weigh noise like real differences."""
    best_rows = []
    for dataset in config.DATASETS:
        files = sorted((OUT / dataset).glob("trials_w*.csv"))
        if not files:
            continue
        t = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        per_family = t.loc[t.groupby("family").val_mcc.idxmax()].sort_values("val_mcc", ascending=False)
        per_family.to_csv(OUT / dataset / "best_per_family.csv", index=False)
        best = per_family.iloc[0]
        (OUT / dataset / "best.json").write_text(json.dumps(
            {"family": best.family, "params": json.loads(best.params), "val_mcc": float(best.val_mcc),
             "best_epoch": int(best.best_epoch), "trials": int(len(t))}, indent=2))
        print(f"\n{dataset} ({len(t)} trials): best {best.family} val MCC {best.val_mcc:.4f}")
        print(per_family[["family", "val_mcc", "test_MCC", "best_epoch", "params_count", "seconds_per_epoch"]]
              .round(4).to_string(index=False))
        best_rows.append(per_family)
    if not best_rows:
        return
    table = pd.concat(best_rows).pivot(index="family", columns="dataset", values="val_mcc")
    complete = table.dropna()
    summary = pd.DataFrame({"mean_rank": complete.rank(ascending=False).mean(axis=1),
                            "mean_val_mcc": complete.mean(axis=1)}).join(complete)
    summary = summary.sort_values(["mean_val_mcc", "mean_rank"], ascending=[False, True])
    summary.to_csv(OUT / "family_summary.csv")
    print("\nFamilies across datasets (validation MCC of each family's best configuration):")
    print(summary.round(4).to_string())
    if len(complete.columns) == len(config.DATASETS):
        family = summary.index[0]
        chosen = {}
        for per_family in best_rows:
            row = per_family[per_family.family == family].iloc[0]
            chosen[row.dataset] = {"family": family, "params": json.loads(row.params), "val_mcc": float(row.val_mcc)}
        (OUT / "chosen_family.json").write_text(json.dumps({"family": family, "datasets": chosen}, indent=2))
        print(f"\nChosen family for all datasets: {family}")


RECHECK_FAMILIES = ["cnn1d", "ft_transformer", "mlp", "dcn"]
RECHECK_SEEDS = [1, 2, 3]


def recheck():
    """Retrain each dataset's best configuration of the leading families with more seeds (the
    seed-0 search left them within 0.003 validation MCC). Writes results/arch/<dataset>/recheck.csv."""
    for dataset in config.DATASETS:
        best = pd.read_csv(OUT / dataset / "best_per_family.csv").set_index("family")
        out = OUT / dataset / "recheck.csv"
        data = None
        for family in RECHECK_FAMILIES:
            for seed in RECHECK_SEEDS:
                if out.exists():
                    done = pd.read_csv(out)
                    if ((done.family == family) & (done.seed == seed)).any():
                        continue
                data = data or load(dataset)
                params = json.loads(best.loc[family, "params"])
                row = {"dataset": dataset, "family": family, "seed": seed, "params": json.dumps(params),
                       **run_trial(family, params, data, seed)}
                pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"{dataset} {family} seed {seed}: val MCC {row['val_mcc']:.4f} | test MCC {row['test_MCC']:.4f} "
                      f"({row['seconds_per_epoch']:.1f} s/epoch)", flush=True)


def decide():
    """The family for all datasets, by a rule fixed before the re-check was run:
      m(f, s) = mean over datasets of the validation MCC of family f's configuration with seed s
      (seed 0 from the search, seeds 1-3 from the re-check); M(f), SD(f) = mean and standard
      deviation of m(f, s) over the seeds. Leader = highest M. Tied = every family with
      M(f) >= M(leader) - SD(leader). Among the tied, the cheapest to train (sum over datasets of
      the mean seconds per epoch, measured with the GPU to itself) is chosen.
    Writes results/arch/chosen_family.json and recheck_summary.csv."""
    rows = []
    for dataset in config.DATASETS:
        best = pd.read_csv(OUT / dataset / "best_per_family.csv")
        best = best[best.family.isin(RECHECK_FAMILIES)].assign(seed=0)
        rows += [best, pd.read_csv(OUT / dataset / "recheck.csv")]
    t = pd.concat(rows, ignore_index=True)[["dataset", "family", "seed", "params", "val_mcc", "test_MCC", "seconds_per_epoch"]]
    per_seed = t.groupby(["family", "seed"]).val_mcc.mean().unstack()                  # m(f, s)
    summary = pd.DataFrame({"M": per_seed.mean(axis=1), "SD": per_seed.std(axis=1),
                            "cost_s_per_epoch": t[t.seed > 0].groupby(["family", "dataset"]).seconds_per_epoch.mean()
                            .groupby("family").sum()})
    summary = summary.join(t.groupby(["family", "dataset"]).val_mcc.mean().unstack())
    summary = summary.sort_values("M", ascending=False)
    leader = summary.index[0]
    summary["tied_with_leader"] = summary.M >= summary.M[leader] - summary.SD[leader]
    family = summary[summary.tied_with_leader].cost_s_per_epoch.idxmin()
    summary.to_csv(OUT / "recheck_summary.csv")
    print(summary.round(4).to_string())
    chosen = {}
    for dataset in config.DATASETS:
        b = pd.read_csv(OUT / dataset / "best_per_family.csv").set_index("family").loc[family]
        chosen[dataset] = {"family": family, "params": json.loads(b.params), "val_mcc_seed0": float(b.val_mcc)}
    (OUT / "chosen_family.json").write_text(json.dumps({"family": family, "leader": leader, "rule": decide.__doc__,
                                                        "datasets": chosen}, indent=2))
    print(f"\nleader {leader}; tied: {list(summary.index[summary.tied_with_leader])}; chosen (cheapest tied): {family}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--recheck", action="store_true", help="more seeds for the leading families")
    parser.add_argument("--decide", action="store_true", help="choose the family from search + re-check")
    args = parser.parse_args()
    if args.select:
        select()
    elif args.recheck:
        recheck()
    elif args.decide:
        decide()
    else:
        search(args.worker, args.workers)


if __name__ == "__main__":
    main()
