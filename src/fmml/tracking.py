"""Optional Weights & Biases mirror of the results (the files under results/ stay the source of truth).

Enabled with FMML_WANDB=1 (project: WANDB_PROJECT, default "fmml-ids"; entity: the logged-in
default). scripts/run_fl.py and scripts/train_central.py then log each finished run; scripts/wandb_sync.py
logs the runs already on disk and adds the later evaluation (ROC/PR-AUC, calibrated threshold).

A W&B run is built from one result directory only: config (settings, model, host), summary (the
run's final metrics and timings), history (per round, or per epoch when recorded) and tables
(per-client test metrics, partition, client updates). Each run carries its result path as
config.key (runs are found by it, not by W&B run id) and a fingerprint of its main result file: a
directory whose results changed (a rerun) replaces its W&B run; the later evaluation files only
update the summary.
"""
import hashlib
import json
import math
import os
import re
from pathlib import Path

import pandas as pd

from . import config

PROJECT = os.environ.get("WANDB_PROJECT", "fmml-ids")


def enabled() -> bool:
    return os.environ.get("FMML_WANDB", "").lower() in ("1", "true", "yes")


def run_key(run_dir: Path) -> str:
    """e.g. "main/UNSW-NB15/federated/iid/fedavg/seed0", "random/central/ToN-IoT/seed3"."""
    parts = Path(run_dir).resolve().relative_to(config.MAIN_RESULTS_DIR).parts
    if parts[0] in ("random", "refit"):
        return "/".join(parts)
    return "/".join(("main",) + parts)


def is_central(run_dir: Path) -> bool:
    return run_key(run_dir).split("/")[1] == "central"


def _main_file(run_dir: Path) -> Path:
    return run_dir / ("test_metrics.csv" if is_central(run_dir) else "summary.json")


def fingerprint(run_dir: Path) -> str:
    return hashlib.sha1(_main_file(run_dir).read_bytes()).hexdigest()[:12]


def _clean(d: dict) -> dict:
    return {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def post_metrics(run_dir: Path) -> dict:
    """Evaluation added after the run (scripts/eval_types.py, scripts/calibrate_threshold.py)."""
    out = {}
    if (run_dir / "auc.csv").exists():
        auc = pd.read_csv(run_dir / "auc.csv")
        for m in ("ROC_AUC", "PR_AUC"):
            out[f"test_{m}_mean"], out[f"test_{m}_worst"] = auc[m].mean(), auc[m].min()
    if (run_dir / "calibrated.json").exists():
        calibrated = json.loads((run_dir / "calibrated.json").read_text())
        out.update({f"calibrated_{k}".replace(".", "_"): v for k, v in calibrated.items()})     # W&B nests dotted keys
    return _clean(out)


def describe(run_dir: Path) -> dict:
    """The W&B run for a result directory: {key, group, job_type, config, summary, history, tables}."""
    run_dir = Path(run_dir)
    key = run_key(run_dir)
    experiment, rest = key.split("/", 1)
    if is_central(run_dir):
        _, dataset, seed = rest.split("/")
        row = _clean(pd.read_csv(run_dir / "test_metrics.csv").to_dict("records")[0])
        metrics = ["Accuracy", "Precision", "Recall", "F1", "FPR", "BalancedAccuracy", "MCC"]
        cfg = {"dataset": dataset, "setting": "central", "scaler": "central", "partition": "central",
               "strategy": "central", "seed": int(seed[4:]), "refit": experiment == "refit", "host": row.pop("host", None)}
        # One model: named like the federated runs' client mean (and worst) so both compare in one column.
        summary = {f"test_{m}_mean": row.pop(m) for m in metrics}
        summary.update(test_F1_worst=summary["test_F1_mean"], test_MCC_worst=summary["test_MCC_mean"])
        renamed = {"epoch": "best_epoch", "val_mcc": "val_mcc_best"}
        summary.update({renamed.get(k, k): v for k, v in row.items()})
        history = pd.read_csv(run_dir / "history.csv") if (run_dir / "history.csv").exists() else None
        tables = {}
        job_type = "central"
    else:
        dataset, scaler, partition, strategy_dir, seed = rest.split("/")
        s = json.loads((run_dir / "summary.json").read_text())
        alpha = re.fullmatch(r"dirichlet-a(.+)", partition)
        cfg = {"dataset": dataset, "setting": "federated", "scaler": scaler, "partition": partition,
               "alpha": float(alpha.group(1)) if alpha else None, "strategy": strategy_dir,
               "proximal_mu": s.get("proximal_mu"), "seed": int(seed[4:]), "refit": s.get("refit", False),
               "clients": s.get("clients"), "rounds": s.get("rounds"), "local_epochs": s.get("local_epochs"),
               "host": s.get("host")}
        skip = set(cfg) | {"dataset", "scaler", "partition", "strategy", "seed"}
        summary = {k: v for k, v in s.items() if k not in skip}
        history = pd.read_csv(run_dir / "history.csv")
        partition_stats = json.loads((run_dir / "partition.json").read_text())
        tables = {"test_metrics": pd.read_csv(run_dir / "test_metrics.csv"),
                  "partition": pd.DataFrame([{k: v for k, v in c.items() if k != "types"} for c in partition_stats]),
                  "client_types": pd.DataFrame([{"client": c["client"], **c["types"]} for c in partition_stats]).fillna(0),
                  "client_updates": pd.read_csv(run_dir / "client_updates.csv")}
        job_type = "federated"
    cfg.update(experiment=experiment, key=key, fingerprint=fingerprint(run_dir),
               model=config.MODELS[dataset]["family"], model_params=config.MODELS[dataset]["params"])
    summary.update(post_metrics(run_dir))
    return {"key": key, "group": key.rsplit("/", 1)[0], "job_type": job_type, "config": cfg,
            "summary": _clean(summary), "history": history, "tables": tables}


def existing_runs(api, keys=None) -> dict:
    """{config.key: [runs]} for the project's runs (all, or those with the given keys)."""
    import wandb
    filters = {"config.key": {"$in": list(keys)}} if keys is not None else {}
    try:
        runs = api.runs(f"{api.default_entity}/{PROJECT}", filters=filters, per_page=500, lazy=False)
        found = {}
        for r in runs:
            found.setdefault(r.config.get("key"), []).append(r)
        return found
    except (ValueError, wandb.errors.CommError):         # the project does not exist yet
        return {}


def create(run_dir: Path) -> str:
    """Logs the result directory as a new W&B run; returns its id."""
    import wandb
    d = describe(run_dir)
    wandb_dir = config.MAIN_RUNS_DIR                  # untracked; wandb writes its local files to runs/wandb/
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=PROJECT, name=d["key"], group=d["group"], job_type=d["job_type"],
                     tags=[d["config"]["experiment"], d["config"]["dataset"]], config=d["config"],
                     dir=str(wandb_dir), reinit="finish_previous", settings=wandb.Settings(silent=True))
    if d["history"] is not None:
        step = "round" if "round" in d["history"] else "epoch"
        run.define_metric(step)
        run.define_metric("*", step_metric=step)
        for row in d["history"].to_dict("records"):
            run.log({k: v for k, v in _clean(row).items() if v is not None})
    for name, table in d["tables"].items():
        run.log({name: wandb.Table(dataframe=table)})
    run.summary.update(d["summary"])
    run.finish()
    return run.id


def sync(run_dir: Path, existing: list | None = None, api=None) -> str:
    """Brings W&B in line with one result directory; returns "created", "replaced", "updated" or
    "unchanged". existing: the W&B runs with this directory's key (looked up when None)."""
    import wandb
    run_dir = Path(run_dir)
    api = api or wandb.Api()
    key, fp = run_key(run_dir), fingerprint(run_dir)
    if existing is None:
        existing = existing_runs(api, [key]).get(key, [])
    current = [r for r in existing if r.config.get("fingerprint") == fp]
    for r in existing:                               # older results of this directory (the run was redone)
        if r not in current:
            r.delete()
    if not current:
        create(run_dir)
        return "replaced" if existing else "created"
    run, post = current[0], post_metrics(run_dir)
    for r in current[1:]:
        r.delete()
    changed = {k: v for k, v in post.items() if not _same(run.summary.get(k), v)}
    if changed:
        run.summary.update(changed)
        return "updated"
    return "unchanged"


def _same(a, b) -> bool:
    """Equal, allowing for the last digit W&B's JSON round trip can change in a float."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12)
    return a == b


def maybe_sync(run_dir: Path):
    """For the training scripts: sync the finished run when FMML_WANDB is set; a W&B failure never
    fails the run."""
    if not enabled():
        return
    try:
        print(f"W&B: {sync(run_dir)} {run_key(run_dir)}")
    except Exception as e:                            # noqa: BLE001  (results are on disk; W&B can be synced later)
        print(f"W&B: sync failed for {run_dir} ({type(e).__name__}: {e}); run scripts/wandb_sync.py later")
