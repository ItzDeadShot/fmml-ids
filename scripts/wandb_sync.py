"""Mirror the finished runs under results/ to Weights & Biases (see fmml.tracking).

    python scripts/wandb_sync.py                          # every experiment (main, random, refit)
    python scripts/wandb_sync.py --experiments random --dry-run

Logs the runs not yet in W&B, replaces those whose results changed (a rerun), and adds the later
evaluation (ROC/PR-AUC, calibrated threshold) to the others. Runs in progress (a .running lock or no
result file yet) are left out. Safe to repeat. With --prune, W&B runs whose result directory no
longer exists are deleted.
"""
import argparse
from collections import Counter
from pathlib import Path

from fmml import config, tracking

EXPERIMENTS = {"main": config.MAIN_RESULTS_DIR, "random": config.MAIN_RESULTS_DIR / "random",
               "refit": config.MAIN_RESULTS_DIR / "refit"}


def result_dirs(root: Path):
    for f in sorted(root.glob("*/*/*/*/seed*/summary.json")) + sorted(root.glob("central/*/seed*/test_metrics.csv")):
        yield f.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--dry-run", action="store_true", help="only report what would change")
    parser.add_argument("--prune", action="store_true", help="delete W&B runs whose result directory is gone")
    args = parser.parse_args()

    import wandb
    api = wandb.Api()
    existing = tracking.existing_runs(api)
    print(f"W&B project {api.default_entity}/{tracking.PROJECT}: {sum(map(len, existing.values()))} runs")
    counts, keys = Counter(), set()
    for experiment in args.experiments:
        dirs = list(result_dirs(EXPERIMENTS[experiment]))
        for i, run_dir in enumerate(dirs, 1):
            key = tracking.run_key(run_dir)
            keys.add(key)
            if (run_dir / ".running").exists():     # being redone
                continue
            runs = existing.get(key, [])
            if args.dry_run:
                fp = tracking.fingerprint(run_dir)
                action = "create" if not runs else "keep" if any(r.config.get("fingerprint") == fp for r in runs) else "replace"
            else:
                try:
                    action = tracking.sync(run_dir, runs, api)
                except Exception as e:             # noqa: BLE001  (report and go on; a later sync retries)
                    action = "failed"
                    print(f"  {key}: {type(e).__name__}: {e}")
            counts[action] += 1
            if action not in ("unchanged", "keep"):
                print(f"[{experiment} {i}/{len(dirs)}] {action} {key}", flush=True)
    if args.prune:
        prefixes = tuple(f"{e}/" for e in args.experiments)
        stale = [r for k, rs in existing.items() if k and k.startswith(prefixes) and k not in keys for r in rs]
        for r in stale:
            print(f"{'would delete' if args.dry_run else 'delete'} {r.config.get('key')} ({r.id})")
            if not args.dry_run:
                r.delete()
        counts["pruned"] = len(stale)
    print(dict(counts))


if __name__ == "__main__":
    main()
