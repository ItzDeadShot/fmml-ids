"""Run a whole experiment plan: centralised baseline, FedProx mu selection, the federated grid, and
the evaluation, with several jobs at a time.

    python scripts/run_queue.py --plans main random --parallel 2

Plans (each in its FMML_EXPERIMENT; see fmml.config):
  main    time-ordered splits: centralised seeds; mu selection for both scalers; FedAvg and FedProx
          (chosen mu) x {IID, Dirichlet 1, 0.5, 0.1} x both scalers x seeds
  random  random splits: centralised seeds; mu selection (federated scaler); FedAvg and FedProx x
          4 partitions (federated scaler) and FedAvg x {IID, 0.5, 0.1} (local scaler) x seeds
  refit   time-ordered splits, train+val refit of the main plan's runs (federated scaler)
Parts of the main plan run only on request (not in main:all):
  main:scalers   reference scalers with FedAvg x 4 partitions x seeds: pooled (quantile scaler fit on
                 all clients' flows), zscore-local, zscore-federated
  main:scaffold  SCAFFOLD (local SGD) x 4 partitions x seeds, local and federated quantile scaler,
                 after choosing the local learning rate per dataset and scaler (as FedProx's mu)

Each phase finishes before the next starts (mu must be chosen before the runs that use it); within
a phase up to --parallel jobs run at once, each simulation in its own Ray instance. Jobs whose
output exists are skipped, so the queue resumes after an interruption; a failed job is retried
twice, then reported and left for a later run. Per-job output: runs/logs/jobs/<plan>/<job>.log.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
DATASETS = ["UNSW-NB15", "BoT-IoT", "ToN-IoT", "CIC-2018"]
PARTITIONS = {"iid": ["--partition", "iid"], "a1": ["--partition", "dirichlet", "--alpha", "1.0"],
              "a0.5": ["--partition", "dirichlet", "--alpha", "0.5"], "a0.1": ["--partition", "dirichlet", "--alpha", "0.1"]}
TAG = {"iid": "iid", "a1": "dirichlet-a1", "a0.5": "dirichlet-a0.5", "a0.1": "dirichlet-a0.1"}
MUS = [0.001, 0.01, 0.1]
LRS = [0.01, 0.03, 0.1, 0.3]                           # SCAFFOLD local SGD learning rates
RESULTS = {"main": ROOT / "results", "random": ROOT / "results" / "random", "refit": ROOT / "results" / "refit"}
EXPERIMENT = {"main": "", "random": "random", "refit": "refit"}


# Memory weight of a job's simulation (measured peaks with FT-Transformer, 15.5 GB RAM): CIC ~7.5 GB,
# UNSW ~5-6 GB, BoT/ToN less. Jobs run together only while their weights add up to at most MAX_WEIGHT:
# CIC with BoT/ToN and UNSW with UNSW are allowed; CIC + UNSW reached 94% of RAM with local scalers.
WEIGHT = {"CIC-2018": 2, "UNSW-NB15": 1}
MAX_WEIGHT = 2


class Job:
    def __init__(self, plan, name, args, done, gpu=True, lock=None, weight=0):
        self.plan, self.name, self.args, self.done, self.gpu = plan, name, args, done, gpu
        self.lock, self.weight = lock, weight              # lock: callable -> Path of run_fl's .running file
        self.attempts = 0

    def is_running_elsewhere(self):
        """True while another process (e.g. one left from an earlier dispatcher) runs this job."""
        try:
            f = self.lock() if self.lock else None
            if f is None or not f.exists():
                return False
            os.kill(int(f.read_text()), 0)
            return True
        except (OSError, ValueError, KeyError):
            return False

    def is_done(self):
        try:
            return self.done is not None and self.done()
        except (OSError, KeyError, ValueError):                # e.g. the chosen mu is not there yet
            return False

    def command(self):
        return [PY] + (self.args() if callable(self.args) else self.args)


def fl_job(plan, dataset, scaler, part, strategy, seed, mu=None, refit=False, lr=None):
    """strategy: fedavg, fedprox or scaffold. mu (FedProx) and lr (SCAFFOLD): a number, or None for
    the value chosen for (plan, dataset, scaler), read when the job starts."""
    res = RESULTS[plan]

    def chosen_lr():
        return json.loads((res / dataset / scaler / "scaffold_lr.json").read_text())["local_lr"]

    def chosen_mu():
        f = res / dataset / scaler / "fedprox_mu.json"
        if not f.exists():                                     # refit: the main study's choice
            f = RESULTS["main"] / dataset / scaler / "fedprox_mu.json"
        return json.loads(f.read_text())["proximal_mu"]

    def args():
        a = ["scripts/run_fl.py", "--datasets", dataset, "--scaler", scaler, *PARTITIONS[part], "--seed", str(seed),
             "--skip-existing", "--gpus-per-client", str(GPUS_PER_CLIENT)]
        if strategy == "fedprox":
            a += ["--strategy", "fedprox", "--mu", f"{mu if mu is not None else chosen_mu():g}"]
        elif strategy == "scaffold":
            a += ["--strategy", "scaffold", "--lr", f"{lr if lr is not None else chosen_lr():g}"]
        else:
            a += ["--strategy", strategy]
        return a + (["--refit"] if refit else [])

    def run_dir():
        s = {"fedprox": lambda: f"fedprox-mu{mu if mu is not None else chosen_mu():g}",
             "scaffold": lambda: f"scaffold-lr{lr if lr is not None else chosen_lr():g}"}.get(strategy, lambda: strategy)()
        return res / dataset / scaler / TAG[part] / s / f"seed{seed}"

    name = f"{dataset}.{scaler}.{part}.{strategy}{'' if mu is None else f'-mu{mu:g}'}{'' if lr is None else f'-lr{lr:g}'}.seed{seed}"
    return Job(plan, name, args, lambda: (run_dir() / "summary.json").exists(),
               lock=lambda: run_dir() / ".running", weight=WEIGHT.get(dataset, 0))


def central_jobs(plan, seeds, refit=False):
    res = RESULTS[plan] / "central"
    return [Job(plan, f"central.{d}", ["scripts/train_central.py", "--datasets", d, "--seeds", *map(str, seeds)]
                + (["--refit"] if refit else []),
                lambda d=d: all((res / d / f"seed{s}" / "test_metrics.csv").exists() for s in seeds)) for d in DATASETS]


def mu_phase(plan, scaler):
    return [fl_job(plan, d, scaler, "a0.5", "fedprox", 0, mu=mu) for mu in MUS for d in DATASETS]


def lr_phase(plan):
    return [fl_job(plan, d, sc, "a0.5", "scaffold", 0, lr=lr) for lr in LRS for d in DATASETS for sc in ("local", "federated")]


def choose_lr_job(plan, scaler):
    res = RESULTS[plan]
    return Job(plan, f"choose_lr.{scaler}", ["scripts/choose_lr.py", "--scaler", scaler, "--lrs", *map(str, LRS)],
               lambda: all((res / d / scaler / "scaffold_lr.json").exists() for d in DATASETS), gpu=False)


def choose_job(plan, scaler):
    res = RESULTS[plan]
    return Job(plan, f"choose_mu.{scaler}", ["scripts/choose_mu.py", "--scaler", scaler],
               lambda: all((res / d / scaler / "fedprox_mu.json").exists() for d in DATASETS), gpu=False)


def post_jobs(plan):
    """Evaluation steps over all of the plan's runs. Each counts as done while its marker is newer than
    the plan's newest result, so a restarted queue does not repeat them, but new runs trigger them."""
    res = RESULTS[plan]

    def newest_result():
        files = list(res.glob("*/*/*/*/seed*/summary.json")) + list(res.glob("central/*/seed*/test_metrics.csv"))
        return max((f.stat().st_mtime for f in files), default=0)

    def job(name, script, *extra):
        marker = res / f".done_{name}"
        j = Job(plan, name, [script, *extra], lambda: marker.exists() and marker.stat().st_mtime >= newest_result(),
                gpu=POST_GPU)
        j.marker = marker
        return j
    jobs = [job("eval_types", "scripts/eval_types.py"), job("calibrate", "scripts/calibrate_threshold.py"),
            job("summarize", "scripts/summarize.py")]
    if os.environ.get("FMML_WANDB", "").lower() in ("1", "true", "yes"):     # W&B mirror (fmml.tracking)
        jobs.append(job("wandb_sync", "scripts/wandb_sync.py", "--experiments", plan))
    return jobs


def plan_phases(plan, seeds, part="all"):
    """The plan as a list of phases (lists of jobs). part: "all", or "fed" / "local" for only the
    federated-scaler or only the local-scaler runs (so the most important block can finish first)."""
    if plan == "main" and part == "scalers":
        grid = [fl_job(plan, d, sc, p, "fedavg", s) for s in seeds for d in DATASETS for p in PARTITIONS
                for sc in ("pooled", "zscore-federated", "zscore-local")]
        return [grid, *[[j] for j in post_jobs(plan)]]
    if plan == "main" and part == "scaffold":
        grid = [fl_job(plan, d, sc, p, "scaffold", s) for s in seeds for d in DATASETS for p in PARTITIONS
                for sc in ("local", "federated")]
        return [lr_phase(plan), [choose_lr_job(plan, "local"), choose_lr_job(plan, "federated")], grid,
                *[[j] for j in post_jobs(plan)]]
    if part in ("fed", "local"):
        scaler = "federated" if part == "fed" else "local"
        mu = [] if (plan == "random" and part == "local") else [mu_phase(plan, scaler), [choose_job(plan, scaler)]]
        if plan == "main":
            grid = [fl_job(plan, d, scaler, p, st, s) for s in seeds for d in DATASETS for p in PARTITIONS
                    for st in ("fedavg", "fedprox")]
        elif plan == "random" and part == "fed":
            grid = [fl_job(plan, d, scaler, p, st, s) for s in seeds for d in DATASETS for p in PARTITIONS
                    for st in ("fedavg", "fedprox")]
        elif plan == "random":
            grid = [fl_job(plan, d, scaler, p, "fedavg", s) for s in seeds for d in DATASETS for p in ("iid", "a0.5", "a0.1")]
        else:
            raise ValueError(f"{plan}:{part}")
        return [central_jobs(plan, seeds), *mu, grid, *[[j] for j in post_jobs(plan)]]
    grid = lambda scalers, strategies, parts, refit=False: [
        fl_job(plan, d, sc, p, st, s, refit=refit) for s in seeds for d in DATASETS for p in parts
        for sc in scalers for st in strategies]
    if plan == "main":
        return [central_jobs(plan, seeds), mu_phase(plan, "federated") + mu_phase(plan, "local"),
                [choose_job(plan, "federated"), choose_job(plan, "local")],
                grid(["federated", "local"], ["fedavg", "fedprox"], list(PARTITIONS)), *[[j] for j in post_jobs(plan)]]
    if plan == "random":
        return [central_jobs(plan, seeds), mu_phase(plan, "federated"), [choose_job(plan, "federated")],
                grid(["federated"], ["fedavg", "fedprox"], list(PARTITIONS)) + grid(["local"], ["fedavg"], ["iid", "a0.5", "a0.1"]),
                *[[j] for j in post_jobs(plan)]]
    if plan == "refit":
        return [central_jobs(plan, seeds, refit=True),
                grid(["federated"], ["fedavg", "fedprox"], list(PARTITIONS), refit=True), *[[j] for j in post_jobs(plan)]]
    raise ValueError(plan)


def run_phase(jobs, parallel, log_dir):
    pending = [j for j in jobs if not j.is_done()]
    running, failed = {}, []
    while pending or running:
        while pending:
            elsewhere = [j for j in pending if j.is_running_elsewhere()]      # e.g. left by an earlier dispatcher
            if len(running) + len(elsewhere) >= parallel:
                break
            load = sum(j.weight for j, _, _ in running.values()) + sum(j.weight for j in elsewhere)
            # The heaviest job that fits (earliest first among equals): heavy jobs pair up throughout the
            # phase instead of light ones running ahead and leaving a tail of heavy jobs run one at a time.
            fits = [j for j in pending if not j.is_running_elsewhere() and load + j.weight <= MAX_WEIGHT]
            pick = max(fits, key=lambda j: j.weight) if fits else None
            if pick is None:                                  # all waiting: another heavy job, or runs elsewhere
                break
            pending.remove(pick)
            job = pick
            if job.is_done():
                continue
            job.attempts += 1
            try:
                command = job.command()
            except (OSError, KeyError, ValueError) as e:
                print(f"{time.strftime('%H:%M:%S')} FAILED {job.plan}/{job.name}: cannot build command ({e})", flush=True)
                failed.append(job)
                continue
            env = {**os.environ, "FMML_EXPERIMENT": EXPERIMENT[job.plan], "PYTHONUNBUFFERED": "1",
                   "RAY_ADDRESS": "local", "TF_CPP_MIN_LOG_LEVEL": "2"}
            if not job.gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            else:
                # No XLA kernel autotuning: timed next to other jobs on the same GPU it can lock in far
                # slower kernels; for the FT-Transformer the default kernels are as fast.
                env["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
                # Ray's memory monitor kills workers above 95% of RAM by default; the CIC + UNSW peak is
                # ~92%, so allow up to 98% (swap takes the rest).
                env["RAY_memory_usage_threshold"] = "0.98"
            (log_dir / job.plan).mkdir(parents=True, exist_ok=True)
            log = open(log_dir / job.plan / f"{job.name}.log", "a")
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[proc] = (job, time.time(), log)
            print(f"{time.strftime('%H:%M:%S')} start  {job.plan}/{job.name} (attempt {job.attempts})", flush=True)
        if not running and pending and all(j.is_running_elsewhere() for j in pending):
            time.sleep(20)                                      # only jobs still running in other processes
        time.sleep(2)
        for proc in [p for p in running if p.poll() is not None]:
            job, start, log = running.pop(proc)
            log.close()
            minutes = (time.time() - start) / 60
            if proc.returncode == 0 and getattr(job, "marker", None) is not None:
                job.marker.touch()
            if proc.returncode == 0 and (job.done is None or job.is_done()):
                print(f"{time.strftime('%H:%M:%S')} done   {job.plan}/{job.name} ({minutes:.1f} min)", flush=True)
            elif job.attempts < 3:
                print(f"{time.strftime('%H:%M:%S')} FAILED {job.plan}/{job.name} (exit {proc.returncode}), retrying", flush=True)
                pending.append(job)
            else:
                print(f"{time.strftime('%H:%M:%S')} FAILED {job.plan}/{job.name} (exit {proc.returncode}), giving up", flush=True)
                failed.append(job)
    return failed


def main():
    global GPUS_PER_CLIENT, POST_GPU, MAX_WEIGHT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plans", nargs="+", default=["main"],
                        help="plan[:part], plan in main/random/refit, part in all/fed/local, or for main also "
                             "scalers/scaffold (e.g. main:fed random:fed main:local main:scalers)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--parallel", type=int, default=2, help="simulations (or training jobs) at a time")
    parser.add_argument("--gpus-per-client", type=float, default=0.5,
                        help="GPU share per simulated client within one simulation")
    parser.add_argument("--post-gpu", action="store_true",
                        help="run the evaluation steps on the GPU (they run alone in their phase); for hosts with a slow CPU")
    parser.add_argument("--max-weight", type=int, default=MAX_WEIGHT,
                        help="memory budget for jobs running together, in WEIGHT units (2 for ~16 GB RAM)")
    parser.add_argument("--dry-run", action="store_true", help="list the jobs still to run")
    args = parser.parse_args()
    GPUS_PER_CLIENT, POST_GPU, MAX_WEIGHT = args.gpus_per_client, args.post_gpu, args.max_weight
    log_dir = ROOT / "runs" / "logs" / "jobs"
    failed = []
    for spec in args.plans:
        plan, _, part = spec.partition(":")
        for i, phase in enumerate(plan_phases(plan, args.seeds, part or "all")):
            if args.dry_run:
                todo = [j.name for j in phase if not j.is_done()]
                print(f"{spec} phase {i}: {len(todo)}/{len(phase)} to run" + (f", e.g. {todo[:3]}" if todo else ""))
                continue
            print(f"{time.strftime('%H:%M:%S')} == {spec} phase {i} ({len(phase)} jobs)", flush=True)
            failed += run_phase(phase, args.parallel, log_dir)
    print(f"{time.strftime('%H:%M:%S')} == finished; {len(failed)} failed: {[f'{j.plan}/{j.name}' for j in failed]}", flush=True)
    sys.exit(1 if failed else 0)


GPUS_PER_CLIENT = 0.5
POST_GPU = False

if __name__ == "__main__":
    main()
