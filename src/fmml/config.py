"""Paths, datasets and per-dataset model settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Experiment variant, from the environment (FMML_EXPERIMENT):
#   ""        main study: time-ordered splits (data/split-data), results/ and runs/
#   "random"  the same flows split at random, stratified by attack type (data/split-random),
#             for comparison with the usual random-split evaluation: results/random/, runs/random/
#   "refit"   time-ordered splits; final models retrained on train+val with the settings chosen
#             in the main study (results/refit/, runs/refit/)
EXPERIMENT = os.environ.get("FMML_EXPERIMENT", "")
assert EXPERIMENT in ("", "random", "refit"), EXPERIMENT
MAIN_RESULTS_DIR = ROOT / "results"
MAIN_RUNS_DIR = ROOT / "runs"
SPLIT_DIR = ROOT / "data" / ("split-random" if EXPERIMENT == "random" else "split-data")
RUNS_DIR = MAIN_RUNS_DIR / EXPERIMENT if EXPERIMENT else MAIN_RUNS_DIR      # untracked: client data, models
RESULTS_DIR = MAIN_RESULTS_DIR / EXPERIMENT if EXPERIMENT else MAIN_RESULTS_DIR

LABEL_COL = "label"      # 1 = any attack, 0 = benign
ATTACK_COL = "attack"    # attack type ("Benign" for benign flows)
N_FEATURES = 45

# Dataset name -> split file stem (built from NetFlow v3 by scripts/build_dataset.py).
DATASETS = {
    "UNSW-NB15": "UNSW-NB15",
    "BoT-IoT": "BoT-IoT",
    "ToN-IoT": "ToN-IoT",
    "CIC-2018": "CIC-2018",
}

# Model per dataset: one family for all datasets, chosen by the v3 architecture search
# (scripts/tune_arch.py: seeded random search per family on the time-ordered validation split, then
# a 4-seed re-check of the leading families; rule and results in results/arch/chosen_family.json).
# Each dataset keeps its own best configuration of that family.
MODELS = {
    "UNSW-NB15": {"family": "ft_transformer", "params": {"learning_rate": 0.0003, "batch_size": 512, "d_token": 32, "blocks": 2, "heads": 4, "dropout": 0.0},
                  "learning_rate": 0.0003, "batch_size": 512},
    "BoT-IoT": {"family": "ft_transformer", "params": {"learning_rate": 0.001, "batch_size": 1024, "d_token": 32, "blocks": 1, "heads": 4, "dropout": 0.0},
                  "learning_rate": 0.001, "batch_size": 1024},
    "ToN-IoT": {"family": "ft_transformer", "params": {"learning_rate": 0.001, "batch_size": 1024, "d_token": 32, "blocks": 1, "heads": 4, "dropout": 0.0},
                  "learning_rate": 0.001, "batch_size": 1024},
    "CIC-2018": {"family": "ft_transformer", "params": {"learning_rate": 0.001, "batch_size": 1024, "d_token": 64, "blocks": 3, "heads": 4, "dropout": 0.0},
                  "learning_rate": 0.001, "batch_size": 1024},
}

# The models of the first v3 runs: the centralised specialists of the companion HDSE work (built
# for NetFlow v2), inputs widened to the 45 v3 features. Kept to reproduce the archived results.
V2_MODELS = {
    "UNSW-NB15": {"arch": "DNN", "learning_rate": 0.001, "batch_size": 128},
    "BoT-IoT": {"arch": "MLP", "learning_rate": 0.001, "batch_size": 128},
    "ToN-IoT": {"arch": "LSTM", "learning_rate": 0.01, "batch_size": 128},
    "CIC-2018": {"arch": "GRU", "learning_rate": 0.001, "batch_size": 128},
}

MIN_CLIENT_SIZE = 50            # Dirichlet partitions are redrawn until every client has this many flows


def split_path(dataset: str, split: str) -> Path:
    return SPLIT_DIR / f"{DATASETS[dataset]}_{split}.csv"
