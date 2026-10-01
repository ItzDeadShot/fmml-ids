# FMML-IDS: Federated learning of NetFlow intrusion detectors under non-IID clients

Federated training of binary (benign vs attack) intrusion detectors on four NetFlow v3 datasets
(UNSW-NB15, BoT-IoT, ToN-IoT, CSE-CIC-IDS2018), with 10 simulated clients whose data is IID or
skewed over attack types (Dirichlet α = 1, 0.5, 0.1). The study compares FedAvg and FedProx
against a centralised baseline and, in particular, how features are scaled: each client's own
quantile scaler (*local*) or one scaler merged from the clients' quantile sketches (*federated*).

## Repository layout

```
├── src/fmml/
│   ├── config.py              paths, datasets, experiment variant (FMML_EXPERIMENT), per-dataset models
│   ├── data.py                loading, client partitions, local and merged (federated) quantile scalers
│   ├── architectures.py       model families of the architecture search (FT-Transformer, MLP, CNN, ...)
│   ├── models.py              the chosen model per dataset
│   ├── fl.py                  Flower client and server (FedAvg, FedProx), per-round recording
│   ├── metrics.py             binary metrics, MCC
│   └── tracking.py            optional Weights & Biases mirror of the results
├── scripts/
│   ├── build_dataset.py       raw NetFlow v3 -> selected flows and time-ordered 72/18/10 splits
│   ├── make_random_split.py   the same flows split at random (comparison)
│   ├── tune_arch.py           architecture search on the validation split; tree_reference.py for trees
│   ├── train_central.py       centralised baseline
│   ├── run_fl.py              one federated run (dataset, scaler, partition, strategy, seed)
│   ├── choose_mu.py           FedProx mu by validation MCC
│   ├── eval_types.py          per-attack-type detection rates, ROC-AUC, PR-AUC
│   ├── calibrate_threshold.py decision threshold from the clients' validation data
│   ├── summarize.py           mean and std over seeds
│   ├── run_queue.py           runs a whole experiment plan (resumable, several jobs at a time)
│   └── wandb_sync.py          mirrors results/ to Weights & Biases
├── results/                   per-run and summarised results (random/: the random-split study)
└── data/                      not tracked; see data/README.md
```

## Setup

Python 3.10, TensorFlow 2.17 / Keras 3.5, Flower 1.23:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt     # the exact environment of the reported runs (CUDA 12 wheels)
pip install -e .
```

## Reproducing the results

```bash
# 1. Data: place the raw CSVs in data/raw/ (see data/README.md), then
python scripts/build_dataset.py
python scripts/make_random_split.py

# 2. Experiments: centralised seeds, FedProx mu selection, the federated grid (5 seeds), evaluation
python scripts/run_queue.py --plans main random --parallel 2
```

`run_queue.py` skips finished runs, so it can be stopped and restarted. A plan can be run in parts
(`main:fed`, `main:local`, ...); `--post-gpu` runs the evaluation steps on the GPU. Single runs:

```bash
python scripts/train_central.py --seeds 0 1 2 3 4
python scripts/run_fl.py --datasets UNSW-NB15 --scaler federated --partition dirichlet --alpha 0.1 --strategy fedavg --seed 0
FMML_EXPERIMENT=random python scripts/run_fl.py ...     # the random-split study (results/random/)
```

Each federated run writes `results/<dataset>/<scaler>/<partition>/<strategy>/seed<N>/`:
`summary.json`, `history.csv` (per round), `test_metrics.csv` (per client), `client_updates.csv`
(per client and round: update norm and cosine to the mean update), `partition.json`, and after
evaluation `type_metrics.csv`, `auc.csv`, `calibrated.json`. Aggregated tables:
`results/summary.csv`, `results/type_summary.csv`, `results/mu_selection.csv`.

## Experiment tracking

With `FMML_WANDB=1` (after `wandb login`), each finished run is also logged to Weights & Biases
(project `fmml-ids`, or `WANDB_PROJECT`); `python scripts/wandb_sync.py` logs runs already on disk.
The files under `results/` remain the record.

## Method

- **Data.** NetFlow v3, 45 flow features; flows de-duplicated, ambiguous feature vectors dropped,
  up to 1M flows per dataset. Within each class, flows are split in time order (72/18/10), so
  validation and test data are later than the training data.
- **Model.** One family for all datasets, FT-Transformer, chosen by a pre-declared rule from an
  architecture search on the validation split (`results/arch/`); each dataset keeps its own tuned
  configuration (`config.MODELS`).
- **Federated training.** 10 clients, 20 rounds of 1 local epoch; a client's flows keep their split.
  The round is chosen by validation MCC pooled over the clients' confusion counts. The test split is
  only evaluated, as each client sees it through its scaler; the mean and the worst client are reported.
- **Metrics.** MCC is the primary metric; F1, FPR, ROC-AUC and PR-AUC are also reported.
