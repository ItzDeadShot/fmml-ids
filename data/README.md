# Data

The data files are not tracked in git. They are built from the four public raw datasets:

```
data/
├── raw/            the published NetFlow v3 CSVs (download, see below)
├── processed/      selected flows per dataset, written by scripts/build_dataset.py
├── split-data/     time-ordered 72/18/10 train/val/test splits (main study), build_dataset.py
└── split-random/   random 72/18/10 splits of the same flows, scripts/make_random_split.py
```

```bash
python scripts/build_dataset.py       # raw -> processed + split-data
python scripts/make_random_split.py   # processed -> split-random
```

## Source datasets

The NetFlow v3 datasets are published by the University of Queensland at
https://staff.itee.uq.edu.au/marius/NIDS_datasets/ (check the dataset terms before redistributing).
Place the CSVs in `data/raw/`:

| File | Raw flows | SHA-1 |
|---|---:|---|
| NF-UNSW-NB15-v3.csv | 2,365,424 | `2cba4879ead17c51e3268a01c06af41186cd90e4` |
| NF-BoT-IoT-v3.csv | 16,933,808 | `357cc7ffe46ef57ba8193a58d7a66eff5cf44db8` |
| NF-ToN-IoT-v3.csv | 27,520,260 | `24854ca3072ab7c2ade9ebfb202345a666a94cfe` |
| NF-CICIDS2018-v3.csv | 20,115,529 | `ddcad6d826b73a128a447a0189137efd6c8f70c9` |

## Selected flows

| Dataset | Distinct vectors | Ambiguous dropped | Selected | Benign share |
|---|---:|---:|---:|---:|
| UNSW-NB15 | 1,230,871 | 0 | 1,000,000 | 90.2% |
| BoT-IoT | 674,403 | 466 | 243,281 | 6.2% |
| ToN-IoT | 5,886,034 | 65,497 | 1,000,000 | 43.5% |
| CIC-2018 | 14,508,426 | 8,928 | 1,000,000 | 49.6% |

Selection (details in `scripts/build_dataset.py`): flow identifiers and timestamps removed (45
features remain); duplicate feature vectors dropped, and vectors seen under both labels removed;
attack types with fewer than 50 distinct flows dropped; each attack type capped at 100,000 flows,
benign filling up to 1,000,000; capped classes sampled evenly over their time span.
