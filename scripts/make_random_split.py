"""Random (non-temporal) train/validation/test splits of the selected flows, for comparison.

    python scripts/make_random_split.py

Takes the flows chosen by build_dataset.py (data/processed/<name>.csv) and splits them 72/18/10
at random, stratified by attack type, into data/split-random/<name>_{train,val,test}.csv (same
columns as data/split-data). This is the protocol most prior work uses; the main study splits in
time order instead, so the comparison shows how much a random split flatters the results.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from fmml import config

SEED = 0


def main():
    out = config.ROOT / "data" / "split-random"
    out.mkdir(parents=True, exist_ok=True)
    for name in config.DATASETS.values():
        df = pd.read_csv(config.ROOT / "data" / "processed" / f"{name}.csv")
        columns = list(pd.read_csv(config.ROOT / "data" / "split-data" / f"{name}_test.csv", nrows=0).columns)
        df = df[columns]
        rest, test = train_test_split(df, test_size=0.10, stratify=df[config.ATTACK_COL], random_state=SEED)
        train, val = train_test_split(rest, test_size=0.20, stratify=rest[config.ATTACK_COL], random_state=SEED)
        for split, part in (("train", train), ("val", val), ("test", test)):
            part.to_csv(out / f"{name}_{split}.csv", index=False)
        print(f"{name}: train {len(train)}, val {len(val)}, test {len(test)}")


if __name__ == "__main__":
    main()
