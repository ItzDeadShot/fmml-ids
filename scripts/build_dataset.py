"""Build the benign-vs-attack datasets from the raw NetFlow v3 CSVs.

    python scripts/build_dataset.py                          # all four datasets
    python scripts/build_dataset.py --datasets UNSW-NB15

Reads  data/raw/<file>.csv
Writes data/processed/<name>.csv         selected flows: 45 features, flow_start_ms, label, attack
       data/split-data/<name>_{train,val,test}.csv   45 features, label, attack
       data/processed/<name>_summary.json

Per dataset:
  1. Features: all NetFlow v3 fields except the flow identifiers (IPV4_SRC_ADDR, L4_SRC_PORT,
     IPV4_DST_ADDR, L4_DST_PORT), the timestamps (FLOW_START/END_MILLISECONDS, used only for
     splitting) and SRC_TO_DST_SECOND_BYTES / DST_TO_SRC_SECOND_BYTES (bytes / duration: redundant,
     undefined for zero-duration flows). 45 features remain.
  2. Label: 1 for any attack, 0 for benign; the attack type is kept in the "attack" column.
  3. De-duplication: keep the first occurrence of each feature vector; drop vectors that occur
     under both labels. Attack types with fewer than 50 distinct flows are dropped.
  4. Size: each attack type is capped at 100,000 flows and benign fills the rest up to
     1,000,000; capped classes are sampled evenly over their time span.
  5. Split: within each attack type (and benign), flows are ordered by start time; the earliest
     72% are training, the next 18% validation and the latest 10% test.
"""
import argparse
import json

import numpy as np
import pandas as pd

from fmml import config

RAW_FILES = {
    "UNSW-NB15": "NF-UNSW-NB15-v3.csv",
    "BoT-IoT": "NF-BoT-IoT-v3.csv",
    "ToN-IoT": "NF-ToN-IoT-v3.csv",
    "CIC-2018": "NF-CICIDS2018-v3.csv",
}
IDENTIFIERS = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT"]
TIMESTAMPS = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]
DROPPED = ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]
MIN_CLASS = 50
CLASS_CAP = 100_000
TOTAL_CAP = 1_000_000
SPLITS = (("train", 0.72), ("val", 0.18), ("test", 0.10))
CHUNK = 2_000_000


def evenly(idx_sorted_by_time: np.ndarray, k: int) -> np.ndarray:
    """k rows spread evenly over a class's time-ordered rows."""
    if len(idx_sorted_by_time) <= k:
        return idx_sorted_by_time
    return idx_sorted_by_time[np.unique(np.linspace(0, len(idx_sorted_by_time) - 1, k).round().astype(int))]


def build(name: str):
    path = config.ROOT / "data" / "raw" / RAW_FILES[name]
    header = pd.read_csv(path, nrows=0).columns
    features = [c for c in header if c not in IDENTIFIERS + TIMESTAMPS + DROPPED + ["Label", "Attack"]]
    assert len(features) == config.N_FEATURES, len(features)

    # Pass 1: fingerprint every flow.
    hashes, labels, attacks, starts = [], [], [], []
    for ch in pd.read_csv(path, chunksize=CHUNK, usecols=features + ["FLOW_START_MILLISECONDS", "Label", "Attack"],
                          low_memory=False):
        hashes.append(pd.util.hash_pandas_object(ch[features].astype("float64"), index=False).to_numpy())
        labels.append(ch["Label"].to_numpy(dtype=np.int8))
        attacks.append(ch["Attack"].to_numpy())
        starts.append(ch["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.int64))
    h, lab, att, start = (np.concatenate(a) for a in (hashes, labels, attacks, starts))
    n_raw = len(h)

    # De-duplicate: first occurrence of each vector; drop vectors seen with both labels.
    _, first, inverse = np.unique(h, return_index=True, return_inverse=True)
    label_mix = np.zeros(len(first), dtype=np.int8)
    np.bitwise_or.at(label_mix, inverse, (1 << lab).astype(np.int8))    # bit 0: benign seen, bit 1: attack seen
    keep = first[label_mix != 3]
    n_ambiguous = int(np.sum(label_mix == 3))

    # Per-class selection, evenly over time.
    kept_att = att[keep]
    counts = pd.Series(kept_att).value_counts()
    attack_types = [a for a in counts.index if a != "Benign" and counts[a] >= MIN_CLASS]
    dropped_types = {a: int(counts[a]) for a in counts.index if a != "Benign" and counts[a] < MIN_CLASS}
    order = lambda idx: idx[np.lexsort((idx, start[idx]))]               # by start time, then file order
    selected = {a: evenly(order(keep[kept_att == a]), CLASS_CAP) for a in attack_types}
    budget = TOTAL_CAP - sum(len(v) for v in selected.values())
    selected["Benign"] = evenly(order(keep[kept_att == "Benign"]), budget)

    # Time-ordered split within each class.
    split_of = {}
    for a, idx in selected.items():
        cuts = np.cumsum([int(round(f * len(idx))) for _, f in SPLITS])[:-1]
        for (split, _), part in zip(SPLITS, np.split(idx, cuts)):
            split_of.update(dict.fromkeys(part.tolist(), split))
    rows = np.sort(np.fromiter(split_of.keys(), dtype=np.int64))

    # Pass 2: read the selected rows.
    parts, offset = [], 0
    for ch in pd.read_csv(path, chunksize=CHUNK, usecols=features + ["FLOW_START_MILLISECONDS", "Label", "Attack"],
                          low_memory=False):
        in_chunk = rows[(rows >= offset) & (rows < offset + len(ch))]
        if len(in_chunk):
            parts.append(ch.iloc[in_chunk - offset].set_index(pd.Index(in_chunk, name="raw_row")))
        offset += len(ch)
    df = pd.concat(parts)
    df = df.rename(columns={"FLOW_START_MILLISECONDS": "flow_start_ms", "Label": config.LABEL_COL,
                            "Attack": config.ATTACK_COL})[features + ["flow_start_ms", config.LABEL_COL, config.ATTACK_COL]]
    df["split"] = [split_of[i] for i in df.index]

    out_proc, out_split = config.ROOT / "data" / "processed", config.SPLIT_DIR
    out_proc.mkdir(parents=True, exist_ok=True)
    out_split.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["split"]).to_csv(out_proc / f"{name}.csv")
    for split, _ in SPLITS:
        df[df.split == split].drop(columns=["split", "flow_start_ms"]).to_csv(out_split / f"{name}_{split}.csv", index=False)

    summary = {"raw_flows": n_raw, "distinct_vectors": int(len(first)), "ambiguous_vectors_dropped": n_ambiguous,
               "attack_types_dropped_lt_50": dropped_types, "selected_flows": int(len(df)),
               "benign_share": round(float((df[config.LABEL_COL] == 0).mean()), 4),
               "per_class": {a: {s: int(((df[config.ATTACK_COL] == a) & (df.split == s)).sum()) for s, _ in SPLITS}
                             for a in df[config.ATTACK_COL].value_counts().index}}
    (out_proc / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{name}: {n_raw:,} raw -> {len(first):,} distinct ({n_ambiguous:,} ambiguous dropped) -> {len(df):,} selected, "
          f"benign {summary['benign_share']:.1%}; dropped types {dropped_types}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(RAW_FILES), default=list(RAW_FILES))
    args = parser.parse_args()
    for name in args.datasets:
        build(name)


if __name__ == "__main__":
    main()
