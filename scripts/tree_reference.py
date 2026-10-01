"""Centralised gradient-boosted trees as a reference ceiling for the architecture search.

    python scripts/tree_reference.py

Not a federated candidate (trees cannot be averaged like network weights); it shows how far the
networks are from a strong tabular learner on the same 45 features. Four configurations of
scikit-learn's HistGradientBoostingClassifier (learning rate x leaves, 500 iterations), trained on
the training split and chosen by validation MCC; the test split is only evaluated.
Writes results/arch/<dataset>/trees.csv.
"""
import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from fmml import config
from fmml.data import load_xy
from fmml.metrics import binary_metrics

GRID = [{"learning_rate": lr, "max_leaf_nodes": leaves} for lr in (0.05, 0.1) for leaves in (31, 127)]


def main():
    for dataset in config.DATASETS:
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = (load_xy(config.split_path(dataset, s)) for s in ("train", "val", "test"))
        rows = []
        for params in GRID:
            start = time.perf_counter()
            clf = HistGradientBoostingClassifier(max_iter=500, random_state=0, **params).fit(X_tr, y_tr)
            val = binary_metrics(y_va, clf.predict(X_va))
            test = binary_metrics(y_te, clf.predict(X_te))
            rows.append({**params, "val_mcc": val["MCC"], "train_seconds": time.perf_counter() - start,
                         **{f"test_{k}": v for k, v in test.items()}})
            print(f"{dataset} {params}: val MCC {val['MCC']:.4f} | test MCC {test['MCC']:.4f}", flush=True)
        out = config.RESULTS_DIR / "arch" / dataset
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "trees.csv", index=False)


if __name__ == "__main__":
    main()
