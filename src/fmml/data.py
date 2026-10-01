"""Loading the splits, partitioning them across clients, and per-client preparation."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from . import config


def load_split(path):
    """Return (X, y, attack): the 45 features, the binary label and the attack type."""
    df = pd.read_csv(path)
    y = df.pop(config.LABEL_COL).to_numpy()
    attack = df.pop(config.ATTACK_COL).to_numpy()
    X = df.to_numpy(dtype="float64")
    assert X.shape[1] == config.N_FEATURES, path
    return X, y, attack


def load_xy(path):
    X, y, _ = load_split(path)
    return X, y


def load_pool(dataset: str):
    """Everything outside the test split (the train and validation splits): the data that gets
    distributed across the clients. Returns (X, y, attack, is_val)."""
    parts = [load_split(config.split_path(dataset, s)) for s in ("train", "val")]
    is_val = np.repeat([False, True], [len(parts[0][1]), len(parts[1][1])])
    return (*(np.concatenate([p[i] for p in parts]) for i in range(3)), is_val)


def iid_partition(groups, n_clients: int, seed: int):
    idx = np.random.default_rng(seed).permutation(len(groups))
    return [np.sort(part) for part in np.array_split(idx, n_clients)]


def dirichlet_partition(groups, n_clients: int, alpha: float, seed: int, min_size: int = config.MIN_CLIENT_SIZE):
    """Skewed partition: for each group (here: attack type, with benign as one group), the
    group's flows are divided among clients in shares drawn from Dir(alpha).

    Small alpha gives clients very different mixes of traffic (some see an attack type often,
    others never); large alpha approaches IID. Redrawn until every client holds at least
    min_size flows.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(groups)
    for _ in range(1000):
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(groups == c))
            shares = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(shares)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p)) for p in parts]
    raise RuntimeError("could not draw a Dirichlet partition with the minimum client size")


def merge_quantile_scalers(scalers, weights, seed: int) -> QuantileTransformer:
    """One QuantileTransformer for all clients, built from the clients' own fitted ones.

    Each client shares only its per-feature quantiles (its QuantileTransformer's quantiles_). The
    server treats every client's quantiles as a sample of its data, each point weighted by the
    client's size over its number of quantiles, and takes the weighted quantiles of the union at
    the usual reference levels: an approximation of the scaler that would be fit on the pooled
    training data, without any client sharing flows. Repeated values (discrete features) are kept
    exactly, as in a pooled fit.
    """
    references = np.linspace(0, 1, max(s.n_quantiles_ for s in scalers))
    point_weights = np.concatenate([np.full(s.n_quantiles_, w / s.n_quantiles_) for s, w in zip(scalers, weights)])
    quantiles = np.empty((len(references), scalers[0].quantiles_.shape[1]))
    for j in range(quantiles.shape[1]):
        values = np.concatenate([s.quantiles_[:, j] for s in scalers])
        order = np.argsort(values, kind="stable")
        v, w = values[order], point_weights[order]
        position = (np.cumsum(w) - w / 2) / w.sum()           # weighted plotting positions in (0, 1)
        quantiles[:, j] = np.interp(references, position, v)
    merged = QuantileTransformer(n_quantiles=len(references), output_distribution="normal", random_state=seed)
    merged.fit(quantiles)                                     # sets the fitted attributes; replaced below
    merged.quantiles_, merged.references_ = np.maximum.accumulate(quantiles, axis=0), references
    return merged


def prepare_clients(dataset: str, partition: str, alpha: float, n_clients: int, seed: int, out_dir: Path,
                    scaler: str = "local", refit: bool = False):
    """Partition the pool, fit the clients' scalers on their training data, and save everything
    under out_dir.

    scaler="local": every client scales with its own QuantileTransformer. scaler="federated": the
    clients' QuantileTransformers are merged into one common scaler (merge_quantile_scalers).
    refit=True: each client trains on its train and validation flows together (for retraining the
    final model with settings already chosen); its validation flows are kept only for monitoring.

    A client's flows keep the split they came from: its training data is its share of the train
    split and its validation data its share of the (later) validation split. The clients together
    therefore hold exactly the centralised baseline's train and validation data.

    Writes client_<k>.npz (scaled X_train, y_train, X_val, y_val), scaler_<k>.pkl, partition.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, attack, is_val = load_pool(dataset)
    parts = (iid_partition(attack, n_clients, seed) if partition == "iid"
             else dirichlet_partition(attack, n_clients, alpha, seed))
    splits = []
    for k, idx in enumerate(parts):
        tr, va = idx[~is_val[idx]], idx[is_val[idx]]
        assert len(tr) and len(va), f"client {k} has no training or no validation flows"
        if refit:
            tr = idx
        splits.append((tr, va))
    scalers = [QuantileTransformer(output_distribution="normal", random_state=seed).fit(X[tr]) for tr, _ in splits]
    if scaler == "federated":
        scalers = [merge_quantile_scalers(scalers, [len(tr) for tr, _ in splits], seed)] * n_clients
    stats = []
    for k, (idx, (tr, va), sc) in enumerate(zip(parts, splits, scalers)):
        yk, y_tr, y_va = y[idx], y[tr], y[va]
        np.savez(out_dir / f"client_{k}.npz", X_train=sc.transform(X[tr]).astype("float32"), y_train=y_tr,
                 X_val=sc.transform(X[va]).astype("float32"), y_val=y_va)
        with open(out_dir / f"scaler_{k}.pkl", "wb") as f:
            pickle.dump(sc, f)
        types = pd.Series(attack[idx]).value_counts()
        stats.append({"client": k, "flows": int(len(yk)), "train": int(len(y_tr)), "val": int(len(y_va)),
                      "attack_share": round(float(yk.mean()), 4),
                      "attack_types": int(sum(1 for t in types.index if t != "Benign")),
                      "types": {str(t): int(c) for t, c in types.items()}})
    (out_dir / "partition.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_client(out_dir: Path, k: int):
    z = np.load(out_dir / f"client_{k}.npz")
    return z["X_train"], z["y_train"], z["X_val"], z["y_val"]


def load_scaler(out_dir: Path, k: int) -> QuantileTransformer:
    with open(out_dir / f"scaler_{k}.pkl", "rb") as f:
        return pickle.load(f)
