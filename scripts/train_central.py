"""Centralised baseline: each dataset's model trained on all of its training data.

    python scripts/train_central.py --grid              # grid search, seed 0, all datasets
    python scripts/train_central.py --seeds 0 1 2 3 4   # final baseline with the chosen settings

Features are scaled with a QuantileTransformer (normal output) fit on the training split. Each
configuration trains for --epochs epochs, keeping the epoch with the highest validation MCC (as
the federated runs choose their round; validation loss favours undecided early epochs under the
drift between the time-ordered splits). With --grid, every learning rate x batch size combination
is trained and the one with the highest validation MCC is written to
results/central/<dataset>/chosen.json; without --grid, the chosen configuration is trained for
every seed. The test split is only evaluated, never used for choices. (The v3 models themselves,
with their learning rate and batch size, come from scripts/tune_arch.py via config.MODELS; --grid
is the earlier search for the original architectures.)

Writes results/central/<dataset>/{grid.csv, chosen.json, seed<N>/test_metrics.csv, seed<N>/history.csv}.
With FMML_WANDB=1 each seed's run is also logged to Weights & Biases (fmml.tracking).
"""
import argparse
import json
import os
import socket
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer

import fmml.architectures  # noqa: F401  (registers custom layers for saved models)
from fmml import config, tracking
from fmml.data import load_xy
from fmml.metrics import binary_metrics
from fmml.models import build_uncompiled

LEARNING_RATES = [0.001, 0.01]
BATCH_SIZES = [128, 512]



class BestValidationMCC(tf.keras.callbacks.Callback):
    """Saves the model after every epoch that improves the validation MCC, stops after `patience`
    epochs without improvement, and times the training part of each epoch (excluding this
    validation). history: one row per epoch (Keras's metrics, val_mcc, epoch_seconds)."""

    def __init__(self, X_va, y_va, path, patience=3):
        super().__init__()
        self.X_va, self.y_va, self.path, self.patience = X_va, y_va, str(path), patience
        self.best, self.best_epoch = -np.inf, None
        self.epoch_seconds, self.history = [], []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_seconds.append(time.perf_counter() - self.epoch_start)
        pred = (self.model.predict(self.X_va, verbose=0, batch_size=8192).ravel() > 0.5).astype(int)
        score = binary_metrics(self.y_va, pred)["MCC"]
        logs["val_mcc"] = score
        self.history.append({"epoch": epoch + 1, **logs, "epoch_seconds": self.epoch_seconds[-1]})
        print(f"epoch {epoch + 1}: val_mcc {score:.4f}")
        if score > self.best:
            self.best, self.best_epoch = score, epoch + 1
            self.model.save(self.path)
        elif epoch + 1 - self.best_epoch >= self.patience:
            self.model.stop_training = True


def train_one(dataset, data, lr, batch_size, epochs, seed, ckpt):
    """Returns (validation MCC at the chosen epoch, the chosen epoch, test metrics, per-epoch history);
    the test metrics include the training time (all epochs, and the median epoch)."""
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = data
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model = build_uncompiled(dataset)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"], jit_compile=True)
    best = BestValidationMCC(X_va, y_va, ckpt)
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=batch_size, verbose=2,
              callbacks=[best])
    chosen = tf.keras.models.load_model(ckpt)
    test = binary_metrics(y_te, (chosen.predict(X_te, verbose=0, batch_size=8192).ravel() > 0.5).astype(int))
    test.update(train_seconds=float(np.sum(best.epoch_seconds)), epoch_seconds_median=float(np.median(best.epoch_seconds)))
    tf.keras.backend.clear_session()
    return float(best.best), best.best_epoch, test, pd.DataFrame(best.history)


def refit(dataset, out, ckpt_dir, seeds):
    """FMML_EXPERIMENT=refit: retrain on train+val for the number of epochs the main study chose
    for this seed (no validation; the test split is only evaluated)."""
    assert config.EXPERIMENT == "refit", "run --refit with FMML_EXPERIMENT=refit"
    main = config.MAIN_RESULTS_DIR / "central" / dataset
    chosen = config.MODELS[dataset]                     # learning_rate, batch_size of the dataset's model
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = (load_xy(config.split_path(dataset, s)) for s in ("train", "val", "test"))
    X, y = np.concatenate([X_tr, X_va]), np.concatenate([y_tr, y_va])
    for seed in seeds:
        epochs = int(pd.read_csv(main / f"seed{seed}" / "test_metrics.csv").epoch.iloc[0])
        scaler = QuantileTransformer(output_distribution="normal", random_state=seed).fit(X)
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        model = build_uncompiled(dataset)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=chosen["learning_rate"]),
                      loss="binary_crossentropy", metrics=["accuracy"], jit_compile=True)
        start = time.perf_counter()
        model.fit(scaler.transform(X), y, epochs=epochs, batch_size=chosen["batch_size"], verbose=2)
        seconds = time.perf_counter() - start
        model.save(ckpt_dir / f"seed{seed}.keras")
        test = binary_metrics(y_te, (model.predict(scaler.transform(X_te), verbose=0, batch_size=8192).ravel() > 0.5).astype(int))
        (out / f"seed{seed}").mkdir(exist_ok=True)
        pd.DataFrame([{"epoch": epochs, **test, "train_seconds": seconds}]).to_csv(
            out / f"seed{seed}" / "test_metrics.csv", index=False)
        tf.keras.backend.clear_session()
        print(f"{dataset} seed {seed}: refit {epochs} epochs | test MCC {test['MCC']:.4f}, F1 {test['F1']:.4f}")
        tracking.maybe_sync(out / f"seed{seed}")


def load_scaled(dataset, seed):
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = (load_xy(config.split_path(dataset, s)) for s in ("train", "val", "test"))
    scaler = QuantileTransformer(output_distribution="normal", random_state=seed).fit(X_tr)
    return (scaler.transform(X_tr), y_tr), (scaler.transform(X_va), y_va), (scaler.transform(X_te), y_te)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    parser.add_argument("--grid", action="store_true", help="grid-search learning rate x batch size (seed 0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=20, help="maximum epochs (early stopping, patience 3)")
    parser.add_argument("--refit", action="store_true",
                        help="(FMML_EXPERIMENT=refit) retrain on train+val for the epochs chosen in the main study")
    args = parser.parse_args()

    for dataset in args.datasets:
        out = config.RESULTS_DIR / "central" / dataset
        out.mkdir(parents=True, exist_ok=True)
        ckpt_dir = config.RUNS_DIR / "central" / dataset
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if args.grid:
            data = load_scaled(dataset, 0)
            rows = []
            for lr in LEARNING_RATES:
                for bs in BATCH_SIZES:
                    val_mcc, epoch, test, _ = train_one(dataset, data, lr, bs, args.epochs, 0,
                                                     ckpt_dir / f"grid_lr{lr}_bs{bs}.keras")
                    rows.append({"learning_rate": lr, "batch_size": bs, "val_mcc": val_mcc, "epoch": epoch, **test})
                    print(f"{dataset} lr={lr} bs={bs}: val MCC {val_mcc:.4f} (epoch {epoch}) | test MCC {test['MCC']:.4f}")
            grid = pd.DataFrame(rows)
            grid.to_csv(out / "grid.csv", index=False)
            best = grid.loc[grid.val_mcc.idxmax()]
            chosen = {"learning_rate": float(best.learning_rate), "batch_size": int(best.batch_size),
                      "val_mcc": float(best.val_mcc)}
            (out / "chosen.json").write_text(json.dumps(chosen, indent=2))
            print(f"{dataset}: chosen {chosen}")
        elif args.refit:
            refit(dataset, out, ckpt_dir, args.seeds)
        else:
            chosen = config.MODELS[dataset]             # the dataset's model and its learning rate, batch size
            for seed in args.seeds:
                data = load_scaled(dataset, seed)
                val_mcc, epoch, test, history = train_one(dataset, data, chosen["learning_rate"], chosen["batch_size"],
                                                          args.epochs, seed, ckpt_dir / f"seed{seed}.keras")
                (out / f"seed{seed}").mkdir(exist_ok=True)
                history.to_csv(out / f"seed{seed}" / "history.csv", index=False)
                pd.DataFrame([{"val_mcc": val_mcc, "epoch": epoch, **test, "host": socket.gethostname()}]).to_csv(
                    out / f"seed{seed}" / "test_metrics.csv", index=False)
                print(f"{dataset} seed {seed}: epoch {epoch} | test MCC {test['MCC']:.4f}, F1 {test['F1']:.4f}")
                tracking.maybe_sync(out / f"seed{seed}")


if __name__ == "__main__":
    main()
