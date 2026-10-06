"""Choose SCAFFOLD's local SGD learning rate per dataset and scaler on client validation MCC.

    python scripts/choose_lr.py --scaler local --lrs 0.01 0.03 0.1 0.3

Reads the seed-0, Dirichlet(0.5) SCAFFOLD runs (results/<dataset>/<scaler>/dirichlet-a0.5/scaffold-lr<lr>/seed0/)
and writes the rate with the highest validation MCC (at the chosen round) to results/<dataset>/<scaler>/scaffold_lr.json.
"""
import argparse
import json

from fmml import config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    parser.add_argument("--scaler", choices=["local", "federated"], default="local")
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    args = parser.parse_args()
    for dataset in args.datasets:
        val = {}
        for lr in args.lrs:
            summary = config.RESULTS_DIR / dataset / args.scaler / "dirichlet-a0.5" / f"scaffold-lr{lr:g}" / "seed0" / "summary.json"
            val[lr] = json.loads(summary.read_text())["val_mcc_best"]
        lr = max(val, key=val.get)
        (config.RESULTS_DIR / dataset / args.scaler / "scaffold_lr.json").write_text(
            json.dumps({"local_lr": lr, "val_mcc": {f"{r:g}": v for r, v in val.items()}}, indent=2))
        print(f"{dataset} ({args.scaler}): lr {lr:g}  ({', '.join(f'{r:g}: {v:.4f}' for r, v in val.items())})")


if __name__ == "__main__":
    main()
