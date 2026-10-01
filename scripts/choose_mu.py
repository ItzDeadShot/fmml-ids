"""Choose the FedProx proximal coefficient per dataset on client validation loss.

    python scripts/choose_mu.py --scaler local --mus 0.001 0.01 0.1

Reads the seed-0, Dirichlet(0.5) FedProx runs (results/<dataset>/<scaler>/dirichlet-a0.5/fedprox-mu<mu>/seed0/)
and writes the mu with the highest validation MCC (at the chosen round) to results/<dataset>/<scaler>/fedprox_mu.json.
"""
import argparse
import json

from fmml import config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    parser.add_argument("--scaler", choices=["local", "federated"], default="local")
    parser.add_argument("--mus", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    args = parser.parse_args()
    for dataset in args.datasets:
        val = {}
        for mu in args.mus:
            summary = config.RESULTS_DIR / dataset / args.scaler / "dirichlet-a0.5" / f"fedprox-mu{mu:g}" / "seed0" / "summary.json"
            val[mu] = json.loads(summary.read_text())["val_mcc_best"]
        mu = max(val, key=val.get)
        (config.RESULTS_DIR / dataset / args.scaler / "fedprox_mu.json").write_text(
            json.dumps({"proximal_mu": mu, "val_mcc": {f"{m:g}": v for m, v in val.items()}}, indent=2))
        print(f"{dataset} ({args.scaler}): mu {mu:g}  ({', '.join(f'{m:g}: {v:.4f}' for m, v in val.items())})")


if __name__ == "__main__":
    main()
