# SCAFFOLD with local Adam (superseded)

The first SCAFFOLD runs (time split, 5 seeds, local and federated quantile scaler), with the local
Adam of FedAvg/FedProx. Not used: the constant correction c - c_i passed through Adam's per-coordinate
normalisation inflated client updates 2-6x over FedAvg under attack-type skew, saturated the output and
froze 34 of 160 runs (every client update exactly zero from some round on; CIC-2018 17, ToN-IoT 9,
UNSW-NB15 8, BoT-IoT 0). Replaced by SCAFFOLD with local SGD at a learning rate chosen on validation
(`<dataset>/<scaler>/<partition>/scaffold-lr<lr>/`, see `src/fmml/fl.py` `_ScaffoldTrainer`).
