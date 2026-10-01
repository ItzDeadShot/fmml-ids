"""Candidate architectures for the v3 detectors, their search spaces, and a builder.

Every family is a plain parametric network, so it can be trained federatedly by averaging weights
(FedAvg / FedProx). No BatchNorm: its running statistics do not average meaningfully across
non-IID clients; LayerNorm is used where normalisation helps.

    logreg          logistic regression (linear baseline)
    mlp             feed-forward network
    resmlp          residual MLP with LayerNorm (Gorishniy et al., 2021)
    cnn1d           1-D convolutions over the feature vector (common in NIDS work)
    dcn             Deep & Cross Network v2 (Wang et al., 2021): cross layers, then an MLP
    ft_transformer  Feature Tokenizer + Transformer (Gorishniy et al., 2021)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .config import N_FEATURES

# Search spaces: each trial draws one value per key. learning_rate and batch_size are drawn for
# every family.
TRAINING = {"learning_rate": [3e-4, 1e-3, 3e-3], "batch_size": [256, 1024]}
SPACES = {
    "logreg": {"l2": [0.0, 1e-5, 1e-4], "learning_rate": [1e-3, 3e-3, 1e-2]},
    "mlp": {"depth": [2, 3, 4], "width": [64, 128, 256, 512], "dropout": [0.0, 0.1, 0.2]},
    "resmlp": {"blocks": [2, 3, 4], "width": [64, 128, 256], "dropout": [0.0, 0.1, 0.2]},
    "cnn1d": {"conv_layers": [2, 3], "filters": [32, 64], "kernel": [3, 5], "dense": [64, 128]},
    "dcn": {"cross_layers": [2, 3], "deep_layers": [2, 3], "width": [128, 256], "dropout": [0.0, 0.1]},
    "ft_transformer": {"d_token": [32, 64], "blocks": [1, 2, 3], "heads": [4, 8], "dropout": [0.0, 0.1],
                       "learning_rate": [3e-4, 1e-3], "batch_size": [512, 1024]},
}
TRIALS = {"logreg": 6, "mlp": 8, "resmlp": 8, "cnn1d": 8, "dcn": 8, "ft_transformer": 4}
MAX_EPOCHS = {"ft_transformer": 12}             # others: 20


def sample_configs(family: str, n: int, seed: int = 0):
    """n distinct configurations of a family, drawn reproducibly."""
    space = {**TRAINING, **SPACES[family]}
    rng = np.random.default_rng([seed, sum(map(ord, family))])
    configs, seen = [], set()
    for _ in range(1000):
        c = {k: v[rng.integers(len(v))] for k, v in space.items()}
        c = {k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()}
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            configs.append(c)
        if len(configs) == n:
            break
    return configs


@tf.keras.utils.register_keras_serializable(package="fmml")
class _CrossLayer(layers.Layer):
    """DCN-v2 cross layer: x_{l+1} = x_0 * (W x_l + b) + x_l."""

    def build(self, input_shape):
        self.dense = layers.Dense(input_shape[0][-1])

    def call(self, inputs):
        x0, x = inputs
        return x0 * self.dense(x) + x


@tf.keras.utils.register_keras_serializable(package="fmml")
class _FeatureTokenizer(layers.Layer):
    """FT-Transformer tokenizer: feature j -> x_j * W_j + b_j, plus a learned [CLS] token."""

    def __init__(self, d_token, **kwargs):
        super().__init__(**kwargs)
        self.d_token = d_token

    def build(self, input_shape):
        n = input_shape[-1]
        self.w = self.add_weight(name="w", shape=(n, self.d_token), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(n, self.d_token), initializer="zeros")
        self.cls = self.add_weight(name="cls", shape=(1, 1, self.d_token), initializer="glorot_uniform")

    def call(self, x):
        tokens = x[:, :, None] * self.w + self.b
        cls = tf.repeat(self.cls, tf.shape(x)[0], axis=0)
        return tf.concat([cls, tokens], axis=1)

    def get_config(self):
        return {**super().get_config(), "d_token": self.d_token}


def build(family: str, params: dict, n_features: int = N_FEATURES) -> tf.keras.Model:
    """Uncompiled model with one sigmoid output (attack probability)."""
    inp = layers.Input((n_features,), name="flow")
    p = params
    if family == "logreg":
        out = layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(p["l2"]))(inp)
        return tf.keras.Model(inp, out, name="logreg")
    if family == "mlp":
        x = inp
        for _ in range(p["depth"]):
            x = layers.Dense(p["width"], activation="relu")(x)
            x = layers.Dropout(p["dropout"])(x)
    elif family == "resmlp":
        x = layers.Dense(p["width"])(inp)
        for _ in range(p["blocks"]):
            h = layers.LayerNormalization()(x)
            h = layers.Dense(2 * p["width"], activation="relu")(h)
            h = layers.Dropout(p["dropout"])(h)
            h = layers.Dense(p["width"])(h)
            x = layers.Add()([x, layers.Dropout(p["dropout"])(h)])
        x = layers.Activation("relu")(layers.LayerNormalization()(x))
    elif family == "cnn1d":
        x = layers.Reshape((n_features, 1))(inp)
        for _ in range(p["conv_layers"]):
            x = layers.Conv1D(p["filters"], p["kernel"], padding="same", activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(p["dense"], activation="relu")(x)
    elif family == "dcn":
        x = inp
        for _ in range(p["cross_layers"]):
            x = _CrossLayer()([inp, x])
        for _ in range(p["deep_layers"]):
            x = layers.Dense(p["width"], activation="relu")(x)
            x = layers.Dropout(p["dropout"])(x)
    elif family == "ft_transformer":
        d = p["d_token"]
        x = _FeatureTokenizer(d)(inp)
        for _ in range(p["blocks"]):
            h = layers.LayerNormalization()(x)
            h = layers.MultiHeadAttention(p["heads"], d // p["heads"], dropout=p["dropout"])(h, h)
            x = layers.Add()([x, layers.Dropout(p["dropout"])(h)])
            h = layers.LayerNormalization()(x)
            h = layers.Dense(2 * d, activation="relu")(h)
            h = layers.Dropout(p["dropout"])(h)
            x = layers.Add()([x, layers.Dense(d)(h)])
        x = layers.Activation("relu")(layers.LayerNormalization()(x[:, 0]))      # the [CLS] token
    else:
        raise ValueError(family)
    return tf.keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name=family)
