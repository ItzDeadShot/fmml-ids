"""Model architectures (identical to the centralised specialists)."""
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, Reshape
from tensorflow.keras.models import Model

from .config import MODELS, N_FEATURES


def _mlp():
    inputs = Input(shape=(N_FEATURES,), name="MLP_input")
    x = Dense(24, activation="relu", name="mlp_dense_1")(inputs)
    x = Dense(16, activation="relu", name="mlp_dense_2")(x)
    return Model(inputs, Dense(1, activation="sigmoid", name="mlp_output")(x), name="MLP")


def _dnn():
    inputs = Input(shape=(N_FEATURES,), name="DNN_input_layer")
    x = inputs
    for i, units in enumerate([32, 16, 8, 4, 2], start=1):
        x = Dense(units, activation="relu", name=f"DNN_dense_layer_{i}")(x)
    return Model(inputs, Dense(1, activation="sigmoid", name="DNN_output_layer")(x), name="DNN")


def _recurrent(layer, prefix):
    inputs = Input(shape=(N_FEATURES,), name=f"{prefix}_input")
    x = Reshape((1, N_FEATURES), name=f"{prefix}_reshape")(inputs)
    x = layer(24, activation="relu", return_sequences=True, name=f"{prefix}_1")(x)
    x = layer(16, activation="relu", return_sequences=False, name=f"{prefix}_2")(x)
    x = Dense(8, activation="relu", name=f"{prefix}_dense")(x)
    return Model(inputs, Dense(1, activation="sigmoid", name=f"{prefix}_output")(x), name=prefix.upper())


BUILDERS = {"MLP": _mlp, "DNN": _dnn, "GRU": lambda: _recurrent(GRU, "gru"), "LSTM": lambda: _recurrent(LSTM, "lstm")}


def build_uncompiled(dataset: str) -> Model:
    """The dataset's model: a family from fmml.architectures ({"family", "params", ...}), or one of the
    original builders above ({"arch", ...})."""
    spec = MODELS[dataset]
    if "family" in spec:
        from .architectures import build
        return build(spec["family"], spec["params"])
    return BUILDERS[spec["arch"]]()


def build_model(dataset: str) -> Model:
    spec = MODELS[dataset]
    model = build_uncompiled(dataset)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=spec["learning_rate"]),
                  loss="binary_crossentropy", metrics=["accuracy"], jit_compile=True)
    return model
