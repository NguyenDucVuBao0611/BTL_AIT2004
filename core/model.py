
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.regularizers import l2


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SEQUENCE_LENGTH = 30      # frames per sample
NUM_FEATURES    = 126     # keypoints per frame  (from Person 1)
# NUM_CLASSES is passed dynamically so the model works for any label set


# ──────────────────────────────────────────────
# LSTM  (baseline – primary model)
# ──────────────────────────────────────────────
def build_lstm(num_classes: int,
               sequence_length: int = SEQUENCE_LENGTH,
               num_features: int    = NUM_FEATURES,
               dropout_rate: float  = 0.3) -> Sequential:
    """
    3-layer stacked LSTM with dropout + batch norm.
    Returns: compiled Keras model.
    """
    model = Sequential([
        Input(shape=(sequence_length, num_features)),

        LSTM(128, return_sequences=True,
             kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout_rate),

        LSTM(64, return_sequences=True,
             kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout_rate),

        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(32, activation="relu"),
        Dropout(dropout_rate),

        Dense(num_classes, activation="softmax"),
    ], name="sign_lstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────
# GRU  (experiment – usually faster than LSTM)
# ──────────────────────────────────────────────
def build_gru(num_classes: int,
              sequence_length: int = SEQUENCE_LENGTH,
              num_features: int    = NUM_FEATURES,
              dropout_rate: float  = 0.3) -> Sequential:
    """
    3-layer stacked GRU – compare with LSTM on same dataset.
    """
    model = Sequential([
        Input(shape=(sequence_length, num_features)),

        GRU(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout_rate),

        GRU(64, return_sequences=True, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout_rate),

        GRU(32, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(32, activation="relu"),
        Dropout(dropout_rate),

        Dense(num_classes, activation="softmax"),
    ], name="sign_gru")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────
# MLP  (experiment – flat features, fast baseline)
# ──────────────────────────────────────────────
def build_mlp(num_classes: int,
              sequence_length: int = SEQUENCE_LENGTH,
              num_features: int    = NUM_FEATURES,
              dropout_rate: float  = 0.4) -> tf.keras.Model:
    """
    Flatten the (30,126) input → fully-connected MLP.
    Useful as a simple non-sequential baseline.
    """
    flat_dim = sequence_length * num_features   # 3780

    inputs = tf.keras.Input(shape=(sequence_length, num_features))
    x = tf.keras.layers.Flatten()(inputs)

    for units in [512, 256, 128, 64]:
        x = Dense(units, activation="relu", kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="sign_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────
BUILDERS = {
    "lstm": build_lstm,
    "gru" : build_gru,
    "mlp" : build_mlp,
}

def get_model(arch: str, num_classes: int, **kwargs) -> tf.keras.Model:
    """
    Usage:
        model = get_model("lstm", num_classes=5)
    """
    arch = arch.lower()
    if arch not in BUILDERS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(BUILDERS)}")
    return BUILDERS[arch](num_classes=num_classes, **kwargs)