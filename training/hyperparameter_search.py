"""
training/hyperparameter_search.py
Purpose : Tự động tìm hyperparameters tốt nhất cho LSTM bằng Keras Tuner.
          Tìm kiếm: units, dropout, learning_rate, num_layers.
          Kết quả lưu vào logs/hp_search_results.json

Cài thêm: pip install keras-tuner
Usage   : python training/hyperparameter_search.py [--trials 20] [--arch lstm|gru]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "dataset"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30
NUM_FEATURES    = 126


# ── Check keras-tuner installed ────────────────────────────────────────────
try:
    import keras_tuner as kt
except ImportError:
    print("❌  keras-tuner chưa được cài.")
    print("    Chạy:  pip install keras-tuner")
    sys.exit(1)


# ── Load data ──────────────────────────────────────────────────────────────
def load_data():
    needed = ["X_train", "Y_train", "X_val", "Y_val"]
    missing = [n for n in needed if not (DATA_DIR / f"{n}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu files: {missing}\n"
            "Chạy  python training/build_dataset.py  trước."
        )
    return {k: np.load(DATA_DIR / f"{k}.npy") for k in needed}


# ══════════════════════════════════════════════════════════════════════════
# HyperModel builders
# ══════════════════════════════════════════════════════════════════════════

class LSTMHyperModel(kt.HyperModel):
    """Tìm units, layers, dropout, learning_rate cho LSTM."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        # ── Hyperparameters space ──────────────────────────────────────────
        units_1      = hp.Choice("units_layer1", [64, 128, 256])
        units_2      = hp.Choice("units_layer2", [32, 64, 128])
        num_layers   = hp.Int  ("num_lstm_layers", min_value=2, max_value=3)
        dropout_rate = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
        lr           = hp.Choice("learning_rate", [1e-2, 1e-3, 5e-4, 1e-4])
        use_bn       = hp.Boolean("batch_norm", default=True)

        model = tf.keras.Sequential(name="lstm_hp")
        model.add(tf.keras.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)))

        # Layer 1
        model.add(tf.keras.layers.LSTM(units_1, return_sequences=(num_layers > 1)))
        if use_bn:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout_rate))

        # Layer 2 (optional layer 3)
        if num_layers >= 2:
            return_seq = (num_layers == 3)
            model.add(tf.keras.layers.LSTM(units_2, return_sequences=return_seq))
            if use_bn:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout_rate))

        if num_layers == 3:
            model.add(tf.keras.layers.LSTM(32, return_sequences=False))
            if use_bn:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout_rate))

        # Dense head
        dense_units = hp.Choice("dense_units", [32, 64])
        model.add(tf.keras.layers.Dense(dense_units, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class GRUHyperModel(kt.HyperModel):
    """Tìm units, layers, dropout, learning_rate cho GRU."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        units_1      = hp.Choice("units_layer1", [64, 128, 256])
        units_2      = hp.Choice("units_layer2", [32, 64, 128])
        num_layers   = hp.Int  ("num_gru_layers", min_value=2, max_value=3)
        dropout_rate = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
        lr           = hp.Choice("learning_rate", [1e-2, 1e-3, 5e-4, 1e-4])

        model = tf.keras.Sequential(name="gru_hp")
        model.add(tf.keras.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)))

        model.add(tf.keras.layers.GRU(units_1, return_sequences=(num_layers > 1)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout_rate))

        if num_layers >= 2:
            return_seq = (num_layers == 3)
            model.add(tf.keras.layers.GRU(units_2, return_sequences=return_seq))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout_rate))

        if num_layers == 3:
            model.add(tf.keras.layers.GRU(32, return_sequences=False))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(32, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


# ── Run search ─────────────────────────────────────────────────────────────
def run_search(arch: str = "lstm",
               max_trials: int = 20,
               epochs_per_trial: int = 40):

    data        = load_data()
    num_classes = data["Y_train"].shape[1]

    print(f"\n{'='*55}")
    print(f"  Hyperparameter Search: {arch.upper()}")
    print(f"  Trials={max_trials}  Epochs/trial={epochs_per_trial}")
    print(f"{'='*55}\n")

    # ── Choose HyperModel ──────────────────────────────────────────────────
    hypermodel = (LSTMHyperModel if arch == "lstm" else GRUHyperModel)(num_classes)

    # ── Tuner: Bayesian Optimization ──────────────────────────────────────
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective       = kt.Objective("val_accuracy", direction="max"),
        max_trials      = max_trials,
        num_initial_points = max(5, max_trials // 4),
        directory       = str(LOGS_DIR / "hp_tuner"),
        project_name    = f"sign_{arch}",
        overwrite       = False,
    )

    tuner.search_space_summary()

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    # ── Search ────────────────────────────────────────────────────────────
    tuner.search(
        data["X_train"], data["Y_train"],
        validation_data = (data["X_val"], data["Y_val"]),
        epochs          = epochs_per_trial,
        batch_size      = 32,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # ── Results ───────────────────────────────────────────────────────────
    tuner.results_summary(num_trials=5)

    best_hp     = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_values = best_hp.values

    print(f"\n✅  Best hyperparameters ({arch.upper()}):")
    for k, v in best_values.items():
        print(f"   {k:<22} = {v}")

    # ── Save results ───────────────────────────────────────────────────────
    # Top 5 trials
    top_trials = []
    for trial in tuner.oracle.get_best_trials(num_trials=5):
        top_trials.append({
            "trial_id" : trial.trial_id,
            "val_acc"  : round(trial.score, 6),
            "params"   : trial.hyperparameters.values,
        })

    output = {
        "arch"       : arch,
        "max_trials" : max_trials,
        "best_params": best_values,
        "top_trials" : top_trials,
    }
    out_path = LOGS_DIR / f"hp_search_results_{arch}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[hp_search] Results saved → {out_path}")

    # ── Retrain best model to full epochs ─────────────────────────────────
    print(f"\n[hp_search] Retraining best {arch.upper()} with full 100 epochs…")
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        data["X_train"], data["Y_train"],
        validation_data = (data["X_val"], data["Y_val"]),
        epochs          = 100,
        batch_size      = 32,
        callbacks       = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath  = str(ROOT / "models" / f"sign_{arch}_tuned.h5"),
                monitor   = "val_accuracy",
                save_best_only = True,
                verbose   = 1,
            ),
        ],
        verbose = 1,
    )
    print(f"[hp_search] Best model saved → models/sign_{arch}_tuned.h5")


# ── Entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",   choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--trials", type=int, default=20,
                        help="Số lượng trials (default: 20)")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Epochs mỗi trial (default: 40)")
    args = parser.parse_args()

    run_search(arch=args.arch, max_trials=args.trials,
               epochs_per_trial=args.epochs)


if __name__ == "__main__":
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    main()