"""
training/train.py
Purpose : Train LSTM (baseline), then GRU and MLP for comparison.
          Saves:
            models/sign_lstm.h5              ← best LSTM weights
            models/sign_gru.h5
            models/sign_mlp.h5
            models/training_history.json     ← loss/accuracy curves
Usage   : python training/train.py [--arch lstm|gru|mlp|all]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)

# allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.model import get_model

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
LOGS_DIR   = ROOT / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────────────────
EPOCHS     = 100
BATCH_SIZE = 32
PATIENCE   = 15          # early stopping


# ── Helpers ────────────────────────────────────────────────────────────────
def load_data():
    needed = ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"]
    missing = [n for n in needed if not (DATA_DIR / f"{n}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing: {missing}.\n"
            "Run  python training/build_dataset.py  first."
        )
    data = {k: np.load(DATA_DIR / f"{k}.npy") for k in needed}
    print("[train] Dataset loaded:")
    for k, v in data.items():
        print(f"  {k}: {v.shape}")
    return data


def get_callbacks(arch: str) -> list:
    model_path = MODELS_DIR / f"sign_{arch}.h5"
    csv_path   = LOGS_DIR   / f"train_{arch}.csv"

    return [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(str(csv_path), append=False),
    ]


def train_arch(arch: str, data: dict) -> dict:
    """Train one architecture; return history dict."""
    print(f"\n{'='*55}")
    print(f"  Training: {arch.upper()}")
    print(f"{'='*55}")

    num_classes = data["Y_train"].shape[1]
    model = get_model(arch, num_classes=num_classes)
    model.summary()

    history = model.fit(
        data["X_train"], data["Y_train"],
        validation_data=(data["X_val"], data["Y_val"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(arch),
        verbose=1,
    )

    # ── evaluate on test set ───────────────────────────────────────────────
    loss, acc = model.evaluate(data["X_test"], data["Y_test"], verbose=0)
    print(f"\n[{arch.upper()}] Test loss={loss:.4f}  Test acc={acc:.4f}")

    return {
        "arch"      : arch,
        "test_loss" : round(float(loss), 6),
        "test_acc"  : round(float(acc),  6),
        "history"   : {k: [round(v, 6) for v in vals]
                       for k, vals in history.history.items()},
    }


def save_history(results: list[dict]):
    out = MODELS_DIR / "training_history.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[train] History saved → {out}")

    # ── print comparison table ─────────────────────────────────────────────
    print("\n" + "─"*40)
    print(f"{'ARCH':<8} {'TEST LOSS':>12} {'TEST ACC':>10}")
    print("─"*40)
    for r in results:
        print(f"{r['arch'].upper():<8} {r['test_loss']:>12.4f} "
              f"{r['test_acc']:>10.4f}")
    print("─"*40)

    best = max(results, key=lambda r: r["test_acc"])
    print(f"\n✅  Best model: {best['arch'].upper()}  "
          f"(acc={best['test_acc']:.4f})")


# ── Entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train sign language models")
    parser.add_argument(
        "--arch",
        choices=["lstm", "gru", "mlp", "all"],
        default="all",
        help="Architecture to train (default: all)",
    )
    args = parser.parse_args()

    data    = load_data()
    archs   = ["lstm", "gru", "mlp"] if args.arch == "all" else [args.arch]
    results = [train_arch(arch, data) for arch in archs]

    save_history(results)


if __name__ == "__main__":
    # GPU memory growth (prevents OOM on shared machines)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    main()