"""
training/evaluate.py
Purpose : Load best saved model, run on test set, generate:
            - Confusion matrix plot  (logs/confusion_matrix_<arch>.png)
            - Classification report  (logs/report_<arch>.txt)
            - Per-sample error log   (logs/errors_<arch>.csv)
Usage   : python training/evaluate.py [--arch lstm|gru|mlp]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tensorflow.keras.models import load_model

# ── Project root on sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR   = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
LOGS_DIR   = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────
def load_labels() -> dict[int, str]:
    with open(DATA_DIR / "labels.json", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_test_data():
    X = np.load(DATA_DIR / "X_test.npy")
    Y = np.load(DATA_DIR / "Y_test.npy")   # one-hot
    return X, Y


def plot_confusion_matrix(cm: np.ndarray,
                           labels: list[str],
                           arch: str):
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix – {arch.upper()}", fontsize=14)
    plt.tight_layout()

    out_path = LOGS_DIR / f"confusion_matrix_{arch}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {out_path}")


def save_report(report_str: str, arch: str):
    out_path = LOGS_DIR / f"report_{arch}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[evaluate] Classification report saved → {out_path}")


def save_error_log(X: np.ndarray,
                   y_true: np.ndarray,
                   y_pred: np.ndarray,
                   labels: list[str],
                   arch: str):
    """
    CSV with one row per mis-classified sample:
      index, true_label, pred_label, confidence
    """
    out_path = LOGS_DIR / f"errors_{arch}.csv"
    errors = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "pred_label", "confidence"])
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            if yt != yp:
                errors += 1
    print(f"[evaluate] {errors} errors saved → {out_path}")


def plot_training_curves(arch: str):
    """Plot loss/accuracy curves from training_history.json if available."""
    hist_path = MODELS_DIR / "training_history.json"
    if not hist_path.exists():
        return

    with open(hist_path) as f:
        all_results = json.load(f)

    result = next((r for r in all_results if r["arch"] == arch), None)
    if result is None:
        return

    h = result["history"]
    epochs = range(1, len(h["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, h["loss"],     label="Train loss")
    ax1.plot(epochs, h["val_loss"], label="Val loss",  linestyle="--")
    ax1.set_title(f"{arch.upper()} – Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, h["accuracy"],     label="Train acc")
    ax2.plot(epochs, h["val_accuracy"], label="Val acc", linestyle="--")
    ax2.set_title(f"{arch.upper()} – Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    out_path = LOGS_DIR / f"training_curves_{arch}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Training curves saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────
def evaluate(arch: str = "lstm"):
    model_path = MODELS_DIR / f"sign_{arch}.h5"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run  python training/train.py --arch {arch}  first."
        )

    print(f"\n[evaluate] Loading model: {model_path}")
    model = load_model(str(model_path))

    X_test, Y_test = load_test_data()
    label_map      = load_labels()
    class_names    = [label_map[i] for i in range(len(label_map))]

    # ── predictions ────────────────────────────────────────────────────────
    probs   = model.predict(X_test, verbose=0)           # (N, C)
    y_pred  = np.argmax(probs,   axis=1)
    y_true  = np.argmax(Y_test,  axis=1)
    confs   = probs[np.arange(len(probs)), y_pred]       # confidence scores

    # ── overall accuracy ────────────────────────────────────────────────────
    acc = (y_pred == y_true).mean()
    print(f"[evaluate] Test accuracy: {acc:.4f}  ({int(acc*len(y_true))}/{len(y_true)})")

    # ── confusion matrix ────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, arch)

    # ── classification report ───────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n" + report)
    save_report(report, arch)

    # ── error log ───────────────────────────────────────────────────────────
    save_error_log(X_test, y_true, y_pred, class_names, arch)

    # ── training curves ─────────────────────────────────────────────────────
    plot_training_curves(arch)

    # ── per-class summary ────────────────────────────────────────────────────
    print("\n[evaluate] Per-class accuracy:")
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        class_acc = (y_pred[mask] == i).mean()
        bar = "█" * int(class_acc * 20)
        print(f"  {name:<12} {class_acc:.2%}  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", choices=["lstm", "gru", "mlp"], default="lstm"
    )
    args = parser.parse_args()
    evaluate(args.arch)


if __name__ == "__main__":
    main()