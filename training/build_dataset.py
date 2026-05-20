"""
training/build_dataset.py
Purpose : Load raw .npy sequences (from Person 1) and build
          X.npy  shape (N, 30, 126)
          y.npy  shape (N, num_classes)   one-hot encoded
          Also saves labels.json mapping index → gesture name.
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
RAW_DIR      = ROOT / "dataset" / "raw"
LABELS_FILE  = ROOT / "dataset" / "labels.json"
OUT_DIR      = ROOT / "dataset"

SEQUENCE_LENGTH = 30
NUM_FEATURES    = 126
TEST_SIZE       = 0.2
VAL_SIZE        = 0.1   # fraction of training set
RANDOM_SEED     = 42


def discover_gestures(raw_dir: Path) -> list[str]:
    """Return sorted list of gesture folder names."""
    gestures = sorted([
        d.name for d in raw_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    print(f"[build_dataset] Found {len(gestures)} gestures: {gestures}")
    return gestures


def load_sequences(raw_dir: Path, gestures: list[str]):
    """
    Read every .npy file under raw_dir/<gesture>/.
    Expected shape per file: (SEQUENCE_LENGTH, NUM_FEATURES).
    Returns X (N, 30, 126) and y (N,) as integer class indices.
    """
    X, y = [], []
    skipped = 0

    for class_idx, gesture in enumerate(gestures):
        gesture_dir = raw_dir / gesture
        files = sorted(gesture_dir.glob("*.npy"))

        if not files:
            print(f"  [WARN] No .npy files in {gesture_dir}")
            continue

        for fp in files:
            seq = np.load(fp)

            # ── shape guard ────────────────────────────────────────────────
            if seq.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                print(f"  [SKIP] {fp.name}: shape {seq.shape} ≠ "
                      f"({SEQUENCE_LENGTH},{NUM_FEATURES})")
                skipped += 1
                continue

            X.append(seq)
            y.append(class_idx)

        print(f"  Loaded {len(files) - skipped} sequences for '{gesture}'")

    X = np.array(X, dtype=np.float32)   # (N, 30, 126)
    y = np.array(y, dtype=np.int32)     # (N,)

    print(f"\n[build_dataset] Total: X={X.shape}  y={y.shape}  "
          f"skipped={skipped}")
    return X, y


def save_labels(gestures: list[str], out_path: Path):
    """Save index→gesture mapping to JSON (used by inferencer)."""
    mapping = {str(i): g for i, g in enumerate(gestures)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"[build_dataset] Labels saved → {out_path}")


def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-feature z-score normalisation computed on the FULL dataset.
    Returns (X_norm, mean, std) so Person 3 can reuse mean/std at inference.
    """
    mean = X.mean(axis=(0, 1), keepdims=True)   # (1,1,126)
    std  = X.std (axis=(0, 1), keepdims=True) + 1e-8
    return (X - mean) / std, mean.squeeze(), std.squeeze()


def build(test_size=TEST_SIZE, val_size=VAL_SIZE, seed=RANDOM_SEED):
    """Main pipeline: raw → X/y splits → saved to disk."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gestures = discover_gestures(RAW_DIR)
    assert gestures, "No gesture folders found in dataset/raw/"

    save_labels(gestures, LABELS_FILE)

    X, y = load_sequences(RAW_DIR, gestures)

    num_classes = len(gestures)
    Y = to_categorical(y, num_classes=num_classes)   # one-hot (N, C)

    # ── train / test split ─────────────────────────────────────────────────
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed, stratify=y
    )

    # ── train / val split ──────────────────────────────────────────────────
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train,
        test_size=val_size, random_state=seed
    )

    # ── normalise (fit on train only) ──────────────────────────────────────
    X_train_n, mean, std = normalize(X_train)
    X_val_n  = (X_val  - mean) / (std + 1e-8)
    X_test_n = (X_test - mean) / (std + 1e-8)

    # ── save ───────────────────────────────────────────────────────────────
    splits = {
        "X_train": X_train_n,
        "Y_train": Y_train,
        "X_val"  : X_val_n,
        "Y_val"  : Y_val,
        "X_test" : X_test_n,
        "Y_test" : Y_test,
        "mean"   : mean,
        "std"    : std,
    }
    for name, arr in splits.items():
        path = OUT_DIR / f"{name}.npy"
        np.save(path, arr)
        print(f"  Saved {name}.npy  shape={arr.shape}")

    print(f"\n[build_dataset] Done. "
          f"Train={len(X_train_n)}  Val={len(X_val_n)}  "
          f"Test={len(X_test_n)}  Classes={num_classes}")
    return splits


if __name__ == "__main__":
    build()