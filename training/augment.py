"""
training/augment.py
Purpose : Data augmentation cho sequences (30, 126) để model robust hơn
          khi gesture bị lệch, nhiễu, hoặc tốc độ khác nhau.

Kỹ thuật:
  1. Gaussian noise        – thêm nhiễu nhỏ vào keypoints
  2. Time shift            – dịch sequence sang trái/phải
  3. Time scale (stretch)  – kéo dài / rút ngắn rồi resample về 30 frames
  4. Frame dropout         – đặt ngẫu nhiên vài frame về 0 (giả lập mất frame)
  5. Mirror / flip         – lật trái-phải keypoints tay

Usage:
  from training.augment import augment_dataset
  X_aug, Y_aug = augment_dataset(X_train, Y_train, multiplier=3)
"""

import numpy as np
from scipy.interpolate import interp1d

# ── Constants ──────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
NUM_FEATURES    = 126
RNG             = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════
# Individual augmentation functions
# Each takes a single sequence (30, 126) and returns (30, 126)
# ══════════════════════════════════════════════════════════════════════════

def add_gaussian_noise(seq: np.ndarray,
                       std: float = 0.02) -> np.ndarray:
    """Thêm Gaussian noise vào tất cả keypoints."""
    noise = RNG.normal(0, std, size=seq.shape).astype(np.float32)
    return seq + noise


def time_shift(seq: np.ndarray,
               max_shift: int = 5) -> np.ndarray:
    """
    Dịch sequence theo trục thời gian.
    Phần thiếu được fill bằng frame đầu hoặc cuối (edge padding).
    """
    shift = RNG.integers(-max_shift, max_shift + 1)
    if shift == 0:
        return seq.copy()

    result = np.zeros_like(seq)
    if shift > 0:
        result[shift:]  = seq[:-shift]
        result[:shift]  = seq[0]          # pad bằng frame đầu
    else:
        result[:shift]  = seq[-shift:]
        result[shift:]  = seq[-1]         # pad bằng frame cuối
    return result


def time_scale(seq: np.ndarray,
               scale_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """
    Kéo dài hoặc rút ngắn sequence rồi resample về SEQUENCE_LENGTH frames.
    scale < 1 → nhanh hơn  |  scale > 1 → chậm hơn
    """
    scale = RNG.uniform(*scale_range)
    src_len = int(SEQUENCE_LENGTH * scale)
    src_len = max(2, src_len)

    # Resample seq → src_len frames
    old_idx = np.linspace(0, SEQUENCE_LENGTH - 1, src_len)
    new_idx = np.linspace(0, SEQUENCE_LENGTH - 1, SEQUENCE_LENGTH)

    result = np.zeros_like(seq)
    for feat in range(NUM_FEATURES):
        f = interp1d(old_idx, seq[:, feat], kind="linear",
                     fill_value="extrapolate")
        # Sample src_len points, then resample back to SEQUENCE_LENGTH
        sampled = f(np.linspace(0, SEQUENCE_LENGTH - 1, src_len))
        f2 = interp1d(np.linspace(0, 1, src_len), sampled,
                      kind="linear", fill_value="extrapolate")
        result[:, feat] = f2(np.linspace(0, 1, SEQUENCE_LENGTH))

    return result.astype(np.float32)


def frame_dropout(seq: np.ndarray,
                  drop_rate: float = 0.1) -> np.ndarray:
    """
    Đặt ngẫu nhiên một số frame về 0.
    Giả lập mất frame do webcam lag hoặc MediaPipe không detect được.
    """
    result = seq.copy()
    mask   = RNG.random(SEQUENCE_LENGTH) < drop_rate
    result[mask] = 0.0
    return result


def mirror_horizontal(seq: np.ndarray) -> np.ndarray:
    """
    Lật trái-phải: x → 1 - x cho tất cả x-coordinates.
    Giả định features là (x, y) xen kẽ nhau: [x0,y0, x1,y1, ...].
    Hữu ích khi người dùng dùng tay trái thay vì tay phải.
    """
    result = seq.copy()
    # x coordinates ở các index chẵn: 0, 2, 4, ...
    result[:, 0::2] = 1.0 - result[:, 0::2]
    return result


def scale_keypoints(seq: np.ndarray,
                    scale_range: tuple = (0.85, 1.15)) -> np.ndarray:
    """
    Scale toàn bộ keypoints quanh tâm (0.5, 0.5).
    Giả lập người đứng gần/xa camera.
    """
    scale  = RNG.uniform(*scale_range)
    result = seq.copy()
    result = (result - 0.5) * scale + 0.5
    return result.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# Augmentation pipeline
# ══════════════════════════════════════════════════════════════════════════

# Danh sách (function, probability_apply)
AUGMENTATIONS = [
    (add_gaussian_noise,  0.8),
    (time_shift,          0.7),
    (time_scale,          0.5),
    (frame_dropout,       0.4),
    (mirror_horizontal,   0.3),
    (scale_keypoints,     0.5),
]


def augment_one(seq: np.ndarray) -> np.ndarray:
    """Áp dụng ngẫu nhiên các augmentation lên 1 sequence."""
    result = seq.copy()
    for fn, prob in AUGMENTATIONS:
        if RNG.random() < prob:
            result = fn(result)
    return result


def augment_dataset(X: np.ndarray,
                    Y: np.ndarray,
                    multiplier: int = 3,
                    verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Tạo thêm (multiplier - 1) bản augmented cho mỗi sample.
    Kết quả: (X_aug, Y_aug) với size = N * multiplier.

    Args:
        X          : (N, 30, 126) float32
        Y          : (N, num_classes) one-hot float32
        multiplier : số lần nhân dataset (3 = gốc + 2 augmented)

    Returns:
        X_aug (N*multiplier, 30, 126), Y_aug (N*multiplier, num_classes)
    """
    N = len(X)
    X_list = [X]
    Y_list = [Y]

    for i in range(multiplier - 1):
        X_new = np.array([augment_one(seq) for seq in X], dtype=np.float32)
        X_list.append(X_new)
        Y_list.append(Y)
        if verbose:
            print(f"[augment] Generated batch {i+1}/{multiplier-1}  "
                  f"shape={X_new.shape}")

    X_aug = np.concatenate(X_list, axis=0)
    Y_aug = np.concatenate(Y_list, axis=0)

    # Shuffle
    idx   = RNG.permutation(len(X_aug))
    X_aug = X_aug[idx]
    Y_aug = Y_aug[idx]

    if verbose:
        print(f"[augment] Final dataset: {N} → {len(X_aug)} samples  "
              f"(×{multiplier})")

    return X_aug, Y_aug


# ── Standalone test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path

    ROOT   = Path(__file__).resolve().parent.parent
    X      = np.load(ROOT / "dataset" / "X_train.npy")
    Y      = np.load(ROOT / "dataset" / "Y_train.npy")

    print(f"Original : X={X.shape}  Y={Y.shape}")
    X_aug, Y_aug = augment_dataset(X, Y, multiplier=3)
    print(f"Augmented: X={X_aug.shape}  Y={Y_aug.shape}")

    # Save augmented dataset
    np.save(ROOT / "dataset" / "X_train_aug.npy", X_aug)
    np.save(ROOT / "dataset" / "Y_train_aug.npy", Y_aug)
    print("Saved X_train_aug.npy + Y_train_aug.npy")