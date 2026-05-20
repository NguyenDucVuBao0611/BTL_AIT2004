"""
training/export_model.py
Purpose : Convert best .h5 model sang TFLite để Person 3 dùng inference realtime.
          Tự động cập nhật models/model_config.json sau khi export.
Usage   : python training/export_model.py [--arch lstm|gru|mlp]
Output  : models/sign_<arch>.tflite
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "dataset"
CONFIG_FILE = MODELS_DIR / "model_config.json"


# ── Helpers ────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[export] model_config.json updated → {CONFIG_FILE}")


def get_representative_dataset(arch: str):
    """
    Dùng X_test để calibrate quantization (representative dataset).
    Yields từng sample shape (1, 30, 126).
    """
    x_test_path = DATA_DIR / "X_test.npy"
    if not x_test_path.exists():
        print("[export] X_test.npy not found – skipping representative dataset")
        return None

    X = np.load(x_test_path).astype(np.float32)

    def generator():
        for sample in X[:100]:                      # 100 samples đủ để calibrate
            yield [sample[np.newaxis, ...]]         # shape: (1, 30, 126)

    return generator


# ── Export full-precision TFLite ───────────────────────────────────────────
def export_float32(model: tf.keras.Model, out_path: Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    size_kb = out_path.stat().st_size / 1024
    print(f"[export] Float32 TFLite → {out_path}  ({size_kb:.1f} KB)")
    return out_path


# ── Export INT8 quantized TFLite (nhỏ hơn ~4x, nhanh hơn) ─────────────────
def export_int8(model: tf.keras.Model,
                out_path: Path,
                arch: str) -> Path | None:
    rep_dataset = get_representative_dataset(arch)
    if rep_dataset is None:
        print("[export] Skipping INT8 – no representative dataset")
        return None

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32   # giữ float input cho dễ dùng
    converter.inference_output_type = tf.float32

    try:
        tflite_model = converter.convert()
        out_path.write_bytes(tflite_model)
        size_kb = out_path.stat().st_size / 1024
        print(f"[export] INT8 quantized TFLite → {out_path}  ({size_kb:.1f} KB)")
        return out_path
    except Exception as e:
        print(f"[export] INT8 failed: {e}")
        return None


# ── Verify exported model ──────────────────────────────────────────────────
def verify_tflite(tflite_path: Path, arch: str):
    """Chạy 1 inference thử để xác nhận model không bị lỗi sau convert."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    inp  = interpreter.get_input_details()[0]
    out  = interpreter.get_output_details()[0]
    print(f"[verify] Input  shape : {inp['shape']}  dtype={inp['dtype']}")
    print(f"[verify] Output shape : {out['shape']}  dtype={out['dtype']}")

    dummy = np.zeros(inp["shape"], dtype=np.float32)
    interpreter.set_tensor(inp["index"], dummy)
    interpreter.invoke()
    result = interpreter.get_tensor(out["index"])

    print(f"[verify] Test inference OK – output shape {result.shape}  "
          f"sum={result.sum():.4f}")


# ── Compare .h5 vs TFLite accuracy ────────────────────────────────────────
def compare_accuracy(model: tf.keras.Model, tflite_path: Path):
    x_path = DATA_DIR / "X_test.npy"
    y_path = DATA_DIR / "Y_test.npy"
    if not x_path.exists():
        return

    X = np.load(x_path).astype(np.float32)
    Y = np.load(y_path)
    y_true = np.argmax(Y, axis=1)

    # ── Keras accuracy ─────────────────────────────────────────────────────
    keras_preds = np.argmax(model.predict(X, verbose=0), axis=1)
    keras_acc   = (keras_preds == y_true).mean()

    # ── TFLite accuracy ────────────────────────────────────────────────────
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    tflite_preds = []
    for sample in X:
        interpreter.set_tensor(inp_idx, sample[np.newaxis, ...])
        interpreter.invoke()
        tflite_preds.append(np.argmax(interpreter.get_tensor(out_idx)))
    tflite_acc = (np.array(tflite_preds) == y_true).mean()

    print(f"\n[compare] Keras  accuracy : {keras_acc:.4f}")
    print(f"[compare] TFLite accuracy : {tflite_acc:.4f}")
    diff = abs(keras_acc - tflite_acc)
    print(f"[compare] Accuracy drop   : {diff:.4f}  "
          f"{'✅ OK' if diff < 0.01 else '⚠️  Check quantization'}")


# ── Main ───────────────────────────────────────────────────────────────────
def export(arch: str = "lstm"):
    h5_path = MODELS_DIR / f"sign_{arch}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"{h5_path} not found.\n"
            f"Run  python training/train.py --arch {arch}  first."
        )

    print(f"\n[export] Loading {h5_path}")
    model = tf.keras.models.load_model(str(h5_path))
    model.summary()

    # ── Float32 export ─────────────────────────────────────────────────────
    fp32_path = MODELS_DIR / f"sign_{arch}.tflite"
    export_float32(model, fp32_path)
    verify_tflite(fp32_path, arch)

    # ── INT8 export ────────────────────────────────────────────────────────
    int8_path = MODELS_DIR / f"sign_{arch}_int8.tflite"
    int8_result = export_int8(model, int8_path, arch)

    # ── Accuracy comparison ────────────────────────────────────────────────
    compare_accuracy(model, fp32_path)

    # ── Update model_config.json ───────────────────────────────────────────
    config = load_config()
    config["tflite_file"]      = str(fp32_path.relative_to(ROOT))
    config["tflite_int8_file"] = str(int8_path.relative_to(ROOT)) if int8_result else None
    config["best_arch"]        = arch
    config["model_file"]       = str(h5_path.relative_to(ROOT))
    save_config(config)

    print(f"\n✅  Export hoàn thành!")
    print(f"   Float32 : {fp32_path}")
    if int8_result:
        print(f"   INT8    : {int8_path}")
    print(f"   Config  : {CONFIG_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["lstm", "gru", "mlp"], default="lstm")
    args = parser.parse_args()
    export(args.arch)


if __name__ == "__main__":
    main()