"""
Dataset Compiler
Compiles individual .npy landmark sequences into a unified train/val/test dataset file (dataset.npz)
and generates label mappings (label_map.json).
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Auto re-exec inside virtual environment if invoked with global system Python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PY = _PROJECT_ROOT / ".venv" / "bin" / "python"
if not _VENV_PY.exists():
    _VENV_PY = _PROJECT_ROOT / "venv" / "bin" / "python"

if _VENV_PY.exists() and sys.prefix == sys.base_prefix:
    os.execv(str(_VENV_PY), [str(_VENV_PY)] + sys.argv)

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(str(_PROJECT_ROOT))
import config


def build_dataset(test_size=0.2, random_state=42):
    print("\n" + "=" * 60)
    print("  ISL GESTURE DATASET COMPILER")
    print("=" * 60)

    gesture_dirs = [d for d in config.LANDMARKS_DIR.iterdir() if d.is_dir()]
    if not gesture_dirs:
        print("[ERROR] No gesture folders found in data/landmarks/. Please record samples first.")
        return False

    gesture_names = sorted([d.name for d in gesture_dirs])
    label_to_idx = {name: idx for idx, name in enumerate(gesture_names)}
    idx_to_label = {idx: name for idx, name in enumerate(gesture_names)}

    X = []
    y = []
    class_counts = {}

    for gesture in gesture_names:
        g_dir = config.LANDMARKS_DIR / gesture
        sample_files = sorted(list(g_dir.glob("sample_*.npy")))
        class_counts[gesture] = len(sample_files)

        for s_file in sample_files:
            arr = np.load(str(s_file))
            # Validate shape (config.SEQUENCE_LENGTH, 126)
            if arr.shape != (config.SEQUENCE_LENGTH, config.TOTAL_FEATURES_PER_FRAME):
                print(f"[WARN] Sample {s_file.name} has unexpected shape {arr.shape}. Skipping.")
                continue
            X.append(arr)
            y.append(label_to_idx[gesture])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    total_samples = len(X)
    print(f"\nDiscovered {len(gesture_names)} classes with {total_samples} valid sequences:")
    for g, count in class_counts.items():
        print(f"  - {g:<15}: {count} samples")

    if total_samples < len(gesture_names) * 2:
        print("[WARN] Dataset is very small! Collect more samples per gesture for reliable training.")

    # Train / Test split
    if total_samples >= 5:
        # Check if minimum class count allows stratified split
        min_class_count = min(class_counts.values()) if class_counts else 0
        stratify = y if min_class_count >= 2 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    else:
        # Fallback for minimal testing
        X_train, y_train = X, y
        X_test, y_test = X, y

    # Save dataset.npz
    npz_path = config.DATA_DIR / "dataset.npz"
    np.savez_compressed(
        str(npz_path),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    # Save label_map.json
    label_map_path = config.DATA_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "gestures": gesture_names
        }, f, indent=2)

    print("\n" + "-" * 60)
    print(f"Dataset successfully compiled!")
    print(f"  Train shape : X={X_train.shape}, y={y_train.shape}")
    print(f"  Test shape  : X={X_test.shape}, y={y_test.shape}")
    print(f"  Files saved :")
    print(f"    - {npz_path}")
    print(f"    - {label_map_path}")
    print("-" * 60 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile gesture landmarks into training dataset.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    args = parser.parse_args()

    build_dataset(test_size=args.test_size, random_state=args.seed)
