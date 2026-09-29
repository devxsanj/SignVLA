"""Dataset compilation and split logic (single source of truth for both)."""
import json

import numpy as np
from sklearn.model_selection import train_test_split

from signvla import config

TRAIN, VAL, TEST = 0, 1, 2
SPLIT_STRATEGIES = ("block", "random")


def block_split(labels, ids, val_frac=0.15, test_frac=0.20):
    """Per class, the LAST samples (recorded later = a later 'session') go to test,
    the ones before them to val, the rest to train. Guards against the near-duplicate
    leakage a random split has on a single-signer, back-to-back recording."""
    split = np.full(len(labels), TRAIN, dtype=np.int8)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx[np.argsort(ids[idx])]
        n = len(idx)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        split[idx[n - n_test:]] = TEST
        split[idx[n - n_test - n_val:n - n_test]] = VAL
    return split


def random_split(labels, val_frac=0.15, test_frac=0.20, seed=42):
    idx = np.arange(len(labels))
    rest, test = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=labels)
    train, val = train_test_split(rest, test_size=val_frac / (1 - test_frac),
                                  random_state=seed, stratify=labels[rest])
    split = np.zeros(len(labels), dtype=np.int8)
    split[val] = VAL
    split[test] = TEST
    return split


def compile_dataset():
    """Read data/landmarks/*/sample_*.npy -> dataset.npz + label_map.json."""
    names = sorted(d.name for d in config.LANDMARKS_DIR.iterdir() if d.is_dir())
    label_to_idx = {n: i for i, n in enumerate(names)}
    X, y, ids = [], [], []
    for name in names:
        for f in sorted((config.LANDMARKS_DIR / name).glob("sample_*.npy")):
            arr = np.load(f)
            if arr.shape != (config.SEQUENCE_LENGTH, config.FEATURES_PER_FRAME):
                print(f"[WARN] skipping {f.name}: shape {arr.shape}")
                continue
            X.append(arr)
            y.append(label_to_idx[name])
            ids.append(int(f.stem.split("_")[1]))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    np.savez_compressed(
        config.DATASET_PATH, X=X, y=y, sample_id=ids,
        split_block=block_split(y, ids), split_random=random_split(y),
    )
    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump({"gestures": names, "label_to_idx": label_to_idx}, f, indent=2)
    return X, y, names


def load_dataset(strategy="block"):
    """-> dict(X_train, y_train, X_val, ..., gestures)."""
    assert strategy in SPLIT_STRATEGIES, strategy
    d = np.load(config.DATASET_PATH)
    split = d[f"split_{strategy}"]
    with open(config.LABEL_MAP_PATH) as f:
        gestures = json.load(f)["gestures"]
    out = {"gestures": gestures}
    for name, code in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        m = split == code
        out[f"X_{name}"], out[f"y_{name}"] = d["X"][m], d["y"][m]
    return out
