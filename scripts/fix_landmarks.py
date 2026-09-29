"""Repair legacy landmark samples so every .npy uses the canonical feature space.

Why: in v0.1 eight classes (back, close, down, front, left, open, right, up) were
recorded with `record_new_gestures.py`, which saved RAW MediaPipe coordinates, while
the other six used normalized coordinates. Live inference always used normalized
coordinates, so 8/14 classes were trained on features the live system never produces.

Usage:  python -m scripts.fix_landmarks            # backs up originals, repairs, verifies
        python -m scripts.fix_landmarks --verify   # only check the invariants
"""
import argparse
import shutil
import sys

import numpy as np

from signvla import config
from signvla.perception.features import canonicalize_sequence, is_normalized_sample

BACKUP_DIR = config.DATA_DIR / "_legacy_landmarks_backup"


def check_canonical(seq):
    """Invariants: wrist == 0, palm scale == 1 for every present hand; lone hand in slot 0."""
    for slot in range(2):
        h = seq[:, slot * 63:(slot + 1) * 63]
        present = np.abs(h).sum(axis=1) > 0
        if not present.any():
            continue
        pts = h[present].reshape(-1, 21, 3)
        if np.abs(pts[:, 0]).max() > 1e-5:
            return f"wrist not at origin (slot {slot})"
        if np.abs(np.linalg.norm(pts[:, 9], axis=1) - 1.0).max() > 1e-3:
            return f"palm scale != 1 (slot {slot})"
    slot0 = (np.abs(seq[:, :63]).sum(axis=1) > 0)
    slot1 = (np.abs(seq[:, 63:]).sum(axis=1) > 0)
    if (slot1 & ~slot0).any():
        return "frame with hand only in slot 1"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    files = sorted(config.LANDMARKS_DIR.glob("*/sample_*.npy"))
    if not files:
        sys.exit("No landmark files found in data/landmarks/")

    raw_n = norm_n = 0
    if not args.verify:
        for f in files:
            seq = np.load(f)
            if seq.shape != (config.SEQUENCE_LENGTH, config.FEATURES_PER_FRAME):
                sys.exit(f"Unexpected shape {seq.shape} in {f}")
            if is_normalized_sample(seq):
                norm_n += 1
            else:
                raw_n += 1
            dest = BACKUP_DIR / f.relative_to(config.LANDMARKS_DIR)
            if not dest.exists():  # never overwrite an earlier backup
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
            np.save(f, canonicalize_sequence(seq))
        print(f"Repaired {len(files)} samples ({raw_n} were raw, {norm_n} were normalized).")
        print(f"Originals backed up to {BACKUP_DIR}")

    bad = 0
    for f in files:
        err = check_canonical(np.load(f))
        if err:
            bad += 1
            print(f"[FAIL] {f.relative_to(config.DATA_DIR)}: {err}")
    if bad:
        sys.exit(f"{bad}/{len(files)} samples violate the canonical-feature invariants")
    print(f"OK: all {len(files)} samples satisfy the canonical-feature invariants.")


if __name__ == "__main__":
    main()
