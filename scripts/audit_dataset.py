"""Data-quality audit: class balance, hand presence, empty samples, duplicates, outliers.
Writes reports/dataset_audit.md.

Usage: python -m scripts.audit_dataset
"""
import json
from collections import Counter

import numpy as np

from signvla import config


def main():
    d = np.load(config.DATASET_PATH)
    X, y = d["X"], d["y"]
    names = json.load(open(config.LABEL_MAP_PATH))["gestures"]
    slot0 = (np.abs(X[:, :, :63]).sum(2) > 0)
    slot1 = (np.abs(X[:, :, 63:]).sum(2) > 0)
    any_hand = slot0 | slot1

    lines = ["# Dataset audit", "",
             f"{len(X)} sequences x {X.shape[1]} frames x {X.shape[2]} features, {len(names)} classes.", "",
             "| class | n | frames w/ hand | mostly-two-hand samples | samples <50% hand frames |",
             "|---|---|---|---|---|"]
    weak = []
    for i, n in enumerate(names):
        m = y == i
        cover = any_hand[m].mean(1)
        two = ((slot0[m] & slot1[m]).mean(1) > 0.5).sum()
        low = int((cover < 0.5).sum())
        weak += [(n, int(j)) for j in np.where(m)[0][cover < 0.5]]
        lines.append(f"| {n} | {m.sum()} | {any_hand[m].mean():.2f} | {two} | {low} |")

    dup = sum(v - 1 for v in Counter(x.tobytes() for x in X).values() if v > 1)
    counts = np.bincount(y)
    lines += ["", "## Findings",
              f"- Exact duplicate sequences: {dup}",
              f"- Class imbalance (max/min): {counts.max() / counts.min():.2f}",
              f"- Samples with <50% hand frames (likely tracking failures, candidates to drop/re-record): {len(weak)}",
              "- Normalization invariant is enforced by `python -m scripts.fix_landmarks --verify`.",
              "- All samples come from ONE signer/setup: random-split accuracy is optimistic. "
              "Use the `block` split and see docs/EVALUATION.md."]
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "dataset_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
