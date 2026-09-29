"""Evaluate the trained encoder on the held-out TEST split and probe the rejection gates.
Writes reports/evaluation.md.   python -m scripts.evaluate

Rejection probes are PROXIES (no real idle footage exists yet):
  - no-hand windows (all zeros)          -> must be rejected
  - spliced windows (half of class A + half of class B, i.e. a transition) -> should be rejected
  - accepted-and-correct / accepted-and-wrong rates on real test windows
"""
import numpy as np
import torch

from signvla import config
from signvla.encoder.data import load_dataset
from signvla.encoder.model import load_checkpoint
from signvla.live.recognizer import Recognizer


def confusion(y, p, n):
    m = np.zeros((n, n), dtype=int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model, gestures = load_checkpoint()
    ckpt = torch.load(str(config.MODEL_PATH), weights_only=False)
    d = load_dataset(ckpt["split_strategy"])
    X, y = d["X_test"], d["y_test"]
    rec = Recognizer(model, gestures)

    probs = rec.probabilities(X)
    pred = probs.argmax(1)
    acc = (pred == y).mean()
    cm = confusion(y, pred, len(gestures))
    gates = [rec.gate(w, p) for w, p in zip(X, probs)]
    accepted = np.array([g.accepted for g in gates])
    correct = pred == y

    zero = np.zeros((50, config.SEQUENCE_LENGTH, config.FEATURES_PER_FRAME), dtype=np.float32)
    zero_rej = np.mean([not rec.gate(w, p).accepted for w, p in zip(zero, rec.probabilities(zero))])

    h = config.SEQUENCE_LENGTH // 2
    spliced = []
    for _ in range(300):
        a, b = rng.choice(len(X), 2, replace=False)
        if y[a] != y[b]:
            spliced.append(np.concatenate([X[a][:h], X[b][h:]]))
    spliced = np.asarray(spliced)
    sp_rej = np.mean([not rec.gate(w, p).accepted for w, p in zip(spliced, rec.probabilities(spliced))])

    L = [f"# Evaluation (split = {ckpt['split_strategy']}, n_test = {len(y)})", "",
         f"- Top-1 accuracy: **{acc * 100:.1f}%**  (val at checkpoint: {ckpt['val_acc'] * 100:.1f}%)",
         f"- Gates: conf >= {rec.conf_threshold}, margin >= {rec.margin_threshold}, hand frames >= {rec.min_hand_fraction}",
         f"- Test windows accepted: {accepted.mean() * 100:.1f}%  | accepted AND correct: {(accepted & correct).mean() * 100:.1f}%"
         f"  | accepted but WRONG (false command): {(accepted & ~correct).mean() * 100:.1f}%",
         f"- No-hand windows rejected: {zero_rej * 100:.0f}%",
         f"- Spliced transition windows rejected (proxy for ambiguous input): {sp_rej * 100:.0f}%  (n={len(spliced)})",
         "", "## Per-class recall", "", "| class | n | recall | most confused with |", "|---|---|---|---|"]
    for i, g in enumerate(gestures):
        n = cm[i].sum()
        off = cm[i].copy(); off[i] = 0
        conf = f"{gestures[off.argmax()]} ({off.max()})" if off.max() else "-"
        L.append(f"| {g} | {n} | {cm[i, i] / max(n, 1) * 100:.0f}% | {conf} |")
    L += ["", "## Confusion matrix (rows = true, cols = predicted)", "", "```",
          "         " + " ".join(f"{g[:4]:>4}" for g in gestures)]
    for i, g in enumerate(gestures):
        L.append(f"{g:<8} " + " ".join(f"{v:>4}" for v in cm[i]))
    L += ["```", "", "Caveat: single signer, single setup. See docs/EVALUATION.md for what is NOT yet measured."]
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "evaluation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
