"""Streaming false-activation evaluation (the main-paper metric).
Replays recorded IDLE streams frame by frame through the exact live logic
(sliding window -> Recognizer gates -> Debouncer) and reports false commands per hour.
  python -m scripts.eval_stream data/streams/idle_*.npz
Every fired command on an idle stream is a false activation.
"""
import argparse
from collections import Counter, deque

import numpy as np

from signvla import config
from signvla.encoder.model import load_checkpoint
from signvla.live.recognizer import Debouncer, Recognizer


def replay(features, timestamps, rec):
    deb, win, fired = Debouncer(), deque(maxlen=config.SEQUENCE_LENGTH), []
    for f, t in zip(features, timestamps):
        win.append(f)
        if len(win) == config.SEQUENCE_LENGTH:
            label = deb.update(rec.predict(np.asarray(win)), t)
            if label:
                fired.append((t, label))
    return fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("streams", nargs="+")
    a = ap.parse_args()
    model, gestures = load_checkpoint()
    rec = Recognizer(model, gestures)
    total_h, all_fired = 0.0, []
    for p in a.streams:
        d = np.load(p)
        fired = replay(d["features"], d["timestamps"], rec)
        hours = d["timestamps"][-1] / 3600
        total_h += hours
        all_fired += fired
        print(f"{p}: {hours * 60:.1f} min, {len(fired)} false commands ({len(fired) / hours:.1f}/h)")
    n = len(all_fired)
    # exact Poisson 95% CI via chi-square quantiles would need scipy; use the normal approx on log-rate
    lo, hi = (max(n - 1.96 * np.sqrt(n), 0) / total_h, (n + 1.96 * np.sqrt(n)) / total_h) if n else (0, 3.0 / total_h)
    print(f"\nTOTAL {total_h * 60:.1f} min idle: {n} false commands = {n / total_h:.1f}/h  (approx 95% CI {lo:.1f}-{hi:.1f}/h)")
    print("by label:", dict(Counter(l for _, l in all_fired)))


if __name__ == "__main__":
    main()
