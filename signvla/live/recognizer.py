"""Camera-free recognition logic: window -> gated prediction -> debounced command.

Kept free of OpenCV/MediaPipe so it can be unit-tested and reused in evaluation.
"""
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import torch

from signvla import config


@dataclass
class Prediction:
    label: str
    confidence: float
    margin: float
    hand_fraction: float
    accepted: bool   # passed all gates -> may count towards a command
    reason: str      # "ok" or the gate that rejected it


class Recognizer:
    """Runs the encoder on one (T, 126) window and applies rejection gates:
    hand-presence, confidence and top1-top2 margin. Anything failing a gate is
    'unknown' - the robot must not move on it."""

    def __init__(self, model, gestures,
                 conf_threshold=config.CONFIDENCE_THRESHOLD,
                 margin_threshold=config.MARGIN_THRESHOLD,
                 min_hand_fraction=config.MIN_HAND_FRAMES):
        self.model, self.gestures = model, gestures
        self.conf_threshold = conf_threshold
        self.margin_threshold = margin_threshold
        self.min_hand_fraction = min_hand_fraction

    @torch.no_grad()
    def probabilities(self, windows):
        """(N, T, 126) -> (N, C) softmax probabilities."""
        x = torch.as_tensor(np.asarray(windows), dtype=torch.float32)
        return torch.softmax(self.model(x), dim=1).numpy()

    def gate(self, window, probs):
        hand_fraction = float((np.abs(window).sum(axis=1) > 0).mean())
        order = np.argsort(probs)[::-1]
        conf = float(probs[order[0]])
        margin = conf - float(probs[order[1]])
        if self.gestures[order[0]] == config.REST_LABEL:
            reason = "rest"
        elif hand_fraction < self.min_hand_fraction:
            reason = "no_hand"
        elif conf < self.conf_threshold:
            reason = "low_confidence"
        elif margin < self.margin_threshold:
            reason = "low_margin"
        else:
            reason = "ok"
        return Prediction(self.gestures[order[0]], conf, margin, hand_fraction, reason == "ok", reason)

    def predict(self, window):
        window = np.asarray(window, dtype=np.float32)
        return self.gate(window, self.probabilities(window[None])[0])


class Debouncer:
    """Fires a label only after `stable` consecutive accepted agreeing predictions,
    then holds off for `cooldown` seconds and needs the sign to change/lapse to re-fire."""

    def __init__(self, stable=config.STABLE_FRAMES, cooldown=config.COOLDOWN_SECONDS):
        self.votes = deque(maxlen=stable)
        self.stable, self.cooldown = stable, cooldown
        self.cooldown_until = 0.0
        self.last_fired = None

    def update(self, pred: Prediction, now: float):
        """-> label to fire, or None."""
        if not pred.accepted:
            self.votes.append(None)
            if self.votes.count(None) == self.stable:
                self.last_fired = None  # sign has lapsed -> same sign may fire again
            return None
        self.votes.append(pred.label)
        if len(self.votes) < self.stable or Counter(self.votes).most_common(1)[0][1] < self.stable:
            return None
        label = pred.label
        if now < self.cooldown_until or label == self.last_fired:
            return None
        self.last_fired, self.cooldown_until = label, now + self.cooldown
        self.votes.clear()
        return label
