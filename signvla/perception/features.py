"""Canonical landmark -> feature conversion.

This is the ONE place that defines the feature space. Recording, dataset repair,
training and live inference must all go through these functions; a mismatch
between them silently ruins real-time accuracy (it did, in v0.1).

Feature layout per frame (126 floats):  [hand slot 0 (63) | hand slot 1 (63)]
Each hand: 21 landmarks x (x, y, z), translated so the wrist is the origin and
divided by the wrist->middle-MCP distance (scale invariant).

Hand slot rule (does NOT depend on MediaPipe's Left/Right label, which is
unreliable and was not saved for old samples):
    1 hand  -> slot 0
    2 hands -> slot 0 = the hand whose wrist is further left in the image
"""
import numpy as np

from signvla.config import FEATURES_PER_FRAME, FEATURES_PER_HAND, LANDMARKS_PER_HAND

MIDDLE_MCP = 9
_EPS = 1e-6


def normalize_hand(points):
    """(21, 3) raw landmarks -> (63,) wrist-centred, palm-scaled vector."""
    pts = np.asarray(points, dtype=np.float64).reshape(LANDMARKS_PER_HAND, 3)
    shifted = pts - pts[0]
    scale = np.linalg.norm(pts[MIDDLE_MCP] - pts[0])
    if scale < _EPS:
        scale = np.max(np.abs(shifted)) + _EPS
    return (shifted / scale).reshape(-1).astype(np.float32)


def frame_features(hands):
    """List of 0-2 raw (21, 3) hands -> (126,) canonical frame vector."""
    hands = [np.asarray(h, dtype=np.float64).reshape(LANDMARKS_PER_HAND, 3) for h in hands[:2]]
    hands.sort(key=lambda h: h[0, 0])  # wrist x, left-most first
    out = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    for slot, hand in enumerate(hands):
        out[slot * FEATURES_PER_HAND:(slot + 1) * FEATURES_PER_HAND] = normalize_hand(hand)
    return out


def hands_from_mediapipe(results):
    """MediaPipe Hands result -> list of raw (21, 3) arrays."""
    if not results.multi_hand_landmarks:
        return []
    return [np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            for hand in results.multi_hand_landmarks]


def extract_features(results):
    """MediaPipe Hands result -> (126,) canonical frame vector."""
    return frame_features(hands_from_mediapipe(results))


# ---------------- repair helpers for legacy .npy samples ----------------

def hand_present(vec63):
    return bool(np.any(np.abs(vec63) > 0))


def is_normalized_sample(seq):
    """True if every present hand in a (T, 126) sample is wrist-centred (wrist == 0)."""
    for slot in range(2):
        h = seq[:, slot * FEATURES_PER_HAND:(slot + 1) * FEATURES_PER_HAND]
        present = np.abs(h).sum(axis=1) > 0
        if present.any() and np.abs(h[present][:, :3]).max() > 1e-6:
            return False
    return True


def canonicalize_sequence(seq):
    """Convert a legacy (T, 126) sample (raw OR already normalized) to canonical form.

    Raw samples (image coords, slots in detection order) are normalized and re-ordered
    by wrist x. Already-normalized samples (old recorder.py, slots = MediaPipe
    Left/Right) keep their two-hand order (Left == image-left in the mirrored view)
    and a lone hand is moved to slot 0.
    """
    seq = np.asarray(seq, dtype=np.float32)
    normalized = is_normalized_sample(seq)
    out = np.zeros_like(seq)
    for t, frame in enumerate(seq):
        slots = [frame[i * FEATURES_PER_HAND:(i + 1) * FEATURES_PER_HAND] for i in range(2)]
        present = [s for s in slots if hand_present(s)]
        if normalized:
            vecs = present  # already Left, Right ordered
        else:
            raw = sorted(present, key=lambda s: s[0])  # s[0] == wrist x
            vecs = [normalize_hand(s) for s in raw]
        for slot, v in enumerate(vecs):
            out[t, slot * FEATURES_PER_HAND:(slot + 1) * FEATURES_PER_HAND] = v
    return out
