import numpy as np
import pytest

from signvla import config
from signvla.encoder.data import block_split, TEST
from signvla.live.recognizer import Debouncer, Prediction
from signvla.perception.features import canonicalize_sequence, frame_features, normalize_hand
from signvla.robot import roarm_sim
from signvla.semantics.concepts import CONCEPT_TO_PRIMITIVE, LEXICONS, Concept, resolve


def _hand(seed=0, offset=(0.5, 0.5, 0.0)):
    return np.random.default_rng(seed).random((21, 3)) * 0.2 + np.array(offset)


def test_normalize_hand_invariants():
    v = normalize_hand(_hand()).reshape(21, 3)
    assert np.allclose(v[0], 0) and np.isclose(np.linalg.norm(v[9]), 1.0)


def test_frame_features_slot_rules():
    left, right = _hand(1, (0.2, 0.5, 0)), _hand(2, (0.7, 0.5, 0))
    f = frame_features([right, left])            # order given must not matter
    assert np.allclose(f[:63], normalize_hand(left)) and np.allclose(f[63:], normalize_hand(right))
    one = frame_features([right])
    assert np.any(one[:63]) and not np.any(one[63:])  # lone hand -> slot 0


def test_canonicalize_raw_equals_live_path():
    """Legacy raw samples must map to exactly what live inference would produce."""
    a, b = _hand(3, (0.6, 0.4, 0)), _hand(4, (0.3, 0.5, 0))
    raw = np.zeros((config.SEQUENCE_LENGTH, 126), np.float32)
    raw[:, :63], raw[:, 63:] = a.reshape(-1), b.reshape(-1)  # detection order: a first
    live = frame_features([a, b])
    assert np.allclose(canonicalize_sequence(raw)[5], live, atol=1e-5)


def test_canonicalize_is_idempotent():
    raw = np.zeros((config.SEQUENCE_LENGTH, 126), np.float32)
    raw[:, 63:] = _hand(5).reshape(-1)  # lone hand in slot 1
    once = canonicalize_sequence(raw)
    assert np.any(once[:, :63]) and not np.any(once[:, 63:])
    assert np.allclose(canonicalize_sequence(once), once)


def test_block_split_holds_out_latest_per_class():
    y = np.repeat([0, 1], 20)
    ids = np.tile(np.arange(1, 21), 2)
    s = block_split(y, ids)
    for c in (0, 1):
        assert ids[(y == c) & (s == TEST)].min() > ids[(y == c) & (s != TEST)].max()


def test_lexicons_cover_every_gesture_and_primitive():
    assert set(LEXICONS["isl"]) == set(config.GESTURES) - {config.REST_LABEL}
    assert set(CONCEPT_TO_PRIMITIVE) == set(Concept)
    assert resolve("isl", "left") is Concept.MOVE_LEFT and resolve("asl", "left") is None


def _p(label, ok=True):
    return Prediction(label, 0.9, 0.8, 1.0, ok, "ok" if ok else "low_confidence")


def test_debouncer_fires_once_then_needs_lapse():
    d = Debouncer(stable=3, cooldown=1.0)
    fired = [d.update(_p("yes"), t) for t in (0.0, 0.1, 0.2, 0.3, 0.4)]
    assert fired == [None, None, "yes", None, None]
    assert d.update(_p("yes"), 5.0) is None          # held sign does not re-fire
    for t in (5.1, 5.2, 5.3):
        d.update(_p("x", ok=False), t)               # sign lapses
    out = [d.update(_p("yes"), 6.0 + i * 0.1) for i in range(3)]
    assert out[-1] == "yes"


def test_debouncer_ignores_flicker():
    d = Debouncer(stable=3)
    assert [d.update(_p(l), i) for i, l in enumerate(["yes", "no", "yes", "no"])] == [None] * 4


@pytest.mark.parametrize("name", list(roarm_sim.PRIMITIVES))
def test_every_primitive_reaches_target_in_sim(name):
    assert roarm_sim.run_headless(name)["success"]
