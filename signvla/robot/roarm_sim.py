"""RoArm-M2-Pro MuJoCo simulation: primitive library, trajectory player, and a
UDP-driven live viewer. Also a headless runner used by tests/evaluation.

Actuator order: 0 base yaw, 1 shoulder, 2 elbow, 3 end-of-arm tool (gripper).
Receives *semantic concepts* over UDP (never raw sign labels).
"""
import math
import socket
import time

import mujoco
import numpy as np

from signvla import config
from signvla.semantics.concepts import CONCEPT_TO_PRIMITIVE, Concept

HOME = [0.00, 0.00, 0.00, 0.00]
_UP = [0.00, -0.65, 0.65, 0.00]

# primitive -> [(target pose or "HOME", duration_s), ...]
PRIMITIVES = {
    "move_left":     [([-0.65, 0.00, 0.00, 0.00], 0.8), ("HOME", 0.6)],
    "move_right":    [([0.65, 0.00, 0.00, 0.00], 0.8), ("HOME", 0.6)],
    "move_up":       [(_UP, 0.8), ("HOME", 0.6)],
    "move_down":     [([0.00, 0.35, 0.30, 0.00], 0.8), ("HOME", 0.6)],
    "move_front":    [([0.00, -0.80, 1.15, 0.00], 0.8), ("HOME", 0.6)],
    "move_back":     [([0.00, 0.35, 0.35, 0.00], 0.8), ("HOME", 0.6)],
    "open_gripper":  [([0.00, 0.00, 0.00, 0.00], 0.5)],
    "close_gripper": [([0.00, 0.00, 0.00, 1.15], 0.5)],
    "pick_object":   [([0.00, -0.70, 1.20, 0.00], 0.6), ([0.00, -1.15, 1.65, 0.85], 0.8),
                      ([0.00, -0.70, 1.20, 1.15], 0.5)],
    "place_object":  [([0.75, -0.95, 1.35, 1.15], 0.8), ([0.75, -0.95, 1.35, 0.00], 0.5), ("HOME", 0.7)],
    "wave":          [([0.00, -0.50, 1.20, 0.00], 0.6)]
                     + [([s * 0.45, -0.50, 1.25 if s > 0 else 1.10, 0.00], 0.35) for s in (1, -1, 1, -1)]
                     + [("HOME", 0.6)],
    "nod_head":      [([0.00, -0.35, 0.65, 0.00], 0.35), ("HOME", 0.35)] * 2,
    "shake_head":    [([0.45, 0.00, 0.00, 0.00], 0.30), ([-0.45, 0.00, 0.00, 0.00], 0.30),
                      ([0.45, 0.00, 0.00, 0.00], 0.30), ("HOME", 0.5)],
    "home":          [("HOME", 0.8)],
}
assert set(CONCEPT_TO_PRIMITIVE.values()) == set(PRIMITIVES), "concept table and primitive table disagree"


def load_model(xml=config.ROARM_XML):
    return mujoco.MjModel.from_xml_path(str(xml))


def clamp_pose(pose, model):
    pose = np.array(pose, dtype=np.float64)
    lo, hi = model.actuator_ctrlrange[:4].T
    return np.clip(pose, lo, hi)


def build_trajectories(model):
    home = clamp_pose(HOME, model)
    return {name: [(home if p == "HOME" else clamp_pose(p, model), d) for p, d in steps]
            for name, steps in PRIMITIVES.items()}


class TrajectoryPlayer:
    """Cosine-eased waypoint follower. Time is passed in, so it works with wall or sim time."""

    def __init__(self, home):
        self.ctrl = np.array(home, dtype=np.float64)
        self.active = False
        self.action = "IDLE"

    def play(self, action, trajectories, now):
        if action not in trajectories:
            print(f"[WARN] unknown primitive: {action}")
            return False
        self.waypoints, self.index = trajectories[action], 0
        self.start_pose, self.start_time = self.ctrl.copy(), now
        self.active, self.action = True, action
        return True

    def update(self, now):
        if not self.active:
            return self.ctrl
        target, duration = self.waypoints[self.index]
        t = min((now - self.start_time) / max(duration, 1e-3), 1.0)
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        self.ctrl = (1.0 - s) * self.start_pose + s * target
        if t >= 1.0:
            self.index += 1
            if self.index >= len(self.waypoints):
                self.active, self.action, self.ctrl = False, "IDLE", target.copy()
            else:
                self.start_pose, self.start_time = self.ctrl.copy(), now
        return self.ctrl


def run_headless(primitive, model=None):
    """Execute one primitive in physics without a viewer.
    -> dict(success, sim_seconds, max_tracking_error, final_error). success = final joint
    pose within 0.05 rad of the last waypoint."""
    model = model or load_model()
    data = mujoco.MjData(model)
    trajs = build_trajectories(model)
    player = TrajectoryPlayer(clamp_pose(HOME, model))
    player.play(primitive, trajs, 0.0)
    max_err = 0.0
    while player.active:
        data.ctrl[:4] = player.update(data.time)
        mujoco.mj_step(model, data)
        max_err = max(max_err, float(np.abs(data.qpos[:4] - data.ctrl[:4]).max()))
    for _ in range(int(0.5 / model.opt.timestep)):  # settle
        mujoco.mj_step(model, data)
    final_err = float(np.abs(data.qpos[:4] - player.ctrl[:4]).max())
    return {"success": final_err < 0.05, "sim_seconds": data.time,
            "max_tracking_error": max_err, "final_error": final_err}


def run_viewer(port=config.UDP_PORT):
    """Open the viewer and execute concepts received over UDP. macOS: run via `mjpython`."""
    import mujoco.viewer

    model = load_model()
    data = mujoco.MjData(model)
    trajs = build_trajectories(model)
    player = TrajectoryPlayer(clamp_pose(HOME, model))
    data.ctrl[:4] = player.ctrl

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((config.UDP_HOST, port))
    sock.setblocking(False)
    print(f"RoArm sim listening for semantic concepts on udp://{config.UDP_HOST}:{port}")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                try:
                    msg = sock.recvfrom(1024)[0].decode().strip()
                    if msg == "QUIT":
                        break
                    if not player.active:
                        try:
                            primitive = CONCEPT_TO_PRIMITIVE[Concept(msg)]
                        except ValueError:
                            print(f"[WARN] not a concept: {msg!r}")
                        else:
                            print(f"[SIM] {msg} -> {primitive}")
                            player.play(primitive, trajs, time.time())
                except BlockingIOError:
                    pass
                data.ctrl[:4] = player.update(time.time())
                for _ in range(4):
                    mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.005)
    finally:
        sock.close()
