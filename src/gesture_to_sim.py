"""
SignVLA — 14-Class ISL Gesture -> RoArm-M2 Pro MuJoCo

Pipeline:
    Camera
      ↓
    MediaPipe
      ↓
    30-frame landmark sequence
      ↓
    Trained 14-class GRU (vision_tracker.py)
      ↓
    Gesture -> Robot Action
      ↓
    UDP
      ↓
    MuJoCo RoArm-M2 Pro
"""

import os
import sys
import time
import math
import socket
import subprocess
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT / VENV
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"

if not VENV_PY.exists():
    VENV_PY = PROJECT_ROOT / "venv" / "bin" / "python"


if VENV_PY.exists() and sys.prefix == sys.base_prefix:
    os.execv(
        str(VENV_PY),
        [str(VENV_PY)] + sys.argv
    )


# ============================================================
# macOS MUJOCO
# ============================================================

if __name__ == "__main__" and sys.platform == "darwin":

    try:
        import mujoco.viewer

        if mujoco.viewer._MJPYTHON is None:

            mjpython = PROJECT_ROOT / ".venv" / "bin" / "mjpython"

            if not mjpython.exists():
                mjpython = PROJECT_ROOT / "venv" / "bin" / "mjpython"

            if mjpython.exists():

                os.execv(
                    str(mjpython),
                    [
                        str(mjpython),
                        str(Path(__file__).resolve())
                    ]
                )

    except Exception:
        pass


# ============================================================
# IMPORTS
# ============================================================

import mujoco
import mujoco.viewer

sys.path.append(str(PROJECT_ROOT))

import config


# ============================================================
# UDP
# ============================================================

UDP_IP = "127.0.0.1"
UDP_PORT = 9876


# ============================================================
# ROARM-M2 PRO POSES
#
# actuator order:
# 0 = base
# 1 = shoulder
# 2 = elbow
# 3 = eoat/gripper
# ============================================================

HOME = np.array([
    0.00,
    0.00,
    0.00,
    0.00
], dtype=np.float64)


# Basic directional poses

MOVE_LEFT = np.array([
    -0.65,
    0.00,
    0.00,
    0.00
], dtype=np.float64)

MOVE_RIGHT = np.array([
    0.65,
    0.00,
    0.00,
    0.00
], dtype=np.float64)


MOVE_UP = np.array([
    0.00,
    -0.65,
    0.65,
    0.00
], dtype=np.float64)

MOVE_DOWN = np.array([
    0.00,
    0.35,
    0.30,
    0.00
], dtype=np.float64)


MOVE_FRONT = np.array([
    0.00,
    -0.80,
    1.15,
    0.00
], dtype=np.float64)

MOVE_BACK = np.array([
    0.00,
    0.35,
    0.35,
    0.00
], dtype=np.float64)


# Pick / place

PICK = np.array([
    0.00,
    -1.15,
    1.65,
    0.85
], dtype=np.float64)


PLACE = np.array([
    0.75,
    -0.95,
    1.35,
    1.15
], dtype=np.float64)


# Gripper

OPEN_GRIPPER = np.array([
    0.00,
    0.00,
    0.00,
    0.00
], dtype=np.float64)


CLOSE_GRIPPER = np.array([
    0.00,
    0.00,
    0.00,
    1.15
], dtype=np.float64)


# ============================================================
# CLAMP TO REAL ROARM LIMITS
# ============================================================

def clamp_pose(pose, model):

    pose = np.asarray(
        pose,
        dtype=np.float64
    ).copy()

    for i in range(min(4, model.nu)):

        if model.actuator(i).ctrllimited:

            pose[i] = np.clip(
                pose[i],
                model.actuator_ctrlrange[i][0],
                model.actuator_ctrlrange[i][1]
            )

    return pose


# ============================================================
# TRAJECTORY BUILDER
# ============================================================

def build_trajectories(model):

    home = clamp_pose(HOME, model)

    trajectories = {

        # ----------------------------------------------------
        # DIRECTIONS
        # ----------------------------------------------------

        "move_left": [
            (clamp_pose(MOVE_LEFT, model), 0.8),
            (home, 0.6),
        ],

        "move_right": [
            (clamp_pose(MOVE_RIGHT, model), 0.8),
            (home, 0.6),
        ],

        "move_up": [
            (clamp_pose(MOVE_UP, model), 0.8),
            (home, 0.6),
        ],

        "move_down": [
            (clamp_pose(MOVE_DOWN, model), 0.8),
            (home, 0.6),
        ],

        "move_front": [
            (clamp_pose(MOVE_FRONT, model), 0.8),
            (home, 0.6),
        ],

        "move_back": [
            (clamp_pose(MOVE_BACK, model), 0.8),
            (home, 0.6),
        ],


        # ----------------------------------------------------
        # GRIPPER
        # ----------------------------------------------------

        "open_gripper": [
            (clamp_pose(OPEN_GRIPPER, model), 0.5),
        ],

        "close_gripper": [
            (clamp_pose(CLOSE_GRIPPER, model), 0.5),
        ],


        # ----------------------------------------------------
        # PICK / PLACE
        # ----------------------------------------------------

        "pick_object": [

            (clamp_pose(
                np.array([0.00, -0.70, 1.20, 0.00]),
                model
            ), 0.6),

            (clamp_pose(
                PICK,
                model
            ), 0.8),

            (clamp_pose(
                np.array([0.00, -0.70, 1.20, 1.15]),
                model
            ), 0.5),

        ],


        "place_object": [

            (clamp_pose(
                PLACE,
                model
            ), 0.8),

            (clamp_pose(
                np.array([0.75, -0.95, 1.35, 0.00]),
                model
            ), 0.5),

            (home, 0.7),

        ],


        # ----------------------------------------------------
        # HELLO
        # ----------------------------------------------------

        "wave": [

            (
                clamp_pose(
                    np.array([0.00, -0.50, 1.20, 0.00]),
                    model
                ),
                0.6
            ),

            (
                clamp_pose(
                    np.array([0.45, -0.50, 1.25, 0.00]),
                    model
                ),
                0.35
            ),

            (
                clamp_pose(
                    np.array([-0.45, -0.50, 1.10, 0.00]),
                    model
                ),
                0.35
            ),

            (
                clamp_pose(
                    np.array([0.45, -0.50, 1.25, 0.00]),
                    model
                ),
                0.35
            ),

            (
                clamp_pose(
                    np.array([-0.45, -0.50, 1.10, 0.00]),
                    model
                ),
                0.35
            ),

            (home, 0.6),
        ],


        # ----------------------------------------------------
        # YES
        # ----------------------------------------------------

        "nod_head": [

            (
                clamp_pose(
                    np.array([0.00, -0.35, 0.65, 0.00]),
                    model
                ),
                0.35
            ),

            (home, 0.35),

            (
                clamp_pose(
                    np.array([0.00, -0.35, 0.65, 0.00]),
                    model
                ),
                0.35
            ),

            (home, 0.5),
        ],


        # ----------------------------------------------------
        # NO
        # ----------------------------------------------------

        "shake_head": [

            (
                clamp_pose(
                    np.array([0.45, 0.00, 0.00, 0.00]),
                    model
                ),
                0.30
            ),

            (
                clamp_pose(
                    np.array([-0.45, 0.00, 0.00, 0.00]),
                    model
                ),
                0.30
            ),

            (
                clamp_pose(
                    np.array([0.45, 0.00, 0.00, 0.00]),
                    model
                ),
                0.30
            ),

            (home, 0.5),
        ],


        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

        "home": [
            (home, 0.8)
        ],
    }

    return trajectories


# ============================================================
# TRAJECTORY PLAYER
# ============================================================

class TrajectoryPlayer:

    def __init__(self, home):

        self.home = home.copy()

        self.current_ctrl = home.copy()

        self.waypoints = []

        self.index = 0

        self.start_pose = home.copy()

        self.start_time = 0.0

        self.active = False

        self.action = "IDLE"


    def play(self, action, trajectories):

        if action not in trajectories:

            print(
                f"[WARN] Unknown action: {action}"
            )

            return

        self.waypoints = trajectories[action]

        self.index = 0

        self.start_pose = self.current_ctrl.copy()

        self.start_time = time.time()

        self.active = True

        self.action = action

        print(
            f"\n[MUJOCO] ACTION -> {action.upper()}"
        )


    def update(self):

        if not self.active:

            return self.current_ctrl


        target, duration = self.waypoints[self.index]

        elapsed = time.time() - self.start_time

        t = elapsed / max(duration, 0.001)

        t = min(t, 1.0)


        # cosine interpolation

        s = 0.5 * (
            1.0 - math.cos(math.pi * t)
        )


        self.current_ctrl = (
            (1.0 - s) * self.start_pose
            + s * target
        )


        if t >= 1.0:

            self.index += 1

            if self.index >= len(self.waypoints):

                self.active = False

                self.action = "IDLE"

                self.current_ctrl = target.copy()

            else:

                self.start_pose = self.current_ctrl.copy()

                self.start_time = time.time()


        return self.current_ctrl


# ============================================================
# MAIN
# ============================================================

def run_bridge():

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    xml_path = config.ROARM_MODEL_XML

    if not xml_path.exists():

        print(
            "[ERROR] RoArm MuJoCo model not found:"
        )

        print(xml_path)

        return


    print()
    print("=" * 65)
    print("   SignVLA — 14 CLASS ISL → ROARM-M2 PRO")
    print("=" * 65)
    print()
    print("Model:", xml_path)


    # --------------------------------------------------------
    # LOAD MUJOCO
    # --------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(xml_path)
    )

    data = mujoco.MjData(model)


    print()
    print("Actuators:")

    for i in range(model.nu):

        print(
            f"  {i}: {model.actuator(i).name}"
        )


    # --------------------------------------------------------
    # TRAJECTORIES
    # --------------------------------------------------------

    trajectories = build_trajectories(model)

    player = TrajectoryPlayer(
        clamp_pose(HOME, model)
    )


    data.ctrl[:] = player.current_ctrl


    # --------------------------------------------------------
    # UDP
    # --------------------------------------------------------

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    sock.bind(
        (UDP_IP, UDP_PORT)
    )

    sock.setblocking(False)


    # --------------------------------------------------------
    # START VISION PROCESS
    # --------------------------------------------------------

    tracker_script = (
        PROJECT_ROOT
        / "src"
        / "vision_tracker.py"
    )


    print()
    print("[LAUNCH] Starting trained-model vision tracker...")


    tracker_proc = subprocess.Popen(
        [
            str(VENV_PY),
            str(tracker_script)
        ]
    )


    # --------------------------------------------------------
    # PRINT GESTURES
    # --------------------------------------------------------

    print()
    print("14 GESTURES:")
    print()

    for gesture in config.GESTURES:

        action = config.GESTURE_ACTION_MAP.get(
            gesture
        )

        print(
            f"  {gesture:<8} -> {action}"
        )


    print()
    print("Camera window = gesture recognition")
    print("MuJoCo window = RoArm simulation")
    print()
    print("Press Q in camera window to stop.")
    print("=" * 65)
    print()


    # --------------------------------------------------------
    # MUJOCO
    # --------------------------------------------------------

    try:

        with mujoco.viewer.launch_passive(
            model,
            data
        ) as viewer:

            viewer.sync()


            while viewer.is_running():

                # --------------------------------------------
                # CHECK TRACKER
                # --------------------------------------------

                if tracker_proc.poll() is not None:

                    print(
                        "\n[INFO] Vision tracker stopped."
                    )

                    break


                # --------------------------------------------
                # RECEIVE ACTION
                # --------------------------------------------

                try:

                    packet, _ = sock.recvfrom(
                        1024
                    )

                    msg = packet.decode(
                        "utf-8"
                    ).strip()


                    if msg == "QUIT":

                        break


                    if msg in trajectories:

                        if not player.active:

                            player.play(
                                msg,
                                trajectories
                            )


                except BlockingIOError:

                    pass


                # --------------------------------------------
                # UPDATE ROBOT
                # --------------------------------------------

                ctrl = player.update()

                data.ctrl[:] = ctrl


                # physics

                for _ in range(4):

                    mujoco.mj_step(
                        model,
                        data
                    )


                viewer.sync()

                time.sleep(0.005)


    finally:

        print()
        print("[SHUTDOWN] Stopping controller...")


        sock.close()


        if tracker_proc.poll() is None:

            tracker_proc.terminate()

            try:

                tracker_proc.wait(
                    timeout=2
                )

            except subprocess.TimeoutExpired:

                tracker_proc.kill()


        print("[DONE]")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    run_bridge()
