"""
End-to-End ISL Gesture to MuJoCo RoArm-M2 Controller
Runs the 3D MuJoCo Simulation under mjpython, and spawns the Camera Tracker
in an isolated process to ensure 100% macOS Cocoa compatibility without UI collisions.
"""
import os
import sys
import time
import math
import socket
import subprocess
from pathlib import Path

# Auto re-exec inside virtual environment if invoked with global system Python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PY = _PROJECT_ROOT / ".venv" / "bin" / "python"
if not _VENV_PY.exists():
    _VENV_PY = _PROJECT_ROOT / "venv" / "bin" / "python"

if _VENV_PY.exists() and sys.prefix == sys.base_prefix:
    os.execv(str(_VENV_PY), [str(_VENV_PY)] + sys.argv)

# On macOS, MuJoCo interactive viewer requires execution under mjpython when run directly
if __name__ == "__main__" and sys.platform == "darwin":
    try:
        import mujoco.viewer as _mj_viewer
        if _mj_viewer._MJPYTHON is None:
            _mjpy = _PROJECT_ROOT / ".venv" / "bin" / "mjpython"
            if not _mjpy.exists():
                _mjpy = _PROJECT_ROOT / "venv" / "bin" / "mjpython"
            if _mjpy.exists():
                os.execv(str(_VENV_PY), [str(_VENV_PY), str(_mjpy), str(Path(__file__).resolve())] + sys.argv[1:])
    except Exception:
        pass

import numpy as np
import mujoco
import mujoco.viewer

sys.path.append(str(_PROJECT_ROOT))
import config


UDP_IP = "127.0.0.1"
UDP_PORT = 9876

# Joint and Actuator Indices for Official RoArm-M2-Pro:
# 0: base_motor     (J1 Base Yaw, -3.14 to 3.14 rad)
# 1: shoulder_motor (J2 Shoulder Pitch, -1.57 to 1.57 rad)
# 2: elbow_motor    (J3 Elbow Pitch, -1.0 to 2.95 rad)
# 3: eoat_motor     (J4 Gripper/EoAT, 0.0 to 1.5 rad)

HOME_POSE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

ACTION_TRAJECTORIES = {
    "wave": [
        (np.array([0.0, -0.5, 1.2, 0.0]), 0.6),
        (np.array([0.35, -0.5, 1.4, 0.0]), 0.3),
        (np.array([-0.35, -0.5, 1.0, 0.0]), 0.3),
        (np.array([0.35, -0.5, 1.4, 0.0]), 0.3),
        (np.array([-0.35, -0.5, 1.0, 0.0]), 0.3),
        (HOME_POSE, 0.6)
    ],
    "nod_head": [
        (np.array([0.0, -0.3, 0.6, 0.0]), 0.3),
        (HOME_POSE, 0.3),
        (np.array([0.0, -0.3, 0.6, 0.0]), 0.3),
        (HOME_POSE, 0.4)
    ],
    "shake_head": [
        (np.array([0.4, 0.0, 0.0, 0.0]), 0.25),
        (np.array([-0.4, 0.0, 0.0, 0.0]), 0.3),
        (np.array([0.4, 0.0, 0.0, 0.0]), 0.3),
        (HOME_POSE, 0.35)
    ],
    "pick_object": [
        (np.array([0.0, -0.7, 1.4, 0.0]), 0.7),
        (np.array([0.0, -0.7, 1.4, 0.5]), 0.4),
        (np.array([0.0, -0.2, 0.8, 0.5]), 0.6)
    ],
    "place_object": [
        (np.array([0.65, -0.2, 0.8, 0.5]), 0.6),
        (np.array([0.65, -0.4, 0.65, 0.5]), 0.5),
        (np.array([0.65, -0.4, 0.65, 0.0]), 0.4),
        (np.array([0.65, -0.2, 0.5, 0.0]), 0.4),
        (HOME_POSE, 0.6)
    ],
    "reset_to_home": [
        (HOME_POSE, 0.8)
    ]
}


class NonBlockingTrajectoryPlayer:
    """Manages smooth cosine-interpolated multi-waypoint robot trajectories."""
    def __init__(self):
        self.current_ctrl = HOME_POSE.copy()
        self.waypoints = []
        self.wp_index = 0
        self.wp_start_pose = HOME_POSE.copy()
        self.wp_start_time = 0.0
        self.is_active = False
        self.current_action_name = "IDLE"

    def play(self, action_name):
        if action_name not in ACTION_TRAJECTORIES:
            return
        self.waypoints = ACTION_TRAJECTORIES[action_name]
        self.wp_index = 0
        self.wp_start_pose = self.current_ctrl.copy()
        self.wp_start_time = time.time()
        self.is_active = True
        self.current_action_name = action_name
        print(f"[MUJOCO] Executing Robot Action: {action_name.upper()}")

    def update(self):
        if not self.is_active or not self.waypoints:
            return self.current_ctrl

        target_pose, duration = self.waypoints[self.wp_index]
        elapsed = time.time() - self.wp_start_time
        alpha = min(elapsed / max(duration, 1e-4), 1.0)
        
        smooth_alpha = 0.5 * (1.0 - math.cos(math.pi * alpha))
        self.current_ctrl = (1.0 - smooth_alpha) * self.wp_start_pose + smooth_alpha * target_pose

        if alpha >= 1.0:
            self.wp_index += 1
            if self.wp_index < len(self.waypoints):
                self.wp_start_pose = self.current_ctrl.copy()
                self.wp_start_time = time.time()
            else:
                self.is_active = False
                self.current_action_name = "IDLE"

        return self.current_ctrl


def run_bridge():
    xml_path = getattr(config, "ROARM_MODEL_XML", config.PROJECT_ROOT / "roarm_official" / "roarm_m2_pro.xml")
    if not xml_path.exists():
        xml_path = config.SIM_DIR / "roarm_m2_pro.xml"
    if not xml_path.exists():
        xml_path = config.SIM_DIR / "roarm_m2.xml"

    if not xml_path.exists():
        print(f"[ERROR] MuJoCo model not found at {xml_path}")
        return

    # 1. Setup UDP Receiver Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)

    # 2. Spawn Camera Vision Tracker as Isolated Process (No Cocoa conflicts!)
    tracker_script = _PROJECT_ROOT / "src" / "vision_tracker.py"
    print("\n[LAUNCH] Starting Camera Vision Tracker process...")
    tracker_proc = subprocess.Popen([str(_VENV_PY), str(tracker_script)])

    # 3. Load MuJoCo Model
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    player = NonBlockingTrajectoryPlayer()
    mj_data.ctrl[:] = player.current_ctrl

    print("\n" + "=" * 65)
    print("  ISL GESTURE TO MUJOCO 3D ROARM CONTROLLER")
    print("=" * 65)
    print("Recognized Gestures & Mapped Actions:")
    for g, a in config.GESTURE_ACTION_MAP.items():
        print(f"  • {g.upper():<10} -> {a}")
    print("\nBoth 3D MuJoCo Window & Camera Stream are running!")
    print("Close the 3D window or press [Q] on the camera window to exit.")
    print("=" * 65 + "\n")

    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            while viewer.is_running():
                # Check if camera tracker is still alive
                if tracker_proc.poll() is not None:
                    print("\n[INFO] Camera tracker closed.")
                    break

                # Non-blocking check for action commands from camera process
                try:
                    data, _ = sock.recvfrom(1024)
                    msg = data.decode("utf-8").strip()
                    if msg == "QUIT":
                        break
                    elif msg in ACTION_TRAJECTORIES:
                        if not player.is_active:
                            player.play(msg)
                except BlockingIOError:
                    pass

                # Step robot physics
                ctrl = player.update()
                mj_data.ctrl[:] = ctrl
                for _ in range(4):
                    mujoco.mj_step(mj_model, mj_data)

                viewer.sync()
                time.sleep(0.005)

    finally:
        # Cleanup child process and sockets
        print("\nShutting down controller...")
        sock.close()
        if tracker_proc.poll() is None:
            tracker_proc.terminate()
            try:
                tracker_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                tracker_proc.kill()
        print("[DONE] Exited cleanly.")


if __name__ == "__main__":
    run_bridge()
