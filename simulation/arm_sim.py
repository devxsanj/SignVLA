"""
RoArm-M2 MuJoCo Simulation Controller
Provides high-level kinematic primitives and action execution in a 3D simulated MuJoCo environment.
"""
import os
import sys
import time
import math
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


XML_PATH = _PROJECT_ROOT / "roarm_official" / "roarm_m2_pro.xml"
if not XML_PATH.exists():
    XML_PATH = Path(__file__).resolve().parent / "roarm_m2_pro.xml"
if not XML_PATH.exists():
    XML_PATH = Path(__file__).resolve().parent / "roarm_m2.xml"

# Joint and Actuator Indices for Official RoArm-M2-Pro:
# 0: base_motor     (J1 Base Yaw, -3.14 to 3.14 rad)
# 1: shoulder_motor (J2 Shoulder Pitch, -1.57 to 1.57 rad)
# 2: elbow_motor    (J3 Elbow Pitch, -1.0 to 2.95 rad)
# 3: eoat_motor     (J4 Gripper/EoAT, 0.0 to 1.5 rad)

HOME_POSE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)


class RoArmMuJoCoSim:
    def __init__(self, xml_path=XML_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        
        # Current target controls
        self.current_ctrl = HOME_POSE.copy()
        self.data.ctrl[:] = self.current_ctrl
        self.is_running = True

    def set_target(self, target_ctrl, duration=1.0, viewer=None, steps=100):
        """Smoothly interpolates actuators to target_ctrl using cosine easing."""
        start_ctrl = self.current_ctrl.copy()
        dt = duration / float(steps)

        for step in range(steps + 1):
            alpha = step / float(steps)
            # Smooth cosine easing: 0 -> 1
            smooth_alpha = 0.5 * (1.0 - math.cos(math.pi * alpha))
            interpolated = (1.0 - smooth_alpha) * start_ctrl + smooth_alpha * target_ctrl
            
            self.current_ctrl = interpolated
            self.data.ctrl[:] = self.current_ctrl
            mujoco.mj_step(self.model, self.data)

            if viewer is not None and viewer.is_running():
                viewer.sync()
            time.sleep(dt)

    def reset_to_home(self, viewer=None):
        """Restores arm to neutral home posture."""
        print("[ACTION] Resetting to Home Pose")
        self.set_target(HOME_POSE, duration=0.8, viewer=viewer)

    def wave(self, cycles=3, viewer=None):
        """Performs a friendly waving motion."""
        print("[ACTION] Performing Wave Gesture")
        wave_up = np.array([0.0, -0.5, 1.2, 0.0], dtype=np.float64)
        self.set_target(wave_up, duration=0.6, viewer=viewer)

        for _ in range(cycles):
            left = np.array([0.35, -0.5, 1.4, 0.0], dtype=np.float64)
            self.set_target(left, duration=0.3, viewer=viewer, steps=30)

            right = np.array([-0.35, -0.5, 1.0, 0.0], dtype=np.float64)
            self.set_target(right, duration=0.3, viewer=viewer, steps=30)

        self.reset_to_home(viewer=viewer)

    def nod_head(self, cycles=2, viewer=None):
        """Nods the arm to signify 'Yes'."""
        print("[ACTION] Nodding (Yes)")
        for _ in range(cycles):
            nod_down = np.array([0.0, -0.3, 0.6, 0.0], dtype=np.float64)
            self.set_target(nod_down, duration=0.3, viewer=viewer, steps=30)
            self.set_target(HOME_POSE, duration=0.3, viewer=viewer, steps=30)

    def shake_head(self, cycles=2, viewer=None):
        """Shakes the base back and forth to signify 'No'."""
        print("[ACTION] Shaking (No)")
        for _ in range(cycles):
            left = np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float64)
            self.set_target(left, duration=0.25, viewer=viewer, steps=25)

            right = np.array([-0.4, 0.0, 0.0, 0.0], dtype=np.float64)
            self.set_target(right, duration=0.25, viewer=viewer, steps=25)

        self.reset_to_home(viewer=viewer)

    def pick_object(self, viewer=None):
        """Reaches to pick pose, closes gripper, and lifts up."""
        print("[ACTION] Executing Pick Object")
        # 1. Reach down with open gripper
        reach = np.array([0.0, -0.7, 1.4, 0.0], dtype=np.float64)
        self.set_target(reach, duration=0.7, viewer=viewer)

        # 2. Close gripper
        grasp = np.array([0.0, -0.7, 1.4, 0.5], dtype=np.float64)
        self.set_target(grasp, duration=0.4, viewer=viewer, steps=40)

        # 3. Lift arm up
        lift = np.array([0.0, -0.2, 0.8, 0.5], dtype=np.float64)
        self.set_target(lift, duration=0.6, viewer=viewer)

    def place_object(self, viewer=None):
        """Swings arm to side, lowers, and opens gripper to place object."""
        print("[ACTION] Executing Place Object")
        # 1. Swing arm sideways
        swing = np.array([0.65, -0.2, 0.8, 0.5], dtype=np.float64)
        self.set_target(swing, duration=0.6, viewer=viewer)

        # 2. Lower to place position
        lower = np.array([0.65, -0.4, 0.65, 0.5], dtype=np.float64)
        self.set_target(lower, duration=0.5, viewer=viewer)

        # 3. Open gripper
        release = np.array([0.65, -0.4, 0.65, 0.0], dtype=np.float64)
        self.set_target(release, duration=0.4, viewer=viewer, steps=40)

        # 4. Lift up slightly and return home
        lift_away = np.array([0.65, -0.2, 0.5, 0.0], dtype=np.float64)
        self.set_target(lift_away, duration=0.4, viewer=viewer)
        self.reset_to_home(viewer=viewer)

    def execute_action(self, action_name, viewer=None):
        """Dispatches action string to corresponding method."""
        action_map = {
            "wave": self.wave,
            "nod_head": self.nod_head,
            "shake_head": self.shake_head,
            "pick_object": self.pick_object,
            "place_object": self.place_object,
            "reset_to_home": self.reset_to_home
        }
        handler = action_map.get(action_name)
        if handler:
            handler(viewer=viewer)
        else:
            print(f"[WARN] Unknown action: {action_name}")


def interactive_demo():
    """Runs a standalone 3D interactive viewer testing all action primitives."""
    sim = RoArmMuJoCoSim()
    print("\n" + "=" * 60)
    print("  RoArm-M2 MuJoCo Simulation Interactive Test")
    print("=" * 60)
    print("Commands:")
    print("  [1]: Wave           [2]: Nod (Yes)      [3]: Shake (No)")
    print("  [4]: Pick Object    [5]: Place Object   [0]: Reset Home")
    print("  Close the MuJoCo window to exit.")
    print("=" * 60 + "\n")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        sim.reset_to_home(viewer=viewer)
        
        # Test routine
        sim.wave(cycles=2, viewer=viewer)
        time.sleep(0.5)
        sim.pick_object(viewer=viewer)
        time.sleep(0.5)
        sim.place_object(viewer=viewer)
        
        while viewer.is_running():
            mujoco.mj_step(sim.model, sim.data)
            viewer.sync()
            time.sleep(0.005)


if __name__ == "__main__":
    interactive_demo()
