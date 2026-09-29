import time
import signal
import sys
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

XML_PATH = PROJECT_ROOT / "roarm_official" / "roarm_m2_pro.xml"


# ============================================================
# LOAD MUJOCO MODEL
# ============================================================

print()
print("==============================================")
print("       RoArm-M2 Pro — MuJoCo Simulation")
print("==============================================")
print()

print(f"Loading XML:")
print(f"  {XML_PATH}")

if not XML_PATH.exists():
    raise FileNotFoundError(
        f"\n[ERROR] XML file not found:\n{XML_PATH}"
    )

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

print()
print(f"Actuators detected: {model.nu}")

if model.nu != 4:
    print(
        f"[WARNING] Expected 4 actuators, "
        f"but MuJoCo reports {model.nu}."
    )


# ============================================================
# ACTUATOR INDICES
# ============================================================

BASE = 0
SHOULDER = 1
ELBOW = 2
EOAT = 3


# ============================================================
# GRIPPER
#
# Your XML has ONE eoat_joint.
# Therefore OPEN/CLOSE uses one actuator.
#
# XML:
# eoat_joint range = 0 -> 1.5
# ============================================================

GRIPPER_OPEN = 1.50
GRIPPER_CLOSED = 0.05


# ============================================================
# POSES
#
# IMPORTANT:
# These are for the ORIGINAL 4-DOF XML.
#
# Home is the 90-degree neutral working position.
# ============================================================

HOME = np.array([
    0.00,       # base
    0.00,       # shoulder
    1.5708,     # elbow = 90 degrees
    GRIPPER_OPEN
], dtype=float)


# ------------------------------------------------------------
# PICK POSITION
#
# Lower the arm toward the tabletop while keeping
# the base centered.
# ------------------------------------------------------------

PICK = np.array([
    0.00,
    -0.95,
    1.45,
    GRIPPER_OPEN
], dtype=float)


# ------------------------------------------------------------
# GRAB
# ------------------------------------------------------------

GRAB = np.array([
    0.00,
    -0.95,
    1.45,
    GRIPPER_CLOSED
], dtype=float)


# ------------------------------------------------------------
# LIFT
# ------------------------------------------------------------

LIFT = np.array([
    0.00,
    -0.25,
    0.75,
    GRIPPER_CLOSED
], dtype=float)


# ------------------------------------------------------------
# MOVE RIGHT
# ------------------------------------------------------------

RIGHT = np.array([
    -0.75,
    -0.25,
    0.75,
    GRIPPER_CLOSED
], dtype=float)


# ------------------------------------------------------------
# MOVE LEFT
# ------------------------------------------------------------

LEFT = np.array([
    0.75,
    -0.25,
    0.75,
    GRIPPER_CLOSED
], dtype=float)


# ------------------------------------------------------------
# PLACE
# ------------------------------------------------------------

PLACE = np.array([
    0.75,
    -0.55,
    1.05,
    GRIPPER_CLOSED
], dtype=float)


# ------------------------------------------------------------
# RELEASE
# ------------------------------------------------------------

RELEASE = np.array([
    0.75,
    -0.55,
    1.05,
    GRIPPER_OPEN
], dtype=float)


# ============================================================
# SLEEP POSITION
#
# ONLY used when quitting.
#
# This is the horizontal/sleep pose.
# ============================================================

SLEEP = np.array([
    0.00,
    0.00,
    0.00,
    0.00
], dtype=float)


# ============================================================
# SAFETY LIMITS
# ============================================================

def clamp_pose(pose):

    pose = np.asarray(pose, dtype=float).copy()

    pose[BASE] = np.clip(
        pose[BASE],
        -3.1416,
        3.1416
    )

    pose[SHOULDER] = np.clip(
        pose[SHOULDER],
        -1.5708,
        1.5708
    )

    pose[ELBOW] = np.clip(
        pose[ELBOW],
        -1.0,
        2.95
    )

    pose[EOAT] = np.clip(
        pose[EOAT],
        0.0,
        1.5
    )

    return pose


# ============================================================
# SET ACTUATORS
# ============================================================

def set_target(target):

    target = clamp_pose(target)

    data.ctrl[BASE] = target[BASE]
    data.ctrl[SHOULDER] = target[SHOULDER]
    data.ctrl[ELBOW] = target[ELBOW]
    data.ctrl[EOAT] = target[EOAT]


# ============================================================
# SMOOTH MOTION
# ============================================================

def move_to(target, duration, viewer=None):

    target = clamp_pose(target)

    start = np.array([
        data.ctrl[BASE],
        data.ctrl[SHOULDER],
        data.ctrl[ELBOW],
        data.ctrl[EOAT]
    ], dtype=float)

    steps = max(
        1,
        int(duration / model.opt.timestep)
    )

    for step in range(steps):

        # Smoothstep interpolation
        t = step / float(steps)

        s = t * t * (3.0 - 2.0 * t)

        command = start + (target - start) * s

        set_target(command)

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

        time.sleep(model.opt.timestep)


# ============================================================
# HOLD
# ============================================================

def hold(seconds, viewer=None):

    end_time = time.time() + seconds

    while time.time() < end_time:

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

        time.sleep(model.opt.timestep)


# ============================================================
# HOME
# ============================================================

def go_home(viewer=None):

    print("→ HOME / 90° REST")

    move_to(
        HOME,
        2.0,
        viewer
    )

    hold(0.5, viewer)


# ============================================================
# SLEEP
# ============================================================

def go_sleep(viewer=None):

    print()
    print("→ SLEEP / HORIZONTAL")

    move_to(
        SLEEP,
        3.0,
        viewer
    )

    hold(0.5, viewer)


# ============================================================
# PICK → PLACE TASK
# ============================================================

def pick_and_place(viewer=None):

    print()
    print("==============================================")
    print("         PICK → PLACE TASK")
    print("==============================================")

    # --------------------------------------------------------
    # 1. Start from HOME
    # --------------------------------------------------------

    go_home(viewer)

    # --------------------------------------------------------
    # 2. Move down to cube
    # --------------------------------------------------------

    print("→ LOWERING TO CUBE")

    move_to(
        PICK,
        2.5,
        viewer
    )

    hold(0.5, viewer)

    # --------------------------------------------------------
    # 3. Close gripper
    # --------------------------------------------------------

    print("→ GRIPPING CUBE")

    move_to(
        GRAB,
        1.2,
        viewer
    )

    hold(0.8, viewer)

    # --------------------------------------------------------
    # 4. Lift cube
    # --------------------------------------------------------

    print("→ LIFTING CUBE")

    move_to(
        LIFT,
        2.0,
        viewer
    )

    hold(0.5, viewer)

    # --------------------------------------------------------
    # 5. Move to LEFT side
    # --------------------------------------------------------

    print("→ MOVING TO LEFT")

    move_to(
        LEFT,
        2.5,
        viewer
    )

    hold(0.5, viewer)

    # --------------------------------------------------------
    # 6. Lower to destination
    # --------------------------------------------------------

    print("→ MOVING TO DESTINATION")

    move_to(
        PLACE,
        2.0,
        viewer
    )

    hold(0.5, viewer)

    # --------------------------------------------------------
    # 7. Open gripper
    # --------------------------------------------------------

    print("→ RELEASING CUBE")

    move_to(
        RELEASE,
        1.2,
        viewer
    )

    hold(1.0, viewer)

    # --------------------------------------------------------
    # 8. ALWAYS RETURN TO HOME
    # --------------------------------------------------------

    print("→ RETURNING TO HOME")

    go_home(viewer)

    print()
    print("✓ PICK → PLACE COMPLETE")
    print("✓ ARM IS BACK AT 90° HOME")
    print()


# ============================================================
# CTRL+C HANDLER
# ============================================================

viewer_global = None


def handle_interrupt(signum, frame):

    print()
    print()
    print("==============================================")
    print("         INTERRUPT RECEIVED")
    print("==============================================")

    if viewer_global is not None:

        try:
            go_sleep(viewer_global)
        except Exception as e:
            print(f"[WARNING] Could not move to sleep: {e}")

    print("→ Simulation exiting.")

    sys.exit(0)


signal.signal(
    signal.SIGINT,
    handle_interrupt
)


# ============================================================
# MAIN
# ============================================================

print()
print("Controls:")
print("  Q      = close MuJoCo window")
print("  Ctrl+C = sleep pose + exit")
print()

print("DOF:")
print("  Base")
print("  Shoulder")
print("  Elbow")
print("  End-effector")
print()

print("Gripper:")
print("  OPEN   =", GRIPPER_OPEN)
print("  CLOSED =", GRIPPER_CLOSED)
print()

print("HOME:")
print(HOME)

print()
print("Starting simulation...")
print()


# ============================================================
# VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    viewer_global = viewer

    # --------------------------------------------------------
    # INITIAL HOME
    # --------------------------------------------------------

    print("→ INITIAL HOME")

    set_target(HOME)

    mujoco.mj_forward(
        model,
        data
    )

    viewer.sync()

    hold(
        2.0,
        viewer
    )

    # --------------------------------------------------------
    # TASK LOOP
    # --------------------------------------------------------

    while viewer.is_running():

        pick_and_place(viewer)

        # Always remain at HOME after completing task.
        print("→ WAITING AT HOME")

        hold(
            2.0,
            viewer
        )


# ============================================================
# NORMAL WINDOW CLOSE
# ============================================================

print()
print("MuJoCo window closed.")

try:
    # If the viewer was closed normally, try to put the arm
    # into sleep before terminating.
    go_sleep(None)
except Exception:
    pass

print("✓ Simulation finished.")