import time
import numpy as np
import mujoco
import mujoco.viewer


XML_PATH = "roarm_m2_pro/roarm_m2_pro.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# ============================================================
# ACTUATOR INDICES
# ============================================================

BASE = 0
SHOULDER = 1
ELBOW = 2
WRIST = 3
LEFT_GRIPPER = 4
RIGHT_GRIPPER = 5


# ============================================================
# POSES
# ============================================================

# Gripper OPEN
OPEN = [0.45, -0.45]

# Gripper CLOSED
CLOSED = [-0.35, 0.35]


HOME = [
    0.00,     # base
    0.00,     # shoulder
    0.00,     # elbow
    0.00,     # wrist
    OPEN[0],
    OPEN[1]
]


# Arm lowered toward the table
PICK = [
    0.00,
   -1.15,
    1.35,
    0.20,
    OPEN[0],
    OPEN[1]
]


# Same position, gripper closed
GRAB = [
    0.00,
   -1.15,
    1.35,
    0.20,
    CLOSED[0],
    CLOSED[1]
]


# Lift after grabbing
LIFT = [
    0.00,
   -0.65,
    0.90,
    0.15,
    CLOSED[0],
    CLOSED[1]
]


# Move to the other side
PLACE = [
    0.80,
   -0.65,
    0.90,
    0.15,
    CLOSED[0],
    CLOSED[1]
]


# Release object
RELEASE = [
    0.80,
   -0.65,
    0.90,
    0.15,
    OPEN[0],
    OPEN[1]
]


# ============================================================
# SMOOTH ACTUATOR MOTION
# ============================================================

def move_to(target, duration):

    target = np.array(target, dtype=float)

    # Current actuator targets
    start = np.array([
        data.ctrl[BASE],
        data.ctrl[SHOULDER],
        data.ctrl[ELBOW],
        data.ctrl[WRIST],
        data.ctrl[LEFT_GRIPPER],
        data.ctrl[RIGHT_GRIPPER]
    ])

    steps = max(1, int(duration / model.opt.timestep))

    for step in range(steps):

        t = step / steps

        # Smooth acceleration/deceleration
        s = t * t * (3.0 - 2.0 * t)

        command = start + (target - start) * s

        data.ctrl[BASE] = command[0]
        data.ctrl[SHOULDER] = command[1]
        data.ctrl[ELBOW] = command[2]
        data.ctrl[WRIST] = command[3]
        data.ctrl[LEFT_GRIPPER] = command[4]
        data.ctrl[RIGHT_GRIPPER] = command[5]

        mujoco.mj_step(model, data)

        time.sleep(model.opt.timestep)


# ============================================================
# START
# ============================================================

print()
print("==============================================")
print("        RoArm-M2 Pro — MuJoCo Simulation")
print("==============================================")
print()
print("DOF: 4")
print("Gripper: 2 active fingers")
print()
print("Sequence:")
print("  HOME")
print("    ↓")
print("  LOWER")
print("    ↓")
print("  GRAB")
print("    ↓")
print("  LIFT")
print("    ↓")
print("  PLACE")
print("    ↓")
print("  RELEASE")
print("    ↓")
print("  HOME")
print()


with mujoco.viewer.launch_passive(model, data) as viewer:

    # --------------------------------------------------------
    # INITIAL HOME
    # --------------------------------------------------------

    print("→ HOME")

    for i in range(6):
        data.ctrl[i] = HOME[i]

    mujoco.mj_forward(model, data)

    viewer.sync()

    time.sleep(2)


    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while viewer.is_running():

        # HOME
        print("→ HOME")
        move_to(HOME, 2.0)

        time.sleep(1)


        # LOWER ARM
        print("→ LOWERING TO PICK POSITION")
        move_to(PICK, 3.0)

        time.sleep(1)


        # CLOSE GRIPPER
        print("→ CLOSING GRIPPER")
        move_to(GRAB, 1.5)

        time.sleep(1)


        # LIFT
        print("→ LIFTING")
        move_to(LIFT, 2.5)

        time.sleep(1)


        # MOVE
        print("→ MOVING TO PLACE POSITION")
        move_to(PLACE, 3.0)

        time.sleep(1)


        # OPEN GRIPPER
        print("→ OPENING GRIPPER")
        move_to(RELEASE, 1.5)

        time.sleep(1)


        # RETURN HOME
        print("→ RETURNING HOME")
        move_to(HOME, 3.0)

        print("✓ CYCLE COMPLETE")

        time.sleep(2)
