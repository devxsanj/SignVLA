import time
import mujoco
import mujoco.viewer

XML_PATH = ".venv/share/openarm_mujoco/v2/openarm_bimanual.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Right-arm actuator indices:
# 8-14 = joints 1-7
# 15   = gripper

POSES = {
    "HOME": [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ],

    "PICK": [
        0.0, -0.5, 0.8, -1.2, 0.0, 0.6, 0.0
    ],

    "PLACE": [
        0.5, -0.3, 0.6, -1.0, 0.0, 0.5, 0.0
    ],
}

current_pose = "HOME"
target_pose = "HOME"

print("OpenArm Pose Controller")
print("Available poses:", ", ".join(POSES.keys()))
print("Starting in HOME")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # Apply current target pose to right arm
        target = POSES[target_pose]

        for i in range(7):
            data.ctrl[8 + i] = target[i]

        # Keep gripper open
        data.ctrl[15] = 0.0

        # Physics
        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(0.01)

        # Simple keyboard-style command through terminal
        if target_pose != current_pose:
            print(f"Moving: {current_pose} -> {target_pose}")
            current_pose = target_pose

