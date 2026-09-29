import time
import math
import mujoco
import mujoco.viewer

XML_PATH = ".venv/share/openarm_mujoco/v2/openarm_bimanual.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("OpenArm loaded")
print("Actuators:", model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()

    while viewer.is_running():

        t = time.time() - start

        # Right arm
        data.ctrl[8] = 0.5 * math.sin(t)
        data.ctrl[9] = 0.3 * math.sin(t)
        data.ctrl[10] = 0.2 * math.sin(t)

        # Remaining joints
        data.ctrl[11] = 0
        data.ctrl[12] = 0
        data.ctrl[13] = 0
        data.ctrl[14] = 0

        # Gripper
        data.ctrl[15] = 0

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(0.01)

