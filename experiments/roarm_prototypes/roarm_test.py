import time
import mujoco
import mujoco.viewer

XML = "roarm_m2_pro/roarm_m2_pro.xml"

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

print("================================")
print("RoArm LIVE TEST")
print("================================")
print("Joints:", model.njnt)
print("Actuators:", model.nu)

for i in range(model.nu):
    print(i, model.actuator(i).name)

print("Starting viewer...")


with mujoco.viewer.launch_passive(model, data) as viewer:

    print("VIEWER RUNNING")

    start = time.time()

    while viewer.is_running():

        t = time.time() - start

        # Slowly move the four arm joints
        data.ctrl[0] = 0.5 * __import__("math").sin(t)
        data.ctrl[1] = -0.7 + 0.3 * __import__("math").sin(t)
        data.ctrl[2] = 0.8 + 0.3 * __import__("math").sin(t)
        data.ctrl[3] = 0.2 * __import__("math").sin(t)

        # Open/close both fingers slowly
        grip = 0.35 * __import__("math").sin(t * 1.5)

        data.ctrl[4] = grip
        data.ctrl[5] = -grip

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.005)

print("Viewer closed.")
