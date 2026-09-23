import time
import mujoco
import mujoco.viewer

XML = "roarm_official/roarm_m2_pro.xml"

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

POSES = [
    ("HOME",  [0.0,  0.0, 0.0, 0.0]),
    ("PICK",  [0.0, -0.7, 1.4, 0.5]),
    ("PLACE", [0.65, -0.5, 1.0, 0.7]),
]

print("===================================")
print("   OFFICIAL RoArm-M2 SIMULATION")
print("===================================")
print("Joints:", model.njnt)
print("Actuators:", model.nu)
print()
print("Starting simulation...")

with mujoco.viewer.launch_passive(model, data) as viewer:

    for name, target in POSES:

        print("Moving to:", name, target)

        start = data.ctrl.copy()

        for step in range(300):

            alpha = min(1.0, step / 300.0)

            for i in range(4):
                data.ctrl[i] = (
                    start[i] * (1 - alpha)
                    + target[i] * alpha
                )

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(0.01)

        time.sleep(1)

    print("Simulation complete.")

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
