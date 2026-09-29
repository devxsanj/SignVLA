import time
import threading
import mujoco
import mujoco.viewer

XML_PATH = "assets/roarm_m2_pro/roarm_m2_pro.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

POSES = {
    "HOME":  [0.0,  0.0,  0.0, 0.0],
    "PICK":  [0.0, -0.7, 1.4, 0.5],
    "PLACE": [0.6, -0.5, 1.0, 0.7],
}

target = POSES["HOME"][:]
running = True


def keyboard():
    global target, running

    print()
    print("================================")
    print(" RoArm-M2 Pro Manual Test")
    print("================================")
    print("h = HOME")
    print("p = PICK")
    print("l = PLACE")
    print("q = QUIT")
    print()

    while running:
        cmd = input("Command: ").strip().lower()

        if cmd == "h":
            target = POSES["HOME"][:]
            print("→ HOME")

        elif cmd == "p":
            target = POSES["PICK"][:]
            print("→ PICK")

        elif cmd == "l":
            target = POSES["PLACE"][:]
            print("→ PLACE")

        elif cmd == "q":
            running = False
            break

        else:
            print("Use h / p / l / q")


threading.Thread(
    target=keyboard,
    daemon=True
).start()


print()
print("Official RoArm model loaded")
print("Joints:", model.njnt)
print("Actuators:", model.nu)
print("Simulation started.")
print()


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and running:

        for i in range(4):

            error = target[i] - data.ctrl[i]

            # Smooth movement
            data.ctrl[i] += 0.02 * error

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.005)


running = False
print("Simulation stopped.")

