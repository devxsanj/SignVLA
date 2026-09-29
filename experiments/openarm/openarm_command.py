import time
import threading
import mujoco
import mujoco.viewer

XML_PATH = ".venv/share/openarm_mujoco/v2/openarm_bimanual.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

POSES = {
    "HOME":  [0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0],
    "PICK":  [0.0, -0.5,  0.8, -1.2, 0.0, 0.6, 0.0],
    "PLACE": [0.5, -0.3,  0.6, -1.0, 0.0, 0.5, 0.0],
}

current_target = POSES["HOME"]

commands = {
    "h": "HOME",
    "p": "PICK",
    "l": "PLACE",
}

def command_listener():
    global current_target

    print("\nCommands:")
    print("  h = HOME")
    print("  p = PICK")
    print("  l = PLACE")
    print("  q = QUIT")

    while True:
        command = input("\nCommand: ").strip().lower()

        if command == "q":
            break

        if command in commands:
            pose_name = commands[command]
            current_target = POSES[pose_name]
            print(f"→ Moving to {pose_name}")
        else:
            print("Unknown command.")

listener = threading.Thread(target=command_listener, daemon=True)
listener.start()

print("OpenArm Command Controller")
print("Right arm selected.")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # Right arm joints 1–7
        for i in range(7):
            data.ctrl[8 + i] = current_target[i]

        # Gripper
        data.ctrl[15] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(0.01)

