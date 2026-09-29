import time
import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer


# ============================================================
# OPENARM MUJOCO
# ============================================================

XML_PATH = ".venv/share/openarm_mujoco/v2/openarm_bimanual.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# ============================================================
# ROBOT POSES
# ============================================================

POSES = {
    "HOME": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ],

    "PICK": [
        0.0,
        -0.5,
        0.8,
        -1.2,
        0.0,
        0.6,
        0.0
    ],

    "PLACE": [
        0.5,
        -0.3,
        0.6,
        -1.0,
        0.0,
        0.5,
        0.0
    ],
}


current_target = POSES["HOME"]

last_command = None
last_command_time = 0

COMMAND_DELAY = 1.0


# ============================================================
# MEDIAPIPE HANDS
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    hands.close()
    exit()


print()
print("===================================")
print("     SignVLA → OpenArm Controller")
print("===================================")
print()
print("GESTURES:")
print()
print("  POINT      → PICK")
print("  OPEN PALM  → HOME")
print("  FIST       → PLACE")
print()
print("The camera runs in the background.")
print("Close MuJoCo or press Ctrl+C to stop.")
print()


# ============================================================
# FINGER DETECTION
# ============================================================

def finger_extended(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y


# ============================================================
# GESTURE DETECTION
# ============================================================

def detect_gesture(hand_landmarks):

    lm = hand_landmarks.landmark

    index = finger_extended(lm, 8, 6)
    middle = finger_extended(lm, 12, 10)
    ring = finger_extended(lm, 16, 14)
    pinky = finger_extended(lm, 20, 18)

    # ----------------------------------------
    # POINT
    # Index finger only
    # ----------------------------------------

    if index and not middle and not ring and not pinky:
        return "POINT"

    # ----------------------------------------
    # OPEN PALM
    # Four fingers extended
    # ----------------------------------------

    if index and middle and ring and pinky:
        return "OPEN"

    # ----------------------------------------
    # FIST
    # No fingers extended
    # ----------------------------------------

    if not index and not middle and not ring and not pinky:
        return "FIST"

    return "NONE"


# ============================================================
# GESTURE → ROBOT COMMAND
# ============================================================

def execute_gesture(gesture):

    global current_target
    global last_command
    global last_command_time

    if gesture == "POINT":

        command = "PICK"

    elif gesture == "OPEN":

        command = "HOME"

    elif gesture == "FIST":

        command = "PLACE"

    else:

        return

    now = time.time()

    # Don't repeatedly send the same command
    if command == last_command:
        return

    # Small safety delay
    if now - last_command_time < COMMAND_DELAY:
        return

    current_target = POSES[command]

    last_command = command
    last_command_time = now

    print(
        f"Gesture: {gesture}  →  Moving to {command}",
        flush=True
    )


# ============================================================
# MUJOCO SIMULATION
# ============================================================

try:

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            # =================================================
            # CAMERA
            # =================================================

            success, frame = cap.read()

            if not success:

                print(
                    "ERROR: Could not read camera frame."
                )

                break


            # Mirror camera image

            frame = cv2.flip(frame, 1)


            # BGR → RGB

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # =================================================
            # MEDIAPIPE
            # =================================================

            results = hands.process(rgb_frame)


            gesture = "NONE"


            if results.multi_hand_landmarks:

                hand_landmarks = (
                    results.multi_hand_landmarks[0]
                )


                # Draw landmarks on the frame.
                #
                # We don't display the frame because
                # cv2.imshow conflicts with MuJoCo's
                # macOS viewer when using mjpython.

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )


                # Detect gesture

                gesture = detect_gesture(
                    hand_landmarks
                )


                # Send gesture to OpenArm

                execute_gesture(gesture)


            # =================================================
            # RIGHT ARM CONTROL
            # =================================================

            # Right arm actuators are 8–14

            for i in range(7):

                data.ctrl[8 + i] = current_target[i]


            # Right gripper actuator

            data.ctrl[15] = 0.0


            # =================================================
            # PHYSICS
            # =================================================

            mujoco.mj_step(
                model,
                data
            )


            # Update MuJoCo viewer

            viewer.sync()


            # Small delay

            time.sleep(0.01)


except KeyboardInterrupt:

    print()
    print("Stopping SignVLA → OpenArm...")


finally:

    cap.release()

    hands.close()

    cv2.destroyAllWindows()

    print("Controller stopped.")
