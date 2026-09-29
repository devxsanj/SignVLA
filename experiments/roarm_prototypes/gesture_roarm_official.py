import os
import time
import cv2
import numpy as np
import mujoco
import mujoco.viewer
import mediapipe as mp


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "assets/roarm_m2_pro/roarm_m2_pro.xml"
CAMERA_ID = 0

MOVE_TIME = 1.5
GESTURE_CONFIRM_FRAMES = 8


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print()
print("===================================")
print("     Official RoArm-M2 Pro")
print("===================================")
print()
print("Model:", MODEL_PATH)
print("Joints:", model.njnt)
print("Actuators:", model.nu)
print()

for i in range(model.njnt):
    print(f"Joint {i}: {model.joint(i).name}")

print()

for i in range(model.nu):
    print(f"Actuator {i}: {model.actuator(i).name}")

print()


# ============================================================
# POSES
# ============================================================

HOME = np.array([
    0.00,
    0.00,
    0.00,
    0.00
])

PICK = np.array([
    0.00,
    -1.15,
    1.65,
    0.85
])

PLACE = np.array([
    0.75,
    -0.95,
    1.35,
    1.15
])


# ============================================================
# LIMIT TO ACTUATOR RANGES
# ============================================================

def clamp_pose(pose):

    pose = pose.copy()

    for i in range(4):

        actuator = model.actuator(i)

        if actuator.ctrllimited:

            pose[i] = np.clip(
                pose[i],
                actuator.ctrlrange[0],
                actuator.ctrlrange[1]
            )

    return pose


HOME = clamp_pose(HOME)
PICK = clamp_pose(PICK)
PLACE = clamp_pose(PLACE)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)


# ============================================================
# GESTURE DETECTION
# ============================================================

def finger_extended(lm, tip, pip, wrist):

    tip_distance = np.linalg.norm(
        np.array([
            lm[tip].x - lm[wrist].x,
            lm[tip].y - lm[wrist].y
        ])
    )

    pip_distance = np.linalg.norm(
        np.array([
            lm[pip].x - lm[wrist].x,
            lm[pip].y - lm[wrist].y
        ])
    )

    return tip_distance > pip_distance * 1.10


def detect_gesture(hand):

    lm = hand.landmark

    wrist = mp_hands.HandLandmark.WRIST

    index = finger_extended(
        lm,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        wrist
    )

    middle = finger_extended(
        lm,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        wrist
    )

    ring = finger_extended(
        lm,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        wrist
    )

    pinky = finger_extended(
        lm,
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.PINKY_PIP,
        wrist
    )

    # POINT
    if index and not middle and not ring and not pinky:
        return "POINT"

    # OPEN PALM
    if index and middle and ring and pinky:
        return "OPEN PALM"

    # FIST
    if not index and not middle and not ring and not pinky:
        return "FIST"

    return "UNKNOWN"


# ============================================================
# SMOOTH ROBOT MOVEMENT
# ============================================================

def move_arm(target, viewer):

    start = np.array([
        data.ctrl[0],
        data.ctrl[1],
        data.ctrl[2],
        data.ctrl[3]
    ])

    target = clamp_pose(target)

    start_time = time.time()

    while True:

        elapsed = time.time() - start_time

        t = elapsed / MOVE_TIME

        if t >= 1.0:
            t = 1.0

        # Smoothstep
        s = t * t * (3.0 - 2.0 * t)

        command = start + (target - start) * s

        for i in range(4):
            data.ctrl[i] = command[i]

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.002)

        if t >= 1.0:
            break

    for i in range(4):
        data.ctrl[i] = target[i]

    for _ in range(100):

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.002)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():

    print()
    print("ERROR: Camera could not be opened.")
    print()
    print("Check:")
    print("System Settings")
    print("→ Privacy & Security")
    print("→ Camera")
    print("→ Terminal / Python")
    print()

    raise SystemExit(1)


# ============================================================
# INITIAL STATE
# ============================================================

last_gesture = "NONE"
stable_gesture = "NONE"

gesture_frames = 0

last_command = "HOME"


# ============================================================
# START
# ============================================================

print()
print("===================================")
print("     SignVLA Gesture Control")
print("===================================")
print()
print("POINT      -> PICK")
print("OPEN PALM  -> HOME")
print("FIST       -> PLACE")
print()
print("Press Ctrl+C to quit.")
print()

# Start at HOME

for i in range(4):
    data.ctrl[i] = HOME[i]


# ============================================================
# MUJOCO
# ============================================================

try:

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        viewer.sync()

        while viewer.is_running():

            # =================================================
            # CAMERA
            # =================================================

            success, frame = cap.read()

            if not success:
                continue

            # Mirror image
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)


            gesture = "NO HAND"


            # =================================================
            # HAND DETECTION
            # =================================================

            if results.multi_hand_landmarks:

                hand = results.multi_hand_landmarks[0]

                gesture = detect_gesture(hand)


                # Draw landmarks internally
                # No cv2.imshow() is used.

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )


            # =================================================
            # STABILITY FILTER
            # =================================================

            if gesture == last_gesture:

                gesture_frames += 1

            else:

                last_gesture = gesture
                gesture_frames = 0


            # =================================================
            # CONFIRM GESTURE
            # =================================================

            if gesture_frames >= GESTURE_CONFIRM_FRAMES:

                if gesture != stable_gesture:

                    stable_gesture = gesture


                    # -----------------------------------------
                    # POINT
                    # -----------------------------------------

                    if gesture == "POINT":

                        if last_command != "PICK":

                            print(
                                "\n☝️ POINT  →  PICK"
                            )

                            last_command = "PICK"

                            move_arm(
                                PICK,
                                viewer
                            )


                    # -----------------------------------------
                    # OPEN PALM
                    # -----------------------------------------

                    elif gesture == "OPEN PALM":

                        if last_command != "HOME":

                            print(
                                "\n🖐️ OPEN PALM  →  HOME"
                            )

                            last_command = "HOME"

                            move_arm(
                                HOME,
                                viewer
                            )


                    # -----------------------------------------
                    # FIST
                    # -----------------------------------------

                    elif gesture == "FIST":

                        if last_command != "PLACE":

                            print(
                                "\n✊ FIST  →  PLACE"
                            )

                            last_command = "PLACE"

                            move_arm(
                                PLACE,
                                viewer
                            )


            # =================================================
            # TERMINAL STATUS
            # =================================================

            print(
                f"\rGesture: {gesture:<10} "
                f"| Command: {last_command:<6}",
                end="",
                flush=True
            )


            # =================================================
            # PHYSICS
            # =================================================

            mujoco.mj_step(
                model,
                data
            )

            viewer.sync()

            time.sleep(0.005)


except KeyboardInterrupt:

    print("\n\nStopping...")


finally:

    cap.release()

    hands.close()

    print()
    print("===================================")
    print("     Simulation stopped")
    print("===================================")
