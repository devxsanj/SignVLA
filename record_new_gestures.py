import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time

# ============================================================
# SETTINGS
# ============================================================

GESTURES = [
    "up",
    "down",
    "left",
    "right",
    "front",
    "back",
    "open",
    "close",
]

SEQUENCE_LENGTH = 30
SAMPLES_PER_GESTURE = 40

SAVE_DIR = Path("data/landmarks")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# ============================================================
# LANDMARK EXTRACTION
# 2 hands × 21 landmarks × 3 coordinates = 126
# ============================================================

def extract_landmarks(results):
    data = []

    # Always keep exactly 2 hands
    hands_data = []

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks[:2]:
            landmarks = []

            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            hands_data.append(landmarks)

    while len(hands_data) < 2:
        hands_data.append([0.0] * 63)

    data = hands_data[0] + hands_data[1]

    return np.array(data, dtype=np.float32)


# ============================================================
# RECORD ONE SAMPLE
# ============================================================

def record_sample(gesture, sample_number):

    sequence = []

    print()
    print("=" * 55)
    print(f"Gesture : {gesture.upper()}")
    print(f"Sample  : {sample_number}/{SAMPLES_PER_GESTURE}")
    print("=" * 55)
    print("Get ready...")

    # Countdown
    for countdown in [3, 2, 1]:
        start = time.time()

        while time.time() - start < 1:
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                f"{gesture.upper()}  |  GET READY: {countdown}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

            cv2.imshow("ISL Gesture Recorder", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

    print("RECORDING...")

    while len(sequence) < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks = extract_landmarks(results)

        sequence.append(landmarks)

        # Draw hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

        cv2.putText(
            frame,
            f"{gesture.upper()}  |  Sample {sample_number}/{SAMPLES_PER_GESTURE}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("ISL Gesture Recorder", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    sequence = np.array(sequence, dtype=np.float32)

    # Safety check
    if sequence.shape != (SEQUENCE_LENGTH, 126):
        print("ERROR: Wrong sequence shape:", sequence.shape)
        return False

    gesture_dir = SAVE_DIR / gesture
    gesture_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(gesture_dir.glob("sample_*.npy"))

    next_number = len(existing) + 1

    output_file = gesture_dir / f"sample_{next_number:03d}.npy"

    np.save(output_file, sequence)

    print(f"Saved: {output_file}")
    print(f"Shape: {sequence.shape}")

    return True


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 60)
print("        ISL — NEW GESTURE RECORDER")
print("=" * 60)
print()
print("New gestures:")
for i, gesture in enumerate(GESTURES, 1):
    print(f"  {i}. {gesture}")

print()
print(f"Samples per gesture : {SAMPLES_PER_GESTURE}")
print(f"Frames per sample  : {SEQUENCE_LENGTH}")
print("Features per frame : 126")
print()
print("Press Q anytime to stop.")
print()

for gesture in GESTURES:

    print()
    print("#" * 60)
    print(f"NEXT GESTURE: {gesture.upper()}")
    print("#" * 60)

    input(f"Press ENTER when ready to record '{gesture}'...")

    for sample in range(1, SAMPLES_PER_GESTURE + 1):

        success = record_sample(gesture, sample)

        if not success:
            print("Recording stopped.")
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

        time.sleep(0.3)

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("ALL 8 GESTURES RECORDED")
print("=" * 60)
