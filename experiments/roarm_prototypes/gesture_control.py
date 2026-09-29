import cv2
import mediapipe as mp

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def finger_states(landmarks):
    """
    Returns the state of four fingers:
    index, middle, ring, pinky

    True  = extended
    False = folded
    """

    fingers = []

    # Finger tip and PIP landmark IDs
    pairs = [
        (8, 6),    # Index
        (12, 10),  # Middle
        (16, 14),  # Ring
        (20, 18)   # Pinky
    ]

    for tip, pip in pairs:
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    return fingers


def detect_gesture(landmarks):
    fingers = finger_states(landmarks)

    index, middle, ring, pinky = fingers

    # Four fingers extended = OPEN PALM
    if index and middle and ring and pinky:
        return "OPEN PALM"

    # All four fingers folded = FIST
    if not index and not middle and not ring and not pinky:
        return "FIST"

    # Only index finger extended = POINT
    if index and not middle and not ring and not pinky:
        return "POINT"

    return "UNKNOWN"


# Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("================================")
print("SignVLA Gesture Detection")
print("================================")
print("OPEN PALM → HOME")
print("FIST      → PICK")
print("POINT     → PLACE")
print("Press Q to quit.")
print()

while True:
    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # Mirror view
    frame = cv2.flip(frame, 1)

    # BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb_frame)

    gesture = "NO HAND"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        gesture = detect_gesture(hand_landmarks.landmark)

    # Display gesture
    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("SignVLA - Gesture Control", frame)

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Gesture detection stopped.")
