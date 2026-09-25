"""
Landmark extraction and normalization utilities for ISL gesture tracking.
"""
import numpy as np
import cv2
import mediapipe as mp

# MediaPipe Hands Setup - robust import supporting all package structures
try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
except (ImportError, AttributeError):
    try:
        import mediapipe.solutions.hands as mp_hands
        import mediapipe.solutions.drawing_utils as mp_drawing
        import mediapipe.solutions.drawing_styles as mp_drawing_styles
    except (ImportError, AttributeError):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles


def get_hands_detector(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5):
    """Initializes and returns a MediaPipe Hands detector."""
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )


def normalize_hand_landmarks(landmarks):
    """
    Normalizes a single hand's landmarks (21 landmarks x 3 coords):
    1. Origin translation: Landmark 0 (wrist) becomes (0, 0, 0)
    2. Scale invariance: Divide by distance between wrist (0) and middle MCP (9)
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  # Shape: (21, 3)

    # 1. Translate wrist to origin
    wrist = coords[0, :].copy()
    shifted = coords - wrist

    # 2. Scale normalization based on palm scale (wrist to middle knuckle)
    scale = np.linalg.norm(coords[9] - coords[0])
    if scale < 1e-6:
        # Fallback to max bounding distance if palm scale is too small
        scale = np.max(np.abs(shifted)) + 1e-6

    normalized = shifted / scale
    return normalized.flatten()  # Shape: (63,)


def extract_features(results):
    """
    Extracts fixed 126-dimensional feature vector for two hands (Left & Right):
    - Left Hand: 63 floats (21 x 3)
    - Right Hand: 63 floats (21 x 3)
    If a hand is missing, it is zero-padded.
    """
    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            normalized = normalize_hand_landmarks(hand_landmarks.landmark)

            # Assign to respective slot
            # Note: MediaPipe labels are from camera's perspective
            if label == 'Left':
                left_hand = normalized
            elif label == 'Right':
                right_hand = normalized

    return np.concatenate([left_hand, right_hand])  # Total length: 126


def draw_styled_landmarks(image, results):
    """Draws MediaPipe hand landmarks with stylish modern coloring."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


def draw_hud(image, gesture_name, sample_idx, total_samples, state, countdown=0, progress=0.0):
    """
    Draws a sleek, modern HUD overlay on the camera frame.
    States: 'WAITING', 'COUNTDOWN', 'RECORDING', 'BREAK'
    """
    h, w, _ = image.shape

    # 1. Top banner overlay (translucent dark)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 25), -1)

    # 2. Bottom status bar
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    # Top Banner Text
    cv2.putText(image, f"Gesture: {gesture_name.upper()}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(image, f"Sample: {sample_idx}/{total_samples}", (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # State Badge (Right Side of Top Banner)
    badge_color = (128, 128, 128)  # default grey
    badge_text = state

    if state == "RECORDING":
        badge_color = (0, 0, 255)      # Red
        badge_text = "REC"
        # Draw red recording dot
        cv2.circle(image, (w - 140, 38), 8, (0, 0, 255), -1)
    elif state == "COUNTDOWN":
        badge_color = (0, 165, 255)    # Orange
        badge_text = f"READY IN {countdown}"
    elif state == "READY":
        badge_color = (0, 255, 128)    # Green
        badge_text = "PRESS SPACE"
    elif state == "BREAK":
        badge_color = (255, 180, 0)    # Cyan / Blue
        badge_text = "HOLD ON..."

    cv2.putText(image, badge_text, (w - 120, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2, cv2.LINE_AA)

    # Center Countdown Overlay if in Countdown State
    if state == "COUNTDOWN" and countdown > 0:
        center_x, center_y = w // 2, h // 2
        # Translucent circle in center
        center_overlay = image.copy()
        cv2.circle(center_overlay, (center_x, center_y), 70, (20, 20, 25), -1)
        cv2.addWeighted(center_overlay, 0.7, image, 0.3, 0, image)
        cv2.circle(image, (center_x, center_y), 70, (0, 165, 255), 3)
        cv2.putText(image, str(countdown), (center_x - 18, center_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5, cv2.LINE_AA)

    # Recording Progress Bar
    if state == "RECORDING":
        bar_w = int(w * progress)
        cv2.rectangle(image, (0, 72), (bar_w, 75), (0, 0, 255), -1)

    # Bottom Instructions
    info = "[SPACE]: Record | [N]: Next Gesture | [R]: Retake | [Q]: Quit"
    cv2.putText(image, info, (20, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
