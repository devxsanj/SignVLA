"""Thin MediaPipe Hands wrapper (legacy `mp.solutions` API, needs mediapipe==0.10.21)."""
import mediapipe as mp

from signvla import config

mp_hands = mp.solutions.hands
_drawing = mp.solutions.drawing_utils
_styles = mp.solutions.drawing_styles


def get_hands_detector(max_num_hands=2,
                       min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                       min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE):
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            _drawing.draw_landmarks(
                image, hand, mp_hands.HAND_CONNECTIONS,
                _styles.get_default_hand_landmarks_style(),
                _styles.get_default_hand_connections_style(),
            )
