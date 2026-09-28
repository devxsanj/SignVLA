import sys
import time
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config

from src.utils import (
    get_hands_detector,
    extract_features,
    draw_styled_landmarks
)

from src.model import get_model


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = config.MODELS_DIR / "isl_gesture_model.pth"

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

gestures = checkpoint["gestures"]

model = get_model(
    num_classes=checkpoint["num_classes"],
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print()
print("=" * 60)
print("SIGNVLA — GESTURE DIAGNOSTIC")
print("=" * 60)

print("\nMODEL CLASSES:")

for i, g in enumerate(gestures):
    print(f"{i:2d} -> {g}")

print()
print("Show ONE gesture at a time.")
print("Hold it for 2–3 seconds.")
print("Press Q to quit.")
print()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    config.CAMERA_INDEX
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    config.FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    config.FRAME_HEIGHT
)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit(1)


# ============================================================
# MEDIAPIPE
# ============================================================

detector = get_hands_detector(
    max_num_hands=config.MAX_NUM_HANDS,
    min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
)


# ============================================================
# BUFFER
# ============================================================

sequence = deque(
    maxlen=config.SEQUENCE_LENGTH
)

last_inference = 0.0

INFERENCE_INTERVAL = 0.15


# ============================================================
# LOOP
# ============================================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(
        frame,
        1
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = detector.process(
        rgb
    )

    features = extract_features(
        results
    )

    sequence.append(
        features
    )

    draw_styled_landmarks(
        frame,
        results
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    now = time.time()

    if (
        len(sequence) == config.SEQUENCE_LENGTH
        and now - last_inference >= INFERENCE_INTERVAL
    ):

        last_inference = now

        x = torch.tensor(
            np.asarray(
                sequence,
                dtype=np.float32
            )
        ).unsqueeze(0)

        with torch.inference_mode():

            logits = model(x)

            probs = torch.softmax(
                logits,
                dim=1
            )[0]

        values, indices = torch.topk(
            probs,
            5
        )

        top_results = []

        for value, index in zip(
            values,
            indices
        ):

            gesture = gestures[
                int(index)
            ]

            confidence = float(
                value
            ) * 100

            top_results.append(
                (
                    gesture,
                    confidence
                )
            )

        # ----------------------------------------------------
        # TERMINAL
        # ----------------------------------------------------

        print(
            "\nTOP PREDICTIONS:"
        )

        for gesture, confidence in top_results:

            print(
                f"  {gesture:<8} "
                f"{confidence:6.2f}%"
            )

        # ----------------------------------------------------
        # HUD
        # ----------------------------------------------------

        best_gesture = top_results[0][0]
        best_confidence = top_results[0][1]

        cv2.putText(
            frame,
            f"TOP: {best_gesture.upper()}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 128),
            2
        )

        cv2.putText(
            frame,
            f"CONF: {best_confidence:.1f}%",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2
        )

        y = 110

        for gesture, confidence in top_results:

            text = (
                f"{gesture:<8} "
                f"{confidence:5.1f}%"
            )

            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            y += 23

    cv2.imshow(
        "SignVLA - Diagnostic",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key in (
        ord("q"),
        27
    ):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
detector.close()

print("\nDiagnostic stopped.")
