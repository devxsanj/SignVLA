import os
import sys
import time
import socket
from pathlib import Path
from collections import deque, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"

if not VENV_PY.exists():
    VENV_PY = PROJECT_ROOT / "venv" / "bin" / "python"

if VENV_PY.exists() and sys.prefix == sys.base_prefix:
    os.execv(str(VENV_PY), [str(VENV_PY)] + sys.argv)

import cv2
import numpy as np
import torch

sys.path.append(str(PROJECT_ROOT))

import config

from src.utils import (
    get_hands_detector,
    extract_features,
    draw_styled_landmarks
)

from src.model import get_model


UDP_IP = "127.0.0.1"
UDP_PORT = 9876

SEQUENCE_LENGTH = 30

# Much less aggressive than the old 0.75
CONFIDENCE_THRESHOLD = 0.55

# Number of matching predictions required
STABLE_FRAMES = 3

# Time before same gesture can trigger again
COOLDOWN = 1.2


def run_vision_tracker():

    model_path = config.MODELS_DIR / "isl_gesture_model.pth"

    if not model_path.exists():
        print("[ERROR] Model not found:", model_path)
        return

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    device = torch.device("cpu")

    checkpoint = torch.load(
        str(model_path),
        map_location=device,
        weights_only=False
    )

    gestures = checkpoint["gestures"]
    num_classes = checkpoint["num_classes"]
    hidden_dim = checkpoint["hidden_dim"]

    model = get_model(
        num_classes=num_classes,
        input_dim=config.TOTAL_FEATURES_PER_FRAME,
        hidden_dim=hidden_dim
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print()
    print("=" * 60)
    print("SIGNVLA VISION TRACKER")
    print("=" * 60)

    print("\nLoaded model:")
    for i, g in enumerate(gestures):
        print(f"{i:2d} -> {g}")

    print("\nValidation accuracy:",
          f"{checkpoint.get('val_acc', 0)*100:.2f}%")

    # --------------------------------------------------------
    # UDP
    # --------------------------------------------------------

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        config.FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        config.FRAME_HEIGHT
    )

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    detector = get_hands_detector(
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
    )

    # --------------------------------------------------------
    # BUFFERS
    # --------------------------------------------------------

    feature_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    prediction_buffer = deque(
        maxlen=STABLE_FRAMES
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    displayed_gesture = "WAITING"
    displayed_confidence = 0.0

    last_triggered_gesture = None
    cooldown_until = 0

    print()
    print("Camera started.")
    print("Show a gesture.")
    print("Press Q to quit.")
    print()

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            continue

        now = time.time()

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        results = detector.process(rgb)

        rgb.flags.writeable = True

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        features = extract_features(results)

        feature_buffer.append(features)

        draw_styled_landmarks(
            frame,
            results
        )

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        if len(feature_buffer) == SEQUENCE_LENGTH:

            sequence = np.asarray(
                feature_buffer,
                dtype=np.float32
            )

            tensor = torch.from_numpy(
                sequence
            ).unsqueeze(0)

            with torch.no_grad():

                logits = model(tensor)

                probs = torch.softmax(
                    logits,
                    dim=1
                )[0]

            top_idx = int(
                torch.argmax(probs).item()
            )

            confidence = float(
                probs[top_idx].item()
            )

            candidate = gestures[top_idx]

            # ------------------------------------------------
            # DEBUG
            # ------------------------------------------------

            top_values, top_indices = torch.topk(
                probs,
                min(3, len(gestures))
            )

            top3 = [
                f"{gestures[int(i)]}:{float(v)*100:.0f}%"
                for v, i in zip(top_values, top_indices)
            ]

            # ------------------------------------------------
            # CONFIDENCE FILTER
            # ------------------------------------------------

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_buffer.append(candidate)

                counts = Counter(prediction_buffer)

                stable_gesture, count = counts.most_common(1)[0]

                if count >= STABLE_FRAMES:

                    displayed_gesture = stable_gesture
                    displayed_confidence = confidence

                    # ----------------------------------------
                    # ACTION
                    # ----------------------------------------

                    if now >= cooldown_until:

                        if stable_gesture != last_triggered_gesture:

                            action = config.GESTURE_ACTION_MAP.get(
                                stable_gesture
                            )

                            if action:

                                print(
                                    f"\n[ACTION] "
                                    f"{stable_gesture.upper()} "
                                    f"({confidence*100:.1f}%)"
                                    f" -> {action}"
                                )

                                sock.sendto(
                                    action.encode(),
                                    (
                                        UDP_IP,
                                        UDP_PORT
                                    )
                                )

                                last_triggered_gesture = stable_gesture

                                cooldown_until = (
                                    now + COOLDOWN
                                )

                                prediction_buffer.clear()

            else:

                displayed_confidence = confidence

        # ----------------------------------------------------
        # RESET HELD GESTURE
        # ----------------------------------------------------

        # Once another prediction appears, allow the previous
        # gesture to trigger again later.

        if prediction_buffer:

            current_votes = Counter(
                prediction_buffer
            )

            current_gesture = (
                current_votes.most_common(1)[0][0]
            )

            if (
                last_triggered_gesture is not None
                and current_gesture != last_triggered_gesture
            ):

                last_triggered_gesture = None

        # ----------------------------------------------------
        # HUD
        # ----------------------------------------------------

        h, w, _ = frame.shape

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, 105),
            (20, 20, 25),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.75,
            frame,
            0.25,
            0,
            frame
        )

        cv2.putText(
            frame,
            f"SIGN: {displayed_gesture.upper()}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 128),
            2
        )

        cv2.putText(
            frame,
            f"CONF: {displayed_confidence*100:.1f}%",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = EXIT",
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )

        cv2.imshow(
            "SignVLA - ISL Gesture Feed",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):

            sock.sendto(
                b"QUIT",
                (
                    UDP_IP,
                    UDP_PORT
                )
            )

            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    sock.close()

    print("\n[VISION] Tracker exited.")


if __name__ == "__main__":
    run_vision_tracker()
    