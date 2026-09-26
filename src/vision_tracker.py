"""
Real-time ISL Gesture Recognition Tracker

Camera:
    MediaPipe -> landmark sequence -> trained model

Output:
    UDP gesture action -> gesture_to_sim.py
"""

import os
import sys
import time
import socket
from pathlib import Path
from collections import deque


# ============================================================
# VENV
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"

if not VENV_PY.exists():
    VENV_PY = PROJECT_ROOT / "venv" / "bin" / "python"


if VENV_PY.exists() and sys.prefix == sys.base_prefix:

    os.execv(
        str(VENV_PY),
        [str(VENV_PY)] + sys.argv
    )


# ============================================================
# IMPORTS
# ============================================================

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


# ============================================================
# UDP
# ============================================================

UDP_IP = "127.0.0.1"
UDP_PORT = 9876


# ============================================================
# TRACKER
# ============================================================

def run_vision_tracker():

    model_path = (
        config.MODELS_DIR
        / "isl_gesture_model.pth"
    )


    if not model_path.exists():

        print(
            "[ERROR] Model not found:"
        )

        print(
            model_path
        )

        return


    # ========================================================
    # LOAD MODEL
    # ========================================================

    device = torch.device(
        "cpu"
    )

    checkpoint = torch.load(
        str(model_path),
        map_location=device
    )

    gestures = checkpoint[
        "gestures"
    ]

    num_classes = checkpoint[
        "num_classes"
    ]

    hidden_dim = checkpoint[
        "hidden_dim"
    ]


    model = get_model(
        num_classes=num_classes,
        input_dim=config.TOTAL_FEATURES_PER_FRAME,
        hidden_dim=hidden_dim
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()


    print(
        "\n[VISION] Loaded gestures:"
    )

    print(
        gestures
    )


    # ========================================================
    # UDP
    # ========================================================

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )


    # ========================================================
    # CAMERA
    # ========================================================

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

        print(
            "[ERROR] Could not open camera."
        )

        sock.close()

        return


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    detector = get_hands_detector(
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=(
            config.MIN_DETECTION_CONFIDENCE
        ),
        min_tracking_confidence=(
            config.MIN_TRACKING_CONFIDENCE
        )
    )


    # ========================================================
    # BUFFERS
    # ========================================================

    feature_buffer = deque(
        maxlen=config.SEQUENCE_LENGTH
    )

    recent_preds = deque(
        maxlen=8
    )


    # ========================================================
    # STATE
    # ========================================================

    predicted_gesture = "idle"

    predicted_confidence = 0.0

    active_robot_action = "READY"

    last_triggered_gesture = None

    action_cooldown_until = 0.0

    confidence_threshold = 0.75


    print(
        "\n[VISION] Camera tracker started."
    )

    print(
        "[VISION] Press Q to quit."
    )


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break


        now = time.time()


        # Mirror camera.

        frame = cv2.flip(
            frame,
            1
        )


        h, w, _ = frame.shape


        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        results = detector.process(
            rgb
        )

        rgb.flags.writeable = True


        features = extract_features(
            results
        )

        feature_buffer.append(
            features
        )


        draw_styled_landmarks(
            frame,
            results
        )


        # ====================================================
        # INFERENCE
        # ====================================================

        if (
            len(feature_buffer)
            == config.SEQUENCE_LENGTH
        ):

            sequence = np.array(
                feature_buffer,
                dtype=np.float32
            )


            seq_tensor = torch.tensor(
                sequence,
                dtype=torch.float32
            ).unsqueeze(0)


            with torch.no_grad():

                logits = model(
                    seq_tensor
                )

                probs = torch.softmax(
                    logits,
                    dim=1
                ).squeeze(0).numpy()


            top_idx = int(
                np.argmax(probs)
            )

            conf = float(
                probs[top_idx]
            )


            # =================================================
            # HIGH CONFIDENCE
            # =================================================

            if conf >= confidence_threshold:

                candidate = gestures[
                    top_idx
                ]

                recent_preds.append(
                    candidate
                )


                # Require temporal consistency.

                candidate_count = (
                    recent_preds.count(
                        candidate
                    )
                )


                if candidate_count >= 5:

                    predicted_gesture = (
                        candidate
                    )

                    predicted_confidence = (
                        conf
                    )


                    robot_action = (
                        config.GESTURE_ACTION_MAP.get(
                            predicted_gesture
                        )
                    )


                    # =================================================
                    # TRIGGER
                    # =================================================

                    if (
                        robot_action
                        and now >= action_cooldown_until
                    ):

                        # Don't trigger the exact same held
                        # gesture repeatedly.

                        if (
                            predicted_gesture
                            != last_triggered_gesture
                        ):

                            print(
                                "[ACTION TRIGGER] "
                                f"{predicted_gesture.upper()} "
                                "-> "
                                f"{robot_action}"
                            )


                            sock.sendto(
                                robot_action.encode(
                                    "utf-8"
                                ),
                                (
                                    UDP_IP,
                                    UDP_PORT
                                )
                            )


                            active_robot_action = (
                                robot_action
                            )


                            action_cooldown_until = (
                                now + 1.5
                            )


                            last_triggered_gesture = (
                                predicted_gesture
                            )


                            recent_preds.clear()


            else:

                predicted_confidence = conf

                recent_preds.append(
                    "..."
                )


        # ====================================================
        # ALLOW NEW GESTURE
        # ====================================================

        # After cooldown and enough non-matching frames,
        # the same gesture can be triggered again.

        if now > action_cooldown_until:

            active_robot_action = "READY"


            if (
                len(recent_preds) >= 3
                and predicted_gesture not in recent_preds
            ):

                last_triggered_gesture = None


        # ====================================================
        # HUD
        # ====================================================

        overlay = frame.copy()


        cv2.rectangle(
            overlay,
            (0, 0),
            (w, 80),
            (20, 20, 25),
            -1
        )


        cv2.rectangle(
            overlay,
            (0, h - 45),
            (w, h),
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
            (
                f"SIGN: "
                f"{predicted_gesture.upper()} "
                f"({predicted_confidence * 100:.0f}%)"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 128),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            (
                f"ROBOT: "
                f"{active_robot_action.upper()}"
            ),
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "ISL Gesture Camera | Q = Exit",
            (20, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )


        cv2.imshow(
            "SignVLA - ISL Gesture Feed",
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key in (
            27,
            ord("q")
        ):

            print(
                "\n[VISION] Q pressed."
            )


            sock.sendto(
                b"QUIT",
                (
                    UDP_IP,
                    UDP_PORT
                )
            )


            break


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    detector.close()

    sock.close()


    print(
        "\n[VISION] Camera tracker exited."
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    run_vision_tracker()