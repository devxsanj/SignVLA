"""
Real-time Camera & ISL Gesture Recognition Tracker
Runs in its own process to provide a standalone OpenCV window on macOS,
transmitting detected robot actions via local UDP socket to the MuJoCo simulator.
"""
import os
import sys
import time
import socket
from pathlib import Path
from collections import deque

# Auto re-exec inside virtual environment if invoked with global system Python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PY = _PROJECT_ROOT / ".venv" / "bin" / "python"
if not _VENV_PY.exists():
    _VENV_PY = _PROJECT_ROOT / "venv" / "bin" / "python"

if _VENV_PY.exists() and sys.prefix == sys.base_prefix:
    os.execv(str(_VENV_PY), [str(_VENV_PY)] + sys.argv)

import cv2
import numpy as np
import torch

sys.path.append(str(_PROJECT_ROOT))
import config
from src.utils import get_hands_detector, extract_features, draw_styled_landmarks
from src.model import get_model


UDP_IP = "127.0.0.1"
UDP_PORT = 9876


def run_vision_tracker():
    model_path = config.MODELS_DIR / "isl_gesture_model.pth"
    if not model_path.exists():
        print(f"[ERROR] Trained model file not found at: {model_path}")
        return

    # 1. Load Trained ISL Model
    device = torch.device("cpu")
    checkpoint = torch.load(str(model_path), map_location=device)
    gestures = checkpoint["gestures"]
    num_classes = checkpoint["num_classes"]
    hidden_dim = checkpoint["hidden_dim"]

    model = get_model(num_classes=num_classes, input_dim=config.TOTAL_FEATURES_PER_FRAME, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Setup UDP Socket to communicate with MuJoCo
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 3. Setup Webcam & MediaPipe
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    detector = get_hands_detector(
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
    )

    feature_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    recent_preds = deque(maxlen=6)

    predicted_gesture = "idle"
    predicted_confidence = 0.0
    active_robot_action = "READY"
    action_cooldown_until = 0.0

    print(f"\n[VISION] Camera tracker started. Streaming action commands to {UDP_IP}:{UDP_PORT}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = detector.process(rgb)
        rgb.flags.writeable = True

        features = extract_features(results)
        feature_buffer.append(features)
        draw_styled_landmarks(frame, results)

        # Inference
        if len(feature_buffer) == config.SEQUENCE_LENGTH:
            seq_tensor = torch.tensor(np.array(feature_buffer), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(seq_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

            top_idx = int(np.argmax(probs))
            conf = float(probs[top_idx])

            if conf >= 0.75:
                cand = gestures[top_idx]
                recent_preds.append(cand)
                if recent_preds.count(cand) >= 5:
                    predicted_gesture = cand
                    predicted_confidence = conf

                    if now > action_cooldown_until:
                        robot_act = config.GESTURE_ACTION_MAP.get(predicted_gesture)
                        if robot_act:
                            print(f"[ACTION TRIGGER] Sign: {predicted_gesture.upper()} -> Robot: {robot_act}")
                            sock.sendto(robot_act.encode("utf-8"), (UDP_IP, UDP_PORT))
                            active_robot_action = robot_act
                            action_cooldown_until = now + 1.5
                            recent_preds.clear()
            else:
                recent_preds.append("...")

        # Clear active action label after cooldown
        if now > action_cooldown_until:
            active_robot_action = "READY"

        # Sleek Overlay HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 25), -1)
        cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Gesture & Action Text
        cv2.putText(frame, f"SIGN: {predicted_gesture.upper()} ({predicted_confidence*100:.0f}%)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.putText(frame, f"ROBOT: {active_robot_action.upper()}",
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        info = "ISL Gesture Camera Feed | Press [Q] to Exit"
        cv2.putText(frame, info, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("ISL Gesture Feed (Camera)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            # Send quit signal to MuJoCo
            sock.sendto(b"QUIT", (UDP_IP, UDP_PORT))
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    sock.close()
    print("\n[VISION] Camera tracker exited.")


if __name__ == "__main__":
    run_vision_tracker()
