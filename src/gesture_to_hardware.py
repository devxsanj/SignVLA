"""
Live ISL Gesture Control for Physical RoArm-M2 Pro Robotic Arm
Detects gestures in real time using webcam and controls the physical RoArm via USB-Serial.
"""
import os
import sys
import time
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
from src.roarm_hardware import RoArmM2Hardware, find_roarm_port


def run_hardware_bridge():
    model_path = config.MODELS_DIR / "isl_gesture_model.pth"
    if not model_path.exists():
        print(f"[ERROR] Trained model file not found at: {model_path}")
        return

    # 1. Connect to RoArm-M2 Hardware
    port = find_roarm_port()
    arm = RoArmM2Hardware(port=port)
    if not arm.connect():
        print("[WARN] Could not connect to physical RoArm-M2. Run simulation with `python3 src/gesture_to_sim.py` instead.")
        return

    arm.home()

    # 2. Load Model
    device = torch.device("cpu")
    checkpoint = torch.load(str(model_path), map_location=device)
    gestures = checkpoint["gestures"]
    model = get_model(num_classes=checkpoint["num_classes"], 
                      input_dim=config.TOTAL_FEATURES_PER_FRAME, 
                      hidden_dim=checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Setup Camera
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
    action_cooldown_until = 0.0

    print("\n" + "=" * 60)
    print("  PHYSICAL ROARM-M2 PRO ISL GESTURE CONTROLLER")
    print("=" * 60)
    print("Hold gesture for ~0.3s to trigger robot action.")
    print("Press [Q] or [ESC] on camera feed to exit.")
    print("=" * 60 + "\n")

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
                            print(f"[TRIGGER] Sign: {predicted_gesture.upper()} -> Hardware Action: {robot_act}")
                            arm.execute_action(robot_act)
                            action_cooldown_until = time.time() + 1.5
                            recent_preds.clear()
            else:
                recent_preds.append("...")

        # HUD Overlay
        cv2.putText(frame, f"SIGN: {predicted_gesture.upper()} ({predicted_confidence*100:.0f}%)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)
        cv2.putText(frame, "Hardware RoArm Bridge Active | [Q] to Exit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("RoArm Hardware Controller", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    arm.close()
    print("\nHardware bridge closed.")


if __name__ == "__main__":
    run_hardware_bridge()
