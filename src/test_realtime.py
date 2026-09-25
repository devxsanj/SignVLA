"""
Real-Time Gesture Inference & Testing
Tests the trained ISL model with live webcam feed, featuring:
- Rolling temporal window (30 frames)
- Probability confidence bars
- Temporal debouncing (triggers action only when stable)
- Simulation / Robot Action dispatch hook
"""
import os
import sys
import time
import argparse
from pathlib import Path

# Auto re-exec inside virtual environment if invoked with global system Python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PY = _PROJECT_ROOT / ".venv" / "bin" / "python"
if not _VENV_PY.exists():
    _VENV_PY = _PROJECT_ROOT / "venv" / "bin" / "python"

if _VENV_PY.exists() and sys.prefix == sys.base_prefix:
    os.execv(str(_VENV_PY), [str(_VENV_PY)] + sys.argv)

from collections import deque
import cv2
import numpy as np
import torch

sys.path.append(str(_PROJECT_ROOT))
import config
from src.utils import get_hands_detector, extract_features, draw_styled_landmarks
from src.model import get_model


def run_realtime_inference(model_path=config.MODELS_DIR / "isl_gesture_model.pth", 
                           camera_idx=config.CAMERA_INDEX,
                           conf_threshold=0.75,
                           debounce_frames=5):
    print("\n" + "=" * 60)
    print("  REAL-TIME ISL GESTURE INFERENCE")
    print("=" * 60)

    if not Path(model_path).exists():
        print(f"[ERROR] Trained model file not found at: {model_path}")
        print("Please train a model first using: python3 src/train.py")
        return

    # Load Model
    device = torch.device("cpu")  # CPU is faster for single batch real-time
    checkpoint = torch.load(str(model_path), map_location=device)

    gestures = checkpoint["gestures"]
    num_classes = checkpoint["num_classes"]
    hidden_dim = checkpoint["hidden_dim"]

    model = get_model(num_classes=num_classes, input_dim=config.TOTAL_FEATURES_PER_FRAME, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully! Recognized classes: {gestures}")
    print(f"Confidence threshold: {conf_threshold*100:.0f}%")

    # Camera & Detector
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    detector = get_hands_detector(
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
    )

    # Rolling window buffer for sequence
    feature_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    
    # State tracking
    predicted_gesture = "..."
    predicted_confidence = 0.0
    recent_predictions = deque(maxlen=debounce_frames)
    active_action = "IDLE"

    fps_time = time.time()
    fps = 30.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Calculate FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_time, 1e-4))
        fps_time = now

        # MediaPipe Processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = detector.process(rgb)
        rgb.flags.writeable = True

        features = extract_features(results)
        feature_buffer.append(features)
        draw_styled_landmarks(frame, results)

        # Run inference when buffer is ready
        if len(feature_buffer) == config.SEQUENCE_LENGTH:
            seq_tensor = torch.tensor(np.array(feature_buffer), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(seq_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])

            if top_conf >= conf_threshold:
                candidate_gesture = gestures[top_idx]
                recent_predictions.append(candidate_gesture)

                # Check debounce (stable detection across multiple frames)
                if recent_predictions.count(candidate_gesture) == debounce_frames:
                    predicted_gesture = candidate_gesture
                    predicted_confidence = top_conf
                    active_action = config.GESTURE_ACTION_MAP.get(predicted_gesture, "UNKNOWN")
            else:
                recent_predictions.append("...")
                predicted_confidence = top_conf

        # --- Render Sleek Overlay ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 25), -1)
        cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Top Bar: Recognized Sign & Robot Action
        cv2.putText(frame, f"SIGN: {predicted_gesture.upper()} ({predicted_confidence*100:.1f}%)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.putText(frame, f"ROBOT ACTION -> {active_action}",
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Bottom Bar: Buffer Progress & Controls
        buf_fill = len(feature_buffer) / float(config.SEQUENCE_LENGTH)
        cv2.putText(frame, f"Buffer: {len(feature_buffer)}/{config.SEQUENCE_LENGTH}", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        bar_w = int((w - 40) * buf_fill)
        cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 42), (0, 255, 128), -1)

        cv2.putText(frame, "Press 'Q' to Exit | Live Gesture to MuJoCo / RoArm Bridge",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow("ISL Real-Time Detector", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nInference session stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real-time ISL gesture testing.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold.")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Webcam device index.")
    args = parser.parse_args()

    run_realtime_inference(conf_threshold=args.threshold, camera_idx=args.camera)
