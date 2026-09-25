"""
ISL Gesture Data Recorder
Records video samples and extracts normalized MediaPipe landmarks simultaneously.

Features:
- Live MediaPipe skeleton preview
- Guided state machine (Ready -> Countdown -> Recording -> Break)
- Auto-mode (automatic sequence capture with pause intervals)
- Retake previous sample (R key)
- Saves both raw MP4 videos and normalized NPY landmark sequences
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

import cv2
import numpy as np

# Add project root to path
sys.path.append(str(_PROJECT_ROOT))
import config
from src.utils import get_hands_detector, extract_features, draw_styled_landmarks, draw_hud


def parse_args():
    parser = argparse.ArgumentParser(description="Record ISL gesture videos and extract landmarks.")
    parser.add_argument("--gestures", nargs="+", default=config.GESTURES,
                        help="List of gesture names to record.")
    parser.add_argument("--samples", type=int, default=config.SAMPLES_PER_GESTURE,
                        help="Number of samples to collect per gesture.")
    parser.add_argument("--seq_len", type=int, default=config.SEQUENCE_LENGTH,
                        help="Number of frames per sequence sample.")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="Webcam device index.")
    parser.add_argument("--countdown", type=int, default=config.COUNTDOWN_SECONDS,
                        help="Countdown duration in seconds before recording.")
    parser.add_argument("--auto", action="store_true",
                        help="Start in auto-recording mode.")
    return parser.parse_args()


def count_existing_samples(gesture_name):
    """Counts already recorded samples in landmark directory."""
    gesture_dir = config.LANDMARKS_DIR / gesture_name
    if not gesture_dir.exists():
        return 0
    return len(list(gesture_dir.glob("sample_*.npy")))


def run_recorder():
    args = parse_args()
    
    # Initialize Camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}. Please check permissions or index.")
        return

    # MediaPipe Hands
    detector = get_hands_detector(
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
    )

    print("\n" + "=" * 60)
    print("  ISL GESTURE RECORDER & FEATURE EXTRACTOR")
    print("=" * 60)
    print(f"Gestures to record : {args.gestures}")
    print(f"Samples per gesture: {args.samples}")
    print(f"Sequence length    : {args.seq_len} frames (~1 sec)")
    print(f"Hotkeys:")
    print("  [SPACE]  : Start recording sample")
    print("  [A]      : Toggle Auto-recording mode (continuous with breaks)")
    print("  [R]      : Retake / Delete previous sample")
    print("  [N]      : Next gesture")
    print("  [P]      : Previous gesture")
    print("  [Q/ESC]  : Quit")
    print("=" * 60 + "\n")

    gesture_idx = 0
    auto_mode = args.auto

    # States: READY, COUNTDOWN, RECORDING, BREAK
    state = "READY"
    state_start_time = 0.0

    recorded_frames_raw = []
    recorded_landmarks = []

    last_saved_sample_idx = None

    while cap.isOpened():
        current_gesture = args.gestures[gesture_idx]
        
        # Ensure output directories exist
        raw_gesture_dir = config.RAW_VIDEOS_DIR / current_gesture
        npy_gesture_dir = config.LANDMARKS_DIR / current_gesture
        raw_gesture_dir.mkdir(parents=True, exist_ok=True)
        npy_gesture_dir.mkdir(parents=True, exist_ok=True)

        existing_count = count_existing_samples(current_gesture)

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from webcam.")
            break

        # Flip horizontally for natural mirror display
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = detector.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Extract features for current frame (126,)
        frame_features = extract_features(results)

        # Draw landmarks on screen
        draw_styled_landmarks(frame, results)

        now = time.time()

        # State Machine
        if state == "READY":
            draw_hud(frame, current_gesture, existing_count, args.samples, "READY")
            if auto_mode and existing_count < args.samples:
                state = "COUNTDOWN"
                state_start_time = now

        elif state == "COUNTDOWN":
            elapsed = now - state_start_time
            remaining = int(args.countdown - elapsed) + 1
            if remaining <= 0:
                state = "RECORDING"
                state_start_time = now
                recorded_frames_raw = []
                recorded_landmarks = []
            else:
                draw_hud(frame, current_gesture, existing_count, args.samples, "COUNTDOWN", countdown=remaining)

        elif state == "RECORDING":
            recorded_frames_raw.append(frame.copy())
            recorded_landmarks.append(frame_features)

            progress = len(recorded_landmarks) / float(args.seq_len)
            draw_hud(frame, current_gesture, existing_count + 1, args.samples, "RECORDING", progress=progress)

            if len(recorded_landmarks) >= args.seq_len:
                # Save Data
                sample_num = existing_count + 1
                last_saved_sample_idx = sample_num

                # 1. Save Landmarks (.npy)
                npy_path = npy_gesture_dir / f"sample_{sample_num:03d}.npy"
                np.save(str(npy_path), np.array(recorded_landmarks, dtype=np.float32))

                # 2. Save Raw Video (.mp4)
                mp4_path = raw_gesture_dir / f"sample_{sample_num:03d}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(mp4_path), fourcc, config.FPS, (w, h))
                for f in recorded_frames_raw:
                    out.write(f)
                out.release()

                print(f"[SAVED] {current_gesture.upper()} -> Sample {sample_num:03d} (Video & Landmarks)")

                recorded_frames_raw = []
                recorded_landmarks = []

                state = "BREAK"
                state_start_time = now

        elif state == "BREAK":
            elapsed = now - state_start_time
            draw_hud(frame, current_gesture, existing_count, args.samples, "BREAK")
            if elapsed >= 1.5:  # 1.5 second break between samples
                if auto_mode and existing_count < args.samples:
                    state = "COUNTDOWN"
                    state_start_time = time.time()
                else:
                    state = "READY"

        # Display Auto Mode Pill on Top Right
        if auto_mode:
            cv2.putText(frame, "AUTO: ON", (w - 120, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Show Live View
        cv2.imshow("ISL Gesture Studio - MediaPipe Hands", frame)

        # Key Controls
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):  # ESC or Q
            print("\nExiting recorder...")
            break

        elif key == 32:  # SPACE
            if state in ("READY", "BREAK"):
                state = "COUNTDOWN"
                state_start_time = time.time()

        elif key in (ord('a'), ord('A')):  # Toggle Auto-mode
            auto_mode = not auto_mode
            print(f"[MODE] Auto-mode: {'ENABLED' if auto_mode else 'DISABLED'}")
            if auto_mode and state == "READY":
                state = "COUNTDOWN"
                state_start_time = time.time()

        elif key in (ord('n'), ord('N')):  # Next Gesture
            gesture_idx = (gesture_idx + 1) % len(args.gestures)
            state = "READY"
            print(f"[SWITCH] Next gesture: {args.gestures[gesture_idx]}")

        elif key in (ord('p'), ord('P')):  # Previous Gesture
            gesture_idx = (gesture_idx - 1) % len(args.gestures)
            state = "READY"
            print(f"[SWITCH] Previous gesture: {args.gestures[gesture_idx]}")

        elif key in (ord('r'), ord('R')):  # Retake last sample
            if last_saved_sample_idx and last_saved_sample_idx > 0:
                npy_file = npy_gesture_dir / f"sample_{last_saved_sample_idx:03d}.npy"
                mp4_file = raw_gesture_dir / f"sample_{last_saved_sample_idx:03d}.mp4"
                if npy_file.exists():
                    npy_file.unlink()
                if mp4_file.exists():
                    mp4_file.unlink()
                print(f"[RETAKE] Deleted {current_gesture} sample {last_saved_sample_idx:03d}")
                last_saved_sample_idx -= 1
                state = "READY"

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nSession finished.")


if __name__ == "__main__":
    run_recorder()
