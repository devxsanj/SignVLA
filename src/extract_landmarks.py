"""
Batch Landmark Extractor
Processes recorded MP4 videos and extracts normalized landmark sequences into .npy files.
Allows re-processing videos anytime if normalization logic or parameters change.
"""
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.utils import get_hands_detector, extract_features


def resample_sequence(sequence, target_len):
    """
    Interpolates or trims a variable-length sequence of landmark frames 
    to exactly target_len frames using linear interpolation.
    """
    current_len = len(sequence)
    if current_len == target_len:
        return np.array(sequence, dtype=np.float32)
    
    if current_len == 0:
        return np.zeros((target_len, config.TOTAL_FEATURES_PER_FRAME), dtype=np.float32)

    seq_arr = np.array(sequence, dtype=np.float32)  # (T, D)
    indices = np.linspace(0, current_len - 1, num=target_len)
    resampled = np.zeros((target_len, seq_arr.shape[1]), dtype=np.float32)

    for dim in range(seq_arr.shape[1]):
        resampled[:, dim] = np.interp(indices, np.arange(current_len), seq_arr[:, dim])

    return resampled


def process_video(video_path, detector, target_len=config.SEQUENCE_LENGTH):
    """Reads a video file and extracts normalized landmarks frame-by-frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return None

    frames_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = detector.process(rgb_frame)
        rgb_frame.flags.writeable = True

        features = extract_features(results)
        frames_landmarks.append(features)

    cap.release()

    # Resample to fixed sequence length (e.g. 30 frames)
    fixed_sequence = resample_sequence(frames_landmarks, target_len)
    return fixed_sequence


def batch_extract(raw_videos_dir=config.RAW_VIDEOS_DIR, output_dir=config.LANDMARKS_DIR):
    detector = get_hands_detector(
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
    )

    print("\n" + "=" * 60)
    print("  BATCH VIDEO LANDMARK EXTRACTION")
    print("=" * 60)
    print(f"Reading videos from: {raw_videos_dir}")
    print(f"Saving landmarks to: {output_dir}\n")

    video_files = list(raw_videos_dir.glob("*/*.mp4"))
    if not video_files:
        print("[INFO] No MP4 videos found in data/raw_videos/. Record some videos first using recorder.py.")
        return

    processed = 0
    for video_path in sorted(video_files):
        gesture_name = video_path.parent.name
        sample_name = video_path.stem

        dest_dir = output_dir / gesture_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{sample_name}.npy"

        seq = process_video(video_path, detector)
        if seq is not None:
            np.save(str(dest_file), seq)
            processed += 1
            print(f"[{processed}/{len(video_files)}] Processed: {gesture_name}/{sample_name} -> Shape {seq.shape}")

    detector.close()
    print(f"\n[DONE] Successfully extracted landmarks for {processed} video files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch extract landmarks from videos.")
    parser.add_argument("--videos_dir", type=Path, default=config.RAW_VIDEOS_DIR)
    parser.add_argument("--out_dir", type=Path, default=config.LANDMARKS_DIR)
    args = parser.parse_args()

    batch_extract(args.videos_dir, args.out_dir)
