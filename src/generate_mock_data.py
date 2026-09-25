"""
Mock Data Generator
Generates synthetic landmark sequences for testing pipeline flow, dataset compilation,
model training, and MuJoCo simulation without needing physical webcam recording first.
"""
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def generate_mock_gestures(samples_per_gesture=15):
    print("\n[INFO] Generating synthetic landmark sequences for testing...")
    
    np.random.seed(42)
    t = np.linspace(0, 2 * np.pi, config.SEQUENCE_LENGTH)

    for g_idx, gesture in enumerate(config.GESTURES):
        gesture_dir = config.LANDMARKS_DIR / gesture
        gesture_dir.mkdir(parents=True, exist_ok=True)

        for s_idx in range(1, samples_per_gesture + 1):
            # Base pattern distinct per gesture (sinusoidal frequency + offset)
            freq = (g_idx + 1) * 0.8
            base_signal = np.sin(freq * t)[:, None]  # (30, 1)

            # Replicate across 126 coordinates with noise
            noise = np.random.normal(0, 0.05, size=(config.SEQUENCE_LENGTH, config.TOTAL_FEATURES_PER_FRAME))
            seq = np.repeat(base_signal, config.TOTAL_FEATURES_PER_FRAME, axis=1) + noise
            seq = seq.astype(np.float32)

            out_path = gesture_dir / f"sample_{s_idx:03d}.npy"
            np.save(str(out_path), seq)

        print(f"  Generated {samples_per_gesture} mock samples for gesture: '{gesture}'")

    print("[SUCCESS] Mock data generation complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    generate_mock_gestures(args.samples)
