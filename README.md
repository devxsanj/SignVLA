# ISL Gesture Detector & RoArm-M2 MuJoCo Simulator

An end-to-end pipeline to recognize Sign Language gestures using **MediaPipe**, train a deep sequence recognition model (PyTorch Bi-GRU), and drive a **RoArm-M2 Pro** robotic arm in **MuJoCo physics simulation**.

---

## 📁 Project Structure

```
ISL Detector/
├── config.py                 # Central configurations (gestures, fps, model paths)
├── requirements.txt          # Python dependencies
├── roarm_official/           # Official RoArm-M2 Pro CAD Model & STL Meshes
│   ├── roarm_m2_pro.xml      # Official MuJoCo 4-DOF model (+ end-effector)
│   └── meshes/               # High-precision STL meshes (base, links 1-3, gripper)
├── simulation/
│   ├── roarm_m2_pro.xml      # MuJoCo 3D XML model replica
│   ├── meshes/               # STL meshes
│   └── arm_sim.py            # Robot kinematic primitives (wave, pick, place, nod)
├── data/
│   ├── raw_videos/           # Saved MP4 clips (<gesture>/sample_XXX.mp4)
│   ├── landmarks/            # Extracted normalized landmarks (<gesture>/sample_XXX.npy)
│   ├── dataset.npz           # Compiled train/test dataset
│   └── label_map.json        # Class label indices
├── models/
│   └── isl_gesture_model.pth # Trained PyTorch model weights
└── src/
    ├── utils.py              # Landmark normalization & OpenCV HUD overlays
    ├── recorder.py           # Interactive video & landmark recording studio
    ├── extract_landmarks.py  # Batch re-extract landmarks from existing videos
    ├── create_dataset.py     # Compiles .npy files into dataset.npz
    ├── model.py              # Bi-directional GRU neural network architecture
    ├── train.py              # Model training script
    ├── test_realtime.py      # Real-time webcam gesture testing
    └── gesture_to_sim.py     # Direct bridge: Webcam Gestures -> 3D MuJoCo RoArm
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment
```bash
source venv/bin/activate
```

---

### 2. Step 1: Record Gestures (Phase 1)
Run the interactive recorder:
```bash
python3 src/recorder.py
```
- **Controls**:
  - `[SPACE]`: Start recording a sample (3-second countdown $\to$ 1-second sequence).
  - `[A]`: Toggle **Auto-Mode** (records samples automatically with pauses).
  - `[R]`: Retake / delete previous sample.
  - `[N]`: Switch to next gesture.
  - `[P]`: Switch to previous gesture.
  - `[Q]`: Quit and finish.

*Both the raw `.mp4` video (for your records) and the normalized `.npy` sequence are saved simultaneously.*

---

### 3. Step 2: Compile Dataset
Once you have recorded samples for your gestures:
```bash
python3 src/create_dataset.py
```
This splits the data into train/test sets and generates `data/dataset.npz` and `data/label_map.json`.

---

### 4. Step 3: Train the Gesture Recognition Model
```bash
python3 src/train.py --epochs 30
```
This trains the Bi-GRU model and saves the best checkpoint to `models/isl_gesture_model.pth`.

---

### 5. Step 4: Test in Real-Time
To test your gestures with live webcam feedback:
```bash
python3 src/test_realtime.py
```

---

### 6. Step 5: Drive the RoArm-M2 in 3D MuJoCo Simulation
To run the full end-to-end system where your webcam gestures control the simulated RoArm M2 in real time:
```bash
python3 src/gesture_to_sim.py
```

| Sign Gesture | MuJoCo / Physical RoArm Action |
| :--- | :--- |
| **`hello`** | Smooth friendly wave |
| **`yes`** | End-effector nod |
| **`no`** | Base yaw shake |
| **`pick`** | Reaches down, closes gripper on green cube, lifts up |
| **`place`** | Swings to side, lowers, opens gripper, returns home |
| **`home`** | Resets arm to neutral home posture |

---

### 7. Step 6: Drive the PHYSICAL RoArm-M2 Pro over USB
When you are ready to connect your physical **RoArm-M2 Pro**:
1. Plug the RoArm-M2 Pro USB cable into your Mac.
2. Check that the port is detected:
   ```bash
   python3 src/roarm_hardware.py
   ```
3. Run the live physical robot bridge:
   ```bash
   python3 src/gesture_to_hardware.py
   ```
