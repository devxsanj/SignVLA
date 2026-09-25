"""
Configuration file for ISL Gesture Detection and MuJoCo Robot Arm Control
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
LANDMARKS_DIR = DATA_DIR / "landmarks"
MODELS_DIR = PROJECT_ROOT / "models"
SIM_DIR = PROJECT_ROOT / "simulation"
ROARM_OFFICIAL_DIR = PROJECT_ROOT / "roarm_official"
ROARM_MODEL_XML = ROARM_OFFICIAL_DIR / "roarm_m2_pro.xml"
if not ROARM_MODEL_XML.exists():
    ROARM_MODEL_XML = SIM_DIR / "roarm_m2_pro.xml"
if not ROARM_MODEL_XML.exists():
    ROARM_MODEL_XML = SIM_DIR / "roarm_m2.xml"

# Make sure essential directories exist
RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SIM_DIR.mkdir(parents=True, exist_ok=True)

# Gesture Vocabulary
# You can customize these labels according to your ISL dataset needs
GESTURES = [
    "hello",
    "yes",
    "no",
    "pick",
    "place",
    "home"
]

# Gesture to MuJoCo / RoArm Action Mapping
# Maps detected gesture to a robotic primitive
GESTURE_ACTION_MAP = {
    "hello": "wave",
    "yes": "nod_head",
    "no": "shake_head",
    "pick": "pick_object",
    "place": "place_object",
    "home": "reset_to_home"
}

# Recording Parameters
SEQUENCE_LENGTH = 30         # Number of frames per gesture sequence (approx 1 sec at 30fps)
SAMPLES_PER_GESTURE = 30     # Number of video samples to collect per gesture
COUNTDOWN_SECONDS = 3        # Countdown before recording starts
CAMERA_INDEX = 0             # Default webcam index
FPS = 30                     # Target frame rate
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MediaPipe Configuration
MAX_NUM_HANDS = 2            # Support both 1-handed and 2-handed signs
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Landmark Feature Vector Shape:
# 2 hands * 21 landmarks * 3 coordinates (x, y, z) = 126 floats per frame
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
TOTAL_FEATURES_PER_FRAME = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 126
