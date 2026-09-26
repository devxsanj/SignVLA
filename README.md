# SignVLA — Indian Sign Language Conditioned Vision-Language-Action Control

> Research prototype for grounding Indian Sign Language (ISL) commands into robot manipulation actions using **OpenVLA-7B**, with a **RoArm-M2 Pro** embodiment in **MuJoCo**.

## Overview

SignVLA connects a human sign-language interface to a vision-language-action policy and a simulated robot:

```
Webcam / RGB
      ↓
ISL perception
      ↓
semantic instruction
      ↓
OpenVLA-7B
      ↓
7-DoF VLA action
      ↓
4-DoF RoArm action adapter
      ↓
trajectory + safety controller
      ↓
MuJoCo RoArm-M2 Pro
      ↓
workspace / cube
```

The project uses the pretrained **OpenVLA-7B** checkpoint:

```
openvla/openvla-7b
```

OpenVLA is the foundation VLA; SignVLA adds the ISL-conditioned interface, task/scene grounding, and embodiment-specific action adaptation.

---

## Base VLA

**Model:** OpenVLA-7B  
**Checkpoint:** `openvla/openvla-7b`

Official resources:

- https://github.com/openvla/openvla
- https://huggingface.co/openvla/openvla-7b

OpenVLA takes visual observations plus language instructions and predicts robot actions. Its native action representation is 7-DoF, so SignVLA includes an explicit adapter for the 4-DOF RoArm embodiment.

**This repository does not train OpenVLA from scratch.** The intended training path is parameter-efficient adaptation such as LoRA/OFT using project-specific robot demonstrations.

---

# Project Architecture

```
                    USER
                     │
              Indian Sign Language
                     │
                     ▼
             ┌───────────────┐
             │ ISL Perception│
             │ MediaPipe +   │
             │ sign encoder  │
             └───────┬───────┘
                     │
              semantic intent
                     │
                     ▼
             ┌───────────────┐
 RGB image ─►│  OpenVLA-7B   │
             │ VLA policy     │
             └───────┬───────┘
                     │
                7-DoF action
                     │
                     ▼
             ┌───────────────┐
             │ 4-DoF RoArm   │
             │ action adapter │
             └───────┬───────┘
                     │
             trajectory / safety
                     │
                     ▼
                  MuJoCo
                     │
                     ▼
               RoArm-M2 Pro
```

---

# Robot Embodiment

**RoArm-M2 Pro**

Current embodiment:

```
[base, shoulder, elbow, eoat]
```

OpenVLA action:

```
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

SignVLA:

```
OpenVLA 7-DoF
      ↓
embodiment adapter
      ↓
RoArm 4-DoF
      ↓
trajectory controller
```

The adapter is a project layer and should remain separate from the pretrained VLA.

---

# Simulation Environment

MuJoCo contains:

- RoArm-M2 Pro
- workspace table
- small dynamic cube
- target/drop region
- camera observation
- joint/action limits

The existing trajectory controller remains the deterministic execution and safety layer.

The official RoArm robot XML and meshes should not be modified for ordinary experiments. Environment additions such as the table and cube belong outside the robot mesh definitions.

---

# Repository Structure

```
SignVLA/
├── config.py
├── requirements.txt
├── requirements-vla.txt
│
├── roarm_official/
│   ├── roarm_m2_pro.xml
│   └── meshes/
│
├── simulation/
│   ├── arm_sim.py
│   └── ...
│
├── models/
│   └── ...
│
├── data/
│   └── ...
│
├── src/
│   ├── recorder.py
│   ├── create_dataset.py
│   ├── train.py
│   ├── model.py
│   ├── utils.py
│   ├── vision_tracker.py
│   ├── gesture_to_sim.py
│   ├── vla_policy.py
│   ├── vla_action_adapter.py
│   └── isl_vla.py
│
└── scripts/
    ├── setup_agx_thor.sh
    ├── download_openvla.sh
    ├── prepare_vla_dataset.py
    ├── convert_to_rlds.py
    ├── train_openvla_lora.sh
    ├── evaluate_vla.py
    └── run_signvla.py
```

---

# NVIDIA AGX Thor Setup

The training target is an **NVIDIA Jetson AGX Thor 128 GB** system.

## 1. Create the workspace

```bash
mkdir -p ~/SignVLA
cd ~/SignVLA
```

## 2. Clone OpenVLA

```bash
git clone https://github.com/openvla/openvla.git
cd openvla
```

## 3. Verify the NVIDIA stack

```bash
nvidia-smi
```

Then:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA device count:", torch.cuda.device_count())
PY
```

On Jetson, use a PyTorch build compatible with the installed JetPack/CUDA environment. Do not overwrite the vendor-supported stack with an unrelated desktop CUDA wheel.

## 4. Install OpenVLA

```bash
pip install -e .
```

Install project fine-tuning utilities:

```bash
pip install   peft   draccus   einops   accelerate   safetensors   sentencepiece   timm   wandb   ninja   packaging
```

---

# Download the Base Model

Authenticate:

```bash
huggingface-cli login
```

Then download:

```bash
huggingface-cli download openvla/openvla-7b \
  --local-dir ~/SignVLA/models/openvla-7b
```

Verify:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL = "openvla/openvla-7b"

processor = AutoProcessor.from_pretrained(
    MODEL,
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype="auto",
    low_cpu_mem_usage=True
)

print("OpenVLA-7B loaded successfully.")
```

---

# ISL Conditioning

The ISL subsystem provides a semantic command that can be grounded into language.

Examples:

```
LEFT
RIGHT
FRONT
BACK
UP
DOWN
OPEN
CLOSE
PICK
PLACE
HOME
STOP
```

These should become task instructions such as:

```
Move the robot arm to the left.
Move the robot arm to the right.
Move the robot arm forward.
Move the robot arm backward.
Move the robot arm upward.
Move the robot arm downward.
Open the gripper.
Close the gripper.
Pick up the cube.
Place the cube on the target.
Return the robot arm to its home position.
Stop the robot.
```

For manipulation, the instruction can include scene information:

```
Pick up the blue cube on the right.
Place the cube on the left target.
```

---

# VLA Training Dataset

A VLA dataset is different from the old gesture-classification dataset.

Each sample should contain:

1. RGB observation
2. semantic instruction
3. robot state
4. robot action
5. episode/timestep information

Recommended structure:

```
~/SignVLA/datasets/
└── roarm_isl/
    ├── images/
    │   ├── episode_000001/
    │   ├── episode_000002/
    │   └── ...
    ├── episodes.jsonl
    └── metadata.json
```

Example record:

```json
{
  "image": "images/episode_000001/frame_000120.jpg",
  "instruction": "Move the robot arm to the left",
  "state": [-3.14, -1.57, 1.59, 0.0],
  "action": [-3.05, -1.57, 1.59, 1.0],
  "episode_id": 1,
  "timestep": 120
}
```

For a pick task:

```
instruction:
"Pick up the blue cube on the right."

trajectory:
approach
→ descend
→ grasp
→ lift
```

The dataset should preserve the temporal sequence.

---

# MuJoCo Demonstration Generation

The preferred collection pipeline is:

```
Reset environment
        ↓
Choose task / ISL instruction
        ↓
Reference controller executes task
        ↓
Capture RGB
        ↓
Capture robot state
        ↓
Record action
        ↓
Store episode
```

Example:

```bash
python scripts/prepare_vla_dataset.py   --episodes 2000   --output ~/SignVLA/datasets/roarm_isl
```

Validate:

```bash
python scripts/prepare_vla_dataset.py --check
```

Validation should check:

- image paths
- instruction presence
- action dimensions
- episode ordering
- missing/NaN values
- train/validation split
- task balance

---

# RLDS Conversion

OpenVLA's official training pipeline works with robotics datasets in RLDS format.

The intended conversion is:

```
MuJoCo episodes
      ↓
project episode records
      ↓
RLDS
      ↓
OpenVLA fine-tuning
```

Example:

```bash
python scripts/convert_to_rlds.py   --input ~/SignVLA/datasets/roarm_isl   --output ~/SignVLA/datasets/roarm_isl_rlds
```

---

# Fine-Tuning Strategy

SignVLA should use **parameter-efficient fine-tuning**, not full 7B-parameter training.

Starting configuration:

```
Base model       openvla/openvla-7b
Method           LoRA
LoRA rank        32
Learning rate    5e-4
Batch size       4
Gradient accum.  8
Image augment.   enabled
```

These are starting values. Actual batch size should be adjusted after observing AGX Thor memory usage.

The official OpenVLA repository provides the fine-tuning scripts and supports LoRA/custom robotics datasets.

---

# Training Script

Create:

```
scripts/train_openvla_lora.sh
```

with:

```bash
#!/bin/bash
set -e

MODEL="${MODEL:-openvla/openvla-7b}"
DATA_ROOT="${DATA_ROOT:-$HOME/SignVLA/datasets}"
DATASET="${DATASET:-roarm_isl}"
RUN_ROOT="${RUN_ROOT:-$HOME/SignVLA/checkpoints}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$HOME/SignVLA/adapters}"

mkdir -p "$RUN_ROOT" "$ADAPTER_ROOT"

echo "=============================================="
echo "SignVLA - OpenVLA LoRA Fine-Tuning"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Data root: $DATA_ROOT"
echo "=============================================="

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path "$MODEL" \
  --data_root_dir "$DATA_ROOT" \
  --dataset_name "$DATASET" \
  --run_root_dir "$RUN_ROOT" \
  --adapter_tmp_dir "$ADAPTER_ROOT" \
  --lora_rank 32 \
  --batch_size 4 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 250 \
  --wandb_project "SignVLA"
```

Run:

```bash
bash scripts/train_openvla_lora.sh
```

Before a production run, check the exact command-line arguments against the OpenVLA revision checked out locally because upstream scripts can change.

---

# Inference

The final inference stack is:

```
Webcam
   ↓
ISL perception
   ↓
semantic instruction
   +
RGB scene
   ↓
fine-tuned OpenVLA
   ↓
7-DoF action
   ↓
RoArm 4-DoF adapter
   ↓
trajectory / safety controller
   ↓
MuJoCo
```

Example:

```
ISL:
LEFT

Instruction:
"Move the robot arm to the left."

RGB:
current robot + workspace

             ↓

          OpenVLA

             ↓

         VLA action

             ↓

       4-DoF adapter

             ↓

       MuJoCo RoArm
```

---

# Safety Layer

Raw VLA output should not directly control physical hardware.

Use:

```
VLA action
    ↓
validation
    ↓
workspace bounds
    ↓
joint limits
    ↓
collision checks
    ↓
trajectory interpolation
    ↓
robot controller
```

The MuJoCo trajectory layer is therefore retained as a deterministic execution/safety boundary.

---

# Evaluation

Evaluate the complete system at three levels.

## ISL perception

- sign/command accuracy
- confusion matrix
- inference latency
- signer variation

## VLA policy

- action prediction error
- task success rate
- trajectory smoothness
- inference latency

## Robot task execution

- pick success rate
- place success rate
- cube position error
- collision rate
- completion time
- recovery behavior

Primary experimental comparison:

```
Baseline:
ISL classifier → fixed trajectory

vs.

SignVLA:
ISL + RGB → OpenVLA → 4-DoF adapter → trajectory
```

---

# Reproducibility

Record with every experiment:

```
OpenVLA checkpoint
OpenVLA git revision
dataset version
number of episodes
train/validation split
LoRA configuration
learning rate
batch size
gradient accumulation
random seed
JetPack version
PyTorch version
CUDA version
hardware
```

---

# Current Status

- [x] ISL perception pipeline
- [x] sign vocabulary
- [x] RoArm-M2 Pro MuJoCo environment
- [x] table + cube environment
- [x] trajectory/action layer
- [x] OpenVLA policy scaffold
- [x] 7-DoF → 4-DoF adapter scaffold
- [ ] MuJoCo RGB/action demonstration generator
- [ ] RLDS conversion
- [ ] OpenVLA fine-tuning on AGX Thor
- [ ] fine-tuned checkpoint evaluation
- [ ] ISL → OpenVLA → RoArm closed loop
- [ ] physical RoArm validation

---

# References

- OpenVLA: https://github.com/openvla/openvla
- OpenVLA-7B: https://huggingface.co/openvla/openvla-7b
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- MuJoCo: https://mujoco.org/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/

---

## Project

**SignVLA**  
**Indian Sign Language Conditioned Vision-Language-Action Control**
