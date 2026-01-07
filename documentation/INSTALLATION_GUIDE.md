# Installation & Setup Guide

This section provides a comprehensive guide to setting up and running the Reinforcement Learning project.

## System Requirements

*   **Operating System:** Windows 10/11, macOS, or Linux
*   **Hardware:**
    *   **CPU:** Modern multi-core processor (e.g., Intel i5/Ryzen 5 or equivalent).
    *   **RAM:** 8GB or more recommended.
    *   **GPU (Optional but Recommended for faster training):** NVIDIA GPU with CUDA support for accelerated training with PyTorch. If no CUDA-compatible GPU is available, training will default to CPU.

## Python Environment Setup (Anaconda - Recommended)

We recommend using Anaconda to manage the Python environment, as it ensures compatibility with the pre-trained models included in this project.

### Prerequisites

1. **Install Anaconda or Miniconda:**
   - Download from [anaconda.com](https://www.anaconda.com/download) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - Follow the installation instructions for your operating system

### Quick Setup (Recommended)

1. **Clone or download the repository:**
   ```bash
   git clone https://github.com/Ibonarambarri/infection-rl.git
   cd infection-rl
   ```

2. **Create the Anaconda environment with Python 3.11:**
   ```bash
   conda create -n infection-rl python=3.11 -y
   ```

3. **Activate the environment:**
   ```bash
   conda activate infection-rl
   ```

4. **Install PyTorch (required before other dependencies):**

   **For CPU only:**
   ```bash
   pip install torch==2.9.1
   ```

   **For NVIDIA GPU (CUDA 11.8):**
   ```bash
   pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu118
   ```

   **For NVIDIA GPU (CUDA 12.1):**
   ```bash
   pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Verify installation:**
   ```bash
   python -c "import torch; import stable_baselines3; print('Installation successful!')"
   ```

### Version Requirements

The pre-trained models included in this project were trained with specific library versions. Using different versions may cause compatibility issues (segmentation faults, errors loading models).

| Library | Required Version |
|---------|-----------------|
| Python | 3.11.x |
| PyTorch | 2.9.1 |
| Stable-Baselines3 | 2.7.1 |
| Gymnasium | 1.2.3 |
| NumPy | 2.4.0 |

### Troubleshooting Environment Issues

If you encounter issues with model loading (e.g., segmentation fault), verify your versions:

```bash
python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
import stable_baselines3
print(f'SB3: {stable_baselines3.__version__}')
import numpy
print(f'NumPy: {numpy.__version__}')
"
```

If versions don't match, recreate the environment:
```bash
conda deactivate
conda remove -n infection-rl --all -y
# Then follow the Quick Setup steps again
```

## Running the Project

Ensure your environment is activated before running any commands.

### Playing a Game (Visualization)

Watch trained agents play against each other:

```bash
# Level 3 (40x40 map) with pre-trained models
python scripts/play.py -l 3 -m models/final

# Level 1 (20x20 map) - easier to visualize
python scripts/play.py -l 1 -m models/final

# Adjust speed with --fps
python scripts/play.py -l 3 -m models/final --fps 30
```

**Controls during visualization:**
- `SPACE`: Pause/Resume
- `V`: Toggle infected vision overlay
- `S`: Step-by-step (when paused)
- `Q`: Quit
- `1-5`: Change speed

### Training an Agent

To start the training process:

```bash
# Basic training, output to 'models/pingpong' by default
python scripts/train.py

# Specify output directory
python scripts/train.py --output-dir models/my_run

# Start training from a specific phase (e.g., Phase 2)
python scripts/train.py --output-dir models/my_run --start-phase 2

# Render evaluation episodes during training
python scripts/train.py --output-dir models/my_run --render
```

Training logs and model checkpoints will be saved in the specified `--output-dir`. TensorBoard logs will be generated in `tensorboard_logs/`.

### Monitoring Training with TensorBoard

```bash
tensorboard --logdir tensorboard_logs
```

Then open http://localhost:6006 in your browser.

### Evaluating Trained Models

To evaluate individual models or compare multiple models:

```bash
# Evaluate a single healthy agent model
python scripts/evaluate.py --model path/to/healthy_final.zip --role healthy --episodes 50

# Evaluate a single infected agent model
python scripts/evaluate.py --model path/to/infected_final.zip --role infected --episodes 50

# Compare multiple models
python scripts/evaluate.py --compare models/run1/healthy_final.zip models/run2/healthy_final.zip --episodes 100
```
