# Installation & Setup Guide

This section provides a comprehensive guide to setting up and running the Reinforcement Learning project.

## System Requirements

*   **Operating System:** Windows 10/11 (or compatible Linux distribution)
*   **Hardware:**
    *   **CPU:** Modern multi-core processor (e.g., Intel i5/Ryzen 5 or equivalent).
    *   **RAM:** 8GB or more recommended.
    *   **GPU (Optional but Recommended for faster training):** NVIDIA GPU with CUDA support for accelerated training with PyTorch (which Stable-Baselines3 uses). If no CUDA-compatible GPU is available, training will default to CPU.

## Python Environment Setup

It is highly recommended to use a virtual environment to manage project dependencies.

1.  **Install Python:** Ensure Python 3.8 or newer is installed. You can download it from [python.org](https://www.python.org/).

2.  **Create a Virtual Environment:**
    Navigate to the project's root directory (`infection-rl/`) in your terminal and create a virtual environment:
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **Linux/macOS:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    With the virtual environment activated, install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

The project provides scripts for training and evaluation. Ensure your virtual environment is activated before running any commands.

### Training an Agent

To start the training process, use the `train.py` script. You can specify an output directory for models and logs, and even start from a particular phase or render evaluations.

```bash
# Basic training, output to 'models/pingpong' by default
python scripts/train.py

# Specify output directory
python scripts/train.py --output-dir my_training_run_1

# Start training from a specific phase (e.g., Phase 2)
python scripts/train.py --output-dir my_training_run_2 --start-phase 2

# Render evaluation episodes during training
python scripts/train.py --output-dir my_training_run_3 --render
```

Training logs and model checkpoints will be saved in the specified `--output-dir`. TensorBoard logs will be generated in `tensorboard_logs/`.

### Evaluating Trained Models

To evaluate individual models or compare multiple models, use the `evaluate.py` script.

```bash
# Evaluate a single healthy agent model
python scripts/evaluate.py --model path/to/your/healthy_final.zip --role healthy --episodes 50

# Evaluate a single infected agent model
python scripts/evaluate/py --model path/to/your/infected_final.zip --role infected --episodes 50

# Compare multiple models
python scripts/evaluate.py --compare models/run1/healthy_final.zip models/run2/healthy_final.zip --episodes 100 --output-dir comparisons/healthy_models

# Compare multiple models with a fixed environment seed for reproducibility
python scripts/evaluate.py --compare models/run1/healthy_final.zip models/run2/healthy_final.zip --episodes 100 --output-dir comparisons/healthy_models --env-seed 123
```

Evaluation results, including comparative plots, will be saved in the `--output-dir`.

## Troubleshooting

*   **Dependency Conflicts:** If you encounter issues during `pip install -r requirements.txt`, try creating a fresh virtual environment. If problems persist, consider using `pip check` to identify conflicting packages, or manually install dependencies one by one.
*   **CUDA / CPU Issues:**
    *   If training is slow and you have a compatible NVIDIA GPU, ensure your PyTorch installation (a dependency of `stable-baselines3`) is linked to CUDA. Check the `stable-baselines3` documentation for specific installation instructions for GPU support.
    *   If you don't have a GPU or encounter CUDA errors, `stable-baselines3` will default to CPU. Performance will be slower.
*   **Gymnasium / Environment Rendering Problems:**
    *   Ensure `pygame` is correctly installed (`pip install pygame`).
    *   If rendering (`--render` flag) causes issues, it might be due to display server configuration (especially on headless servers). Try running without the `--render` flag.
*   **TensorBoard not showing logs:**
    *   Make sure you are running TensorBoard from the directory *containing* the `tensorboard_logs` folder (e.g., from the `infection-rl` root directory).
    *   Check if the `tensorboard_logs` directory contains event files (e.g., `events.out.tfevents...`).
    *   Ensure no other TensorBoard instance is running on the same port. Try a different port: `tensorboard --logdir tensorboard_logs --port 6007`.
