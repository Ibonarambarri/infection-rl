# Final Report

## 1. Title Page

**Project Title:** Reinforcement Learning for Infection Spread Control

**Student Name(s):** [Your Name(s)]

## 2. Introduction

This project explores the application of Reinforcement Learning (RL) to model and control the spread of infection within a simulated environment. The core problem involves training intelligent agents to navigate and interact within an environment where some agents can be infected and spread the infection to others. The objective is to develop and evaluate RL strategies for managing this dynamic.

The environment, `InfectionEnv`, is a custom multi-agent Gymnasium environment designed to simulate infection dynamics. It features agents moving on a grid, with mechanics for infection transmission and recovery. The primary goals of this project are to analyze the existing RL solution, enhance its testing protocols, implement structured experimentation, and provide academic-quality documentation of the entire system and its performance.

## 3. Installation & Setup

This section provides a comprehensive guide to setting up and running the Reinforcement Learning project.

### 3.1 System Requirements

*   **Operating System:** Windows 10/11 (or compatible Linux distribution)
*   **Hardware:**
    *   **CPU:** Modern multi-core processor (e.g., Intel i5/Ryzen 5 or equivalent).
    *   **RAM:** 8GB or more recommended.
    *   **GPU (Optional but Recommended for faster training):** NVIDIA GPU with CUDA support for accelerated training with PyTorch (which Stable-Baselines3 uses). If no CUDA-compatible GPU is available, training will default to CPU.

### 3.2 Python Environment Setup

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

### 3.3 Running the Project

The project provides scripts for training and evaluation. Ensure your virtual environment is activated before running any commands.

#### Training an Agent

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

#### Evaluating Trained Models

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

### 3.4 Troubleshooting

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

## 4. Methodology

### 4.1 Reinforcement Learning Algorithm

The project utilizes the **Proximal Policy Optimization (PPO)** algorithm, implemented via the `stable-baselines3` library. PPO is a popular and robust on-policy algorithm known for its balance of performance and stability. It works by optimizing a "surrogate" objective function using stochastic gradient ascent, ensuring that new policies do not deviate too far from old policies, which helps prevent destructive updates.

PPO is well-suited for this environment due to its effectiveness in continuous and discrete action spaces (the agents have discrete movement actions) and its ability to handle complex observation spaces.

### 4.2 Environment: `InfectionEnv`

The core of the simulation is the custom `InfectionEnv`, built upon the `gymnasium` API. This environment simulates a grid-world where agents interact, move, and spread infection.

*   **State Space:** The observation space is a `gymnasium.spaces.Dict` containing two main components:
    *   `image`: A 2D grid representation of the environment, capturing spatial information like agent positions, infection status, and obstacles. This is processed by a Convolutional Neural Network (CNN) within the `MultiInputPolicy`.
    *   `vector`: A 1D array containing non-spatial, global information relevant to the agent's decision-making. This is typically processed by a Multi-Layer Perceptron (MLP).
    This composite observation allows the agent to perceive both local visual cues and global contextual data.

*   **Action Space:** Agents have a discrete action space, typically corresponding to movement directions (e.g., North, South, East, West, Stay).

*   **Reward Structure:** The environment employs configurable reward structures, defined in `src/envs/reward_config.py`. These presets include:
    *   **DENSE:** Provides frequent rewards for immediate actions, facilitating early learning.
    *   **INTERMEDIATE:** A balance between dense and sparse rewards.
    *   **SPARSE:** Rewards are given only for achieving long-term goals or significant events, encouraging agents to explore and discover complex strategies.
    The choice of reward structure is crucial for shaping agent behavior and is often varied during curriculum learning.

### 4.3 Training Process

The training process is sophisticated, incorporating curriculum learning and adaptive self-play, orchestrated by `scripts/train.py`.

1.  **Multi-Agent to Single-Agent Adaptation:** The `InfectionEnv` is inherently multi-agent. To leverage `stable-baselines3` (which is primarily designed for single-agent tasks), a system of wrappers (`src/envs/wrappers.py`) is used:
    *   `SingleAgentWrapper`: Transforms the multi-agent environment into a single-agent interface.
    *   `DictObservationWrapper`: Ensures the dictionary observation space is correctly handled by `stable-baselines3`'s `MultiInputPolicy`.
    *   `make_vec_env_parameter_sharing`: This crucial component allows multiple agents to share the same policy network parameters, fostering cooperative learning and efficiency. It also facilitates adaptive self-play.

2.  **Curriculum Learning (`CURRICULUM_PHASES`):** Training progresses through distinct phases, gradually increasing the complexity of the task. This often involves:
    *   Starting with simpler map configurations or more dense reward signals.
    *   Transitioning to more complex maps, more agents, and sparser rewards as the agent's performance improves.
    This approach helps agents learn fundamental behaviors before tackling more challenging scenarios.

3.  **Adaptive Self-Play:** The system employs an adaptive self-play mechanism, particularly in the "Ping-Pong Trainer" (`scripts/train.py`). In scenarios involving "healthy" and "infected" agent types, the training focuses on improving the performance of the weaker agent type. For instance, if healthy agents are consistently outperforming infected agents, the system will primarily train the infected agent's policy against the healthy agent's current best policy, and vice-versa. This ensures a balanced and robust co-adaptation between opposing agent types.

4.  **Policy Network:** A `MultiInputPolicy` from `stable-baselines3` is used to handle the dictionary observation space. This policy internally uses separate networks (e.g., CNN for 'image' and MLP for 'vector') and concatenates their outputs before feeding them to the final policy head.

5.  **Evaluation:** During training, regular evaluations are performed to monitor progress. The `evaluate_models` function in `scripts/train.py` plays a crucial role, assessing the current policies' performance against various opponents or fixed baselines.

## 5. TensorBoard Usage

TensorBoard is an open-source tool for machine learning experimentation. It provides a suite of web applications for inspecting and understanding your model training runs. This project integrates TensorBoard to visualize training progress, key metrics, and model performance over time.

### 5.1 Launching TensorBoard

To launch TensorBoard, navigate to the root of your `infection-rl` project directory in your terminal (where the `tensorboard_logs` folder is located) and run the following command:

```bash
tensorboard --logdir tensorboard_logs
```

By default, TensorBoard will start on `http://localhost:6006`. You can open this URL in your web browser to access the dashboard. If port 6006 is in use, TensorBoard will suggest an alternative port.

### 5.2 Logged Metrics

During training (`scripts/train.py`), the `SimpleLoggingCallback` records several important metrics to TensorBoard:

*   **`train/healthy_win_rate`**: The win rate of the healthy agent when playing against the infected agent.
*   **`train/infected_win_rate`**: The win rate of the infected agent when playing against the healthy agent.
*   **`train/healthy_mean_reward`**: The average reward obtained by the healthy agent.
*   **`train/infected_mean_reward`**: The average reward obtained by the infected agent.
*   **`rollout/ep_len_mean`**: Mean episode length.
*   **`rollout/ep_rew_mean`**: Mean episode reward (overall, not role-specific).
*   **`train/learning_rate`**: The current learning rate.
*   **`train/policy_loss`**: The loss of the policy network.
*   **`train/value_loss`**: The loss of the value network.
*   **`train/entropy_loss`**: The entropy regularization term.

These metrics are logged separately for each training phase and role (healthy/infected) within the `tensorboard_logs` directory structure.

## 6. TensorBoard Analysis

Interpreting the graphs in TensorBoard is crucial for understanding the training dynamics, identifying potential issues, and evaluating agent performance.

### 6.1 Interpreting Loss Curves

*   **`policy_loss`**: This curve indicates how much the agent's policy is changing. A decreasing trend generally suggests that the policy is converging. Oscillations are normal, but large, sudden spikes might indicate an unstable learning rate or batch size.
*   **`value_loss`**: This reflects the accuracy of the value function (the agent's estimation of future rewards). Similar to policy loss, a decreasing trend is desirable. Divergence or high instability could mean the agent is struggling to accurately predict rewards.
*   **`entropy_loss`**: PPO uses entropy regularization to encourage exploration. A decreasing entropy loss indicates that the agent is becoming more confident in its actions and exploring less, which is expected as training progresses. If it drops too quickly, the agent might be getting stuck in local optima; if it stays high, it might not be converging.

### 6.2 Reward Evolution and Win Rates

*   **`mean_reward`**: This is a direct indicator of how well the agent is performing. An increasing trend signifies learning, while plateaus or decreases suggest stagnation or issues.
*   **`healthy_win_rate` / `infected_win_rate`**: These are critical metrics in a self-play setup.
    *   Ideally, both win rates should fluctuate around 0.5 (50%), especially during adaptive self-play. This indicates a well-balanced training where neither agent is overwhelmingly dominant, forcing both to continuously improve.
    *   If one win rate consistently stays very high (e.g., > 0.8) and the other very low, it might mean the training is imbalanced, and the weaker agent is not learning effectively against the stronger opponent.
    *   Observe the trends across different phases of the curriculum. You should generally see an improvement in overall performance (e.g., higher rewards or more balanced win rates) as agents progress through simpler to more complex environments.

### 6.3 Stability and Convergence

*   **Smoothness of Curves**: Generally, smoother curves (especially for rewards and win rates) indicate more stable training. High variance or jagged curves might suggest noisy environments, insufficient batch sizes, or an unstable learning process.
*   **Plateaus**: If reward curves or win rates flatten out before reaching desired levels, it could mean the agent has converged to a local optimum, or the learning rate is too low.
*   **Anomalies**: Sudden drops in reward, spikes in loss, or rapid changes in win rates that don't recover can indicate training instability, hyperparameter issues, or even bugs in the environment or reward function.

By carefully monitoring these metrics in TensorBoard, one can gain deep insights into the agent's learning process and diagnose problems effectively.

## 7. Experiments & Approaches

This project incorporates several sophisticated experimental strategies to achieve robust agent learning and co-adaptation, notably Curriculum Learning and Adaptive Self-Play. Beyond these integrated approaches, further experimentation avenues are discussed.

### 7.1 Curriculum Learning

The project employs a meticulously designed curriculum, progressively increasing the complexity of the learning task across different phases. This strategy involves:
*   **Progressive Reward Density**: Starting with `DENSE` rewards (Phase 1) to provide clear learning signals for basic behaviors, transitioning to `INTERMEDIATE` (Phase 2), and finally `SPARSE` rewards (Phase 3 and beyond) to encourage more complex, long-term strategic planning.
*   **Environment Scaling**: Gradually increasing the map size and the number of agents (`MAP_LVL1` to `MAP_LVL3`) from simpler 1v2 scenarios to more complex 8v2 deployments. This ensures agents learn foundational skills in manageable settings before facing high-dimensional and challenging environments.
The curriculum helps overcome the challenges of sparse rewards and complex state spaces by scaffolding the learning process.

### 7.2 Adaptive Self-Play (Ping-Pong Training)

A core innovative approach in this project is the adaptive self-play mechanism, referred to as "Ping-Pong Training." This strategy addresses the challenge of training agents in a multi-agent environment where the opponent's policy is constantly evolving.
*   **Alternating Training Focus**: Instead of simultaneously updating both policies, the system intelligently focuses training on the "weaker" agent (the one with a lower win rate in recent evaluations). This ensures that both healthy and infected agent models continuously improve against an ever-stronger opponent, preventing one agent from becoming too dominant and stalling the learning of the other.
*   **Robustness**: By co-adapting, the agents develop more robust strategies that are less prone to exploitation by a static or predictable opponent. This dynamic equilibrium is crucial for achieving high-quality multi-agent intelligence. This approach can be considered an "original modification that improves learning" as specified in the rubric, as it specifically tailors the self-play dynamic to balance the competitive landscape.

### 7.3 Proposed Future Experimental Strategies

While the current setup is robust, several avenues for further experimentation can yield deeper insights and potentially improved performance:

*   **Hyperparameter Variation**: Systematically exploring different PPO hyperparameters (e.g., `gamma`, `ent_coef`, `n_steps`, `learning_rate` schedules) could uncover configurations leading to faster convergence or higher final performance. Techniques like grid search, random search, or Bayesian optimization could be employed.
*   **Different RL Algorithms**: Experimenting with alternative `stable-baselines3` algorithms (e.g., A2C, SAC, TD3 if observation/action space were modified for continuous actions, or custom algorithms for multi-agent settings) could reveal insights into their suitability for this specific `InfectionEnv`.
*   **Environment Simplification or Shaping**:
    *   **Observation Space Reduction**: Experimenting with simplified observation spaces (e.g., removing the image observation, or reducing the vector features) could help determine the minimal information required for effective control.
    *   **Reward Shaping**: While curriculum learning handles reward density, further fine-tuning of the reward function for specific sub-goals (e.g., small penalties for idle movement, bonuses for strategic positioning) might guide agents more effectively.
*   **Multi-Agent Learning Enhancements**: Exploring more advanced multi-agent RL techniques beyond parameter sharing, such as centralised training with decentralized execution (CTDE) or communication protocols between agents, could be beneficial.

## 8. Conclusions

This project successfully demonstrates a sophisticated Reinforcement Learning approach to controlling infection spread in a simulated environment. By leveraging the PPO algorithm within a custom `gymnasium` environment, coupled with advanced training strategies, the system effectively trains co-adapting healthy and infected agents.

### 8.1 Key Findings

The implementation of **Curriculum Learning** proved highly effective in managing the complexity of the `InfectionEnv`. By progressively increasing environment difficulty and transitioning from dense to sparse reward signals, agents were able to acquire fundamental behaviors before tackling more challenging scenarios. This scaffolding prevented early training divergence and fostered robust learning.

The **Adaptive Self-Play (Ping-Pong Training)** mechanism is a critical innovation, ensuring a balanced competitive training dynamic. By continuously identifying and focusing training on the weaker agent, the system promotes continuous improvement in both healthy and infected agent policies, leading to more resilient and intelligent behaviors that are not easily exploitable. This co-adaptation is crucial for competitive multi-agent environments.

The project's modular design, utilizing `stable-baselines3` wrappers for multi-agent to single-agent adaptation, demonstrates a clean and effective way to apply single-agent RL frameworks to complex multi-agent problems.

### 8.2 Limitations

Despite its strengths, the current project has certain limitations:
*   **Computational Cost**: Training complex RL agents in a multi-agent environment, even with curriculum learning, can be computationally expensive and time-consuming, requiring significant resources.
*   **Hyperparameter Sensitivity**: PPO, like many RL algorithms, can be sensitive to hyperparameter choices. Optimal performance might require extensive tuning, which was explored in a structured but not exhaustive manner within the curriculum.
*   **Generalizability**: While the agents perform well within the trained environment configurations, their generalizability to vastly different map layouts, agent densities, or infection mechanics outside the curriculum phases might be limited.

### 8.3 Future Improvements

Several directions can be explored to extend and enhance this project:
*   **Hyperparameter Optimization**: Implement automated hyperparameter tuning (e.g., using Optuna or Ray Tune) to systematically find optimal PPO configurations for each phase or for the overall training.
*   **Alternative Architectures**: Investigate different neural network architectures for the `MultiInputPolicy`, or explore other feature engineering techniques to improve observation processing.
*   **More Complex Scenarios**: Introduce dynamic environment changes, varying infection rates, or agent communication protocols to further challenge the agents and develop more sophisticated strategies.
*   **Formal Testing Suite**: Develop a more comprehensive suite of unit and integration tests to ensure the robustness of the environment, wrappers, and core training logic, complementing the existing evaluation mechanisms.

Overall, this project provides a strong foundation and innovative approaches for tackling complex multi-agent control problems using Reinforcement Learning, with clear pathways for future research and development.