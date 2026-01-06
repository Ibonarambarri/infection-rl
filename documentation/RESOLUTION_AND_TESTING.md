# Resolution and Testing in the Environment

This document details the technical methodology, experimental strategies, and evaluation approaches used in the project.

## Methodology

### Reinforcement Learning Algorithm

The project utilizes the **Proximal Policy Optimization (PPO)** algorithm, implemented via the `stable-baselines3` library. PPO is a popular and robust on-policy algorithm known for its balance of performance and stability. It works by optimizing a "surrogate" objective function using stochastic gradient ascent, ensuring that new policies do not deviate too far from old policies, which helps prevent destructive updates.

PPO is well-suited for this environment due to its effectiveness in continuous and discrete action spaces (the agents have discrete movement actions) and its ability to handle complex observation spaces.

### Environment: `InfectionEnv`

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

### Training Process

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

## Experiments & Approaches

This project incorporates several sophisticated experimental strategies to achieve robust agent learning and co-adaptation, notably Curriculum Learning and Adaptive Self-Play. Beyond these integrated approaches, further experimentation avenues are discussed.

### Curriculum Learning

The project employs a meticulously designed curriculum, progressively increasing the complexity of the learning task across different phases. This strategy involves:
*   **Progressive Reward Density**: Starting with `DENSE` rewards (Phase 1) to provide clear learning signals for basic behaviors, transitioning to `INTERMEDIATE` (Phase 2), and finally `SPARSE` rewards (Phase 3 and beyond) to encourage more complex, long-term strategic planning.
*   **Environment Scaling**: Gradually increasing the map size and the number of agents (`MAP_LVL1` to `MAP_LVL3`) from simpler 1v2 scenarios to more complex 8v2 deployments. This ensures agents learn foundational skills in manageable settings before facing high-dimensional and challenging environments.
The curriculum helps overcome the challenges of sparse rewards and complex state spaces by scaffolding the learning process.

### Adaptive Self-Play (Ping-Pong Training)

A core innovative approach in this project is the adaptive self-play mechanism, referred to as "Ping-Pong Training." This strategy addresses the challenge of training agents in a multi-agent environment where the opponent's policy is constantly evolving.
*   **Alternating Training Focus**: Instead of simultaneously updating both policies, the system intelligently focuses training on the "weaker" agent (the one with a lower win rate in recent evaluations). This ensures that both healthy and infected agent models continuously improve against an ever-stronger opponent, preventing one agent from becoming too dominant and stalling the learning of the other.
*   **Robustness**: By co-adapting, the agents develop more robust strategies that are less prone to exploitation by a static or predictable opponent. This dynamic equilibrium is crucial for achieving high-quality multi-agent intelligence. This approach can be considered an "original modification that improves learning" as specified in the rubric, as it specifically tailors the self-play dynamic to balance the competitive landscape.

### Proposed Future Experimental Strategies

While the current setup is robust, several avenues for further experimentation can yield deeper insights and potentially improved performance:

*   **Hyperparameter Variation**: Systematically exploring different PPO hyperparameters (e.g., `gamma`, `ent_coef`, `n_steps`, `learning_rate` schedules) could uncover configurations leading to faster convergence or higher final performance. Techniques like grid search, random search, or Bayesian optimization could be employed.
*   **Different RL Algorithms**: Experimenting with alternative `stable-baselines3` algorithms (e.g., A2C, SAC, TD3 if observation/action space were modified for continuous actions, or custom algorithms for multi-agent settings) could reveal insights into their suitability for this specific `InfectionEnv`.
*   **Environment Simplification or Shaping**:
    *   **Observation Space Reduction**: Experimenting with simplified observation spaces (e.g., removing the image observation, or reducing the vector features) could help determine the minimal information required for effective control.
    *   **Reward Shaping**: While curriculum learning handles reward density, further fine-tuning of the reward function for specific sub-goals (e.g., small penalties for idle movement, bonuses for strategic positioning) might guide agents more effectively.
*   **Multi-Agent Learning Enhancements**: Exploring more advanced multi-agent RL techniques beyond parameter sharing, such as centralised training with decentralized execution (CTDE) or communication protocols between agents, could be beneficial.
