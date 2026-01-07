# Methodology & Algorithms

## 1. Overview

This document describes the methodology and algorithms used in the **Infection RL** project, a multi-agent reinforcement learning system designed to simulate and control infection spread dynamics. The project implements a competitive game between healthy agents (trying to survive) and infected agents (trying to infect all healthy agents).

## 2. Core Algorithm: Proximal Policy Optimization (PPO)

### 2.1 Why PPO?

PPO was selected as the primary algorithm for this project due to its:

- **Stability**: PPO uses a clipped surrogate objective that prevents destructively large policy updates
- **Sample Efficiency**: Better sample efficiency compared to vanilla policy gradient methods
- **Simplicity**: Easier to tune than algorithms like TRPO while maintaining similar performance
- **Proven Track Record**: Widely used in multi-agent and game-playing scenarios

### 2.2 PPO Algorithm

PPO optimizes a clipped surrogate objective function:

```
L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
```

Where:
- `r_t(θ)` is the probability ratio between new and old policies
- `A_t` is the advantage estimate at timestep t
- `ε` is the clipping parameter (typically 0.2)

The clipping mechanism ensures that the policy doesn't change too drastically in a single update, providing training stability.

### 2.3 Generalized Advantage Estimation (GAE)

GAE is used for computing advantage estimates, balancing bias and variance:

```
A_t^GAE = Σ (γλ)^l * δ_{t+l}
```

Where:
- `γ` is the discount factor
- `λ` is the GAE parameter
- `δ_t = r_t + γV(s_{t+1}) - V(s_t)` is the TD residual

High values of `γ` and `λ` are used to handle sparse reward scenarios effectively.

### 2.4 Learning Rate Schedule

A linear decay schedule transitions from exploration to exploitation:

```python
def linear_schedule(initial_lr, final_lr):
    def schedule(progress_remaining):
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule
```

This allows faster initial learning while ensuring stable convergence toward the end of training.

## 3. Neural Network Architecture

### 3.1 MultiInputPolicy

The policy uses a hybrid CNN+MLP architecture to process multi-modal observations:

```
Observation Space:
├── Image (H×W×C)        → NatureCNN → Feature Vector
├── Direction Vector     → MLP       → Feature Vector
├── State Vector         → MLP       → Feature Vector
├── Position Vector      → MLP       → Feature Vector
└── Nearby Agents Info   → MLP       → Feature Vector
                                       ───────────────
                                       Concatenated → Shared MLP → Policy Head
                                                                → Value Head
```

### 3.2 NatureCNN for Vision Processing

The image observation (circular view) is processed by a CNN based on the DQN Nature paper architecture:

```
NatureCNN:
    Conv2d(in, 32, kernel=8, stride=4)  → ReLU
    Conv2d(32, 64, kernel=4, stride=2)  → ReLU
    Conv2d(64, 64, kernel=3, stride=1)  → ReLU
    Flatten → Linear(*, features)
```

This architecture efficiently extracts spatial features from the agent's visual field.

### 3.3 Feature Encoding

- **Image Channels**: Cell type encoding and distance information
- **Direction**: Circular encoding using (cos, sin) for smooth transitions
- **Position**: Normalized coordinates for spatial awareness
- **Nearby Agents**: Relative positions and states of nearby agents

## 4. Curriculum Learning

### 4.1 Concept

Curriculum learning progressively increases task complexity, allowing the agent to:

1. **Build foundational skills** on simpler scenarios
2. **Transfer knowledge** to more complex situations
3. **Avoid local optima** that might trap agents in difficult scenarios

### 4.2 Progression Strategy

The curriculum progresses along multiple dimensions:

| Dimension | Simple → Complex |
|-----------|------------------|
| Agent Count | Few agents → Many agents |
| Map Size | Small grid → Large grid |
| Reward Density | Dense shaping → Sparse signals |
| Episode Length | Short episodes → Long episodes |

### 4.3 Reward Density Curriculum

Three tiers of reward density:

- **DENSE**: Rich per-step feedback with distance bonuses, survival rewards, and progress signals
- **INTERMEDIATE**: Reduced reward intensities to begin weaning off shaped rewards
- **SPARSE**: Only terminal rewards (victory/defeat), forcing long-term strategic thinking

This transition teaches agents to eventually operate without dense guidance.

## 5. Self-Play Training: Ping-Pong Method

### 5.1 Concept

The Ping-Pong method alternates training between competing agent types:

```
Round 1: Train Role A (Role B = Heuristic/Fixed)
Round 2: Train Role B (Role A = Round 1 model)
Round 3: Train Role A (Role B = Round 2 model)
Round 4: Train Role B (Role A = Round 3 model)
...
```

### 5.2 Benefits

- **Prevents Overfitting**: Agents don't memorize a single opponent's behavior
- **Co-evolution**: Both sides improve together through competitive pressure
- **Diverse Strategies**: Agents learn to handle various opponent tactics
- **Avoids Exploitation**: Neither side can exploit static opponent weaknesses

### 5.3 Adaptive Self-Play

An adaptive variant dynamically selects which role to train based on performance:

```python
def adaptive_training_decision(role_a_win_rate, role_b_win_rate):
    if balanced(role_a_win_rate, role_b_win_rate):
        return "train_both"
    elif role_a_win_rate < role_b_win_rate:
        return "train_role_a"  # Strengthen weaker side
    else:
        return "train_role_b"
```

This ensures competitive equilibrium and prevents one side from dominating.

## 6. Parameter Sharing

### 6.1 Multi-Agent to Single-Agent Conversion

Since PPO is a single-agent algorithm, parameter sharing converts the multi-agent problem:

```
Multiple agents of the same role share a single policy network
```

### 6.2 Implementation

Each parallel environment controls a different agent instance:

```
Environment 0: Controls Agent 0 (Role X)
Environment 1: Controls Agent 1 (Role X)
Environment 2: Controls Agent 2 (Role X)
...
All share the same policy network
```

### 6.3 Benefits

- **Efficiency**: Single policy network serves all agents of the same role
- **Generalization**: Policy learns position-invariant strategies
- **Scalability**: Adding more agents doesn't require additional policy networks
- **Sample Efficiency**: Experience from all agents contributes to learning

## 7. Reward Shaping

### 7.1 Design Principles

Effective reward shaping follows these principles:

1. **Alignment**: Shaped rewards should guide toward the true objective
2. **Density**: Provide feedback when terminal rewards are sparse
3. **Balance**: Avoid overshadowing terminal rewards
4. **Avoiding Pathologies**: Prevent reward hacking or unintended behaviors

### 7.2 Healthy Agent Rewards

| Reward Type | Purpose |
|-------------|---------|
| Survival Bonus | Encourage staying alive each step |
| Distance Bonus | Reward maintaining distance from infected |
| Infection Penalty | Strong negative for getting infected |
| Episode Survival | Bonus for surviving the full episode |
| Stuck Penalty | Discourage staying in the same position |

### 7.3 Infected Agent Rewards

| Reward Type | Purpose |
|-------------|---------|
| Infection Reward | Positive reward for each successful infection |
| Progress Bonus | Reward for reducing distance to targets |
| No Progress Penalty | Penalize not making progress toward targets |
| Victory Bonus | Additional reward for infecting all agents |

### 7.4 Progress-Based vs Proximity-Based Rewards

**Progress-based rewards** (rewarding distance *reduction*) are preferred over proximity-based rewards (rewarding being *close*) because:

- Proximity rewards can cause **orbiting behavior** where agents circle targets
- Progress rewards encourage **directed movement** toward objectives
- The gradient points toward the goal rather than around it

## 8. Environment Design

### 8.1 Observation Space

Each agent receives a multi-modal observation:

| Component | Description |
|-----------|-------------|
| **Image** | Local circular vision with cell types and distance encoding |
| **Direction** | Current facing direction (circular encoding) |
| **State** | Agent's infection status |
| **Position** | Normalized grid coordinates |
| **Nearby Agents** | Information about closest agents (position, state, distance) |

### 8.2 Action Space

Discrete action space:

| Action | Description |
|--------|-------------|
| Turn Left | Rotate 90° counterclockwise |
| Turn Right | Rotate 90° clockwise |
| Move Forward | Move 1 cell in facing direction |
| Stay Still | No movement |

### 8.3 Episode Dynamics

- **Infection Mechanism**: Infection spreads when an infected agent is within a defined radius of a healthy agent
- **Episode Termination**:
  - All healthy agents become infected → Infected team wins
  - Maximum steps reached → Healthy team wins

### 8.4 Grid World Features

- **Walls/Obstacles**: Block movement and line of sight
- **Wrapping**: Optional toroidal wrapping at map edges
- **Collision**: Agents cannot occupy the same cell as walls

## 9. Environment Wrappers

### 9.1 Wrapper Architecture

Gymnasium wrappers transform the base environment for compatibility:

```
InfectionEnv (base multi-agent)
    └── SingleAgentWrapper (single-agent interface)
        └── DictObservationWrapper (MultiInputPolicy format)
            └── VecEnv (parallelization)
```

### 9.2 SingleAgentWrapper

Converts multi-agent environment to single-agent interface:

- **Controlled Agent**: One specific agent receives actions from the policy
- **Other Agents**: Controlled by opponent model or heuristic behavior
- **Force Role**: Maintains agent's role throughout the episode

### 9.3 Opponent Behavior Options

When an agent role is not being trained:

1. **Trained Model**: Use a previously trained policy
2. **Heuristic**: Rule-based behavior (e.g., BFS pathfinding)
3. **Random**: Uniform random action selection

### 9.4 Observation Wrappers

- **DictObservationWrapper**: Formats observations for MultiInputPolicy (CNN + MLP branches)
- **FlattenObservationWrapper**: Alternative flattening for pure MLP policies

## 10. Distance Metrics

### 10.1 Manhattan Distance

Used for reward shaping due to computational efficiency:

```
d(a, b) = |x_a - x_b| + |y_a - y_b|
```

### 10.2 BFS Pathfinding

Used for heuristic opponents to find actual paths around obstacles:

```python
def bfs_distance(start, goal, obstacles):
    # Returns shortest path length considering walls
    # Returns infinity if no path exists
```

### 10.3 Trade-offs

| Metric | Pros | Cons |
|--------|------|------|
| Manhattan | Fast O(1), simple | Ignores obstacles |
| BFS | Accurate paths | Slower O(V+E), compute intensive |

Manhattan distance is preferred for per-step reward calculations, while BFS is used for heuristic agent navigation.

## 11. Key Methodological Contributions

1. **Curriculum-based Multi-Agent RL**: Progressive complexity increase with synchronized reward density transition

2. **Ping-Pong Self-Play**: Efficient alternating training paradigm for competitive multi-agent scenarios

3. **Adaptive Self-Play**: Dynamic role selection based on performance imbalance for balanced training

4. **Parameter Sharing**: Scalable approach to train multiple agents with single-agent algorithms

5. **Progress-based Reward Shaping**: Avoids common pathologies like orbiting behavior in pursuit tasks

6. **Multi-modal Observation Processing**: Hybrid CNN+MLP architecture for combined visual and vector inputs

## 12. References

- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"
- Schulman, J., et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning"
- Bengio, Y., et al. (2009). "Curriculum Learning"
- Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
- Gymnasium Documentation: https://gymnasium.farama.org/
