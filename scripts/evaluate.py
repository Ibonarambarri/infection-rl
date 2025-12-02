#!/usr/bin/env python3
"""
Evaluación de Modelos
====================
Evalúa y compara modelos entrenados.

Uso:
    python scripts/evaluate.py --model models/ppo_healthy/best_model.zip --episodes 100
    python scripts/evaluate.py --compare model1.zip model2.zip model3.zip
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy

from src.envs.wrappers import make_infection_env


def load_model(model_path: str, env=None):
    """Carga un modelo guardado."""
    model_path = Path(model_path)
    path_str = str(model_path).replace(".zip", "")

    for algo_class in [PPO, A2C, DQN]:
        try:
            return algo_class.load(path_str, env=env), algo_class.__name__
        except Exception:
            continue

    raise ValueError(f"Could not load model from {model_path}")


def evaluate_agent(model, env, n_episodes: int = 100, verbose: bool = True):
    """Evalúa un agente y retorna métricas."""
    rewards = []
    lengths = []
    healthy_at_end = []
    infections = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        rewards.append(episode_reward)
        lengths.append(episode_length)

        if hasattr(env.unwrapped, "agents"):
            healthy_at_end.append(env.unwrapped.num_healthy)
            infections.append(len(env.unwrapped.infection_events))

        if verbose and (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, Length={episode_length}")

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_length": np.mean(lengths),
        "survival_rate": np.mean([1 if h > 0 else 0 for h in healthy_at_end]) if healthy_at_end else 0,
        "mean_healthy": np.mean(healthy_at_end) if healthy_at_end else 0,
        "mean_infections": np.mean(infections) if infections else 0,
        "rewards": rewards,
        "lengths": lengths,
    }


def compare_models(model_paths: list, n_episodes: int = 50, output_dir: str = None):
    """Compara múltiples modelos."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"evaluations/comparison_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_infection_env()

    results = {}
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)

    for model_path in model_paths:
        model_name = Path(model_path).stem
        print(f"\n>>> Evaluating: {model_name}")

        model, algo = load_model(model_path, env)
        stats = evaluate_agent(model, env, n_episodes, verbose=False)
        results[model_name] = stats

    # Imprimir tabla
    print("\n" + "-" * 60)
    print(f"{'Model':<25} {'Reward':<15} {'Survival':<15}")
    print("-" * 60)

    for name, stats in results.items():
        print(f"{name:<25} {stats['mean_reward']:<15.2f} {stats['survival_rate']*100:<14.1f}%")

    # Guardar gráfica
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = list(results.keys())
    mean_rewards = [results[n]["mean_reward"] for n in names]
    std_rewards = [results[n]["std_reward"] for n in names]
    survival_rates = [results[n]["survival_rate"] * 100 for n in names]

    axes[0].bar(names, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward Comparison")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(names, survival_rates, alpha=0.7, color="green")
    axes[1].set_ylabel("Survival Rate (%)")
    axes[1].set_title("Survival Rate Comparison")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Guardar JSON
    json_results = {k: {kk: vv for kk, vv in v.items() if kk not in ["rewards", "lengths"]}
                    for k, v in results.items()}
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n>>> Results saved to: {output_dir}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")

    parser.add_argument("--model", type=str, help="Path to model")
    parser.add_argument("--compare", type=str, nargs="+", help="Models to compare")
    parser.add_argument("--episodes", type=int, default=100, help="Eval episodes (default: 100)")
    parser.add_argument("--role", type=str, default="healthy",
                        choices=["healthy", "infected"], help="Role for single model eval")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    if args.compare:
        compare_models(args.compare, args.episodes, args.output_dir)

    elif args.model:
        print("\n" + "=" * 60)
        print("  AGENT EVALUATION")
        print("=" * 60)

        env = make_infection_env(force_role=args.role)

        print(f"\n>>> Loading model: {args.model}")
        model, algo = load_model(args.model, env)
        print(f"  Algorithm: {algo}")

        print(f"\n>>> Evaluating ({args.episodes} episodes)...")
        stats = evaluate_agent(model, env, args.episodes)

        print("\n" + "-" * 40)
        print("  RESULTS")
        print("-" * 40)
        print(f"  Mean Reward:   {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
        print(f"  Min/Max:       {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        print(f"  Mean Length:   {stats['mean_length']:.1f}")
        print(f"  Survival Rate: {stats['survival_rate']*100:.1f}%")
        print(f"  Mean Healthy:  {stats['mean_healthy']:.2f}")

        env.close()

    else:
        print("Use --model or --compare")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
