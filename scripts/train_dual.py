#!/usr/bin/env python3
"""
Entrenamiento Dual - Healthy vs Infected
========================================
Entrena ambos roles alternadamente para que aprendan uno contra el otro.

Uso:
    python scripts/train_dual.py --timesteps 500000
    python scripts/train_dual.py --timesteps 1000000 --switch-freq 20000
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from src.envs.wrappers import make_infection_env


class StatsCallback(BaseCallback):
    """Callback para recoger estadísticas."""
    def __init__(self):
        super().__init__(verbose=0)
        self.rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
        return True

    def mean_reward(self):
        if not self.rewards:
            return 0.0
        return np.mean(self.rewards[-100:])


def create_env(role: str, n_envs: int = 4, seed: int = None):
    """Crea entorno para un rol específico."""
    def make_env():
        return make_infection_env(
            flatten=True,
            force_role=role,
            seed=seed,
        )

    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecMonitor(env)
    return env


def train_dual(
    timesteps: int = 500000,
    checkpoint_freq: int = 50000,
    switch_freq: int = 10000,
    n_envs: int = 4,
    seed: int = 42,
    save_dir: str = None,
):
    """Entrena ambos roles alternadamente."""

    # Directorio de guardado
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(f"models/dual_{timestamp}")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)

    print("=" * 60)
    print("  DUAL TRAINING - HEALTHY vs INFECTED")
    print("=" * 60)
    print(f"  Save dir: {save_dir}")
    print(f"  Timesteps per role: {timesteps:,}")
    print(f"  Switch frequency: {switch_freq:,}")
    print("=" * 60)

    # Crear entornos
    print("\n>>> Creating environments...")
    env_healthy = create_env("healthy", n_envs, seed)
    env_infected = create_env("infected", n_envs, seed)
    print(f"  Obs space: {env_healthy.observation_space.shape}")

    # Crear modelos
    print("\n>>> Creating models...")
    model_healthy = PPO("MlpPolicy", env_healthy, verbose=0, seed=seed)
    model_infected = PPO("MlpPolicy", env_infected, verbose=0, seed=seed)

    cb_healthy = StatsCallback()
    cb_infected = StatsCallback()

    # Entrenamiento alternado
    print("\n>>> Starting dual training...")
    steps_healthy = 0
    steps_infected = 0
    checkpoint_num = 0

    while steps_healthy < timesteps or steps_infected < timesteps:
        # Entrenar healthy
        if steps_healthy < timesteps:
            model_healthy.learn(total_timesteps=switch_freq,
                                callback=cb_healthy, reset_num_timesteps=False)
            steps_healthy += switch_freq

        # Entrenar infected
        if steps_infected < timesteps:
            model_infected.learn(total_timesteps=switch_freq,
                                 callback=cb_infected, reset_num_timesteps=False)
            steps_infected += switch_freq

        # Progreso
        progress = min(steps_healthy, steps_infected) / timesteps * 100
        print(f"\r  Progress: {progress:.1f}% | "
              f"Healthy: {steps_healthy:,} (r={cb_healthy.mean_reward():.1f}) | "
              f"Infected: {steps_infected:,} (r={cb_infected.mean_reward():.1f})", end="")

        # Checkpoint
        total_steps = steps_healthy + steps_infected
        if total_steps // checkpoint_freq > checkpoint_num:
            checkpoint_num = total_steps // checkpoint_freq
            model_healthy.save(str(save_dir / "checkpoints" / f"healthy_{checkpoint_num}"))
            model_infected.save(str(save_dir / "checkpoints" / f"infected_{checkpoint_num}"))
            print(f"\n  >>> Checkpoint {checkpoint_num} saved")

    print()

    # Guardar modelos finales
    print("\n>>> Saving final models...")
    model_healthy.save(str(save_dir / "healthy_final"))
    model_infected.save(str(save_dir / "infected_final"))

    # Evaluación final
    print("\n>>> Final evaluation...")
    mean_h, std_h = evaluate_policy(model_healthy, env_healthy, n_eval_episodes=20)
    mean_i, std_i = evaluate_policy(model_infected, env_infected, n_eval_episodes=20)
    print(f"  Healthy:  {mean_h:.2f} +/- {std_h:.2f}")
    print(f"  Infected: {mean_i:.2f} +/- {std_i:.2f}")

    env_healthy.close()
    env_infected.close()

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETED")
    print("=" * 60)
    print(f"  Models saved to: {save_dir}")
    print(f"  - {save_dir}/healthy_final.zip")
    print(f"  - {save_dir}/infected_final.zip")
    print()
    print(f"  To play: python scripts/play.py --models-dir {save_dir}")
    print()

    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Dual training - Healthy vs Infected")

    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Timesteps per role (default: 500000)")
    parser.add_argument("--checkpoint-freq", type=int, default=50000,
                        help="Checkpoint frequency (default: 50000)")
    parser.add_argument("--switch-freq", type=int, default=10000,
                        help="Switch between roles frequency (default: 10000)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Parallel environments (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save directory")

    args = parser.parse_args()

    train_dual(
        timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
        switch_freq=args.switch_freq,
        n_envs=args.n_envs,
        seed=args.seed,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
