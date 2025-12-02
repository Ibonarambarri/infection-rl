#!/usr/bin/env python3
"""
Script de Entrenamiento
=======================
Entrena agentes (healthy o infected) usando PPO, A2C o DQN.

Uso:
    python scripts/train.py --role healthy --algo ppo --timesteps 500000
    python scripts/train.py --role infected --algo a2c --timesteps 300000
    python scripts/train.py --role healthy --algo dqn --timesteps 500000
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from src.envs.wrappers import make_infection_env


# Configuraciones por defecto para cada algoritmo
ALGO_CONFIGS = {
    "ppo": {
        "class": PPO,
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "a2c": {
        "class": A2C,
        "policy": "MlpPolicy",
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "dqn": {
        "class": DQN,
        "policy": "MlpPolicy",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 32,
        "gamma": 0.99,
        "target_update_interval": 500,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
}


def create_envs(role: str, n_envs: int = 4, seed: int = None):
    """Crea entornos de entrenamiento y evaluación."""
    def make_env(seed_offset=0):
        def _init():
            return make_infection_env(
                flatten=True,
                force_role=role,
                seed=seed + seed_offset if seed else None,
            )
        return _init

    train_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([make_env(1000)])
    eval_env = VecMonitor(eval_env)

    return train_env, eval_env


def train(
    role: str,
    algo: str,
    timesteps: int,
    n_envs: int = 4,
    seed: int = 42,
    save_dir: str = None,
    load_path: str = None,
    eval_freq: int = 10000,
):
    """Entrena un agente."""

    # Directorio de guardado
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(f"models/{algo}_{role}_{timestamp}")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  TRAINING: {algo.upper()} - {role.upper()}")
    print("=" * 60)
    print(f"  Save dir: {save_dir}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Envs: {n_envs}")
    print("=" * 60)

    # Crear entornos
    print("\n>>> Creating environments...")
    train_env, eval_env = create_envs(role, n_envs, seed)
    print(f"  Obs space: {train_env.observation_space.shape}")
    print(f"  Action space: {train_env.action_space}")

    # Crear o cargar modelo
    print("\n>>> Creating model...")
    algo_config = ALGO_CONFIGS[algo.lower()]
    AlgoClass = algo_config["class"]

    if load_path:
        print(f"  Loading from: {load_path}")
        model = AlgoClass.load(load_path, env=train_env)
    else:
        model_params = {k: v for k, v in algo_config.items() if k not in ["class", "policy"]}
        model = AlgoClass(
            policy=algo_config["policy"],
            env=train_env,
            tensorboard_log=str(save_dir / "tensorboard"),
            seed=seed,
            verbose=1,
            **model_params,
        )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=20,
        eval_freq=eval_freq,
        log_path=str(save_dir / "eval_logs"),
        best_model_save_path=str(save_dir / "best_model"),
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq * 2,
        save_path=str(save_dir / "checkpoints"),
        name_prefix=f"{role}_model",
    )

    # Entrenar
    print("\n>>> Training...")
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n!!! Training interrupted")

    # Guardar modelo final
    final_path = save_dir / "final_model"
    model.save(str(final_path))
    print(f"\n>>> Model saved: {final_path}")

    # Evaluación final
    print("\n>>> Final evaluation...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)
    print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    train_env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETED")
    print("=" * 60)
    print(f"  Results: {save_dir}")
    print(f"  TensorBoard: tensorboard --logdir={save_dir / 'tensorboard'}")
    print()

    return model, save_dir


def main():
    parser = argparse.ArgumentParser(description="Train infection agents")

    parser.add_argument("--role", type=str, choices=["healthy", "infected"],
                        default="healthy", help="Agent role (default: healthy)")
    parser.add_argument("--algo", type=str, choices=["ppo", "a2c", "dqn"],
                        default="ppo", help="Algorithm (default: ppo)")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Training timesteps (default: 500000)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Parallel environments (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save directory")
    parser.add_argument("--load", type=str, default=None,
                        help="Load model and continue training")
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluation frequency (default: 10000)")

    args = parser.parse_args()

    train(
        role=args.role,
        algo=args.algo,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        save_dir=args.save_dir,
        load_path=args.load,
        eval_freq=args.eval_freq,
    )


if __name__ == "__main__":
    main()
