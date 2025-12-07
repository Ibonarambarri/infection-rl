#!/usr/bin/env python3
"""
Stage 1: Train Healthy Agents vs Heuristic Bots
================================================
Entrena agentes sanos usando Parameter Sharing.

Los infectados usan heurística (perseguir al sano más cercano).
Este es el primer paso del Curriculum Learning.

Uso:
    python scripts/train_stage1_healthy.py --map-file maps/small.txt --timesteps 500000
    python scripts/train_stage1_healthy.py --map-file maps/medium.txt --num-agents 20 --infected-agents 3
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from src.envs import make_infection_env, make_vec_env_parameter_sharing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: Entrenar Healthy Agents vs Bots (heurística)"
    )

    # Configuración del mapa y agentes
    parser.add_argument(
        "--map-file",
        type=str,
        default="maps/small.txt",
        help="Archivo del mapa a usar"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=10,
        help="Número total de agentes"
    )
    parser.add_argument(
        "--infected-agents",
        type=int,
        default=2,
        help="Número de agentes infectados (bots)"
    )

    # Configuración de entrenamiento
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Pasos totales de entrenamiento"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Número de entornos paralelos (default: num_healthy_agents)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Pasos máximos por episodio"
    )

    # Hiperparámetros PPO
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Pasos por actualización"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Épocas por actualización"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Factor de descuento"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="Lambda para GAE"
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="Clip range para PPO"
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Coeficiente de entropía"
    )

    # Arquitectura de red
    parser.add_argument(
        "--net-arch",
        type=str,
        default="256,256",
        help="Arquitectura de red (ej: 256,256)"
    )

    # Guardado y logging
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directorio base para modelos"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Nombre del experimento (default: auto)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Frecuencia de guardado (steps)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Frecuencia de evaluación (steps)"
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Episodios de evaluación"
    )

    # Otros
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad"
    )
    parser.add_argument(
        "--vec-env",
        type=str,
        choices=["subproc", "dummy"],
        default="dummy",
        help="Tipo de VecEnv"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Nivel de verbosidad"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo (auto, cpu, cuda, mps)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determinar nombre del experimento
    map_name = Path(args.map_file).stem
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"stage1_healthy_{map_name}_{timestamp}"

    # Crear directorios
    model_dir = Path(args.output_dir) / exp_name
    log_dir = Path("logs") / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 1: Training Healthy Agents vs Heuristic Bots")
    print("=" * 60)
    print(f"Map: {args.map_file}")
    print(f"Agents: {args.num_agents} total, {args.infected_agents} infected")
    print(f"Healthy agents: {args.num_agents - args.infected_agents}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Output: {model_dir}")
    print("=" * 60)

    # Crear VecEnv con Parameter Sharing
    print("\nCreando entornos con Parameter Sharing...")
    vec_env = make_vec_env_parameter_sharing(
        map_file=args.map_file,
        num_agents=args.num_agents,
        initial_infected=args.infected_agents,
        force_role="healthy",
        n_envs=args.n_envs,
        seed=args.seed,
        opponent_model=None,  # Stage 1: bots heurísticos
        max_steps=args.max_steps,
        vec_env_cls=args.vec_env,
    )

    n_envs = vec_env.num_envs
    print(f"Entornos paralelos creados: {n_envs}")

    # Crear entorno de evaluación
    print("Creando entorno de evaluación...")
    eval_env = make_infection_env(
        map_file=args.map_file,
        num_agents=args.num_agents,
        initial_infected=args.infected_agents,
        controlled_agent_id=args.infected_agents,  # Primer agente healthy
        force_role="healthy",
        seed=args.seed + 1000,
        max_steps=args.max_steps,
    )
    eval_env = Monitor(eval_env)

    # Parsear arquitectura de red
    net_arch = [int(x) for x in args.net_arch.split(",")]

    # Crear modelo PPO
    print("\nCreando modelo PPO...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": net_arch},
        verbose=args.verbose,
        seed=args.seed,
        tensorboard_log=str(log_dir),
        device=args.device,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // n_envs, 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="healthy_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(args.eval_freq // n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Entrenar
    print("\nIniciando entrenamiento...")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")

    # Guardar modelo final
    final_path = model_dir / "healthy_final"
    model.save(str(final_path))
    print(f"\nModelo final guardado: {final_path}")

    # Guardar configuración
    config_path = model_dir / "config.txt"
    with open(config_path, "w") as f:
        f.write("Stage 1: Healthy Agents Training\n")
        f.write("=" * 40 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    print(f"Configuración guardada: {config_path}")

    # Cerrar entornos
    vec_env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Entrenamiento completado!")
    print("=" * 60)
    print(f"\nPróximo paso:")
    print(f"  python scripts/train_stage2_infected.py \\")
    print(f"      --map-file {args.map_file} \\")
    print(f"      --healthy-model {final_path} \\")
    print(f"      --num-agents {args.num_agents} \\")
    print(f"      --infected-agents {args.infected_agents}")


if __name__ == "__main__":
    main()
