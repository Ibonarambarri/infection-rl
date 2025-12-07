#!/usr/bin/env python3
"""
Stage 3: Adversarial Training (Iterative Self-Play)
=====================================================
Entrena un bando contra el modelo congelado del otro bando.

Este script permite entrenar iterativamente:
1. Healthy vs Infected_v1 -> Healthy_v2
2. Infected vs Healthy_v2 -> Infected_v2
3. Healthy vs Infected_v2 -> Healthy_v3
... y así sucesivamente

Uso:
    # Entrenar healthy contra el mejor modelo de infected
    python scripts/train_stage3_adversarial.py \\
        --target-role healthy \\
        --opponent-model models/stage2_infected/infected_final \\
        --map-file maps/medium.txt

    # Entrenar infected contra el modelo healthy actualizado
    python scripts/train_stage3_adversarial.py \\
        --target-role infected \\
        --opponent-model models/stage3_healthy_v2/healthy_final \\
        --map-file maps/medium.txt

    # Continuar desde un checkpoint
    python scripts/train_stage3_adversarial.py \\
        --target-role healthy \\
        --opponent-model models/infected_v2/infected_final \\
        --continue-from models/healthy_v2/healthy_final
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
        description="Stage 3: Entrenamiento Adversarial Iterativo"
    )

    # Configuración principal
    parser.add_argument(
        "--target-role",
        type=str,
        required=True,
        choices=["healthy", "infected"],
        help="Rol a entrenar (healthy o infected)"
    )
    parser.add_argument(
        "--opponent-model",
        type=str,
        required=True,
        help="Path al modelo oponente congelado"
    )

    # Continuación de entrenamiento
    parser.add_argument(
        "--continue-from",
        type=str,
        default=None,
        help="Continuar entrenamiento desde este modelo"
    )

    # Configuración del mapa y agentes
    parser.add_argument(
        "--map-file",
        type=str,
        default="maps/medium.txt",
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
        help="Número de agentes infectados"
    )

    # Configuración de entrenamiento
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Pasos de entrenamiento por iteración"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Número de entornos paralelos"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Pasos máximos por episodio"
    )

    # Oponente
    parser.add_argument(
        "--opponent-deterministic",
        action="store_true",
        default=True,
        help="Usar predicciones determinísticas para oponentes"
    )
    parser.add_argument(
        "--opponent-stochastic",
        action="store_true",
        help="Usar predicciones estocásticas para más variación"
    )

    # Hiperparámetros PPO
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (más bajo para fine-tuning)"
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
        default=0.1,
        help="Clip range (más conservador para fine-tuning)"
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.005,
        help="Coeficiente de entropía (más bajo para fine-tuning)"
    )

    # Arquitectura de red (solo si no se continúa)
    parser.add_argument(
        "--net-arch",
        type=str,
        default="256,256",
        help="Arquitectura de red (ignorado si --continue-from)"
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
        help="Nombre del experimento"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Versión del modelo (ej: v2, v3)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Frecuencia de guardado"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Frecuencia de evaluación"
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
        help="Semilla"
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
        help="Verbosidad"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Verificar modelo oponente
    opponent_path = Path(args.opponent_model)
    if not opponent_path.exists() and not (opponent_path.parent / f"{opponent_path.name}.zip").exists():
        print(f"ERROR: Modelo oponente no encontrado: {args.opponent_model}")
        sys.exit(1)

    # Verificar modelo de continuación
    if args.continue_from:
        continue_path = Path(args.continue_from)
        if not continue_path.exists() and not (continue_path.parent / f"{continue_path.name}.zip").exists():
            print(f"ERROR: Modelo para continuar no encontrado: {args.continue_from}")
            sys.exit(1)

    # Determinar nombre del experimento
    map_name = Path(args.map_file).stem
    version = args.version or "v1"

    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"stage3_{args.target_role}_{version}_{map_name}_{timestamp}"

    # Crear directorios
    model_dir = Path(args.output_dir) / exp_name
    log_dir = Path("logs") / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determinar modo de oponente
    opponent_deterministic = not args.opponent_stochastic

    # Calcular número de agentes del rol target
    if args.target_role == "healthy":
        num_target_agents = args.num_agents - args.infected_agents
        first_agent_id = args.infected_agents
    else:
        num_target_agents = args.infected_agents
        first_agent_id = 0

    print("=" * 60)
    print(f"Stage 3: Adversarial Training - {args.target_role.upper()}")
    print("=" * 60)
    print(f"Target role: {args.target_role}")
    print(f"Opponent model: {args.opponent_model}")
    print(f"Continue from: {args.continue_from or 'scratch'}")
    print(f"Map: {args.map_file}")
    print(f"Agents: {args.num_agents} total, {args.infected_agents} infected")
    print(f"Training {num_target_agents} {args.target_role} agents")
    print(f"Opponent mode: {'deterministic' if opponent_deterministic else 'stochastic'}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Output: {model_dir}")
    print("=" * 60)

    # Crear VecEnv con Parameter Sharing
    print("\nCreando entornos con Parameter Sharing...")
    vec_env = make_vec_env_parameter_sharing(
        map_file=args.map_file,
        num_agents=args.num_agents,
        initial_infected=args.infected_agents,
        force_role=args.target_role,
        n_envs=args.n_envs,
        seed=args.seed,
        opponent_model=args.opponent_model,
        opponent_deterministic=opponent_deterministic,
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
        controlled_agent_id=first_agent_id,
        force_role=args.target_role,
        seed=args.seed + 1000,
        opponent_model=args.opponent_model,
        opponent_deterministic=opponent_deterministic,
        max_steps=args.max_steps,
    )
    eval_env = Monitor(eval_env)

    # Crear o cargar modelo
    if args.continue_from:
        print(f"\nCargando modelo desde: {args.continue_from}")
        model = PPO.load(
            args.continue_from,
            env=vec_env,
            verbose=args.verbose,
            tensorboard_log=str(log_dir),
            device=args.device,
        )
        # Actualizar hiperparámetros para fine-tuning
        model.learning_rate = args.learning_rate
        model.clip_range = lambda _: args.clip_range
        model.ent_coef = args.ent_coef
    else:
        print("\nCreando nuevo modelo PPO...")
        net_arch = [int(x) for x in args.net_arch.split(",")]

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
        name_prefix=f"{args.target_role}_model",
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
    print("\nIniciando entrenamiento adversarial...")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not args.continue_from,
        )
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")

    # Guardar modelo final
    final_path = model_dir / f"{args.target_role}_final"
    model.save(str(final_path))
    print(f"\nModelo final guardado: {final_path}")

    # Guardar configuración
    config_path = model_dir / "config.txt"
    with open(config_path, "w") as f:
        f.write(f"Stage 3: Adversarial Training - {args.target_role}\n")
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

    # Sugerir siguiente paso
    next_role = "infected" if args.target_role == "healthy" else "healthy"
    next_version = f"v{int(version[1:]) + 1}" if version.startswith("v") else "v2"

    print(f"\nSiguiente iteración (entrenar {next_role} contra este modelo):")
    print(f"  python scripts/train_stage3_adversarial.py \\")
    print(f"      --target-role {next_role} \\")
    print(f"      --opponent-model {final_path} \\")
    print(f"      --version {next_version} \\")
    print(f"      --map-file {args.map_file}")


if __name__ == "__main__":
    main()
