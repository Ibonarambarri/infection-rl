#!/usr/bin/env python3
"""
Curriculum Learning para Infection RL
======================================
Script de entrenamiento con 3 fases de curriculum + ciclo de refinamiento.

Uso:
    python scripts/train.py
    python scripts/train.py --output-dir models/run1
    python scripts/train.py --start-phase 2

Fases del Curriculum:
    - Fase 1: MAP_LVL1 (20x20), 300k timesteps por rol
    - Fase 2: MAP_LVL2 (30x30), 500k timesteps por rol
    - Fase 3: MAP_LVL3 (40x40), 700k timesteps por rol

Ciclo de Refinamiento (Self-Play):
    - 4 iteraciones en MAP_LVL3
    - 200k timesteps por rol por iteración
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.maps import MAP_LVL1, MAP_LVL2, MAP_LVL3
from src.envs import InfectionEnv, EnvConfig
from src.envs.wrappers import make_infection_env, make_vec_env_parameter_sharing


# ============================================================================
# Configuracion de Fases
# ============================================================================

@dataclass
class PhaseConfig:
    """Configuracion de una fase del curriculum."""
    phase_id: int
    map_data: str
    num_healthy: int
    num_infected: int
    timesteps: int  # Timesteps para ambos roles
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.01

    @property
    def num_agents(self) -> int:
        return self.num_healthy + self.num_infected


# Curriculum de 3 fases
CURRICULUM_PHASES: List[PhaseConfig] = [
    PhaseConfig(
        phase_id=1,
        map_data=MAP_LVL1,
        num_healthy=4,
        num_infected=1,
        timesteps=300_000,
        learning_rate=5e-4,
        ent_coef=0.02,
    ),
    PhaseConfig(
        phase_id=2,
        map_data=MAP_LVL2,
        num_healthy=6,
        num_infected=2,
        timesteps=500_000,
        learning_rate=4e-4,
        ent_coef=0.015,
    ),
    PhaseConfig(
        phase_id=3,
        map_data=MAP_LVL3,
        num_healthy=8,
        num_infected=2,
        timesteps=700_000,
        learning_rate=3e-4,
        ent_coef=0.01,
    ),
]

# Configuracion del ciclo de refinamiento
REFINEMENT_ITERATIONS = 4
REFINEMENT_TIMESTEPS = 200_000


# ============================================================================
# Callbacks
# ============================================================================

class LoggingCallback(BaseCallback):
    """Callback para logging durante entrenamiento."""

    def __init__(self, phase_id: int, role: str, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.phase_id = phase_id
        self.role = role
        self.log_freq = log_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if "episode" in self.locals.get("infos", [{}])[0]:
            info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(info.get("r", 0))

        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            print(f"  [Phase {self.phase_id}][{self.role}] Step {self.n_calls}: "
                  f"Mean reward={mean_reward:.2f}")

        return True


# ============================================================================
# Evaluacion
# ============================================================================

def evaluate_models(
    healthy_model: PPO,
    infected_model: PPO,
    map_data: str,
    num_healthy: int,
    num_infected: int,
    n_episodes: int = 20,
    max_steps: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """Evalua el rendimiento de los modelos."""
    config = EnvConfig(
        map_data=map_data,
        num_agents=num_healthy + num_infected,
        initial_infected=num_infected,
        max_steps=max_steps,
        seed=seed,
    )
    env = InfectionEnv(config)

    results = {
        "episodes": n_episodes,
        "total_healthy_survived": 0,
        "total_healthy_start": 0,
        "total_steps": 0,
        "healthy_wins": 0,
        "infected_wins": 0,
    }

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        done = False
        step_count = 0
        initial_healthy = env.num_healthy
        results["total_healthy_start"] += initial_healthy

        while not done and step_count < max_steps:
            actions = {}

            for agent in env.agents:
                obs = _get_flat_observation(env, agent)
                if agent.is_infected:
                    action, _ = infected_model.predict(obs, deterministic=True)
                else:
                    action, _ = healthy_model.predict(obs, deterministic=True)
                actions[agent.id] = int(action)

            env.step_all(actions)
            step_count += 1

            if env.num_healthy == 0:
                done = True
                results["infected_wins"] += 1
            elif step_count >= max_steps:
                done = True
                if env.num_healthy > 0:
                    results["healthy_wins"] += 1

        results["total_healthy_survived"] += env.num_healthy
        results["total_steps"] += step_count

    results["survival_rate"] = results["total_healthy_survived"] / max(1, results["total_healthy_start"])
    results["avg_steps"] = results["total_steps"] / n_episodes
    results["healthy_win_rate"] = results["healthy_wins"] / n_episodes
    results["infected_win_rate"] = results["infected_wins"] / n_episodes

    return results


def _get_flat_observation(env: InfectionEnv, agent) -> np.ndarray:
    """Obtiene observacion aplanada para un agente."""
    obs_dict = env._get_observation(agent)
    parts = []

    image = obs_dict["image"].astype(np.float32) / 255.0
    parts.append(image.flatten())

    direction = np.zeros(4, dtype=np.float32)
    direction[obs_dict["direction"]] = 1.0
    parts.append(direction)

    state = np.zeros(2, dtype=np.float32)
    state[obs_dict["state"]] = 1.0
    parts.append(state)

    parts.append(obs_dict["position"])
    parts.append(obs_dict["nearby_agents"].flatten())

    return np.concatenate(parts)


# ============================================================================
# Trainer
# ============================================================================

class CurriculumTrainer:
    """Orquestador del entrenamiento con Curriculum Learning."""

    def __init__(
        self,
        output_dir: str = "models/curriculum",
        seed: int = 42,
        n_envs: int = 4,
        verbose: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.n_envs = n_envs
        self.verbose = verbose

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        self.healthy_model: Optional[PPO] = None
        self.infected_model: Optional[PPO] = None
        self.training_log: List[Dict] = []

    def _log(self, message: str):
        if self.verbose > 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def _save_log(self):
        log_path = self.output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

    def train_phase(self, phase: PhaseConfig) -> Dict:
        """Entrena una fase completa del curriculum."""
        self._log("=" * 60)
        self._log(f"FASE {phase.phase_id}")
        self._log(f"  Agentes: {phase.num_healthy} healthy, {phase.num_infected} infected")
        self._log(f"  Timesteps: {phase.timesteps:,} por rol")
        self._log("=" * 60)

        phase_start = time.time()
        phase_results = {
            "phase_id": phase.phase_id,
            "num_healthy": phase.num_healthy,
            "num_infected": phase.num_infected,
            "timesteps": phase.timesteps,
        }

        # Entrenar Healthy
        self._log(f"\n--- Entrenando HEALTHY (Phase {phase.phase_id}) ---")
        self._train_role(
            phase=phase,
            role="healthy",
            opponent_model=self.infected_model,
        )
        healthy_path = self.output_dir / f"phase{phase.phase_id}_healthy.zip"
        self.healthy_model.save(healthy_path)
        self._log(f"  Modelo guardado: {healthy_path}")

        # Entrenar Infected
        self._log(f"\n--- Entrenando INFECTED (Phase {phase.phase_id}) ---")
        self._train_role(
            phase=phase,
            role="infected",
            opponent_model=self.healthy_model,
        )
        infected_path = self.output_dir / f"phase{phase.phase_id}_infected.zip"
        self.infected_model.save(infected_path)
        self._log(f"  Modelo guardado: {infected_path}")

        # Evaluar
        self._log(f"\n--- Evaluando Phase {phase.phase_id} ---")
        eval_results = evaluate_models(
            healthy_model=self.healthy_model,
            infected_model=self.infected_model,
            map_data=phase.map_data,
            num_healthy=phase.num_healthy,
            num_infected=phase.num_infected,
            n_episodes=30,
            seed=self.seed + 2000,
        )

        phase_results["evaluation"] = eval_results
        phase_results["total_time"] = time.time() - phase_start

        self._log(f"  Survival Rate: {eval_results['survival_rate']:.2%}")
        self._log(f"  Avg Steps: {eval_results['avg_steps']:.1f}")
        self._log(f"  Healthy Wins: {eval_results['healthy_win_rate']:.2%}")
        self._log(f"  Infected Wins: {eval_results['infected_win_rate']:.2%}")
        self._log(f"  Tiempo: {phase_results['total_time'] / 60:.1f} minutos")

        self.training_log.append(phase_results)
        self._save_log()

        return phase_results

    def _train_role(
        self,
        phase: PhaseConfig,
        role: str,
        opponent_model: Optional[PPO],
    ):
        """Entrena un rol especifico."""
        vec_env = make_vec_env_parameter_sharing(
            map_data=phase.map_data,
            num_agents=phase.num_agents,
            initial_infected=phase.num_infected,
            force_role=role,
            n_envs=min(self.n_envs, phase.num_healthy if role == "healthy" else phase.num_infected),
            seed=self.seed if role == "healthy" else self.seed + 1000,
            opponent_model=opponent_model,
            opponent_deterministic=True,
            max_steps=500,
            vec_env_cls="dummy",
        )

        model_attr = "healthy_model" if role == "healthy" else "infected_model"
        current_model = getattr(self, model_attr)

        if current_model is None:
            new_model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=phase.learning_rate,
                n_steps=phase.n_steps,
                batch_size=phase.batch_size,
                n_epochs=phase.n_epochs,
                gamma=phase.gamma,
                ent_coef=phase.ent_coef,
                verbose=0,
                seed=self.seed if role == "healthy" else self.seed + 1000,
            )
            setattr(self, model_attr, new_model)
        else:
            if current_model.n_envs != vec_env.num_envs:
                temp_path = self.output_dir / "checkpoints" / f"temp_{role}.zip"
                current_model.save(temp_path)
                new_model = PPO.load(temp_path, env=vec_env)
                setattr(self, model_attr, new_model)
            else:
                current_model.set_env(vec_env)

        model = getattr(self, model_attr)
        callback = LoggingCallback(
            phase_id=phase.phase_id,
            role=role.upper(),
            log_freq=20000,
        )

        model.learn(
            total_timesteps=phase.timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        vec_env.close()

    def run_refinement(self) -> List[Dict]:
        """Ejecuta el ciclo de refinamiento (Self-Play)."""
        self._log("\n" + "=" * 60)
        self._log("CICLO DE REFINAMIENTO (Self-Play)")
        self._log(f"  Iteraciones: {REFINEMENT_ITERATIONS}")
        self._log(f"  Timesteps por rol: {REFINEMENT_TIMESTEPS:,}")
        self._log("=" * 60)

        refinement_results = []

        # Configuracion para refinamiento (usa MAP_LVL3)
        refinement_phase = PhaseConfig(
            phase_id=99,  # ID especial para refinamiento
            map_data=MAP_LVL3,
            num_healthy=8,
            num_infected=2,
            timesteps=REFINEMENT_TIMESTEPS,
            learning_rate=2e-4,
            ent_coef=0.005,
        )

        for iteration in range(1, REFINEMENT_ITERATIONS + 1):
            self._log(f"\n--- Refinamiento Iteracion {iteration}/{REFINEMENT_ITERATIONS} ---")
            iter_start = time.time()

            # Entrenar Healthy contra Infected actual
            self._log(f"  Entrenando HEALTHY...")
            refinement_phase.phase_id = 100 + iteration
            self._train_role(
                phase=refinement_phase,
                role="healthy",
                opponent_model=self.infected_model,
            )

            # Entrenar Infected contra Healthy actualizado
            self._log(f"  Entrenando INFECTED...")
            self._train_role(
                phase=refinement_phase,
                role="infected",
                opponent_model=self.healthy_model,
            )

            # Guardar checkpoints
            healthy_path = self.output_dir / "checkpoints" / f"refinement_{iteration}_healthy.zip"
            infected_path = self.output_dir / "checkpoints" / f"refinement_{iteration}_infected.zip"
            self.healthy_model.save(healthy_path)
            self.infected_model.save(infected_path)

            # Evaluar
            eval_results = evaluate_models(
                healthy_model=self.healthy_model,
                infected_model=self.infected_model,
                map_data=MAP_LVL3,
                num_healthy=8,
                num_infected=2,
                n_episodes=20,
                seed=self.seed + 3000 + iteration,
            )

            iter_results = {
                "iteration": iteration,
                "evaluation": eval_results,
                "time": time.time() - iter_start,
            }
            refinement_results.append(iter_results)

            self._log(f"  Survival Rate: {eval_results['survival_rate']:.2%}")
            self._log(f"  Healthy Wins: {eval_results['healthy_win_rate']:.2%}")
            self._log(f"  Infected Wins: {eval_results['infected_win_rate']:.2%}")
            self._log(f"  Tiempo: {iter_results['time'] / 60:.1f} minutos")

        self.training_log.append({"refinement": refinement_results})
        self._save_log()

        return refinement_results

    def run(self, start_phase: int = 1) -> bool:
        """Ejecuta el curriculum completo + refinamiento."""
        self._log("=" * 60)
        self._log("CURRICULUM LEARNING - INFECTION RL")
        self._log(f"Fases: {len(CURRICULUM_PHASES)}")
        self._log(f"Fase inicial: {start_phase}")
        self._log(f"Directorio de salida: {self.output_dir}")
        self._log("=" * 60)

        total_start = time.time()

        # Ejecutar fases del curriculum
        for phase in CURRICULUM_PHASES:
            if phase.phase_id < start_phase:
                continue
            self.train_phase(phase)

        # Ejecutar ciclo de refinamiento
        self.run_refinement()

        # Guardar modelos finales
        final_healthy = self.output_dir / "healthy_final.zip"
        final_infected = self.output_dir / "infected_final.zip"
        self.healthy_model.save(final_healthy)
        self.infected_model.save(final_infected)

        total_time = time.time() - total_start

        self._log("\n" + "=" * 60)
        self._log("ENTRENAMIENTO COMPLETO")
        self._log(f"Tiempo total: {total_time / 3600:.2f} horas")
        self._log(f"Modelos finales: {self.output_dir}")
        self._log("=" * 60)

        return True

    def load_checkpoint(self, phase_id: int):
        """Carga modelos de un checkpoint."""
        healthy_path = self.output_dir / f"phase{phase_id}_healthy.zip"
        infected_path = self.output_dir / f"phase{phase_id}_infected.zip"

        temp_env = make_infection_env(
            map_data=MAP_LVL1,
            num_agents=5,
            initial_infected=1,
        )

        if healthy_path.exists():
            self.healthy_model = PPO.load(healthy_path, env=temp_env)
            self._log(f"Cargado modelo healthy de Phase {phase_id}")

        if infected_path.exists():
            self.infected_model = PPO.load(infected_path, env=temp_env)
            self._log(f"Cargado modelo infected de Phase {phase_id}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Learning para Infection RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--output-dir", type=str, default="models/curriculum",
                        help="Directorio de salida para modelos")
    parser.add_argument("--start-phase", type=int, default=1,
                        help="Fase inicial (1-3)")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Directorio con checkpoints para continuar")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Numero de entornos paralelos")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Nivel de verbosidad (0-2)")

    args = parser.parse_args()

    trainer = CurriculumTrainer(
        output_dir=args.output_dir,
        seed=args.seed,
        n_envs=args.n_envs,
        verbose=args.verbose,
    )

    if args.continue_from:
        continue_dir = Path(args.continue_from)
        if continue_dir.exists():
            for i in range(3, 0, -1):
                if (continue_dir / f"phase{i}_healthy.zip").exists():
                    trainer.output_dir = continue_dir
                    trainer.load_checkpoint(i)
                    if args.start_phase == 1:
                        args.start_phase = i + 1
                    break

    try:
        trainer.run(start_phase=args.start_phase)
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido.")
        print(f"Modelos guardados en: {trainer.output_dir}")


if __name__ == "__main__":
    main()
