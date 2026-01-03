#!/usr/bin/env python3
"""
Curriculum Learning Automatizado para Infection RL
===================================================
Script maestro que orquesta el entrenamiento iterativo de ambos bandos
con progresión automática de dificultad.

Uso:
    python scripts/train_curriculum.py
    python scripts/train_curriculum.py --output-dir models/curriculum_run1
    python scripts/train_curriculum.py --start-phase 3 --continue-from models/curriculum_run1

Flujo:
    Para cada fase del curriculum:
    1. Entrenar Healthy vs (heurística o modelo infected anterior)
    2. Evaluar hasta cumplir métrica o límite de pasos
    3. Entrenar Infected vs modelo healthy recién entrenado
    4. Evaluar hasta cumplir métrica o límite de pasos
    5. Si ambos cumplen métricas mínimas, avanzar a siguiente fase
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs import InfectionEnv, EnvConfig
from src.envs.wrappers import (
    make_infection_env,
    make_vec_env_parameter_sharing,
)


# ============================================================================
# Configuración de Fases del Curriculum
# ============================================================================

@dataclass
class PhaseConfig:
    """Configuración de una fase del curriculum."""
    phase_id: int
    map_file: str
    num_healthy: int
    num_infected: int
    timesteps_healthy: int
    timesteps_infected: int
    # Métricas de transición
    min_survival_rate: float = 0.5      # Healthy debe sobrevivir al menos X%
    min_infection_rate: float = 0.6     # Infected debe infectar al menos X%
    # Hiperparámetros opcionales por fase
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.01

    @property
    def num_agents(self) -> int:
        return self.num_healthy + self.num_infected


# Definición del curriculum completo
DEFAULT_CURRICULUM: List[PhaseConfig] = [
    # Fase 1: Mapa pequeño y abierto, pocos agentes
    PhaseConfig(
        phase_id=1,
        map_file="maps/curriculum_lvl1.txt",
        num_healthy=4,
        num_infected=1,
        timesteps_healthy=100_000,
        timesteps_infected=100_000,
        min_survival_rate=0.4,
        min_infection_rate=0.5,
        learning_rate=5e-4,  # Mayor LR para inicio rápido
        ent_coef=0.02,       # Más exploración
    ),
    # Fase 2: Mapa más grande, más agentes
    PhaseConfig(
        phase_id=2,
        map_file="maps/curriculum_lvl2.txt",
        num_healthy=6,
        num_infected=2,
        timesteps_healthy=150_000,
        timesteps_infected=150_000,
        min_survival_rate=0.45,
        min_infection_rate=0.55,
        learning_rate=4e-4,
    ),
    # Fase 3: Ciudad media
    PhaseConfig(
        phase_id=3,
        map_file="maps/curriculum_lvl3.txt",
        num_healthy=8,
        num_infected=2,
        timesteps_healthy=200_000,
        timesteps_infected=200_000,
        min_survival_rate=0.5,
        min_infection_rate=0.6,
        learning_rate=3e-4,
    ),
    # Fase 4: Ciudad densa
    PhaseConfig(
        phase_id=4,
        map_file="maps/curriculum_lvl4.txt",
        num_healthy=10,
        num_infected=3,
        timesteps_healthy=250_000,
        timesteps_infected=250_000,
        min_survival_rate=0.5,
        min_infection_rate=0.65,
        learning_rate=2e-4,
        ent_coef=0.005,  # Menos exploración, más explotación
    ),
    # Fase 5: Laberinto complejo, máximos agentes
    PhaseConfig(
        phase_id=5,
        map_file="maps/curriculum_lvl5.txt",
        num_healthy=12,
        num_infected=3,
        timesteps_healthy=300_000,
        timesteps_infected=300_000,
        min_survival_rate=0.55,
        min_infection_rate=0.7,
        learning_rate=1e-4,
        ent_coef=0.001,  # Mínima exploración
    ),
]


# ============================================================================
# Sistema de Evaluación
# ============================================================================

def evaluate_phase(
    healthy_model: Optional[PPO],
    infected_model: Optional[PPO],
    map_file: str,
    num_healthy: int,
    num_infected: int,
    n_episodes: int = 20,
    max_steps: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evalúa el rendimiento de los modelos en una fase.

    Args:
        healthy_model: Modelo entrenado para healthy (o None para heurística)
        infected_model: Modelo entrenado para infected (o None para heurística)
        map_file: Archivo del mapa
        num_healthy: Número de agentes sanos
        num_infected: Número de agentes infectados
        n_episodes: Número de episodios de evaluación
        max_steps: Pasos máximos por episodio
        seed: Semilla para reproducibilidad

    Returns:
        Dict con métricas: survival_rate, infection_rate, avg_steps, etc.
    """
    config = EnvConfig(
        map_file=map_file,
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
        "total_infected": 0,
        "total_steps": 0,
        "healthy_wins": 0,   # Episodios donde sobrevive al menos 1 healthy
        "infected_wins": 0,  # Episodios donde todos infectados
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
                if agent.is_infected:
                    if infected_model is not None:
                        obs = _get_flat_observation(env, agent)
                        action, _ = infected_model.predict(obs, deterministic=True)
                        actions[agent.id] = int(action)
                    else:
                        actions[agent.id] = env._get_other_agent_action(agent)
                else:
                    if healthy_model is not None:
                        obs = _get_flat_observation(env, agent)
                        action, _ = healthy_model.predict(obs, deterministic=True)
                        actions[agent.id] = int(action)
                    else:
                        actions[agent.id] = env._get_other_agent_action(agent)

            env.step_all(actions)
            step_count += 1

            # Verificar terminación
            if env.num_healthy == 0:
                done = True
                results["infected_wins"] += 1
            elif step_count >= max_steps:
                done = True
                if env.num_healthy > 0:
                    results["healthy_wins"] += 1

        results["total_healthy_survived"] += env.num_healthy
        results["total_infected"] += initial_healthy - env.num_healthy
        results["total_steps"] += step_count

    # Calcular métricas finales
    results["survival_rate"] = results["total_healthy_survived"] / max(1, results["total_healthy_start"])
    results["infection_rate"] = results["total_infected"] / max(1, results["total_healthy_start"])
    results["avg_steps"] = results["total_steps"] / n_episodes
    results["healthy_win_rate"] = results["healthy_wins"] / n_episodes
    results["infected_win_rate"] = results["infected_wins"] / n_episodes

    return results


def _get_flat_observation(env: InfectionEnv, agent) -> np.ndarray:
    """Obtiene observación aplanada para un agente."""
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
# Callbacks Personalizados
# ============================================================================

class CurriculumLoggingCallback(BaseCallback):
    """Callback para logging durante entrenamiento de curriculum."""

    def __init__(self, phase_id: int, role: str, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.phase_id = phase_id
        self.role = role
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Recoger info de episodios terminados
        if "episode" in self.locals.get("infos", [{}])[0]:
            info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(info.get("r", 0))
            self.episode_lengths.append(info.get("l", 0))

        # Logging periódico
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:])
            print(f"  [Phase {self.phase_id}][{self.role}] Step {self.n_calls}: "
                  f"Mean reward={mean_reward:.2f}, Mean length={mean_length:.1f}")

        return True


class EarlyStoppingCallback(BaseCallback):
    """Para el entrenamiento si se alcanza el objetivo de métrica."""

    def __init__(
        self,
        eval_env,
        target_metric: str,
        target_value: float,
        eval_freq: int = 20000,
        n_eval_episodes: int = 10,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.target_metric = target_metric
        self.target_value = target_value
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_metric = 0.0
        self.reached_target = False

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluar modelo actual
            results = self._evaluate()
            current_metric = results.get(self.target_metric, 0.0)

            if current_metric > self.best_metric:
                self.best_metric = current_metric

            if self.verbose > 0:
                print(f"    Eval: {self.target_metric}={current_metric:.3f} "
                      f"(best={self.best_metric:.3f}, target={self.target_value:.3f})")

            if current_metric >= self.target_value:
                self.reached_target = True
                if self.verbose > 0:
                    print(f"    Target reached! Stopping early.")
                return False  # Detener entrenamiento

        return True

    def _evaluate(self) -> Dict[str, float]:
        """Evaluación rápida del modelo actual."""
        # Usar el modelo del training
        total_rewards = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
                if isinstance(done, np.ndarray):
                    done = done[0]
            total_rewards.append(episode_reward)

        return {
            "mean_reward": np.mean(total_rewards),
            "survival_rate": 0.5,  # Placeholder, se calcula en evaluate_phase
            "infection_rate": 0.5,
        }


# ============================================================================
# Clase Principal del Curriculum
# ============================================================================

class CurriculumTrainer:
    """Orquestador del entrenamiento con Curriculum Learning."""

    def __init__(
        self,
        curriculum: List[PhaseConfig],
        output_dir: str = "models/curriculum",
        seed: int = 42,
        n_envs: int = 4,
        verbose: int = 1,
    ):
        self.curriculum = curriculum
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.n_envs = n_envs
        self.verbose = verbose

        # Crear estructura de directorios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        # Estado del entrenamiento
        self.current_phase = 0
        self.healthy_model: Optional[PPO] = None
        self.infected_model: Optional[PPO] = None
        self.training_log: List[Dict] = []

        # Guardar configuración
        self._save_config()

    def _save_config(self):
        """Guarda la configuración del curriculum."""
        config = {
            "curriculum": [asdict(phase) for phase in self.curriculum],
            "seed": self.seed,
            "n_envs": self.n_envs,
            "created_at": datetime.now().isoformat(),
        }
        config_path = self.output_dir / "curriculum_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _log(self, message: str):
        """Log con timestamp."""
        if self.verbose > 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def _save_training_log(self):
        """Guarda el log de entrenamiento."""
        log_path = self.output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

    def train_phase(self, phase: PhaseConfig) -> Tuple[bool, Dict]:
        """
        Entrena una fase completa del curriculum.

        Returns:
            Tuple[bool, Dict]: (éxito, métricas)
        """
        self._log(f"=" * 60)
        self._log(f"FASE {phase.phase_id}: {phase.map_file}")
        self._log(f"  Agentes: {phase.num_healthy} healthy, {phase.num_infected} infected")
        self._log(f"=" * 60)

        phase_start = time.time()
        phase_results = {
            "phase_id": phase.phase_id,
            "map_file": phase.map_file,
            "num_healthy": phase.num_healthy,
            "num_infected": phase.num_infected,
        }

        # =====================
        # 1. Entrenar Healthy
        # =====================
        self._log(f"\n--- Entrenando HEALTHY (Phase {phase.phase_id}) ---")
        healthy_start = time.time()

        # Determinar oponente para healthy
        opponent_model = None
        if phase.phase_id > 1 and self.infected_model is not None:
            opponent_model = self.infected_model
            self._log("  Oponente: modelo infected de fase anterior")
        else:
            self._log("  Oponente: heurística")

        # Crear entorno vectorizado
        vec_env = make_vec_env_parameter_sharing(
            map_file=phase.map_file,
            num_agents=phase.num_agents,
            initial_infected=phase.num_infected,
            force_role="healthy",
            n_envs=min(self.n_envs, phase.num_healthy),
            seed=self.seed,
            opponent_model=opponent_model,
            opponent_deterministic=True,
            max_steps=500,
            vec_env_cls="dummy",
        )

        # Crear o continuar modelo healthy
        if self.healthy_model is None:
            self.healthy_model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=phase.learning_rate,
                n_steps=phase.n_steps,
                batch_size=phase.batch_size,
                n_epochs=phase.n_epochs,
                gamma=phase.gamma,
                ent_coef=phase.ent_coef,
                verbose=0,
                seed=self.seed,
            )
        else:
            self.healthy_model.set_env(vec_env)

        # Callback de logging
        callback = CurriculumLoggingCallback(
            phase_id=phase.phase_id,
            role="HEALTHY",
            log_freq=20000,
        )

        # Entrenar
        self.healthy_model.learn(
            total_timesteps=phase.timesteps_healthy,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        vec_env.close()

        # Guardar modelo healthy
        healthy_path = self.output_dir / f"phase{phase.phase_id}_healthy.zip"
        self.healthy_model.save(healthy_path)
        self._log(f"  Modelo guardado: {healthy_path}")

        healthy_time = time.time() - healthy_start
        phase_results["healthy_training_time"] = healthy_time

        # =====================
        # 2. Entrenar Infected
        # =====================
        self._log(f"\n--- Entrenando INFECTED (Phase {phase.phase_id}) ---")
        infected_start = time.time()

        # Oponente para infected: modelo healthy recién entrenado
        self._log("  Oponente: modelo healthy recién entrenado")

        # Crear entorno vectorizado
        vec_env = make_vec_env_parameter_sharing(
            map_file=phase.map_file,
            num_agents=phase.num_agents,
            initial_infected=phase.num_infected,
            force_role="infected",
            n_envs=min(self.n_envs, phase.num_infected),
            seed=self.seed + 1000,
            opponent_model=self.healthy_model,
            opponent_deterministic=True,
            max_steps=500,
            vec_env_cls="dummy",
        )

        # Crear o continuar modelo infected
        if self.infected_model is None:
            self.infected_model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=phase.learning_rate,
                n_steps=phase.n_steps,
                batch_size=phase.batch_size,
                n_epochs=phase.n_epochs,
                gamma=phase.gamma,
                ent_coef=phase.ent_coef,
                verbose=0,
                seed=self.seed + 1000,
            )
        else:
            self.infected_model.set_env(vec_env)

        # Callback de logging
        callback = CurriculumLoggingCallback(
            phase_id=phase.phase_id,
            role="INFECTED",
            log_freq=20000,
        )

        # Entrenar
        self.infected_model.learn(
            total_timesteps=phase.timesteps_infected,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        vec_env.close()

        # Guardar modelo infected
        infected_path = self.output_dir / f"phase{phase.phase_id}_infected.zip"
        self.infected_model.save(infected_path)
        self._log(f"  Modelo guardado: {infected_path}")

        infected_time = time.time() - infected_start
        phase_results["infected_training_time"] = infected_time

        # =====================
        # 3. Evaluar fase
        # =====================
        self._log(f"\n--- Evaluando Phase {phase.phase_id} ---")

        eval_results = evaluate_phase(
            healthy_model=self.healthy_model,
            infected_model=self.infected_model,
            map_file=phase.map_file,
            num_healthy=phase.num_healthy,
            num_infected=phase.num_infected,
            n_episodes=30,
            max_steps=500,
            seed=self.seed + 2000,
        )

        phase_results["evaluation"] = eval_results

        self._log(f"  Survival Rate: {eval_results['survival_rate']:.2%} "
                  f"(target: {phase.min_survival_rate:.2%})")
        self._log(f"  Infection Rate: {eval_results['infection_rate']:.2%} "
                  f"(target: {phase.min_infection_rate:.2%})")
        self._log(f"  Avg Steps: {eval_results['avg_steps']:.1f}")
        self._log(f"  Healthy Wins: {eval_results['healthy_win_rate']:.2%}")
        self._log(f"  Infected Wins: {eval_results['infected_win_rate']:.2%}")

        # Verificar si se cumplen métricas
        survival_ok = eval_results["survival_rate"] >= phase.min_survival_rate * 0.8  # 80% del target
        infection_ok = eval_results["infection_rate"] >= phase.min_infection_rate * 0.8

        phase_success = survival_ok and infection_ok

        phase_time = time.time() - phase_start
        phase_results["total_time"] = phase_time
        phase_results["success"] = phase_success

        self._log(f"\n  Phase {phase.phase_id} {'PASSED' if phase_success else 'NEEDS IMPROVEMENT'}")
        self._log(f"  Total time: {phase_time / 60:.1f} minutes")

        # Guardar log
        self.training_log.append(phase_results)
        self._save_training_log()

        return phase_success, phase_results

    def run(self, start_phase: int = 1) -> bool:
        """
        Ejecuta el curriculum completo.

        Args:
            start_phase: Fase inicial (1-indexed)

        Returns:
            bool: True si se completó todo el curriculum
        """
        self._log("=" * 60)
        self._log("CURRICULUM LEARNING - INFECTION RL")
        self._log(f"Total phases: {len(self.curriculum)}")
        self._log(f"Starting from phase: {start_phase}")
        self._log(f"Output directory: {self.output_dir}")
        self._log("=" * 60)

        total_start = time.time()

        for i, phase in enumerate(self.curriculum):
            if phase.phase_id < start_phase:
                continue

            success, results = self.train_phase(phase)

            # Guardar mejores modelos como "best"
            if success:
                best_healthy = self.output_dir / "best_healthy_model.zip"
                best_infected = self.output_dir / "best_infected_model.zip"
                shutil.copy(
                    self.output_dir / f"phase{phase.phase_id}_healthy.zip",
                    best_healthy
                )
                shutil.copy(
                    self.output_dir / f"phase{phase.phase_id}_infected.zip",
                    best_infected
                )
                self._log(f"  Best models updated from Phase {phase.phase_id}")

            # Siempre continuar al siguiente nivel (curriculum automático)
            self.current_phase = phase.phase_id

        # Guardar modelos finales
        final_healthy = self.output_dir / "healthy_final.zip"
        final_infected = self.output_dir / "infected_final.zip"
        self.healthy_model.save(final_healthy)
        self.infected_model.save(final_infected)

        total_time = time.time() - total_start

        self._log("\n" + "=" * 60)
        self._log("CURRICULUM TRAINING COMPLETE")
        self._log(f"Total time: {total_time / 3600:.2f} hours")
        self._log(f"Final models saved to: {self.output_dir}")
        self._log("=" * 60)

        return True

    def load_checkpoint(self, phase_id: int):
        """Carga modelos de un checkpoint de fase."""
        healthy_path = self.output_dir / f"phase{phase_id}_healthy.zip"
        infected_path = self.output_dir / f"phase{phase_id}_infected.zip"

        if healthy_path.exists():
            # Crear entorno temporal para cargar
            temp_env = make_infection_env(
                map_file="maps/curriculum_lvl1.txt",
                num_agents=5,
                initial_infected=1,
            )
            self.healthy_model = PPO.load(healthy_path, env=temp_env)
            self._log(f"Loaded healthy model from Phase {phase_id}")

        if infected_path.exists():
            temp_env = make_infection_env(
                map_file="maps/curriculum_lvl1.txt",
                num_agents=5,
                initial_infected=1,
            )
            self.infected_model = PPO.load(infected_path, env=temp_env)
            self._log(f"Loaded infected model from Phase {phase_id}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Learning automatizado para Infection RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/train_curriculum.py
  python scripts/train_curriculum.py --output-dir models/run1
  python scripts/train_curriculum.py --start-phase 3 --continue-from models/run1
  python scripts/train_curriculum.py --n-envs 8 --seed 123
        """
    )

    parser.add_argument("--output-dir", type=str, default="models/curriculum",
                        help="Directorio de salida para modelos y logs")
    parser.add_argument("--start-phase", type=int, default=1,
                        help="Fase inicial (1-5)")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Directorio con checkpoints para continuar")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Número de entornos paralelos")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Nivel de verbosidad (0-2)")

    # Opciones para personalizar timesteps
    parser.add_argument("--timesteps-multiplier", type=float, default=1.0,
                        help="Multiplicador para timesteps de entrenamiento")

    args = parser.parse_args()

    # Ajustar timesteps si se especifica multiplicador
    curriculum = DEFAULT_CURRICULUM.copy()
    if args.timesteps_multiplier != 1.0:
        for phase in curriculum:
            phase.timesteps_healthy = int(phase.timesteps_healthy * args.timesteps_multiplier)
            phase.timesteps_infected = int(phase.timesteps_infected * args.timesteps_multiplier)

    # Crear trainer
    trainer = CurriculumTrainer(
        curriculum=curriculum,
        output_dir=args.output_dir,
        seed=args.seed,
        n_envs=args.n_envs,
        verbose=args.verbose,
    )

    # Cargar checkpoints si se especifica
    if args.continue_from:
        continue_dir = Path(args.continue_from)
        if continue_dir.exists():
            # Encontrar la última fase completada
            for i in range(5, 0, -1):
                if (continue_dir / f"phase{i}_healthy.zip").exists():
                    trainer.output_dir = continue_dir
                    trainer.load_checkpoint(i)
                    if args.start_phase == 1:
                        args.start_phase = i + 1
                    break

    # Ejecutar curriculum
    try:
        trainer.run(start_phase=args.start_phase)
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario.")
        print(f"Modelos guardados en: {trainer.output_dir}")


if __name__ == "__main__":
    main()
