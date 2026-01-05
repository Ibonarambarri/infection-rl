#!/usr/bin/env python3
"""
Entrenamiento con Rewards Sparse + Curriculum
=============================================
Solo recompensa victoria/derrota, pero con curriculum progresivo para
facilitar el aprendizaje.

Filosofia:
- Infected: +10 si ganan (infectan a todos), 0 si no
- Healthy: +10 si sobreviven hasta max_steps, 0 si no
- Sin rewards intermedios (no approach, progress, survive_step, etc.)
- Curriculum: mapas pequeños → grandes para que infected aprendan a ganar

Uso:
    python scripts/train_sparse.py --output-dir models/sparse_v1
    python scripts/train_sparse.py --output-dir models/sparse_v1 --render
    python scripts/train_sparse.py --start-phase 2
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.maps import MAP_LVL1, MAP_LVL2, MAP_LVL3
from src.envs import InfectionEnv, EnvConfig, RewardConfig, RewardPreset
from src.envs.wrappers import make_vec_env_parameter_sharing, FlattenObservationWrapper


# ============================================================================
# Configuracion ULTRA-SPARSE
# ============================================================================

# Rewards: SOLO victoria/derrota - igual en TODAS las fases
SPARSE_REWARDS = RewardConfig(
    preset=RewardPreset.SPARSE,
    # Healthy - gana si sobrevive, penalizado si es infectado
    reward_survive_step=0.0,
    reward_distance_bonus=0.0,
    reward_infected_penalty=-10.0,  # Penalizacion por ser infectado
    reward_not_moving_penalty=0.0,
    reward_stuck_penalty=0.0,
    reward_survive_episode=10.0,  # Reward por sobrevivir
    # Infected - SOLO gana si infecta a todos
    reward_infect_agent=0.0,
    reward_approach_bonus=0.0,
    reward_progress_bonus=0.0,
    reward_no_progress_penalty=0.0,
    reward_step_penalty=0.0,
    reward_all_infected_bonus=10.0,  # UNICO reward para infected
    reward_exploration=0.0,
)


# ============================================================================
# Curriculum de Fases
# ============================================================================

@dataclass
class PhaseConfig:
    """Configuracion de una fase del curriculum."""
    phase_id: int
    map_data: str
    num_healthy: int
    num_infected: int
    timesteps_infected: int  # Infected primero, mas timesteps
    timesteps_healthy: int
    max_steps: int = 500

    @property
    def num_agents(self) -> int:
        return self.num_healthy + self.num_infected


# Curriculum progresivo: mapas pequeños → grandes
# Balance 50/50 entre healthy e infected en todas las fases
CURRICULUM_PHASES: List[PhaseConfig] = [
    PhaseConfig(
        phase_id=1,
        map_data=MAP_LVL1,  # 20x20
        num_healthy=3,
        num_infected=3,
        timesteps_infected=500_000,
        timesteps_healthy=500_000,
        max_steps=300,
    ),
    PhaseConfig(
        phase_id=2,
        map_data=MAP_LVL2,  # 30x30
        num_healthy=4,
        num_infected=4,
        timesteps_infected=700_000,
        timesteps_healthy=700_000,
        max_steps=400,
    ),
    PhaseConfig(
        phase_id=3,
        map_data=MAP_LVL3,  # 40x40
        num_healthy=5,
        num_infected=5,
        timesteps_infected=1_000_000,
        timesteps_healthy=1_000_000,
        max_steps=500,
    ),
]

# Refinamiento adaptativo
ADAPTIVE_CYCLES = 10
ADAPTIVE_TIMESTEPS = 300_000
ADAPTIVE_THRESHOLD = 0.60

# Ventajas para infectados (para balancear el juego)
INFECTED_SPEED = 2  # Los infectados se mueven 2 celdas por acción
INFECTED_GLOBAL_VISION = True  # Los infectados ven a todos los healthy


# ============================================================================
# Callbacks
# ============================================================================

class SparseLoggingCallback(BaseCallback):
    """Callback para logging durante entrenamiento sparse con TensorBoard."""

    def __init__(self, phase_id: int, role: str, log_freq: int = 50000, verbose: int = 1):
        super().__init__(verbose)
        self.phase_id = phase_id
        self.role = role
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.survival_rates = []
        self.infected_percentages = []
        self.infection_counts = []
        self.wins = 0
        self.total_episodes = 0
        self.heartbeat_freq = 100000  # Print minimalista cada 100k steps

    def _on_step(self) -> bool:
        # Recopilar métricas de episodios completados
        for info in self.locals.get("infos", []):
            if "episode" in info:
                episode_info = info["episode"]
                reward = episode_info.get("r", 0)
                length = episode_info.get("l", 0)
                survival_rate = episode_info.get("survival_rate", 0.0)
                infected_pct = episode_info.get("infected_percentage", 0.0)
                infection_count = episode_info.get("infection_count", 0)

                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.survival_rates.append(survival_rate)
                self.infected_percentages.append(infected_pct)
                self.infection_counts.append(infection_count)
                self.total_episodes += 1

                if reward > 0:
                    self.wins += 1

        # Registrar métricas en TensorBoard cada log_freq steps
        if self.n_calls % self.log_freq == 0 and self.total_episodes > 0:
            win_rate = self.wins / self.total_episodes

            # Calcular métricas promedio recientes (últimos 100 episodios)
            recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
            recent_lengths = self.episode_lengths[-100:] if self.episode_lengths else [0]
            recent_survival = self.survival_rates[-100:] if self.survival_rates else [0]
            recent_infected_pct = self.infected_percentages[-100:] if self.infected_percentages else [0]
            recent_infections = self.infection_counts[-100:] if self.infection_counts else [0]

            # Registrar en TensorBoard con prefijo por rol
            role_prefix = f"train/{self.role}"
            self.logger.record(f"{role_prefix}_win_rate", win_rate)
            self.logger.record(f"{role_prefix}_episode_reward", np.mean(recent_rewards))
            self.logger.record(f"{role_prefix}_episode_length", np.mean(recent_lengths))
            self.logger.record(f"{role_prefix}_survival_rate", np.mean(recent_survival))
            self.logger.record(f"{role_prefix}_infected_percentage", np.mean(recent_infected_pct))
            self.logger.record(f"{role_prefix}_infection_count", np.mean(recent_infections))
            self.logger.record(f"{role_prefix}_total_episodes", self.total_episodes)

        # Print minimalista (heartbeat) para saber que el proceso sigue vivo
        if self.n_calls % self.heartbeat_freq == 0:
            win_rate = self.wins / self.total_episodes if self.total_episodes > 0 else 0
            print(f"  [Phase {self.phase_id}][{self.role}] Step {self.n_calls:,} | "
                  f"WR: {win_rate:.1%} | Eps: {self.total_episodes}", flush=True)

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
    render: bool = False,
    renderer=None,
    phase_id: int = 0,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evalua el rendimiento de los modelos en modo multi-agente.

    Usa FlattenObservationWrapper para garantizar que las observaciones
    se aplanen EXACTAMENTE igual que durante el entrenamiento.

    Args:
        deterministic: Si True, usa predicciones determinísticas.
                       Si False, añade estocasticidad (útil si los agentes
                       se quedan atascados en bucles en grid worlds).
    """
    config = EnvConfig(
        map_data=map_data,
        num_agents=num_healthy + num_infected,
        initial_infected=num_infected,
        max_steps=max_steps,
        seed=seed,
        reward_config=SPARSE_REWARDS,
        infected_speed=INFECTED_SPEED,
        infected_global_vision=INFECTED_GLOBAL_VISION,
    )
    env = InfectionEnv(config)

    # Crear wrapper para aplanar observaciones - garantiza compatibilidad con entrenamiento
    flatten_wrapper = FlattenObservationWrapper(env)

    if render and renderer is not None:
        renderer.set_env(env)

    results = {
        "healthy_wins": 0,
        "infected_wins": 0,
        "total_steps": 0,
    }

    skip_all = False

    for ep in range(n_episodes):
        if skip_all:
            break

        env.reset(seed=seed + ep)
        done = False
        step_count = 0

        if render and renderer is not None:
            renderer.set_context(phase_id, ep + 1, n_episodes)

        while not done and step_count < max_steps:
            actions = {}
            for agent in env.agents:
                # Obtener observación dict y aplanarla usando el wrapper
                obs_dict = env._get_observation(agent)
                obs_flat = flatten_wrapper.observation(obs_dict)

                if agent.is_infected:
                    action, _ = infected_model.predict(obs_flat, deterministic=deterministic)
                else:
                    action, _ = healthy_model.predict(obs_flat, deterministic=deterministic)
                actions[agent.id] = int(action)

            env.step_all(actions)
            step_count += 1

            if render and renderer is not None:
                if not renderer.handle_events():
                    if renderer.skip_requested:
                        skip_all = True
                    break
                renderer.render_frame(step_count, env.num_healthy, env.num_infected)
                renderer.wait_frame()

            if env.num_healthy == 0:
                done = True
                results["infected_wins"] += 1
            elif step_count >= max_steps:
                done = True
                results["healthy_wins"] += 1

        results["total_steps"] += step_count

    results["healthy_win_rate"] = results["healthy_wins"] / n_episodes
    results["infected_win_rate"] = results["infected_wins"] / n_episodes
    results["avg_steps"] = results["total_steps"] / n_episodes

    return results


# ============================================================================
# Trainer
# ============================================================================

class SparseTrainer:
    """Entrenador con rewards sparse + curriculum progresivo."""

    def __init__(
        self,
        output_dir: str = "models/sparse",
        seed: int = 42,
        n_envs: int = 4,
        verbose: int = 1,
        render: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.n_envs = n_envs
        self.verbose = verbose
        self.render = render

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        self.healthy_model: Optional[PPO] = None
        self.infected_model: Optional[PPO] = None
        self.training_log: List[Dict] = []
        self.renderer = None

    def _log(self, message: str):
        if self.verbose > 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def _save_log(self):
        log_path = self.output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

    def _init_renderer(self, phase: PhaseConfig):
        """Inicializa el renderer para visualizacion."""
        try:
            from src.utils.evaluation_renderer import EvaluationRenderer

            config = EnvConfig(
                map_data=phase.map_data,
                num_agents=phase.num_agents,
                initial_infected=phase.num_infected,
                infected_speed=INFECTED_SPEED,
                infected_global_vision=INFECTED_GLOBAL_VISION,
            )
            temp_env = InfectionEnv(config)
            temp_env.reset()

            self.renderer = EvaluationRenderer(
                width=temp_env.width,
                height=temp_env.height,
            )

            if not self.renderer.init_pygame():
                self._log("  Warning: pygame no disponible")
                self.render = False
                self.renderer = None
        except Exception as e:
            self._log(f"  Warning: No se pudo inicializar renderer: {e}")
            self.render = False
            self.renderer = None

    def _train_role(
        self,
        phase: PhaseConfig,
        role: str,
        opponent_model: Optional[PPO],
        timesteps: int,
    ):
        """Entrena un rol especifico."""
        vec_env = make_vec_env_parameter_sharing(
            map_data=phase.map_data,
            num_agents=phase.num_agents,
            initial_infected=phase.num_infected,
            force_role=role,
            n_envs=min(self.n_envs, phase.num_healthy if role == "healthy" else phase.num_infected),
            seed=self.seed + (0 if role == "healthy" else 1000) + phase.phase_id * 100,
            opponent_model=opponent_model,
            opponent_deterministic=True,
            max_steps=phase.max_steps,
            vec_env_cls="dummy",
            reward_config=SPARSE_REWARDS,
            infected_speed=INFECTED_SPEED,
            infected_global_vision=INFECTED_GLOBAL_VISION,
        )

        model_attr = "healthy_model" if role == "healthy" else "infected_model"
        current_model = getattr(self, model_attr)

        if current_model is None:
            # Parametros optimizados para sparse rewards
            new_model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=1e-4,
                n_steps=4096,
                batch_size=128,
                n_epochs=10,
                gamma=0.999,
                ent_coef=0.05,
                clip_range=0.2,
                gae_lambda=0.98,
                verbose=0,
                seed=self.seed + (0 if role == "healthy" else 1000),
                tensorboard_log="./tensorboard_logs/",
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
        callback = SparseLoggingCallback(
            phase_id=phase.phase_id,
            role=role.upper(),
            log_freq=50000,
        )

        # Nombre único para TensorBoard: phase1_infected, phase2_healthy, etc.
        tb_log_name = f"phase{phase.phase_id}_{role}"

        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True,
            tb_log_name=tb_log_name,
        )

        vec_env.close()

    def train_phase(self, phase: PhaseConfig) -> Dict:
        """Entrena una fase del curriculum."""
        self._log("=" * 60)
        self._log(f"FASE {phase.phase_id} (SPARSE REWARDS)")
        self._log(f"  Mapa: {phase.num_healthy + phase.num_infected} agentes")
        self._log(f"  Agentes: {phase.num_healthy} healthy vs {phase.num_infected} infected")
        self._log(f"  Max steps: {phase.max_steps}")
        self._log(f"  Timesteps infected: {phase.timesteps_infected:,}")
        self._log(f"  Timesteps healthy: {phase.timesteps_healthy:,}")
        self._log("=" * 60)

        phase_start = time.time()

        # Entrenar INFECTED primero
        # IMPORTANTE: Si healthy_model es None (primera fase), usar heurística
        # en lugar de un modelo aleatorio. Esto fuerza a Infected a aprender
        # estrategias reales de persecución desde el principio.
        infected_opponent = None if self.healthy_model is None else self.healthy_model

        self._log(f"\n--- Entrenando INFECTED (Phase {phase.phase_id}) ---")
        if infected_opponent is None:
            self._log("  Oponente: Heurística (agentes huyen activamente)")
        else:
            self._log("  Oponente: Modelo Healthy entrenado")

        self._train_role(
            phase=phase,
            role="infected",
            opponent_model=infected_opponent,
            timesteps=phase.timesteps_infected,
        )
        infected_path = self.output_dir / f"phase{phase.phase_id}_infected.zip"
        self.infected_model.save(infected_path)

        # Entrenar HEALTHY contra Infected ya entrenado
        self._log(f"\n--- Entrenando HEALTHY (Phase {phase.phase_id}) ---")
        self._log("  Oponente: Modelo Infected entrenado")
        self._train_role(
            phase=phase,
            role="healthy",
            opponent_model=self.infected_model,
            timesteps=phase.timesteps_healthy,
        )
        healthy_path = self.output_dir / f"phase{phase.phase_id}_healthy.zip"
        self.healthy_model.save(healthy_path)

        # Evaluar
        self._log(f"\n--- Evaluando Phase {phase.phase_id} ---")

        if self.render and self.renderer is None:
            self._init_renderer(phase)

        eval_results = evaluate_models(
            healthy_model=self.healthy_model,
            infected_model=self.infected_model,
            map_data=phase.map_data,
            num_healthy=phase.num_healthy,
            num_infected=phase.num_infected,
            n_episodes=30,
            max_steps=phase.max_steps,
            seed=self.seed + 2000 + phase.phase_id,
            render=self.render,
            renderer=self.renderer,
            phase_id=phase.phase_id,
        )

        phase_time = time.time() - phase_start

        self._log(f"  Healthy Win Rate: {eval_results['healthy_win_rate']:.1%}")
        self._log(f"  Infected Win Rate: {eval_results['infected_win_rate']:.1%}")
        self._log(f"  Avg Steps: {eval_results['avg_steps']:.1f}")
        self._log(f"  Tiempo: {phase_time / 60:.1f} minutos")

        phase_results = {
            "phase_id": phase.phase_id,
            "evaluation": eval_results,
            "time": phase_time,
        }
        self.training_log.append(phase_results)
        self._save_log()

        return phase_results

    def run_adaptive_refinement(self) -> List[Dict]:
        """Ciclo de refinamiento adaptativo."""
        self._log("\n" + "=" * 60)
        self._log("REFINAMIENTO ADAPTATIVO (SPARSE)")
        self._log(f"  Ciclos: {ADAPTIVE_CYCLES}")
        self._log(f"  Timesteps por ciclo: {ADAPTIVE_TIMESTEPS:,}")
        self._log(f"  Umbral: {ADAPTIVE_THRESHOLD:.0%}")
        self._log("=" * 60)

        results = []

        refinement_phase = PhaseConfig(
            phase_id=200,
            map_data=MAP_LVL3,
            num_healthy=5,
            num_infected=5,
            timesteps_infected=ADAPTIVE_TIMESTEPS,
            timesteps_healthy=ADAPTIVE_TIMESTEPS,
            max_steps=500,
        )

        if self.render and self.renderer is None:
            self._init_renderer(refinement_phase)

        for cycle in range(1, ADAPTIVE_CYCLES + 1):
            self._log(f"\n{'─' * 50}")
            self._log(f"CICLO {cycle}/{ADAPTIVE_CYCLES}")
            self._log(f"{'─' * 50}")

            cycle_start = time.time()

            # Evaluar estado actual
            self._log("  Evaluando...")
            pre_eval = evaluate_models(
                healthy_model=self.healthy_model,
                infected_model=self.infected_model,
                map_data=MAP_LVL3,
                num_healthy=5,
                num_infected=5,
                n_episodes=20,
                seed=self.seed + 4000 + cycle,
                render=self.render,
                renderer=self.renderer,
                phase_id=200 + cycle,
            )

            healthy_wr = pre_eval['healthy_win_rate']
            infected_wr = pre_eval['infected_win_rate']

            self._log(f"  Healthy: {healthy_wr:.1%}, Infected: {infected_wr:.1%}")

            # Decidir quién entrenar
            if infected_wr > ADAPTIVE_THRESHOLD:
                role_to_train = "healthy"
                reason = f"Infected dominan ({infected_wr:.0%})"
            elif healthy_wr > ADAPTIVE_THRESHOLD:
                role_to_train = "infected"
                reason = f"Healthy dominan ({healthy_wr:.0%})"
            else:
                role_to_train = "infected"
                reason = "Equilibrado → infected por defecto"

            self._log(f"  Entrenando: {role_to_train.upper()} ({reason})")

            # Entrenar
            refinement_phase.phase_id = 200 + cycle
            self._train_role(
                phase=refinement_phase,
                role=role_to_train,
                opponent_model=self.healthy_model if role_to_train == "infected" else self.infected_model,
                timesteps=ADAPTIVE_TIMESTEPS,
            )

            # Guardar checkpoint
            self.healthy_model.save(self.output_dir / "checkpoints" / f"adaptive_{cycle}_healthy.zip")
            self.infected_model.save(self.output_dir / "checkpoints" / f"adaptive_{cycle}_infected.zip")

            cycle_time = time.time() - cycle_start
            results.append({
                "cycle": cycle,
                "pre_eval": {"healthy_wr": healthy_wr, "infected_wr": infected_wr},
                "trained": role_to_train,
                "time": cycle_time,
            })

            self._log(f"  Tiempo: {cycle_time / 60:.1f} min")

        # Evaluación final
        self._log(f"\n{'─' * 50}")
        self._log("EVALUACIÓN FINAL")
        final_eval = evaluate_models(
            healthy_model=self.healthy_model,
            infected_model=self.infected_model,
            map_data=MAP_LVL3,
            num_healthy=5,
            num_infected=5,
            n_episodes=30,
            seed=self.seed + 5000,
            render=self.render,
            renderer=self.renderer,
            phase_id=299,
        )
        self._log(f"  Healthy: {final_eval['healthy_win_rate']:.1%}")
        self._log(f"  Infected: {final_eval['infected_win_rate']:.1%}")

        self.training_log.append({
            "adaptive_refinement": results,
            "final_evaluation": final_eval,
        })
        self._save_log()

        return results

    def run(self, start_phase: int = 1) -> bool:
        """Ejecuta el curriculum completo + refinamiento."""
        self._log("=" * 60)
        self._log("ENTRENAMIENTO SPARSE + CURRICULUM")
        self._log("=" * 60)
        self._log(f"  Fases: {len(CURRICULUM_PHASES)}")
        self._log(f"  Fase inicial: {start_phase}")
        self._log(f"  Rewards: SOLO victoria (+10) o derrota (0)")
        self._log(f"  Output: {self.output_dir}")
        self._log("=" * 60)

        total_start = time.time()

        # Ejecutar fases del curriculum
        for phase in CURRICULUM_PHASES:
            if phase.phase_id < start_phase:
                continue
            self.train_phase(phase)

        # Ejecutar refinamiento adaptativo
        self.run_adaptive_refinement()

        # Guardar modelos finales
        self.healthy_model.save(self.output_dir / "healthy_final.zip")
        self.infected_model.save(self.output_dir / "infected_final.zip")

        total_time = time.time() - total_start

        self._log("\n" + "=" * 60)
        self._log("ENTRENAMIENTO COMPLETO")
        self._log(f"  Tiempo total: {total_time / 3600:.2f} horas")
        self._log(f"  Modelos: {self.output_dir}")
        self._log("=" * 60)

        return True

    def load_checkpoint(self, phase_id: int):
        """Carga modelos de un checkpoint."""
        healthy_path = self.output_dir / f"phase{phase_id}_healthy.zip"
        infected_path = self.output_dir / f"phase{phase_id}_infected.zip"

        temp_env = make_vec_env_parameter_sharing(
            map_data=MAP_LVL1,
            num_agents=5,
            initial_infected=1,
            force_role="healthy",
            reward_config=SPARSE_REWARDS,
            infected_speed=INFECTED_SPEED,
            infected_global_vision=INFECTED_GLOBAL_VISION,
        )

        if healthy_path.exists():
            self.healthy_model = PPO.load(healthy_path, env=temp_env)
            self._log(f"Cargado healthy de Phase {phase_id}")

        if infected_path.exists():
            self.infected_model = PPO.load(infected_path, env=temp_env)
            self._log(f"Cargado infected de Phase {phase_id}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento Sparse con Curriculum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/train_sparse.py --output-dir models/sparse_v1
  python scripts/train_sparse.py --output-dir models/sparse_v1 --start-phase 2
  python scripts/train_sparse.py --output-dir models/sparse_v1 --render
        """
    )

    parser.add_argument("--output-dir", type=str, default="models/sparse",
                        help="Directorio de salida")
    parser.add_argument("--start-phase", type=int, default=1,
                        help="Fase inicial (1-3)")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Directorio con checkpoints para continuar")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Entornos paralelos")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosidad")
    parser.add_argument("--render", action="store_true",
                        help="Visualizar evaluaciones")

    args = parser.parse_args()

    trainer = SparseTrainer(
        output_dir=args.output_dir,
        seed=args.seed,
        n_envs=args.n_envs,
        verbose=args.verbose,
        render=args.render,
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
