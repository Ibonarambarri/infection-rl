#!/usr/bin/env python3
"""
Entrenamiento con Curriculum Learning + Ping-Pong Self-Play
===========================================================
Sistema de entrenamiento que usa:
- Curriculum: DENSE rewards (Fase 1) -> INTERMEDIATE (Fase 2) -> SPARSE (Fase 3)
- Ping-Pong: Alternancia frecuente entre roles para evitar overfitting
- Logs simples y limpios sin barras de progreso complejas

Uso:
    python scripts/train.py --output-dir models/v1
    python scripts/train.py --output-dir models/v1 --render
    python scripts/train.py --start-phase 2
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.maps import MAP_LVL1, MAP_LVL2, MAP_LVL3
from src.envs import InfectionEnv, EnvConfig, RewardConfig, RewardPreset
from src.envs.wrappers import make_vec_env_parameter_sharing


# ============================================================================
# Learning Rate Schedule
# ============================================================================

def linear_schedule(initial_lr: float, final_lr: float = 5e-5):
    """
    Linear learning rate schedule.

    Decae linealmente desde initial_lr hasta final_lr.
    """
    def schedule(progress_remaining: float) -> float:
        # progress_remaining va de 1.0 (inicio) a 0.0 (fin)
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


# ============================================================================
# Configuracion de Rewards por Fase (Curriculum)
# ============================================================================

def get_reward_config_for_phase(phase_id: int) -> RewardConfig:
    """
    Retorna la configuración de rewards según la fase del curriculum.

    - Fase 1: DENSE (señales completas para aprender movimiento básico)
    - Fase 2: INTERMEDIATE (reducción gradual de señales)
    - Fase 3: SPARSE (solo victoria/derrota)
    """
    if phase_id == 1:
        return RewardConfig.dense()
    elif phase_id == 2:
        return RewardConfig.intermediate()
    else:  # Fase 3 y superiores
        return RewardConfig.sparse()


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
    total_timesteps: int  # Total para la fase (se divide en ping-pong)
    max_steps: int = 500
    ping_pong_interval: int = 50_000  # Alternar cada 50k steps

    @property
    def num_agents(self) -> int:
        return self.num_healthy + self.num_infected

    @property
    def reward_config(self) -> RewardConfig:
        return get_reward_config_for_phase(self.phase_id)


# Curriculum progresivo con ratio 1:1 (equilibrado)
# Rewards: DENSE → INTERMEDIATE → SPARSE
CURRICULUM_PHASES: List[PhaseConfig] = [
    PhaseConfig(
        phase_id=1,
        map_data=MAP_LVL1,  # 20x20
        num_healthy=2,      # 2v2 equilibrado
        num_infected=2,
        total_timesteps=800_000,
        max_steps=150,      # Episodios cortos = más iteraciones
        ping_pong_interval=50_000,
    ),
    PhaseConfig(
        phase_id=2,
        map_data=MAP_LVL1,  # 20x20
        num_healthy=3,      # 3v3 equilibrado
        num_infected=3,
        total_timesteps=1_000_000,
        max_steps=200,
        ping_pong_interval=50_000,
    ),
    PhaseConfig(
        phase_id=3,
        map_data=MAP_LVL2,  # 30x30
        num_healthy=4,      # 4v4 equilibrado
        num_infected=4,
        total_timesteps=1_200_000,
        max_steps=300,
        ping_pong_interval=50_000,
    ),
    PhaseConfig(
        phase_id=4,
        map_data=MAP_LVL3,  # 40x40
        num_healthy=5,      # 5v5 equilibrado (target deployment)
        num_infected=5,
        total_timesteps=1_500_000,
        max_steps=400,
        ping_pong_interval=50_000,
    ),
]

# Refinamiento adaptativo
ADAPTIVE_CYCLES = 10
ADAPTIVE_TIMESTEPS = 300_000
ADAPTIVE_THRESHOLD = 0.60

# Ventajas para infectados (para balancear el juego)
INFECTED_SPEED = 1  # Misma velocidad que healthy
INFECTED_GLOBAL_VISION = True  # Los infectados ven a todos los healthy


# ============================================================================
# Callback Simple de Logging
# ============================================================================

class SimpleLoggingCallback(BaseCallback):
    """
    Callback de logging simple y limpio.

    Imprime cada log_freq steps:
    [Phase X][Role] Step: N | Win Rate: XX% | Mean Reward: XX.X | FPS: XXX
    """

    def __init__(self, phase_id: int, role: str, log_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.phase_id = phase_id
        self.role = role
        self.log_freq = log_freq

        # Métricas
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.total_episodes = 0

        # Para calcular FPS
        self.last_log_time = None
        self.last_log_steps = 0

    def _on_training_start(self) -> None:
        self.last_log_time = time.time()
        self.last_log_steps = 0

    def _on_step(self) -> bool:
        # Recopilar métricas de episodios completados
        for info in self.locals.get("infos", []):
            if "episode" in info:
                episode_info = info["episode"]
                reward = episode_info.get("r", 0)
                length = episode_info.get("l", 0)

                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.total_episodes += 1

                # Victoria basada en supervivencia real, no en reward
                # healthy_survived: número de agentes sanos que quedaron vivos
                # FIX: No asumir default 0 (causaba 100% win rate falso para infected)
                healthy_survived = episode_info.get("healthy_survived")

                if healthy_survived is None:
                    # Key missing - no contar como victoria para ninguno
                    continue

                if self.role.upper() == "HEALTHY":
                    # Healthy gana si sobrevive (healthy_survived > 0)
                    if healthy_survived > 0:
                        self.wins += 1
                else:
                    # Infected gana si infecta a todos (healthy_survived == 0)
                    if healthy_survived == 0:
                        self.wins += 1

        # Imprimir cada log_freq steps
        if self.n_calls % self.log_freq == 0 and self.n_calls > 0:
            self._print_metrics()

        return True

    def _print_metrics(self):
        """Imprime métricas de forma simple y limpia."""
        # Calcular win rate
        win_rate = (self.wins / self.total_episodes * 100) if self.total_episodes > 0 else 0

        # Calcular mean reward (últimos 100 episodios)
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
        mean_reward = np.mean(recent_rewards)

        # Calcular FPS
        current_time = time.time()
        if self.last_log_time is not None:
            elapsed = current_time - self.last_log_time
            steps_done = self.n_calls - self.last_log_steps
            fps = int(steps_done / elapsed) if elapsed > 0 else 0
        else:
            fps = 0

        self.last_log_time = current_time
        self.last_log_steps = self.n_calls

        # Imprimir línea limpia
        print(
            f"[Phase {self.phase_id}][{self.role}] "
            f"Step: {self.n_calls:,} | "
            f"Win Rate: {win_rate:.0f}% | "
            f"Mean Reward: {mean_reward:.1f} | "
            f"FPS: {fps}",
            flush=True
        )

        # Registrar en TensorBoard
        if self.logger is not None:
            role_prefix = f"train/{self.role}"
            self.logger.record(f"{role_prefix}_win_rate", win_rate / 100)
            self.logger.record(f"{role_prefix}_mean_reward", mean_reward)
            self.logger.record(f"{role_prefix}_episodes", self.total_episodes)


# ============================================================================
# Evaluacion
# ============================================================================

def evaluate_models(
    healthy_model: PPO,
    infected_model: PPO,
    map_data: str,
    num_healthy: int,
    num_infected: int,
    reward_config: RewardConfig,
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

    Procesa las observaciones manualmente para garantizar que el formato
    sea IDÉNTICO al usado durante el entrenamiento (MultiInputPolicy: 'image' y 'vector').
    """
    from src.envs.wrappers import make_infection_env

    # Crear entorno usando la misma función que el entrenamiento
    env = make_infection_env(
        map_data=map_data,
        num_agents=num_healthy + num_infected,
        initial_infected=num_infected,
        max_steps=max_steps,
        seed=seed,
        flatten=False,  # MultiInputPolicy necesita dict con 'image' y 'vector'
        reward_config=reward_config,
        infected_speed=INFECTED_SPEED,
        infected_global_vision=INFECTED_GLOBAL_VISION,
    )

    # Obtener referencia al entorno base para acceder a config y _get_observation
    base_env = env.unwrapped

    def _process_obs(raw_obs: Dict) -> Dict[str, np.ndarray]:
        """
        Convierte observación cruda al formato MultiInputPolicy.

        Replica EXACTAMENTE el procesamiento de DictObservationWrapper
        para garantizar consistencia entre entrenamiento y evaluación.
        """
        # Imagen: normalizar a [0, 1] como float32
        image = raw_obs["image"].astype(np.float32) / 255.0

        # Construir vector de features
        parts = []

        # Direction como encoding circular [cos, sin] (2 elementos)
        angle = raw_obs["direction"] * (np.pi / 2)  # 0, pi/2, pi, 3pi/2
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        parts.append(direction)

        # State one-hot (2 elementos: 0=healthy, 1=infected)
        state = np.zeros(2, dtype=np.float32)
        state[raw_obs["state"]] = 1.0
        parts.append(state)

        # Position (ya viene normalizada del environment: x/width, y/height)
        position = raw_obs["position"].astype(np.float32)
        parts.append(position)

        # Nearby agents (5 features por agente)
        nearby = raw_obs["nearby_agents"].flatten().astype(np.float32)
        parts.append(nearby)

        vector = np.concatenate(parts)

        return {
            "image": image,
            "vector": vector,
        }

    if render and renderer is not None:
        renderer.set_env(base_env)

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

            for agent in base_env.agents:
                # Obtener observación cruda del entorno base
                raw_obs = base_env._get_observation(agent)
                # Procesar al formato que espera el modelo
                processed_obs = _process_obs(raw_obs)

                if agent.is_infected:
                    if infected_model is not None:
                        action, _ = infected_model.predict(processed_obs, deterministic=deterministic)
                    else:
                        action = np.random.randint(0, 4)
                else:
                    if healthy_model is not None:
                        action, _ = healthy_model.predict(processed_obs, deterministic=deterministic)
                    else:
                        action = np.random.randint(0, 4)

                actions[agent.id] = int(action)

            base_env.step_all(actions)
            step_count += 1

            if base_env.num_healthy == 0:
                done = True
                results["infected_wins"] += 1
            elif step_count >= max_steps:
                done = True
                results["healthy_wins"] += 1

            if render and renderer is not None:
                if not renderer.handle_events():
                    if renderer.skip_requested:
                        skip_all = True
                    break
                renderer.render_frame(step_count, base_env.num_healthy, base_env.num_infected)
                renderer.wait_frame()

        results["total_steps"] += step_count

    env.close()

    results["healthy_win_rate"] = results["healthy_wins"] / n_episodes
    results["infected_win_rate"] = results["infected_wins"] / n_episodes
    results["avg_steps"] = results["total_steps"] / n_episodes

    return results
# ============================================================================
# Trainer con Ping-Pong
# ============================================================================

class PingPongTrainer:
    """
    Entrenador con Ping-Pong Self-Play.

    Alterna el entrenamiento entre infected y healthy cada N steps
    para evitar overfitting a un oponente estático.
    """

    def __init__(
        self,
        output_dir: str = "models/pingpong",
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
            print(f"[{timestamp}] {message}", flush=True)

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

    def _create_model(
        self,
        vec_env,
        role: str,
    ) -> PPO:
        """Crea un nuevo modelo PPO con hiperparámetros optimizados."""
        return PPO(
            "MultiInputPolicy",  # CNN para imagen + MLP para vector
            vec_env,
            learning_rate=linear_schedule(1e-4, 5e-5),  # Decae de 1e-4 a 5e-5
            n_steps=4096,  # Horizonte largo para sparse rewards
            batch_size=128,
            n_epochs=10,
            gamma=0.995,  # Reducido de 0.999 para mejor credit assignment
            ent_coef=0.05,  # Mayor exploración para sparse rewards
            clip_range=0.2,
            gae_lambda=0.98,
            verbose=0,
            seed=self.seed + (0 if role == "healthy" else 1000),
            tensorboard_log="./tensorboard_logs/",
        )

    def _create_vec_env(
        self,
        phase: PhaseConfig,
        role: str,
        opponent_model: Optional[PPO],
    ):
        """Crea el entorno vectorizado para un rol."""
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
            reward_config=phase.reward_config,
            infected_speed=INFECTED_SPEED,
            infected_global_vision=INFECTED_GLOBAL_VISION,
        )

        # NOTA: VecNormalize eliminado porque:
        # 1. Nuestros wrappers ya normalizan imágenes a [0,1]
        # 2. VecNormalize cambia estadísticas durante entrenamiento pero no se aplica igual en evaluación
        # 3. Esto causaba desajuste de observaciones → evaluación con resultados falsos

        return vec_env

    def _train_steps(
        self,
        phase: PhaseConfig,
        role: str,
        opponent_model: Optional[PPO],
        timesteps: int,
        round_num: int,
    ):
        """Entrena un rol por un número específico de steps."""
        vec_env = self._create_vec_env(phase, role, opponent_model)

        model_attr = "healthy_model" if role == "healthy" else "infected_model"
        current_model = getattr(self, model_attr)

        if current_model is None:
            new_model = self._create_model(vec_env, role)
            setattr(self, model_attr, new_model)
        else:
            # Actualizar el entorno del modelo existente
            if current_model.n_envs != vec_env.num_envs:
                temp_path = self.output_dir / "checkpoints" / f"temp_{role}.zip"
                current_model.save(temp_path)
                new_model = PPO.load(temp_path, env=vec_env)
                setattr(self, model_attr, new_model)
            else:
                current_model.set_env(vec_env)

        model = getattr(self, model_attr)
        callback = SimpleLoggingCallback(
            phase_id=phase.phase_id,
            role=role.upper(),
            log_freq=10_000,  # Print cada 10k steps
        )

        tb_log_name = f"phase{phase.phase_id}_{role}_r{round_num}"

        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,  # Sin barra de progreso, usamos nuestros prints
            tb_log_name=tb_log_name,
        )

        vec_env.close()

    def train_phase_pingpong(self, phase: PhaseConfig) -> Dict:
        """
        Entrena una fase usando ADAPTIVE Self-Play.

        Antes de cada ronda, evalúa el rendimiento de ambos bandos y entrena
        SOLO al agente más débil para equilibrar la balanza.

        Lógica:
        - Si healthy_win_rate < 0.45: Entrena SOLO healthy
        - Si infected_win_rate < 0.45: Entrena SOLO infected
        - Si ambos entre 0.45-0.55: Entrena AMBOS (ping-pong clásico)
        """
        self._log("=" * 60)
        self._log(f"FASE {phase.phase_id} ({phase.reward_config.preset.value.upper()} REWARDS)")
        self._log(f"  Mapa: {phase.num_agents} agentes ({phase.num_healthy}H vs {phase.num_infected}I)")
        self._log(f"  Max steps: {phase.max_steps}")
        self._log(f"  Total timesteps: {phase.total_timesteps:,}")
        self._log(f"  Adaptive interval: {phase.ping_pong_interval:,}")
        self._log("=" * 60)

        phase_start = time.time()

        # Calcular número de rondas
        timesteps_per_role = phase.total_timesteps // 2
        num_rounds = timesteps_per_role // phase.ping_pong_interval

        self._log(f"\nAdaptive Self-Play: {num_rounds} rondas de {phase.ping_pong_interval:,} steps")

        # Umbrales para decisión adaptativa
        WEAK_THRESHOLD = 0.45  # Por debajo = muy débil, necesita entrenamiento
        BALANCED_MIN = 0.45
        BALANCED_MAX = 0.55

        for round_num in range(1, num_rounds + 1):
            self._log(f"\n{'─' * 50}")
            self._log(f"RONDA {round_num}/{num_rounds}")
            self._log(f"{'─' * 50}")

            # === EVALUACIÓN RÁPIDA PARA DECIDIR QUIÉN ENTRENAR ===
            if self.healthy_model is not None and self.infected_model is not None:
                self._log("  Evaluando balance actual...")
                quick_eval = evaluate_models(
                    healthy_model=self.healthy_model,
                    infected_model=self.infected_model,
                    map_data=phase.map_data,
                    num_healthy=phase.num_healthy,
                    num_infected=phase.num_infected,
                    reward_config=phase.reward_config,
                    n_episodes=20,  # Evaluación rápida
                    max_steps=phase.max_steps,
                    seed=self.seed + 3000 + round_num,
                    render=False,
                    deterministic=True,
                )
                healthy_wr = quick_eval['healthy_win_rate']
                infected_wr = quick_eval['infected_win_rate']
                self._log(f"  Win Rates: Healthy={healthy_wr:.0%} | Infected={infected_wr:.0%}")
            else:
                # Primera ronda: no hay modelos todavía, entrenar ambos
                healthy_wr = 0.5
                infected_wr = 0.5
                self._log("  Primera ronda: entrenando ambos agentes")

            # === DECISIÓN ADAPTATIVA ===
            # Siempre entrenar al más débil para balancear
            balance_margin = 0.05  # 5% de diferencia = equilibrado
            if abs(healthy_wr - infected_wr) < balance_margin:
                # Muy equilibrado → entrenar ambos
                train_healthy = True
                train_infected = True
                decision = f"Equilibrado ({healthy_wr:.0%} vs {infected_wr:.0%}) → Entrenando AMBOS"
            elif healthy_wr < infected_wr:
                # Healthy más débil → entrenar SOLO healthy
                train_healthy = True
                train_infected = False
                decision = f"HEALTHY más débil ({healthy_wr:.0%} vs {infected_wr:.0%}) → Entrenando SOLO Healthy"
            else:
                # Infected más débil → entrenar SOLO infected
                train_healthy = False
                train_infected = True
                decision = f"INFECTED más débil ({infected_wr:.0%} vs {healthy_wr:.0%}) → Entrenando SOLO Infected"

            self._log(f"  Decisión: {decision}")

            # === ENTRENAMIENTO SEGÚN DECISIÓN ===
            if train_infected:
                infected_opponent = self.healthy_model
                self._log(f"  Training INFECTED... (vs {'Model' if infected_opponent else 'Heuristic'})")
                self._train_steps(
                    phase=phase,
                    role="infected",
                    opponent_model=infected_opponent,
                    timesteps=phase.ping_pong_interval,
                    round_num=round_num,
                )

            if train_healthy:
                self._log(f"  Training HEALTHY... (vs {'Model' if self.infected_model else 'Heuristic'})")
                self._train_steps(
                    phase=phase,
                    role="healthy",
                    opponent_model=self.infected_model,
                    timesteps=phase.ping_pong_interval,
                    round_num=round_num,
                )

            # === GUARDAR CHECKPOINTS ===
            if self.infected_model is not None:
                self.infected_model.save(self.output_dir / "checkpoints" / f"phase{phase.phase_id}_r{round_num}_infected.zip")
            if self.healthy_model is not None:
                self.healthy_model.save(self.output_dir / "checkpoints" / f"phase{phase.phase_id}_r{round_num}_healthy.zip")

        # Guardar modelos finales de la fase
        self.infected_model.save(self.output_dir / f"phase{phase.phase_id}_infected.zip")
        self.healthy_model.save(self.output_dir / f"phase{phase.phase_id}_healthy.zip")

        # Evaluación final de la fase
        self._log(f"\n--- Evaluación Final Phase {phase.phase_id} ---")

        if self.render and self.renderer is None:
            self._init_renderer(phase)

        eval_results = evaluate_models(
            healthy_model=self.healthy_model,
            infected_model=self.infected_model,
            map_data=phase.map_data,
            num_healthy=phase.num_healthy,
            num_infected=phase.num_infected,
            reward_config=phase.reward_config,
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
            "reward_preset": phase.reward_config.preset.value,
            "evaluation": eval_results,
            "time": phase_time,
        }
        self.training_log.append(phase_results)
        self._save_log()

        return phase_results

    def run_adaptive_refinement(self) -> List[Dict]:
        """Ciclo de refinamiento adaptativo post-curriculum."""
        self._log("\n" + "=" * 60)
        self._log("REFINAMIENTO ADAPTATIVO")
        self._log(f"  Ciclos: {ADAPTIVE_CYCLES}")
        self._log(f"  Timesteps por ciclo: {ADAPTIVE_TIMESTEPS:,}")
        self._log(f"  Umbral: {ADAPTIVE_THRESHOLD:.0%}")
        self._log("=" * 60)

        results = []

        # Usar SPARSE rewards para refinamiento
        refinement_phase = PhaseConfig(
            phase_id=200,
            map_data=MAP_LVL3,
            num_healthy=5,
            num_infected=5,
            total_timesteps=ADAPTIVE_TIMESTEPS * 2,
            max_steps=500,
            ping_pong_interval=ADAPTIVE_TIMESTEPS,
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
                reward_config=RewardConfig.sparse(),
                n_episodes=20,
                seed=self.seed + 4000 + cycle,
                render=self.render,
                renderer=self.renderer,
                phase_id=200 + cycle,
            )

            healthy_wr = pre_eval['healthy_win_rate']
            infected_wr = pre_eval['infected_win_rate']

            self._log(f"  Healthy: {healthy_wr:.1%}, Infected: {infected_wr:.1%}")

            # Decidir quién entrenar (siempre el más débil)
            balance_margin = 0.05
            if abs(healthy_wr - infected_wr) < balance_margin:
                # Muy equilibrado → alternar (infected en ciclos impares)
                role_to_train = "infected" if cycle % 2 == 1 else "healthy"
                reason = f"Equilibrado ({healthy_wr:.0%} vs {infected_wr:.0%}) → alternando"
            elif healthy_wr < infected_wr:
                role_to_train = "healthy"
                reason = f"Healthy más débil ({healthy_wr:.0%} vs {infected_wr:.0%})"
            else:
                role_to_train = "infected"
                reason = f"Infected más débil ({infected_wr:.0%} vs {healthy_wr:.0%})"

            self._log(f"  Entrenando: {role_to_train.upper()} ({reason})")

            # Entrenar
            refinement_phase.phase_id = 200 + cycle
            opponent = self.healthy_model if role_to_train == "infected" else self.infected_model

            self._train_steps(
                phase=refinement_phase,
                role=role_to_train,
                opponent_model=opponent,
                timesteps=ADAPTIVE_TIMESTEPS,
                round_num=cycle,
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
            reward_config=RewardConfig.sparse(),
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
        """Ejecuta el curriculum completo con ping-pong + refinamiento."""
        self._log("=" * 60)
        self._log("ENTRENAMIENTO PING-PONG + CURRICULUM")
        self._log("=" * 60)
        self._log(f"  Fases: {len(CURRICULUM_PHASES)}")
        self._log(f"  Fase inicial: {start_phase}")
        self._log(f"  Curriculum: DENSE -> INTERMEDIATE -> SPARSE")
        self._log(f"  Output: {self.output_dir}")
        self._log("=" * 60)

        total_start = time.time()

        # Ejecutar fases del curriculum con ping-pong
        for phase in CURRICULUM_PHASES:
            if phase.phase_id < start_phase:
                continue
            self.train_phase_pingpong(phase)

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

        # Crear entorno temporal para cargar modelos
        phase = CURRICULUM_PHASES[0]  # Usar fase 1 como referencia
        temp_env = self._create_vec_env(phase, "healthy", None)

        if healthy_path.exists():
            self.healthy_model = PPO.load(healthy_path, env=temp_env)
            self._log(f"Cargado healthy de Phase {phase_id}")

        if infected_path.exists():
            self.infected_model = PPO.load(infected_path, env=temp_env)
            self._log(f"Cargado infected de Phase {phase_id}")

        temp_env.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento Ping-Pong con Curriculum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/train.py --output-dir models/v1
  python scripts/train.py --output-dir models/v1 --start-phase 2
  python scripts/train.py --output-dir models/v1 --render
        """
    )

    parser.add_argument("--output-dir", type=str, default="models/pingpong",
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

    trainer = PingPongTrainer(
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
        print("\n\nEntrenamiento interrumpido.", flush=True)
        print(f"Modelos guardados en: {trainer.output_dir}", flush=True)


if __name__ == "__main__":
    main()
