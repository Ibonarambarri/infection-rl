"""
Custom Callbacks for Training
=============================
Callbacks especializados para monitorear el entrenamiento en el entorno de infección.
"""

from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class InfectionEvalCallback(EvalCallback):
    """
    Callback de evaluación extendido con métricas específicas del entorno de infección.
    """

    def __init__(
        self,
        eval_env,
        n_eval_episodes: int = 10,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose,
        )

        # Métricas adicionales
        self.survival_rates = []
        self.infection_times = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Registrar métricas adicionales después de evaluación
        if self.n_calls % self.eval_freq == 0:
            self._log_infection_metrics()

        return result

    def _log_infection_metrics(self):
        """Registra métricas específicas del entorno de infección."""
        survival_count = 0
        total_episodes = 0
        total_infection_events = 0
        total_steps = 0

        # Evaluar episodios
        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            steps = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                steps += 1

            total_episodes += 1
            total_steps += steps

            # Extraer métricas del info
            if "episode" in info:
                ep_info = info["episode"]
                if ep_info.get("healthy_survived", 0) > 0:
                    survival_count += 1
                total_infection_events += ep_info.get("infection_events", 0)

        # Calcular promedios
        survival_rate = survival_count / total_episodes if total_episodes > 0 else 0
        avg_steps = total_steps / total_episodes if total_episodes > 0 else 0
        avg_infections = total_infection_events / total_episodes if total_episodes > 0 else 0

        # Registrar en TensorBoard
        if self.logger is not None:
            self.logger.record("infection/survival_rate", survival_rate)
            self.logger.record("infection/avg_episode_length", avg_steps)
            self.logger.record("infection/avg_infections_per_episode", avg_infections)

        # Guardar historial
        self.survival_rates.append(survival_rate)
        self.episode_lengths.append(avg_steps)

        if self.verbose > 0:
            print(f"  Survival rate: {survival_rate:.2%}")
            print(f"  Avg episode length: {avg_steps:.1f}")
            print(f"  Avg infections/episode: {avg_infections:.2f}")


class TensorBoardInfectionCallback(BaseCallback):
    """
    Callback para registrar métricas detalladas del entorno de infección en TensorBoard.
    """

    def __init__(
        self,
        verbose: int = 0,
        log_freq: int = 1000,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Acumuladores
        self.episode_rewards = []
        self.episode_lengths = []
        self.healthy_survived = []
        self.infection_events = []

    def _on_step(self) -> bool:
        # Obtener info del step actual
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                ep_info = info["episode"]

                self.episode_rewards.append(ep_info.get("r", 0))
                self.episode_lengths.append(ep_info.get("l", 0))
                self.healthy_survived.append(ep_info.get("healthy_survived", 0))
                self.infection_events.append(ep_info.get("infection_events", 0))

        # Registrar periódicamente
        if self.n_calls % self.log_freq == 0 and self.episode_rewards:
            self._log_metrics()

        return True

    def _log_metrics(self):
        """Registra métricas acumuladas."""
        if not self.episode_rewards:
            return

        # Calcular estadísticas
        metrics = {
            "infection/mean_reward": np.mean(self.episode_rewards[-100:]),
            "infection/mean_length": np.mean(self.episode_lengths[-100:]),
            "infection/mean_healthy_survived": np.mean(self.healthy_survived[-100:]),
            "infection/mean_infection_events": np.mean(self.infection_events[-100:]),
            "infection/survival_rate": np.mean([1 if h > 0 else 0 for h in self.healthy_survived[-100:]]),
        }

        for key, value in metrics.items():
            self.logger.record(key, value)

        if self.verbose > 0:
            print(f"\n[Infection Metrics @ step {self.num_timesteps}]")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")


class CurriculumCallback(BaseCallback):
    """
    Callback para implementar curriculum learning.

    Aumenta progresivamente la dificultad del entorno basándose en el rendimiento.
    """

    def __init__(
        self,
        env_configs: list,  # Lista de configuraciones de dificultad creciente
        performance_threshold: float = 0.7,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env_configs = env_configs
        self.performance_threshold = performance_threshold
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.current_level = 0
        self.level_performance = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_update()

        return True

    def _evaluate_and_update(self):
        """Evalúa el rendimiento y actualiza el nivel de dificultad."""
        # Evaluar rendimiento actual
        performance = self._evaluate_performance()
        self.level_performance.append(performance)

        if self.verbose > 0:
            print(f"\n[Curriculum @ step {self.num_timesteps}]")
            print(f"  Current level: {self.current_level + 1}/{len(self.env_configs)}")
            print(f"  Performance: {performance:.2%}")

        # Verificar si debe avanzar de nivel
        if performance >= self.performance_threshold:
            if self.current_level < len(self.env_configs) - 1:
                self.current_level += 1
                self._update_environment()

                if self.verbose > 0:
                    print(f"  → Advancing to level {self.current_level + 1}!")

        # Registrar en TensorBoard
        self.logger.record("curriculum/level", self.current_level + 1)
        self.logger.record("curriculum/performance", performance)

    def _evaluate_performance(self) -> float:
        """Evalúa el rendimiento actual como tasa de éxito."""
        successes = 0

        for _ in range(self.n_eval_episodes):
            obs, info = self.training_env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.training_env.step(action)
                done = terminated or truncated

            # Considerar éxito si el agente sobrevivió
            if "episode" in info:
                if info["episode"].get("healthy_survived", 0) > 0:
                    successes += 1

        return successes / self.n_eval_episodes

    def _update_environment(self):
        """Actualiza la configuración del entorno al nuevo nivel."""
        new_config = self.env_configs[self.current_level]

        # Actualizar configuración del entorno de entrenamiento
        if hasattr(self.training_env.unwrapped, "config"):
            for key, value in new_config.items():
                setattr(self.training_env.unwrapped.config, key, value)

        if self.verbose > 0:
            print(f"  Environment updated with config: {new_config}")


class SaveBestAgentCallback(BaseCallback):
    """
    Callback que guarda el mejor modelo para cada rol (healthy/infected).
    """

    def __init__(
        self,
        save_dir: str,
        role: str = "healthy",  # "healthy" o "infected"
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.best_mean_reward = -np.inf
        self.best_model_path = None

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_save()

        return True

    def _evaluate_and_save(self):
        """Evalúa y guarda si es el mejor modelo."""
        # Use SB3's evaluate_policy for proper VecEnv handling
        from stable_baselines3.common.evaluation import evaluate_policy

        try:
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.training_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_model_path = self.save_dir / f"best_{self.role}_model"
                self.model.save(str(self.best_model_path))

                if self.verbose > 0:
                    print(f"\n[New best {self.role} model saved!]")
                    print(f"  Mean reward: {mean_reward:.2f}")
                    print(f"  Path: {self.best_model_path}")

            self.logger.record(f"best_{self.role}/mean_reward", mean_reward)
            self.logger.record(f"best_{self.role}/best_so_far", self.best_mean_reward)
        except Exception as e:
            if self.verbose > 0:
                print(f"  Warning: Could not evaluate for best model: {e}")
