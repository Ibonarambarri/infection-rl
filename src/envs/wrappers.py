"""
Gymnasium Wrappers for Infection Environment
============================================
Wrappers para compatibilidad con Stable-Baselines3.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .environment import InfectionEnv, EnvConfig
from ..agents import BaseAgent, HealthyAgent, InfectedAgent, Direction


class FlattenObservationWrapper(gym.ObservationWrapper):
    """Convierte la observación dict a un vector plano."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        config = env.unwrapped.config
        view_size = config.view_size

        image_size = view_size * view_size * 3
        direction_size = 4
        state_size = 2
        position_size = 2
        nearby_size = config.max_nearby_agents * 4

        total_size = image_size + direction_size + state_size + position_size + nearby_size

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_size,),
            dtype=np.float32
        )

        self._config = config

    def observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Aplana la observación."""
        parts = []

        image = obs["image"].astype(np.float32) / 255.0
        parts.append(image.flatten())

        direction = np.zeros(4, dtype=np.float32)
        direction[obs["direction"]] = 1.0
        parts.append(direction)

        state = np.zeros(2, dtype=np.float32)
        state[obs["state"]] = 1.0
        parts.append(state)

        parts.append(obs["position"])
        parts.append(obs["nearby_agents"].flatten())

        return np.concatenate(parts)


class SingleAgentWrapper(gym.Wrapper):
    """
    Convierte el entorno multi-agente en single-agent.

    Permite entrenar un agente específico (HealthyAgent o InfectedAgent)
    mientras los demás siguen políticas heurísticas.
    """

    def __init__(
        self,
        env: InfectionEnv,
        controlled_agent_id: int = 0,
        force_role: Optional[str] = None,  # "healthy" o "infected"
    ):
        super().__init__(env)
        self.controlled_agent_id = controlled_agent_id
        self.force_role = force_role

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reinicia el entorno y asegura el rol del agente controlado."""
        obs, info = self.env.reset(**kwargs)

        if self.force_role:
            self._ensure_role()

        return self._get_obs(), info

    def _ensure_role(self):
        """Asegura que el agente controlado tenga el rol correcto."""
        agents = self.env.agents
        controlled = agents.get(self.controlled_agent_id)

        if controlled is None:
            return

        if self.force_role == "healthy" and controlled.is_infected:
            # El agente 0 es infectado pero queremos healthy
            # Intercambiar posiciones con un healthy
            healthy_list = agents.healthy
            if healthy_list:
                other = healthy_list[0]
                # Intercambiar posiciones
                controlled_pos = controlled.position
                other_pos = other.position
                controlled.position = other_pos
                other.position = controlled_pos

        elif self.force_role == "infected" and controlled.is_healthy:
            # El agente 0 es healthy pero queremos infected
            infected_list = agents.infected
            if infected_list:
                other = infected_list[0]
                controlled_pos = controlled.position
                other_pos = other.position
                controlled.position = other_pos
                other.position = controlled_pos

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Ejecuta un paso con la acción del agente controlado."""
        obs, reward, terminated, truncated, info = self.env.step(
            action, agent_id=self.controlled_agent_id
        )
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> Dict[str, Any]:
        """Obtiene observación del agente controlado."""
        controlled = self.env.agents.get(self.controlled_agent_id)
        if controlled:
            return self.env._get_observation(controlled)
        return self.env._get_observation(self.env.agents.all[0])


class RecordEpisodeStatisticsWrapper(gym.Wrapper):
    """Registra estadísticas de cada episodio."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_returns = []
        self.episode_lengths = []
        self.current_return = 0.0
        self.current_length = 0

    def reset(self, **kwargs):
        if self.current_length > 0:
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)

        self.current_return = 0.0
        self.current_length = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.current_return += reward
        self.current_length += 1

        info["episode_return"] = self.current_return
        info["episode_length"] = self.current_length

        if terminated or truncated:
            info["episode"] = {
                "r": self.current_return,
                "l": self.current_length,
            }

            if hasattr(self.env.unwrapped, "agents"):
                env = self.env.unwrapped
                info["episode"]["healthy_survived"] = env.num_healthy
                info["episode"]["total_infected"] = env.num_infected
                info["episode"]["infection_events"] = len(env.infection_events)

        return obs, reward, terminated, truncated, info


class MultiAgentToSingleAgentWrapper(gym.Wrapper):
    """Wrapper completo: SingleAgent + Flatten + Statistics."""

    def __init__(
        self,
        env: InfectionEnv,
        controlled_agent_id: int = 0,
        force_role: Optional[str] = None,
        flatten: bool = True,
    ):
        env = SingleAgentWrapper(
            env,
            controlled_agent_id=controlled_agent_id,
            force_role=force_role,
        )

        if flatten:
            env = FlattenObservationWrapper(env)

        env = RecordEpisodeStatisticsWrapper(env)

        super().__init__(env)


def make_infection_env(
    num_agents: int = 15,
    render_mode: str = None,
    flatten: bool = True,
    force_role: str = None,
    seed: int = None,
) -> gym.Env:
    """
    Crea un entorno de infección configurado.

    Args:
        num_agents: Número de agentes
        render_mode: Modo de renderizado
        flatten: Si aplanar observaciones
        force_role: Forzar rol ("healthy" o "infected")
        seed: Semilla

    Returns:
        Entorno configurado
    """
    config = EnvConfig(
        num_agents=num_agents,
        render_mode=render_mode,
        seed=seed,
    )

    env = InfectionEnv(config)
    env = MultiAgentToSingleAgentWrapper(
        env,
        force_role=force_role,
        flatten=flatten,
    )

    return env
