"""
Gymnasium Wrappers for Infection Environment
============================================
Wrappers para compatibilidad con Stable-Baselines3.

Incluye soporte para:
- Parameter Sharing (controlled_agent_id variable)
- Opponent Model (usar modelo entrenado como oponente)
- Configuración dinámica de mapas y agentes
"""

from typing import Dict, Any, Tuple, Optional, Union, Callable
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .environment import InfectionEnv, EnvConfig
from .reward_config import RewardConfig
from ..agents import BaseAgent, HealthyAgent, InfectedAgent, Direction


class FlattenObservationWrapper(gym.ObservationWrapper):
    """Convierte la observación dict a un vector plano."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        config = env.unwrapped.config
        view_size = config.view_size  # 15 con radius=7

        # Vista circular: 2 canales (tipo + distancia)
        image_size = view_size * view_size * 2
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
    mientras los demás siguen políticas heurísticas o un modelo oponente.

    Args:
        env: Entorno InfectionEnv
        controlled_agent_id: ID del agente controlado (para Parameter Sharing)
        force_role: Forzar rol "healthy" o "infected"
        opponent_model: Modelo para controlar agentes del bando contrario
            - Si es str/Path: carga el modelo desde archivo
            - Si es objeto: usa el modelo directamente
            - Si es None: usa heurística por defecto
        opponent_deterministic: Si usar predicciones determinísticas para oponente
    """

    def __init__(
        self,
        env: InfectionEnv,
        controlled_agent_id: int = 0,
        force_role: Optional[str] = None,
        opponent_model: Optional[Union[str, Path, Any]] = None,
        opponent_deterministic: bool = True,
    ):
        super().__init__(env)
        self.controlled_agent_id = controlled_agent_id
        self.force_role = force_role
        self.opponent_deterministic = opponent_deterministic

        # Cargar modelo oponente si se proporciona
        self._opponent_model = None
        self._opponent_role = None
        if opponent_model is not None:
            self._load_opponent_model(opponent_model)

    def _load_opponent_model(self, model: Union[str, Path, Any]) -> None:
        """Carga el modelo oponente."""
        if isinstance(model, (str, Path)):
            try:
                from stable_baselines3 import PPO
                self._opponent_model = PPO.load(str(model))
                print(f"Modelo oponente cargado: {model}")
            except Exception as e:
                print(f"Error cargando modelo oponente: {e}")
                self._opponent_model = None
        else:
            # Asumir que es un modelo ya cargado
            self._opponent_model = model

        # Determinar rol del oponente basado en force_role
        if self.force_role == "healthy":
            self._opponent_role = "infected"
        elif self.force_role == "infected":
            self._opponent_role = "healthy"

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
            healthy_list = list(agents.healthy)
            if healthy_list:
                other = healthy_list[0]
                controlled_pos = controlled.position
                other_pos = other.position
                controlled.position = other_pos
                other.position = controlled_pos

        elif self.force_role == "infected" and controlled.is_healthy:
            infected_list = list(agents.infected)
            if infected_list:
                other = infected_list[0]
                controlled_pos = controlled.position
                other_pos = other.position
                controlled.position = other_pos
                other.position = controlled_pos

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Ejecuta un paso con la acción del agente controlado.

        Si hay un modelo oponente, lo usa para predecir acciones del bando contrario.
        """
        if self.env.done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        self.env.current_step += 1

        # Guardar posiciones anteriores
        old_positions = {a.id: a.position for a in self.env.agents}

        # Ejecutar acción del agente controlado
        controlled_agent = self.env.agents.get(self.controlled_agent_id)
        if controlled_agent:
            self.env._execute_action(controlled_agent, action)

        # Ejecutar acciones de otros agentes
        for agent in self.env.agents:
            if agent.id != self.controlled_agent_id:
                other_action = self._get_other_agent_action(agent)
                self.env._execute_action(agent, other_action)

        # Actualizar contadores de movimiento
        for agent in self.env.agents:
            if agent.position == old_positions.get(agent.id):
                self.env._steps_in_same_cell[agent.id] = self.env._steps_in_same_cell.get(agent.id, 0) + 1
            else:
                self.env._steps_in_same_cell[agent.id] = 0

        self.env._last_positions = old_positions

        # Detectar y ejecutar infecciones
        new_infections = self.env._check_infections()

        # Obtener agente controlado (puede haber sido transformado)
        controlled_agent = self.env.agents.get(self.controlled_agent_id)

        # Calcular recompensa
        reward = self.env._calculate_reward(controlled_agent, new_infections, self.controlled_agent_id)

        # Actualizar estadísticas de agentes
        for agent in self.env.agents:
            agent.step()

        # Verificar terminación
        terminated = self.env._check_termination()
        truncated = self.env.current_step >= self.env.config.max_steps
        self.env.done = terminated or truncated

        if self.env.done:
            reward += self.env._calculate_episode_end_reward(controlled_agent, terminated)

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_other_agent_action(self, agent: BaseAgent) -> int:
        """
        Determina la acción de un agente no controlado.

        Usa el modelo oponente si está disponible y el agente es del rol oponente.
        De lo contrario, usa heurística.
        """
        # Verificar si debemos usar el modelo oponente
        if self._opponent_model is not None and self._opponent_role is not None:
            agent_role = "infected" if agent.is_infected else "healthy"

            if agent_role == self._opponent_role:
                return self._predict_opponent_action(agent)

        # Fallback a heurística
        return self._heuristic_action(agent)

    def _predict_opponent_action(self, agent: BaseAgent) -> int:
        """Predice la acción usando el modelo oponente."""
        try:
            # Obtener observación para el agente
            obs_dict = self.env._get_observation(agent)

            # Aplanar observación (el modelo espera observación plana)
            obs_flat = self._flatten_observation(obs_dict)

            # Predecir acción
            action, _ = self._opponent_model.predict(
                obs_flat,
                deterministic=self.opponent_deterministic
            )

            return int(action)

        except Exception as e:
            # Fallback a heurística si hay error
            return self._heuristic_action(agent)

    def _flatten_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Aplana una observación dict a vector."""
        config = self.env.config
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

    def _heuristic_action(self, agent: BaseAgent) -> int:
        """Acción heurística para un agente."""
        if isinstance(agent, InfectedAgent):
            target = self.env.agents.find_nearest_healthy(agent)
            if target:
                return self.env._move_towards(agent, target.position)
        else:
            threat = self.env.agents.find_nearest_infected(agent)
            if threat:
                return self.env._move_away_from(agent, threat.position)

        return self.env._np_random.integers(0, 4)

    def _get_obs(self) -> Dict[str, Any]:
        """Obtiene observación del agente controlado."""
        controlled = self.env.agents.get(self.controlled_agent_id)
        if controlled:
            return self.env._get_observation(controlled)
        return self.env._get_observation(self.env.agents.all[0])

    def _get_info(self) -> Dict[str, Any]:
        """Obtiene info del entorno."""
        return self.env._get_info()


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

                # Pasar las nuevas métricas estadísticas desde _get_info()
                info["episode"]["survival_rate"] = info.get("survival_rate", 0.0)
                info["episode"]["infected_percentage"] = info.get("infected_percentage", 0.0)
                info["episode"]["infection_count"] = info.get("infection_count", 0)

        return obs, reward, terminated, truncated, info


class MultiAgentToSingleAgentWrapper(gym.Wrapper):
    """Wrapper completo: SingleAgent + Flatten + Statistics."""

    def __init__(
        self,
        env: InfectionEnv,
        controlled_agent_id: int = 0,
        force_role: Optional[str] = None,
        flatten: bool = True,
        opponent_model: Optional[Union[str, Path, Any]] = None,
        opponent_deterministic: bool = True,
    ):
        env = SingleAgentWrapper(
            env,
            controlled_agent_id=controlled_agent_id,
            force_role=force_role,
            opponent_model=opponent_model,
            opponent_deterministic=opponent_deterministic,
        )

        if flatten:
            env = FlattenObservationWrapper(env)

        env = RecordEpisodeStatisticsWrapper(env)

        super().__init__(env)


def make_infection_env(
    map_file: str = None,
    map_data: str = None,
    num_agents: int = 15,
    initial_infected: int = 1,
    controlled_agent_id: int = 0,
    render_mode: str = None,
    flatten: bool = True,
    force_role: str = None,
    seed: int = None,
    opponent_model: Optional[Union[str, Path, Any]] = None,
    opponent_deterministic: bool = True,
    max_steps: int = 1000,
    view_radius: int = 7,
    reward_config: Optional[RewardConfig] = None,
    infected_speed: int = 1,
    infected_global_vision: bool = False,
) -> gym.Env:
    """
    Crea un entorno de infección configurado.

    Args:
        map_file: Archivo del mapa (si None, usa default)
        map_data: String con el contenido del mapa (tiene prioridad sobre map_file)
        num_agents: Número total de agentes
        initial_infected: Número de agentes infectados al inicio
        controlled_agent_id: ID del agente a controlar (para Parameter Sharing)
        render_mode: Modo de renderizado
        flatten: Si aplanar observaciones
        force_role: Forzar rol ("healthy" o "infected")
        seed: Semilla para reproducibilidad
        opponent_model: Modelo para agentes oponentes
        opponent_deterministic: Si usar predicciones determinísticas para oponente
        max_steps: Pasos máximos por episodio
        view_radius: Radio de visión circular (default=7, diámetro=15)
        reward_config: Configuración de rewards progresiva (opcional)
        infected_speed: Velocidad de infectados (celdas por movimiento, default=1)
        infected_global_vision: Si infectados ven todo el mapa (default=False)

    Returns:
        Entorno configurado compatible con SB3
    """
    config_kwargs = {
        "num_agents": num_agents,
        "initial_infected": initial_infected,
        "render_mode": render_mode,
        "seed": seed,
        "max_steps": max_steps,
        "view_radius": view_radius,
        "infected_speed": infected_speed,
        "infected_global_vision": infected_global_vision,
    }

    if map_data is not None:
        config_kwargs["map_data"] = map_data
    elif map_file is not None:
        config_kwargs["map_file"] = map_file

    if reward_config is not None:
        config_kwargs["reward_config"] = reward_config

    config = EnvConfig(**config_kwargs)

    env = InfectionEnv(config)
    env = MultiAgentToSingleAgentWrapper(
        env,
        controlled_agent_id=controlled_agent_id,
        force_role=force_role,
        flatten=flatten,
        opponent_model=opponent_model,
        opponent_deterministic=opponent_deterministic,
    )

    return env


def make_vec_env_parameter_sharing(
    map_file: str = None,
    map_data: str = None,
    num_agents: int = 10,
    initial_infected: int = 1,
    force_role: str = "healthy",
    n_envs: int = None,
    seed: int = None,
    opponent_model: Optional[Union[str, Path, Any]] = None,
    opponent_deterministic: bool = True,
    max_steps: int = 1000,
    vec_env_cls: str = "subproc",
    reward_config: Optional[RewardConfig] = None,
    infected_speed: int = 1,
    infected_global_vision: bool = False,
) -> Any:
    """
    Crea un VecEnv con Parameter Sharing.

    Cada entorno paralelo controla un agente diferente del mismo rol.
    Todos comparten la misma red neuronal.

    Args:
        map_file: Archivo del mapa
        map_data: String con el contenido del mapa (tiene prioridad sobre map_file)
        num_agents: Número total de agentes
        initial_infected: Número de infectados iniciales
        force_role: Rol a entrenar ("healthy" o "infected")
        n_envs: Número de entornos paralelos (default: número de agentes del rol)
        seed: Semilla base
        opponent_model: Modelo para oponentes
        opponent_deterministic: Si usar predicciones determinísticas
        max_steps: Pasos máximos
        vec_env_cls: Tipo de VecEnv ("subproc" o "dummy")
        reward_config: Configuración de rewards progresiva (opcional)
        infected_speed: Velocidad de infectados (celdas por movimiento, default=1)
        infected_global_vision: Si infectados ven todo el mapa (default=False)

    Returns:
        VecEnv configurado
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.vec_env import VecMonitor

    # Determinar número de entornos
    if force_role == "healthy":
        num_role_agents = num_agents - initial_infected
    else:  # infected
        num_role_agents = initial_infected

    if n_envs is None:
        n_envs = num_role_agents

    # Limitar n_envs al número de agentes del rol
    n_envs = min(n_envs, num_role_agents)

    def make_env(rank: int) -> Callable[[], gym.Env]:
        """
        Crea función factory para un entorno.

        El rank determina qué agente del rol se controla.
        """
        def _init() -> gym.Env:
            # Calcular controlled_agent_id basado en rol y rank
            if force_role == "healthy":
                # Los agentes healthy empiezan después de los infected
                controlled_id = initial_infected + rank
            else:  # infected
                controlled_id = rank

            env = make_infection_env(
                map_file=map_file,
                map_data=map_data,
                num_agents=num_agents,
                initial_infected=initial_infected,
                controlled_agent_id=controlled_id,
                force_role=force_role,
                seed=seed + rank if seed else None,
                opponent_model=opponent_model,
                opponent_deterministic=opponent_deterministic,
                max_steps=max_steps,
                flatten=True,
                reward_config=reward_config,
                infected_speed=infected_speed,
                infected_global_vision=infected_global_vision,
            )
            return env

        return _init

    # Crear lista de factories
    env_fns = [make_env(i) for i in range(n_envs)]

    # Crear VecEnv
    if vec_env_cls == "subproc":
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Añadir monitor
    vec_env = VecMonitor(vec_env)

    return vec_env
