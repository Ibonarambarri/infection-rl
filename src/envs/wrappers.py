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


class DictObservationWrapper(gym.ObservationWrapper):
    """
    Convierte la observación dict a formato compatible con MultiInputPolicy.

    MultiInputPolicy de SB3 detecta automáticamente:
    - Keys con shape (H, W, C) → CNN (NatureCNN)
    - Keys con shape (N,) → MLP

    Este wrapper:
    1. Mantiene 'image' para CNN (normalizada a [0,1])
    2. Agrupa features vectoriales (direction, state, position, nearby) en 'vector'
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        config = env.unwrapped.config
        view_size = config.view_size

        # Calcular tamaño del vector concatenado
        direction_size = 4  # one-hot
        state_size = 2  # one-hot
        position_size = 2
        nearby_size = config.max_nearby_agents * 4

        vector_size = direction_size + state_size + position_size + nearby_size

        # Observation space para MultiInputPolicy
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0,
                shape=(view_size, view_size, 2),
                dtype=np.float32
            ),
            "vector": spaces.Box(
                low=-1.0, high=1.0,
                shape=(vector_size,),
                dtype=np.float32
            ),
        })

        self._config = config

    def observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convierte observación a formato MultiInputPolicy."""
        # Imagen normalizada para CNN
        image = obs["image"].astype(np.float32) / 255.0

        # Construir vector de features
        parts = []

        # Direction one-hot
        direction = np.zeros(4, dtype=np.float32)
        direction[obs["direction"]] = 1.0
        parts.append(direction)

        # State one-hot
        state = np.zeros(2, dtype=np.float32)
        state[obs["state"]] = 1.0
        parts.append(state)

        # Position normalizada
        position = obs["position"].astype(np.float32) / 100.0  # Normalizar
        parts.append(position)

        # Nearby agents
        parts.append(obs["nearby_agents"].flatten().astype(np.float32))

        vector = np.concatenate(parts)

        return {
            "image": image,
            "vector": vector,
        }


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

        IMPORTANTE: Si force_role="healthy" y el agente es infectado durante este paso,
        se fuerza terminación local para cortar el flujo de experiencia y evitar que
        el agente aprenda de rewards de comportamiento infectado.
        """
        if self.env.done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        self.env.current_step += 1

        # Guardar posiciones anteriores
        old_positions = {a.id: a.position for a in self.env.agents}

        # Guardar estado de infección ANTES de ejecutar acciones
        controlled_agent = self.env.agents.get(self.controlled_agent_id)
        was_healthy = controlled_agent is not None and controlled_agent.is_healthy

        # Ejecutar acción del agente controlado
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

        # === DETECCIÓN DE CAMBIO DE ROL ===
        # Si force_role="healthy" y el agente ERA healthy pero AHORA está infectado,
        # forzar terminación local con penalización
        agent_was_infected = (
            self.force_role == "healthy" and
            was_healthy and
            controlled_agent is not None and
            controlled_agent.is_infected
        )

        if agent_was_infected:
            # Cortar flujo de experiencia: terminación forzada con penalización
            reward = self.env.config.reward_config.reward_infected_penalty
            terminated = True
            truncated = False

            # Actualizar estadísticas de agentes igualmente
            for agent in self.env.agents:
                agent.step()

            obs = self._get_obs()
            info = self._get_info()
            info["agent_infected"] = True  # Flag para debugging/logging

            return obs, reward, terminated, truncated, info

        # Calcular recompensa (flujo normal)
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
        """
        Predice la acción usando el modelo oponente.

        Procesa la observación cruda al formato exacto que espera MultiInputPolicy
        ({image, vector}) para garantizar consistencia con el entrenamiento.
        """
        try:
            # 1. Obtener observación cruda del entorno
            raw_obs = self.env._get_observation(agent)

            # 2. Procesar al formato MultiInputPolicy {image, vector}
            # Imagen: normalizar a [0, 1] como float32
            image = raw_obs["image"].astype(np.float32) / 255.0

            # Construir vector de features
            parts = []

            # Direction one-hot (4 elementos)
            direction = np.zeros(4, dtype=np.float32)
            direction[raw_obs["direction"]] = 1.0
            parts.append(direction)

            # State one-hot (2 elementos: 0=healthy, 1=infected)
            state = np.zeros(2, dtype=np.float32)
            state[raw_obs["state"]] = 1.0
            parts.append(state)

            # Position normalizada (dividir por 100.0 como en DictObservationWrapper)
            position = raw_obs["position"].astype(np.float32) / 100.0
            parts.append(position)

            # Nearby agents (flatten)
            nearby = raw_obs["nearby_agents"].flatten().astype(np.float32)
            parts.append(nearby)

            vector = np.concatenate(parts)

            processed_obs = {
                "image": image,
                "vector": vector,
            }

            # 3. Predecir con el modelo oponente
            action, _ = self._opponent_model.predict(
                processed_obs,
                deterministic=self.opponent_deterministic
            )

            return int(action)

        except Exception as e:
            # Fallback a heurística si hay error (ej. desajuste de dimensiones)
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
        """
        Acción heurística mejorada usando BFS para rodear obstáculos.

        En lugar de moverse directamente hacia el objetivo (lo cual causa
        que los agentes se atasquen en paredes), usa BFS para encontrar
        el camino real y da el primer paso de ese camino.
        """
        if isinstance(agent, InfectedAgent):
            target = self.env.agents.find_nearest_healthy(agent)
            if target:
                return self._bfs_move_towards(agent, target.position)
        else:
            threat = self.env.agents.find_nearest_infected(agent)
            if threat:
                return self._bfs_move_away_from(agent, threat.position)

        return self.env._np_random.integers(0, 4)

    def _bfs_move_towards(self, agent: BaseAgent, target_pos: Tuple[int, int]) -> int:
        """
        Usa BFS para encontrar el siguiente paso hacia el objetivo,
        rodeando obstáculos y otros agentes dinámicos.
        """
        from collections import deque

        start = agent.position
        goal = target_pos

        if start == goal:
            return self.env._np_random.integers(0, 4)

        # BFS para encontrar el camino
        queue = deque([(start[0], start[1], [])])  # (x, y, path)
        visited = {start}

        # Direcciones: UP=0, RIGHT=1, DOWN=2, LEFT=3 en términos de movimiento
        # Pero las acciones son: 0=turn_left, 1=turn_right, 2=forward, 3=backward
        directions = [
            (0, -1),   # UP (dy=-1)
            (1, 0),    # RIGHT (dx=+1)
            (0, 1),    # DOWN (dy=+1)
            (-1, 0),   # LEFT (dx=-1)
        ]

        max_search = min(100, self.env.width * self.env.height // 4)

        while queue and len(visited) < max_search:
            x, y, path = queue.popleft()

            for dir_idx, (dx, dy) in enumerate(directions):
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.env.width and 0 <= ny < self.env.height):
                    continue
                if (nx, ny) in visited:
                    continue
                # Verificar si es una celda válida (no muro)
                if not self.env.map_generator.is_valid_position(nx, ny):
                    continue
                # Verificar si hay otro agente ocupando la celda (excepto el objetivo)
                if (nx, ny) != goal and self.env.agents.get_agent_at((nx, ny)) is not None:
                    continue

                new_path = path + [(dx, dy)]

                if (nx, ny) == goal:
                    # Encontramos el camino, convertir primer paso a acción
                    return self._direction_to_action(agent, new_path[0])

                visited.add((nx, ny))
                queue.append((nx, ny, new_path))

        # Si no hay camino, movimiento aleatorio como fallback
        return self.env._np_random.integers(0, 4)

    def _bfs_move_away_from(self, agent: BaseAgent, threat_pos: Tuple[int, int]) -> int:
        """
        Evalúa direcciones posibles para huir,
        maximizando la distancia al enemigo y evitando otros agentes.
        """
        start = agent.position

        # Evaluar cada dirección posible y elegir la que maximice distancia
        directions = [
            (0, -1),   # UP
            (1, 0),    # RIGHT
            (0, 1),    # DOWN
            (-1, 0),   # LEFT
        ]

        best_action = None
        best_distance = -1

        for dx, dy in directions:
            nx, ny = start[0] + dx, start[1] + dy

            if not (0 <= nx < self.env.width and 0 <= ny < self.env.height):
                continue
            # Verificar si es una celda válida (no muro)
            if not self.env.map_generator.is_valid_position(nx, ny):
                continue
            # Verificar si hay otro agente ocupando la celda
            if self.env.agents.get_agent_at((nx, ny)) is not None:
                continue

            # Calcular distancia Manhattan al enemigo desde esta posición
            dist = abs(nx - threat_pos[0]) + abs(ny - threat_pos[1])

            if dist > best_distance:
                best_distance = dist
                best_action = self._direction_to_action(agent, (dx, dy))

        if best_action is not None:
            return best_action

        # Fallback: movimiento aleatorio
        return self.env._np_random.integers(0, 4)

    def _direction_to_action(self, agent: BaseAgent, direction: Tuple[int, int]) -> int:
        """
        Convierte una dirección de movimiento (dx, dy) en una acción del agente.

        Acciones: 0=turn_left, 1=turn_right, 2=forward, 3=backward
        """
        dx, dy = direction

        # Determinar la dirección objetivo
        if dy < 0:
            target_dir = Direction.UP
        elif dy > 0:
            target_dir = Direction.DOWN
        elif dx > 0:
            target_dir = Direction.RIGHT
        else:
            target_dir = Direction.LEFT

        current_dir = agent.direction

        # Si ya está mirando en la dirección correcta, avanzar
        if current_dir == target_dir:
            return 2  # forward

        # Calcular la rotación necesaria
        diff = (target_dir.value - current_dir.value) % 4

        if diff == 1:
            return 1  # turn_right
        elif diff == 3:
            return 0  # turn_left
        elif diff == 2:
            # Necesita dar la vuelta: elegir aleatoriamente izq o derecha
            return self.env._np_random.choice([0, 1])

        return 2  # forward por defecto

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
    """Wrapper completo: SingleAgent + Dict/Flatten + Statistics."""

    def __init__(
        self,
        env: InfectionEnv,
        controlled_agent_id: int = 0,
        force_role: Optional[str] = None,
        flatten: bool = False,
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
            # MlpPolicy: observación aplanada
            env = FlattenObservationWrapper(env)
        else:
            # MultiInputPolicy: imagen (CNN) + vector (MLP)
            env = DictObservationWrapper(env)

        env = RecordEpisodeStatisticsWrapper(env)

        super().__init__(env)


def make_infection_env(
    map_file: str = None,
    map_data: str = None,
    num_agents: int = 15,
    initial_infected: int = 1,
    controlled_agent_id: int = 0,
    render_mode: str = None,
    flatten: bool = False,
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
                flatten=False,
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
