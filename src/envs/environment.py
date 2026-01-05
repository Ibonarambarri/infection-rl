"""
Multi-Agent Infection Environment
=================================
Entorno de RL donde agentes sanos huyen de agentes infectados.

Usa clases separadas para HealthyAgent e InfectedAgent.
Cuando un agente sano es infectado, se transforma en InfectedAgent.
"""

from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..agents import BaseAgent, HealthyAgent, InfectedAgent, AgentCollection, Direction
from .map_generator import MapGenerator, MapConfig, CellType
from .reward_config import RewardConfig, RewardPreset

# Mapa por defecto (hardcodeado)
DEFAULT_MAP_FILE = "maps/vacio_60x60.txt"


@dataclass
class EnvConfig:
    """Configuración del entorno de infección."""
    # Agentes
    num_agents: int = 15
    initial_infected: int = 1

    # Mapa (se puede cargar desde archivo o string)
    map_file: Optional[str] = DEFAULT_MAP_FILE
    map_data: Optional[str] = None  # String con el contenido del mapa (tiene prioridad)

    # Mecánicas
    infection_radius: int = 1
    max_steps: int = 1000

    # Vista circular del agente
    view_radius: int = 7  # Radio de visión (diámetro = 2*radius + 1 = 15)
    max_nearby_agents: int = 8

    # Ventajas para infectados (para balancear el juego)
    infected_speed: int = 1  # Celdas por movimiento (1 = normal, 2 = doble velocidad)
    infected_global_vision: bool = False  # Si True, infectados ven todo el mapa

    @property
    def view_size(self) -> int:
        """Tamaño de la vista (diámetro)."""
        return self.view_radius * 2 + 1

    # Configuracion de rewards progresiva (opcional)
    # Si se proporciona, tiene prioridad sobre los campos individuales
    reward_config: Optional[RewardConfig] = None

    # Recompensas para agentes SANOS (HealthyAgent) - fallback si no hay reward_config
    reward_survive_step: float = 0.1
    reward_distance_bonus: float = 0.1
    reward_infected_penalty: float = -10.0
    reward_not_moving_penalty: float = -0.05
    reward_stuck_penalty: float = -0.2
    stuck_threshold: int = 3
    reward_survive_episode: float = 5.0

    # Recompensas para agentes INFECTADOS (InfectedAgent) - fallback si no hay reward_config
    reward_infect_agent: float = 15.0
    reward_approach_bonus: float = 0.2
    reward_step_penalty: float = -0.01
    reward_all_infected_bonus: float = 20.0
    reward_exploration: float = 0.02

    # Otros
    seed: Optional[int] = None
    render_mode: Optional[str] = None
    fixed_map: bool = True

    def __post_init__(self):
        """Inicializa reward_config si no se proporciona (backwards compatibility)."""
        if self.reward_config is None:
            # Crear RewardConfig desde los valores individuales (DENSE por defecto)
            self.reward_config = RewardConfig(
                preset=RewardPreset.DENSE,
                reward_survive_step=self.reward_survive_step,
                reward_distance_bonus=self.reward_distance_bonus,
                reward_infected_penalty=self.reward_infected_penalty,
                reward_not_moving_penalty=self.reward_not_moving_penalty,
                reward_stuck_penalty=self.reward_stuck_penalty,
                stuck_threshold=self.stuck_threshold,
                reward_survive_episode=self.reward_survive_episode,
                reward_infect_agent=self.reward_infect_agent,
                reward_approach_bonus=self.reward_approach_bonus,
                reward_step_penalty=self.reward_step_penalty,
                reward_all_infected_bonus=self.reward_all_infected_bonus,
                reward_exploration=self.reward_exploration,
            )


class InfectionEnv(gym.Env):
    """
    Entorno Multi-Agente de Infección.

    Usa HealthyAgent e InfectedAgent como clases separadas.
    Cuando un HealthyAgent es infectado, se transforma en InfectedAgent.

    Acciones:
        0: Girar izquierda
        1: Girar derecha
        2: Avanzar
        3: Quedarse quieto
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}

    ACTIONS = {
        0: "turn_left",
        1: "turn_right",
        2: "forward",
        3: "stay",
    }

    def __init__(self, config: EnvConfig = None, **kwargs):
        super().__init__()

        if config is None:
            config = EnvConfig(**kwargs)
        self.config = config

        self._np_random = np.random.default_rng(config.seed)

        self.map_config = MapConfig(
            map_file=config.map_file,
            map_data=config.map_data,
            seed=config.seed,
        )
        self.map_generator: MapGenerator = None
        self.grid: np.ndarray = None
        self._fixed_grid: np.ndarray = None

        self.width = 60
        self.height = 60

        # Colección de agentes (HealthyAgent e InfectedAgent)
        self.agents = AgentCollection()

        self.current_step = 0
        self.done = False
        self.infection_events: List[Dict] = []

        self.render_mode = config.render_mode
        self._window = None
        self._clock = None

        self.action_space = spaces.Discrete(4)

        # Vista circular: (view_size, view_size, 2) donde view_size = 2*radius + 1
        view_size = config.view_size  # 15 con radius=7
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(view_size, view_size, 2),  # 2 canales: tipo + distancia
                dtype=np.uint8
            ),
            "direction": spaces.Discrete(4),
            "state": spaces.Discrete(2),  # 0=healthy, 1=infected
            "position": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
            "nearby_agents": spaces.Box(
                low=-1, high=1,
                shape=(config.max_nearby_agents, 4),
                dtype=np.float32
            ),
        })

        self._last_positions: Dict[int, Tuple[int, int]] = {}
        self._steps_in_same_cell: Dict[int, int] = {}
        self._prev_distances: Dict[int, float] = {}  # Para tracking de progreso de infected

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reinicia el entorno."""
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self.map_config.seed = seed

        # Generar mapa
        if self.config.fixed_map and self._fixed_grid is not None:
            self.grid = self._fixed_grid.copy()
        else:
            self.map_generator = MapGenerator(self.map_config)
            self.grid = self.map_generator.grid.copy()
            self.width = self.map_config.width
            self.height = self.map_config.height
            if self.config.fixed_map:
                self._fixed_grid = self.grid.copy()

        # Limpiar y crear agentes
        self.agents.clear()

        valid_positions = self.map_generator.get_random_valid_positions(
            self.config.num_agents
        )

        # Crear agentes: primero infectados, luego sanos
        for i, pos in enumerate(valid_positions):
            direction = Direction(self._np_random.integers(0, 4))

            if i < self.config.initial_infected:
                # Crear InfectedAgent
                self.agents.add_infected(pos, direction, infection_time=0)
            else:
                # Crear HealthyAgent
                self.agents.add_healthy(pos, direction)

        # Reiniciar estado
        self.current_step = 0
        self.done = False
        self.infection_events = []
        self._last_positions = {a.id: a.position for a in self.agents}
        self._steps_in_same_cell = {a.id: 0 for a in self.agents}
        self._prev_distances = {}  # Reset tracking de distancias

        obs = self._get_observation(self.agents.all[0])
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int,
        agent_id: int = 0
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Ejecuta un paso en el entorno."""
        if self.done:
            return self._get_observation(self.agents[agent_id]), 0.0, True, False, self._get_info()

        self.current_step += 1

        # Guardar posiciones anteriores
        old_positions = {a.id: a.position for a in self.agents}

        # Ejecutar acción del agente controlado
        controlled_agent = self.agents.get(agent_id)
        if controlled_agent:
            self._execute_action(controlled_agent, action)

        # Ejecutar acciones de otros agentes
        for agent in self.agents:
            if agent.id != agent_id:
                other_action = self._get_other_agent_action(agent)
                self._execute_action(agent, other_action)

        # Actualizar contadores de movimiento
        for agent in self.agents:
            if agent.position == old_positions.get(agent.id):
                self._steps_in_same_cell[agent.id] = self._steps_in_same_cell.get(agent.id, 0) + 1
            else:
                self._steps_in_same_cell[agent.id] = 0

        self._last_positions = old_positions

        # Detectar y ejecutar infecciones (transforma HealthyAgent -> InfectedAgent)
        new_infections = self._check_infections()

        # Obtener agente controlado (puede haber sido transformado)
        controlled_agent = self.agents.get(agent_id)

        # Calcular recompensa
        reward = self._calculate_reward(controlled_agent, new_infections, agent_id)

        # Actualizar estadísticas de agentes
        for agent in self.agents:
            agent.step()

        # Verificar terminación
        terminated = self._check_termination()
        truncated = self.current_step >= self.config.max_steps
        self.done = terminated or truncated

        if self.done:
            reward += self._calculate_episode_end_reward(controlled_agent, terminated)

        obs = self._get_observation(controlled_agent)
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def step_all(self, actions: Dict[int, int]) -> Tuple[Dict[int, Dict], Dict[int, float], bool, bool, Dict[str, Any]]:
        """
        Ejecuta un paso para todos los agentes (multi-agent).

        Args:
            actions: Diccionario {agent_id: action}

        Returns:
            (observations, rewards, terminated, truncated, info)
        """
        if self.done:
            obs = {a.id: self._get_observation(a) for a in self.agents}
            rewards = {a.id: 0.0 for a in self.agents}
            return obs, rewards, True, False, self._get_info()

        self.current_step += 1
        old_positions = {a.id: a.position for a in self.agents}

        # Ejecutar acciones en orden: HEALTHY primero, luego INFECTED
        # Esto permite que los sanos liberen casillas para que los infectados puedan ocuparlas
        healthy_actions = []
        infected_actions = []

        for agent_id, action in actions.items():
            agent = self.agents.get(agent_id)
            if agent:
                if agent.is_healthy:
                    healthy_actions.append((agent, action))
                else:
                    infected_actions.append((agent, action))

        # Primero ejecutar acciones de agentes HEALTHY
        for agent, action in healthy_actions:
            self._execute_action(agent, action)

        # Después ejecutar acciones de agentes INFECTED
        for agent, action in infected_actions:
            self._execute_action(agent, action)

        # Actualizar contadores
        for agent in self.agents:
            if agent.position == old_positions.get(agent.id):
                self._steps_in_same_cell[agent.id] = self._steps_in_same_cell.get(agent.id, 0) + 1
            else:
                self._steps_in_same_cell[agent.id] = 0

        self._last_positions = old_positions

        # Detectar infecciones
        new_infections = self._check_infections()

        # Calcular recompensas
        rewards = {}
        for agent in self.agents:
            rewards[agent.id] = self._calculate_reward(agent, new_infections, agent.id)
            agent.step()

        # Verificar terminación
        terminated = self._check_termination()
        truncated = self.current_step >= self.config.max_steps
        self.done = terminated or truncated

        obs = {a.id: self._get_observation(a) for a in self.agents}
        info = self._get_info()

        return obs, rewards, terminated, truncated, info

    def _execute_action(self, agent: BaseAgent, action: int) -> None:
        """Ejecuta una acción para un agente."""
        if action == 0:
            agent.turn_left()
        elif action == 1:
            agent.turn_right()
        elif action == 2:
            # Determinar velocidad (infectados pueden moverse más rápido)
            speed = self.config.infected_speed if agent.is_infected else 1
            for _ in range(speed):
                new_x, new_y = agent.move_forward()
                if self._is_valid_move(new_x, new_y, agent.id):
                    agent.set_position(new_x, new_y)
                else:
                    break  # Parar si hay obstáculo

    def _is_valid_move(self, x: int, y: int, agent_id: int) -> bool:
        """Verifica si un movimiento es válido."""
        if not self.map_generator.is_valid_position(x, y):
            return False

        for agent in self.agents:
            if agent.id != agent_id and agent.position == (x, y):
                return False

        return True

    def _check_infections(self) -> List[int]:
        """
        Verifica y ejecuta infecciones.
        Transforma HealthyAgent en InfectedAgent.

        Returns:
            Lista de IDs de agentes recién infectados
        """
        new_infections = []

        infected_agents = self.agents.infected
        healthy_agents = list(self.agents.healthy)  # Copia para iterar

        for infected in infected_agents:
            for healthy in healthy_agents:
                if healthy.id in new_infections:
                    continue

                distance = infected.distance_to(healthy)
                if distance <= self.config.infection_radius:
                    # Transformar HealthyAgent -> InfectedAgent
                    new_infected = self.agents.infect_agent(
                        healthy.id,
                        self.current_step,
                        infected_by=infected.id
                    )

                    if new_infected:
                        new_infections.append(healthy.id)

                        self.infection_events.append({
                            "step": self.current_step,
                            "infected_by": infected.id,
                            "newly_infected": healthy.id,
                            "position": healthy.position,
                        })

        return new_infections

    def _check_termination(self) -> bool:
        """Termina si todos están infectados."""
        return self.agents.num_healthy == 0

    def _calculate_reward(self, agent: BaseAgent, new_infections: List[int], original_id: int) -> float:
        """Calcula la recompensa para un agente."""
        rc = self.config.reward_config
        reward = 0.0

        # Si el agente era sano y fue infectado
        if original_id in new_infections:
            reward = rc.reward_infected_penalty
        elif isinstance(agent, HealthyAgent):
            reward = self._calculate_healthy_reward(agent)
        elif isinstance(agent, InfectedAgent):
            reward = self._calculate_infected_reward(agent, new_infections)

        agent.add_reward(reward)
        return reward

    def _calculate_healthy_reward(self, agent: HealthyAgent) -> float:
        """Calcula recompensa para HealthyAgent usando distancia Manhattan."""
        rc = self.config.reward_config
        reward = rc.reward_survive_step

        # Bonus por distancia a infectados (usando distancia Manhattan)
        if rc.reward_distance_bonus > 0:
            nearest_infected = self.agents.find_nearest_infected(agent)
            if nearest_infected:
                # Usar distancia Manhattan (más eficiente que BFS)
                distance = agent.distance_to(nearest_infected)
                max_dist = self.width + self.height
                # Normalizar: más distancia = más bonus
                distance_ratio = min(distance / max_dist, 1.0)
                reward += rc.reward_distance_bonus * distance_ratio

        # Penalización por no moverse (solo si activo)
        steps_stuck = self._steps_in_same_cell.get(agent.id, 0)
        if steps_stuck > 0 and rc.reward_not_moving_penalty < 0:
            reward += rc.reward_not_moving_penalty
            if steps_stuck >= rc.stuck_threshold and rc.reward_stuck_penalty < 0:
                multiplier = min(steps_stuck - rc.stuck_threshold + 1, 5)
                reward += rc.reward_stuck_penalty * multiplier

        return reward

    def _calculate_infected_reward(self, agent: InfectedAgent, new_infections: List[int]) -> float:
        """
        Calcula recompensa para InfectedAgent usando distancia Manhattan.

        El sistema premia:
        - Infectar agentes (reward_infect_agent)
        - Proximidad al healthy más cercano (reward_approach_bonus)
        - Progreso: reducir la distancia Manhattan (reward_progress_bonus)

        Y penaliza:
        - No hacer progreso (reward_no_progress_penalty)
        - Quedarse quieto (reward_not_moving_penalty)
        """
        rc = self.config.reward_config
        reward = 0.0

        # Recompensa por infectar
        for event in self.infection_events:
            if event["step"] == self.current_step and event["infected_by"] == agent.id:
                reward += rc.reward_infect_agent

        # Calcular distancia Manhattan al healthy mas cercano
        nearest_healthy = self.agents.find_nearest_healthy(agent)
        if nearest_healthy:
            # Usar distancia Manhattan (más eficiente que BFS)
            current_distance = agent.distance_to(nearest_healthy)
            max_dist = self.width + self.height

            # Bonus por proximidad (basado en distancia Manhattan)
            if rc.reward_approach_bonus > 0:
                proximity = max(0, (max_dist - current_distance) / max_dist)
                reward += rc.reward_approach_bonus * proximity

            # Bonus/penalty por PROGRESO en distancia
            prev_distance = self._prev_distances.get(agent.id)
            if prev_distance is not None:
                progress = prev_distance - current_distance  # Positivo = se acercó

                if progress > 0:
                    # Bonus por acercarse (premiar cada paso que reduce distancia)
                    progress_bonus = getattr(rc, 'reward_progress_bonus', 0.0)
                    if progress_bonus > 0:
                        reward += progress_bonus * progress
                elif progress < 0:
                    # Penalty por alejarse
                    no_progress_penalty = getattr(rc, 'reward_no_progress_penalty', 0.0)
                    if no_progress_penalty < 0:
                        reward += no_progress_penalty
                else:
                    # progress == 0: no se movió hacia el objetivo
                    no_progress_penalty = getattr(rc, 'reward_no_progress_penalty', 0.0)
                    if no_progress_penalty < 0:
                        reward += no_progress_penalty * 0.5  # Menor penalización

            # Guardar distancia actual para siguiente step
            self._prev_distances[agent.id] = current_distance

        elif rc.reward_exploration > 0:
            # Bonus de exploracion si no hay healthy
            if self._steps_in_same_cell.get(agent.id, 0) == 0:
                reward += rc.reward_exploration

        # Penalizacion por step (solo si activo)
        if rc.reward_step_penalty < 0:
            reward += rc.reward_step_penalty

        # Penalizacion por no moverse (solo si activo)
        steps_stuck = self._steps_in_same_cell.get(agent.id, 0)
        if steps_stuck > 0 and rc.reward_not_moving_penalty < 0:
            reward += rc.reward_not_moving_penalty
            if steps_stuck >= rc.stuck_threshold and rc.reward_stuck_penalty < 0:
                multiplier = min(steps_stuck - rc.stuck_threshold + 1, 5)
                reward += rc.reward_stuck_penalty * multiplier

        return reward

    def _calculate_episode_end_reward(self, agent: BaseAgent, all_infected: bool) -> float:
        """Calcula bonus de fin de episodio."""
        rc = self.config.reward_config
        if isinstance(agent, HealthyAgent):
            return rc.reward_survive_episode
        elif isinstance(agent, InfectedAgent) and all_infected:
            return rc.reward_all_infected_bonus
        return 0.0

    def _get_other_agent_action(self, agent: BaseAgent) -> int:
        """Determina acción de agentes no controlados (heurística)."""
        if isinstance(agent, InfectedAgent):
            target = self.agents.find_nearest_healthy(agent)
            if target:
                return self._move_towards(agent, target.position)
        else:
            threat = self.agents.find_nearest_infected(agent)
            if threat:
                return self._move_away_from(agent, threat.position)

        return self._np_random.integers(0, 4)

    def _move_towards(self, agent: BaseAgent, target_pos: Tuple[int, int]) -> int:
        """Acción para moverse hacia un objetivo."""
        dx = target_pos[0] - agent.x
        dy = target_pos[1] - agent.y

        if abs(dx) > abs(dy):
            desired_dir = Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            desired_dir = Direction.DOWN if dy > 0 else Direction.UP

        if agent.direction == desired_dir:
            new_x, new_y = agent.move_forward()
            if self._is_valid_move(new_x, new_y, agent.id):
                return 2
            return self._np_random.choice([0, 1])

        diff = (desired_dir - agent.direction) % 4
        if diff == 1:
            return 1
        elif diff == 3:
            return 0
        else:
            return self._np_random.choice([0, 1])

    def _move_away_from(self, agent: BaseAgent, threat_pos: Tuple[int, int]) -> int:
        """Acción para alejarse de una amenaza."""
        dx = agent.x - threat_pos[0]
        dy = agent.y - threat_pos[1]

        if abs(dx) > abs(dy):
            desired_dir = Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            desired_dir = Direction.DOWN if dy > 0 else Direction.UP

        if agent.direction == desired_dir:
            new_x, new_y = agent.move_forward()
            if self._is_valid_move(new_x, new_y, agent.id):
                return 2
            return self._np_random.choice([0, 1])

        diff = (desired_dir - agent.direction) % 4
        if diff == 1:
            return 1
        elif diff == 3:
            return 0
        else:
            return self._np_random.choice([0, 1])

    def _get_visible_agents(self, agent: BaseAgent, infected_only: bool = False, healthy_only: bool = False) -> List[BaseAgent]:
        """Obtiene agentes visibles en el campo de visión."""
        visible_cells = set(self._get_visible_cells(agent))
        visible_agents = []

        for other in self.agents:
            if other.id == agent.id:
                continue
            if other.position in visible_cells:
                if infected_only and not other.is_infected:
                    continue
                if healthy_only and other.is_infected:
                    continue
                visible_agents.append(other)

        return visible_agents

    def _get_visible_cells(self, agent: BaseAgent) -> List[Tuple[int, int]]:
        """Obtiene celdas visibles para un agente."""
        view_size = self.config.view_size
        half_width = view_size // 2
        visible = []

        for vy in range(view_size):
            for vx in range(view_size):
                rel_x = vx - half_width
                forward_dist = (view_size - 1) - vy

                world_dx, world_dy = self._rotate_coords_forward(rel_x, forward_dist, agent.direction)
                world_x = agent.x + world_dx
                world_y = agent.y + world_dy

                if 0 <= world_x < self.width and 0 <= world_y < self.height:
                    visible.append((world_x, world_y))

        return visible

    def _rotate_coords_forward(self, rel_x: int, forward: int, direction: Direction) -> Tuple[int, int]:
        """Rota coordenadas según dirección."""
        if direction == Direction.UP:
            return (rel_x, -forward)
        elif direction == Direction.DOWN:
            return (-rel_x, forward)
        elif direction == Direction.RIGHT:
            return (forward, rel_x)
        else:
            return (-forward, -rel_x)

    def _get_pathfinding_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """
        Calcula distancia REAL usando BFS (considerando obstáculos).

        SIEMPRE retorna un valor (nunca None):
        - Si hay camino: retorna distancia real
        - Si no hay camino: retorna distancia muy grande (penaliza fuertemente)

        Esto permite que el reward premie encontrar el camino más corto
        rodeando obstáculos.
        """
        if start == goal:
            return 0

        max_dist = self.width + self.height  # Máxima distancia posible
        queue = deque([(start[0], start[1], 0)])
        visited = {start}
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while queue:
            x, y, dist = queue.popleft()

            # Limitar búsqueda para evitar lag en mapas grandes
            if dist > max_dist:
                break

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in visited:
                    continue
                if self.grid[ny, nx] != CellType.EMPTY.value:
                    continue

                if (nx, ny) == goal:
                    return dist + 1

                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))

        # No hay camino -> retornar distancia máxima (muy penalizado)
        return max_dist * 2

    def _get_observation(self, agent: BaseAgent) -> Dict[str, Any]:
        """Genera la observación para un agente."""
        image = self._get_circular_view(agent)
        nearby = self._get_nearby_agents_info(agent)

        pos = np.array([
            agent.x / self.width,
            agent.y / self.height
        ], dtype=np.float32)

        # state: 0 = healthy, 1 = infected
        state = 1 if agent.is_infected else 0

        return {
            "image": image,
            "direction": int(agent.direction),
            "state": state,
            "position": pos,
            "nearby_agents": nearby,
        }

    def _get_circular_view(self, agent: BaseAgent) -> np.ndarray:
        """
        Genera vista circular centrada en el agente.

        La vista es un cuadrado de (2*radius+1) x (2*radius+1) con el agente en el centro.
        No depende de la dirección del agente (visión 360°).

        Canales:
            0: Tipo de celda (0=vacío, 1=bloqueado, 2=sano, 3=infectado, 4=self)
            1: Distancia Manhattan normalizada (para depth perception)
        """
        radius = self.config.view_radius
        size = radius * 2 + 1  # 15x15 con radius=7
        view = np.zeros((size, size, 2), dtype=np.uint8)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Coordenadas en la vista (0 a size-1)
                vy = dy + radius
                vx = dx + radius

                # Coordenadas en el mundo
                world_x = agent.x + dx
                world_y = agent.y + dy

                # Canal 0: tipo de celda
                view[vy, vx, 0] = self._get_cell_type(world_x, world_y, agent.id)

                # Canal 1: distancia Manhattan normalizada (0-255)
                dist = abs(dx) + abs(dy)
                max_dist = radius * 2  # Máxima distancia en la vista
                view[vy, vx, 1] = min(255, int(dist * (255 / max_dist)))

        return view

    def _get_cell_type(self, x: int, y: int, observer_id: int) -> int:
        """
        Tipo de celda simplificado:
            0 = vacío
            1 = bloqueado (muro u obstáculo)
            2 = agente sano
            3 = agente infectado
            4 = self (el observador)
        """
        # Fuera del mapa = bloqueado
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return 1

        # Muro u obstáculo = bloqueado (unificados)
        if self.grid[y, x] in (CellType.WALL.value, CellType.OBSTACLE.value):
            return 1

        # Verificar agentes
        other_agent = self.agents.get_agent_at((x, y))
        if other_agent:
            if other_agent.id == observer_id:
                return 4  # Self
            elif other_agent.is_infected:
                return 3  # Infectado
            else:
                return 2  # Sano

        return 0  # Vacío

    def _get_nearby_agents_info(self, agent: BaseAgent) -> np.ndarray:
        """
        Información de agentes cercanos.

        Si el agente es infectado y tiene visión global, prioriza mostrar
        TODOS los healthy agents (para que sepa dónde están todos).
        """
        max_agents = self.config.max_nearby_agents
        info = np.zeros((max_agents, 4), dtype=np.float32)
        max_dist = self.width + self.height

        others = [(a, agent.distance_to(a)) for a in self.agents if a.id != agent.id]

        # Si es infectado con visión global, priorizar healthy agents
        if agent.is_infected and self.config.infected_global_vision:
            # Separar healthy y otros infected
            healthy = [(a, d) for a, d in others if not a.is_infected]
            infected = [(a, d) for a, d in others if a.is_infected]
            # Ordenar cada grupo por distancia
            healthy.sort(key=lambda x: x[1])
            infected.sort(key=lambda x: x[1])
            # Priorizar healthy (el infectado necesita saber dónde están TODOS)
            others = healthy + infected
        else:
            others.sort(key=lambda x: x[1])

        for i, (other, dist) in enumerate(others[:max_agents]):
            rel_x = (other.x - agent.x) / self.width
            rel_y = (other.y - agent.y) / self.height
            is_infected = 1.0 if other.is_infected else 0.0
            dist_norm = dist / max_dist
            info[i] = [rel_x, rel_y, is_infected, dist_norm]

        return info

    def _get_info(self) -> Dict[str, Any]:
        """Información adicional del estado."""
        num_healthy = self.agents.num_healthy
        num_infected = self.agents.num_infected

        # Calcular métricas de victoria para el callback
        # healthy_survived: 1 si quedan sanos vivos, 0 si no
        # infected_won: 1 si todos fueron infectados, 0 si no
        healthy_survived = 1 if num_healthy > 0 else 0
        infected_won = 1 if num_healthy == 0 else 0

        return {
            "step": self.current_step,
            "num_healthy": num_healthy,
            "num_infected": num_infected,
            "healthy_survived": healthy_survived,
            "infected_won": infected_won,
            "infection_events": len(self.infection_events),
            "infection_count": len(self.infection_events),
            "survival_rate": num_healthy / max(1, num_healthy + num_infected - self.config.initial_infected),
            "infected_percentage": num_infected / max(1, self.config.num_agents),
            "agents": {a.id: a.to_dict() for a in self.agents},
        }

    def render(self) -> Optional[np.ndarray]:
        """Renderiza el entorno."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode in ["human", "rgb_array"]:
            return self._render_rgb()
        return None

    def _render_ansi(self) -> str:
        """Renderiza en modo texto."""
        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                agent = self.agents.get_agent_at((x, y))
                if agent:
                    line += "I" if agent.is_infected else "H"
                elif self.grid[y, x] == CellType.WALL.value:
                    line += "#"
                elif self.grid[y, x] == CellType.OBSTACLE.value:
                    line += "O"
                else:
                    line += "."
            lines.append(line)

        output = "\n".join(lines)
        output += f"\nStep: {self.current_step} | Healthy: {self.agents.num_healthy} | Infected: {self.agents.num_infected}"
        print(output)
        return output

    def _render_rgb(self, show_vision: bool = True) -> np.ndarray:
        """Renderiza como imagen RGB."""
        cell_size = 20
        width = self.width * cell_size
        height = self.height * cell_size

        img = np.zeros((height, width, 3), dtype=np.uint8)

        colors = {
            "empty": (200, 200, 200),
            "wall": (50, 50, 50),
            "obstacle": (100, 100, 100),
            "healthy": (0, 200, 0),
            "infected": (200, 0, 0),
            "vision_infected": (80, 80, 80),
        }

        # Dibujar grid
        for y in range(self.height):
            for x in range(self.width):
                x1, y1 = x * cell_size, y * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size

                if self.grid[y, x] == CellType.WALL.value:
                    color = colors["wall"]
                elif self.grid[y, x] == CellType.OBSTACLE.value:
                    color = colors["obstacle"]
                else:
                    color = colors["empty"]

                img[y1:y2, x1:x2] = color

        # Dibujar visión de infectados
        if show_vision:
            alpha = 0.5
            for agent in self.agents.infected:
                vision_color = colors["vision_infected"]
                visible_cells = self._get_visible_cells(agent)

                for (world_x, world_y) in visible_cells:
                    if 0 <= world_x < self.width and 0 <= world_y < self.height:
                        x1 = world_x * cell_size
                        y1 = world_y * cell_size
                        x2 = x1 + cell_size
                        y2 = y1 + cell_size

                        for c in range(3):
                            img[y1:y2, x1:x2, c] = (
                                alpha * vision_color[c] +
                                (1 - alpha) * img[y1:y2, x1:x2, c]
                            ).astype(np.uint8)

        # Dibujar agentes
        for agent in self.agents:
            x1 = agent.x * cell_size + 2
            y1 = agent.y * cell_size + 2
            x2 = x1 + cell_size - 4
            y2 = y1 + cell_size - 4

            color = colors["infected"] if agent.is_infected else colors["healthy"]
            img[y1:y2, x1:x2] = color

            # Indicador de dirección
            cx = agent.x * cell_size + cell_size // 2
            cy = agent.y * cell_size + cell_size // 2
            size = 4
            dir_color = (255, 255, 255)

            if agent.direction == Direction.UP and cy - size >= 0:
                img[cy - size:cy - size + 2, cx - 1:cx + 1] = dir_color
            elif agent.direction == Direction.DOWN and cy + size < height:
                img[cy + size - 2:cy + size, cx - 1:cx + 1] = dir_color
            elif agent.direction == Direction.LEFT and cx - size >= 0:
                img[cy - 1:cy + 1, cx - size:cx - size + 2] = dir_color
            elif agent.direction == Direction.RIGHT and cx + size < width:
                img[cy - 1:cy + 1, cx + size - 2:cx + size] = dir_color

        if self.render_mode == "human":
            try:
                import pygame
                if self._window is None:
                    pygame.init()
                    self._window = pygame.display.set_mode((width, height))
                    pygame.display.set_caption("Infection Environment")
                    self._clock = pygame.time.Clock()

                surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                self._window.blit(surf, (0, 0))
                pygame.display.flip()
                self._clock.tick(self.metadata["render_fps"])
            except ImportError:
                pass

        return img

    def close(self) -> None:
        """Cierra recursos."""
        if self._window is not None:
            import pygame
            pygame.quit()
            self._window = None
            self._clock = None

    def get_agent(self, agent_id: int) -> Optional[BaseAgent]:
        """Obtiene un agente por ID."""
        return self.agents.get(agent_id)

    @property
    def num_agents(self) -> int:
        return self.agents.num_total

    @property
    def num_healthy(self) -> int:
        return self.agents.num_healthy

    @property
    def num_infected(self) -> int:
        return self.agents.num_infected
