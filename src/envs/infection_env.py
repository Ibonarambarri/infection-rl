"""
Multi-Agent Infection Environment
=================================
Entorno personalizado de MiniGrid para simular un escenario de infestación multi-agente.

Características:
- Múltiples agentes (N=5-15) coexisten en un mapa grande con obstáculos
- Un agente inicia infectado y debe perseguir a los demás
- Los agentes sanos aprenden a huir usando Reinforcement Learning
- Al ser atrapado, un agente sano se infecta y se une a la persecución
- El episodio termina cuando todos están infectados o se alcanza el límite de pasos
"""

from typing import Tuple, Dict, List, Optional, Any, SupportsFloat
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .agent import Agent, AgentState, AgentCollection, Direction
from .map_generator import MapGenerator, MapConfig, MapType, CellType


@dataclass
class EnvConfig:
    """Configuración del entorno de infección."""
    # Tamaño del mapa (grande por defecto para mejor aprendizaje)
    width: int = 25
    height: int = 25

    # Agentes
    num_agents: int = 8
    initial_infected: int = 1

    # Tipo de mapa
    map_type: MapType = MapType.SIMPLE
    obstacle_density: float = 0.15

    # Mecánicas
    infection_radius: int = 1  # Distancia Manhattan para infección
    max_steps: int = 500

    # Vista parcial del agente
    view_size: int = 11  # Tamaño de la vista parcial (11x11) - mayor visión
    max_nearby_agents: int = 8  # Máximo de agentes cercanos en observación

    # Recompensas para agentes SANOS
    reward_survive_step: float = 0.1
    reward_distance_bonus: float = 0.1  # Por unidad de distancia al infectado
    reward_infected_penalty: float = -10.0
    reward_not_moving_penalty: float = -0.05  # Penalización base por no moverse
    reward_stuck_penalty: float = -0.2  # Penalización extra por estar atascado
    stuck_threshold: int = 3  # Steps sin moverse para considerar "atascado"
    reward_survive_episode: float = 5.0

    # Recompensas para agentes INFECTADOS
    reward_infect_agent: float = 15.0
    reward_approach_bonus: float = 0.2  # Escalado por cercanía (solo si ve al objetivo)
    reward_step_penalty: float = -0.01
    reward_all_infected_bonus: float = 20.0
    reward_exploration: float = 0.02  # Pequeño bonus por moverse cuando no ve objetivos

    # Otros
    seed: Optional[int] = None
    render_mode: Optional[str] = None
    fixed_map: bool = True  # Por defecto usa el mismo mapa en cada reset para mejor aprendizaje


class InfectionEnv(gym.Env):
    """
    Entorno Multi-Agente de Infección.

    Acciones:
        0: Girar izquierda
        1: Girar derecha
        2: Avanzar
        3: Quedarse quieto

    Observaciones (para cada agente):
        - image: Vista parcial del grid (view_size x view_size x 3)
        - direction: Dirección actual (0-3)
        - state: Estado de infección (0=sano, 1=infectado)
        - nearby_agents: Información de agentes cercanos
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}

    # Acciones disponibles
    ACTIONS = {
        0: "turn_left",
        1: "turn_right",
        2: "forward",
        3: "stay",
    }

    def __init__(self, config: EnvConfig = None, **kwargs):
        super().__init__()

        # Configuración
        if config is None:
            config = EnvConfig(**kwargs)
        self.config = config

        # Semilla
        self._np_random = np.random.default_rng(config.seed)

        # Generador de mapas
        self.map_config = MapConfig(
            width=config.width,
            height=config.height,
            map_type=config.map_type,
            obstacle_density=config.obstacle_density,
            seed=config.seed,
        )
        self.map_generator: MapGenerator = None
        self.grid: np.ndarray = None
        self._fixed_grid: np.ndarray = None  # Grid guardado para fixed_map mode

        # Agentes
        self.agents = AgentCollection()

        # Estado del entorno
        self.current_step = 0
        self.done = False
        self.infection_events: List[Dict] = []  # Historial de infecciones

        # Render
        self.render_mode = config.render_mode
        self._window = None
        self._clock = None

        # Espacios de acción y observación
        self.action_space = spaces.Discrete(4)

        # Espacio de observación
        # Vista parcial: (view_size, view_size, 3) donde los canales son:
        #   0: tipo de celda (0=vacío, 1=muro, 2=obstáculo, 3=agente_sano, 4=agente_infectado)
        #   1: dirección del agente (0-4, 4=no hay agente)
        #   2: distancia al agente observador (normalizada)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(config.view_size, config.view_size, 3),
                dtype=np.uint8
            ),
            "direction": spaces.Discrete(4),
            "state": spaces.Discrete(2),
            "position": spaces.Box(
                low=0,
                high=max(config.width, config.height),
                shape=(2,),
                dtype=np.float32
            ),
            "nearby_agents": spaces.Box(
                low=-1,
                high=1,
                shape=(config.max_nearby_agents, 4),  # (rel_x, rel_y, is_infected, distance)
                dtype=np.float32
            ),
        })

        # Para tracking
        self._last_positions: Dict[int, Tuple[int, int]] = {}
        self._steps_in_same_cell: Dict[int, int] = {}  # Contador de steps en misma celda

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reinicia el entorno.

        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales

        Returns:
            Observación inicial y diccionario info
        """
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self.map_config.seed = seed

        # Generar mapa (o reusar el fijo)
        if self.config.fixed_map and self._fixed_grid is not None:
            # Reusar el mapa guardado
            self.grid = self._fixed_grid.copy()
        else:
            # Generar nuevo mapa
            self.map_generator = MapGenerator(self.map_config)
            self.grid = self.map_generator.grid.copy()
            # Guardar si es modo fijo
            if self.config.fixed_map:
                self._fixed_grid = self.grid.copy()

        # Limpiar agentes
        self.agents.clear()

        # Obtener posiciones válidas para agentes
        valid_positions = self.map_generator.get_random_valid_positions(
            self.config.num_agents
        )

        # Crear agentes
        for i, pos in enumerate(valid_positions):
            direction = Direction(self._np_random.integers(0, 4))
            state = AgentState.INFECTED if i < self.config.initial_infected else AgentState.HEALTHY
            agent = self.agents.add(pos, direction, state)

            if state == AgentState.INFECTED:
                agent.infection_time = 0

        # Reiniciar estado
        self.current_step = 0
        self.done = False
        self.infection_events = []
        self._last_positions = {a.id: a.position for a in self.agents}
        self._steps_in_same_cell = {a.id: 0 for a in self.agents}

        # Obtener observación del primer agente (para single-agent training)
        obs = self._get_observation(self.agents.all[0])

        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int,
        agent_id: int = 0
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Ejecuta un paso en el entorno.

        En modo single-agent, solo controla un agente y los demás actúan según
        una política simple (random o heurística).

        Args:
            action: Acción a ejecutar (0-3)
            agent_id: ID del agente que ejecuta la acción

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.done:
            return self._get_observation(self.agents[agent_id]), 0.0, True, False, self._get_info()

        self.current_step += 1

        # Guardar posiciones anteriores
        old_positions = {a.id: a.position for a in self.agents}

        # Ejecutar acción del agente controlado
        controlled_agent = self.agents.get(agent_id)
        if controlled_agent:
            self._execute_action(controlled_agent, action)

        # Ejecutar acciones de otros agentes (política simple)
        for agent in self.agents:
            if agent.id != agent_id:
                other_action = self._get_other_agent_action(agent)
                self._execute_action(agent, other_action)

        # Actualizar contadores de steps en misma celda
        for agent in self.agents:
            if agent.position == old_positions.get(agent.id):
                self._steps_in_same_cell[agent.id] = self._steps_in_same_cell.get(agent.id, 0) + 1
            else:
                self._steps_in_same_cell[agent.id] = 0

        # Actualizar posiciones previas
        self._last_positions = old_positions

        # Detectar infecciones
        new_infections = self._check_infections()

        # Calcular recompensa
        reward = self._calculate_reward(controlled_agent, new_infections)

        # Actualizar estadísticas
        for agent in self.agents:
            if agent.is_healthy:
                agent.steps_survived += 1

        # Verificar condiciones de terminación
        terminated = self._check_termination()
        truncated = self.current_step >= self.config.max_steps

        self.done = terminated or truncated

        # Añadir bonus de fin de episodio
        if self.done:
            reward += self._calculate_episode_end_reward(controlled_agent, terminated)

        obs = self._get_observation(controlled_agent)
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def step_all(
        self,
        actions: Dict[int, int]
    ) -> Tuple[Dict[int, Dict], Dict[int, float], bool, bool, Dict[str, Any]]:
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
        self._last_positions = {a.id: a.position for a in self.agents}

        # Ejecutar todas las acciones
        for agent_id, action in actions.items():
            agent = self.agents.get(agent_id)
            if agent:
                self._execute_action(agent, action)

        # Detectar infecciones
        new_infections = self._check_infections()

        # Calcular recompensas
        rewards = {}
        for agent in self.agents:
            rewards[agent.id] = self._calculate_reward(agent, new_infections)
            if agent.is_healthy:
                agent.steps_survived += 1

        # Verificar terminación
        terminated = self._check_termination()
        truncated = self.current_step >= self.config.max_steps
        self.done = terminated or truncated

        # Bonus de fin de episodio
        if self.done:
            for agent in self.agents:
                rewards[agent.id] += self._calculate_episode_end_reward(agent, terminated)

        obs = {a.id: self._get_observation(a) for a in self.agents}
        info = self._get_info()

        return obs, rewards, terminated, truncated, info

    def _execute_action(self, agent: Agent, action: int) -> None:
        """Ejecuta una acción para un agente."""
        if action == 0:  # turn_left
            agent.turn_left()
        elif action == 1:  # turn_right
            agent.turn_right()
        elif action == 2:  # forward
            new_x, new_y = agent.move_forward()
            if self._is_valid_move(new_x, new_y, agent.id):
                agent.set_position(new_x, new_y)
        # action == 3: stay (no hacer nada)

    def _is_valid_move(self, x: int, y: int, agent_id: int) -> bool:
        """Verifica si un movimiento es válido."""
        # Verificar límites y obstáculos
        if not self.map_generator.is_valid_position(x, y):
            return False

        # Verificar colisión con otros agentes
        for agent in self.agents:
            if agent.id != agent_id and agent.position == (x, y):
                return False

        return True

    def _check_infections(self) -> List[int]:
        """
        Verifica y ejecuta infecciones por proximidad.

        Returns:
            Lista de IDs de agentes recién infectados
        """
        new_infections = []

        infected_agents = self.agents.infected
        healthy_agents = self.agents.healthy

        for infected in infected_agents:
            for healthy in healthy_agents:
                if healthy.id in new_infections:
                    continue

                distance = infected.distance_to(healthy)
                if distance <= self.config.infection_radius:
                    # ¡Infección!
                    healthy.infect(self.current_step)
                    infected.agents_infected += 1
                    new_infections.append(healthy.id)

                    # Registrar evento
                    self.infection_events.append({
                        "step": self.current_step,
                        "infected_by": infected.id,
                        "newly_infected": healthy.id,
                        "position": healthy.position,
                    })

        return new_infections

    def _check_termination(self) -> bool:
        """Verifica si el episodio debe terminar."""
        # Termina si todos están infectados
        return self.agents.num_healthy == 0

    def _calculate_reward(self, agent: Agent, new_infections: List[int]) -> float:
        """Calcula la recompensa para un agente."""
        reward = 0.0

        if agent.is_healthy:
            reward = self._calculate_healthy_reward(agent, new_infections)
        else:
            reward = self._calculate_infected_reward(agent, new_infections)

        agent.add_reward(reward)
        return reward

    def _calculate_healthy_reward(self, agent: Agent, new_infections: List[int]) -> float:
        """Calcula recompensa para agente sano."""
        reward = 0.0

        # ¿Fue infectado este step?
        if agent.id in new_infections:
            return self.config.reward_infected_penalty

        # Recompensa por sobrevivir
        reward += self.config.reward_survive_step

        # Bonus por distancia al infectado más cercano VISIBLE (usando pathfinding)
        visible_infected = self._get_visible_agents(agent, infected_only=True)
        if visible_infected:
            # Encontrar el más cercano entre los visibles
            nearest_visible = min(visible_infected, key=lambda a: agent.distance_to(a))
            # Usar distancia real por pathfinding
            distance = self._pathfinding_distance(agent.position, nearest_visible.position)
            if distance is None:
                distance = agent.distance_to(nearest_visible)
            reward += self.config.reward_distance_bonus * distance

        # Penalización por no moverse (progresiva)
        steps_stuck = self._steps_in_same_cell.get(agent.id, 0)
        if steps_stuck > 0:
            # Penalización base por no moverse
            reward += self.config.reward_not_moving_penalty

            # Penalización extra si está "atascado" (muchos steps sin moverse)
            if steps_stuck >= self.config.stuck_threshold:
                # Penalización progresiva: aumenta con cada step adicional
                multiplier = min(steps_stuck - self.config.stuck_threshold + 1, 5)
                reward += self.config.reward_stuck_penalty * multiplier

        return reward

    def _calculate_infected_reward(self, agent: Agent, new_infections: List[int]) -> float:
        """Calcula recompensa para agente infectado."""
        reward = 0.0

        # Recompensa por infectar
        for event in self.infection_events:
            if event["step"] == self.current_step and event["infected_by"] == agent.id:
                reward += self.config.reward_infect_agent

        # Bonus por acercarse al sano más cercano VISIBLE (usando pathfinding)
        visible_healthy = self._get_visible_agents(agent, healthy_only=True)
        if visible_healthy:
            # Encontrar el más cercano entre los visibles
            nearest_visible = min(visible_healthy, key=lambda a: agent.distance_to(a))
            # Usar distancia real por pathfinding
            distance = self._pathfinding_distance(agent.position, nearest_visible.position)
            if distance is None:
                distance = agent.distance_to(nearest_visible)

            # Recompensa inversamente proporcional a la distancia real
            max_dist = self.config.width + self.config.height
            proximity = (max_dist - distance) / max_dist
            reward += self.config.reward_approach_bonus * proximity
        else:
            # No ve a nadie - dar pequeño bonus por explorar (moverse)
            if self._steps_in_same_cell.get(agent.id, 0) == 0:
                reward += self.config.reward_exploration

        # Penalización por step (presión temporal)
        reward += self.config.reward_step_penalty

        # Penalización por no moverse (también para infectados)
        steps_stuck = self._steps_in_same_cell.get(agent.id, 0)
        if steps_stuck > 0:
            reward += self.config.reward_not_moving_penalty
            if steps_stuck >= self.config.stuck_threshold:
                multiplier = min(steps_stuck - self.config.stuck_threshold + 1, 5)
                reward += self.config.reward_stuck_penalty * multiplier

        return reward

    def _can_see_agent(self, observer: Agent, target: Agent) -> bool:
        """
        Verifica si el observador puede ver al objetivo en su campo de visión.

        Args:
            observer: Agente que observa
            target: Agente objetivo

        Returns:
            True si el objetivo está en el campo de visión del observador
        """
        visible_cells = self._get_visible_cells(observer)
        return target.position in visible_cells

    def _get_visible_agents(self, agent: Agent, infected_only: bool = False, healthy_only: bool = False) -> List[Agent]:
        """
        Obtiene lista de agentes visibles para un agente.

        Args:
            agent: Agente que observa
            infected_only: Solo retornar infectados
            healthy_only: Solo retornar sanos

        Returns:
            Lista de agentes visibles
        """
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

    def _pathfinding_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[int]:
        """
        Calcula la distancia real entre dos puntos usando BFS (pathfinding).
        Considera obstáculos y muros.

        Args:
            start: Posición inicial (x, y)
            goal: Posición objetivo (x, y)

        Returns:
            Distancia en pasos, o None si no hay camino
        """
        if start == goal:
            return 0

        # BFS
        queue = deque([(start[0], start[1], 0)])  # (x, y, distance)
        visited = {start}

        # Direcciones: arriba, abajo, izquierda, derecha
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while queue:
            x, y, dist = queue.popleft()

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # Verificar límites
                if nx < 0 or nx >= self.config.width or ny < 0 or ny >= self.config.height:
                    continue

                # Verificar si ya visitado
                if (nx, ny) in visited:
                    continue

                # Verificar si es muro u obstáculo
                if self.grid[ny, nx] == CellType.WALL.value:
                    continue
                if self.grid[ny, nx] == CellType.OBSTACLE.value:
                    continue

                # ¿Llegamos al objetivo?
                if (nx, ny) == goal:
                    return dist + 1

                # Añadir a la cola
                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))

        # No hay camino
        return None

    def _calculate_episode_end_reward(self, agent: Agent, all_infected: bool) -> float:
        """Calcula bonus de fin de episodio."""
        if agent.is_healthy:
            # El agente sobrevivió el episodio
            return self.config.reward_survive_episode
        else:
            if all_infected:
                # Los infectados ganaron
                return self.config.reward_all_infected_bonus
        return 0.0

    def _get_other_agent_action(self, agent: Agent) -> int:
        """
        Determina la acción de un agente no controlado.
        Usa una política heurística simple.
        """
        if agent.is_infected:
            # Perseguir al sano más cercano
            target = self.agents.find_nearest_healthy(agent)
            if target:
                return self._move_towards(agent, target.position)
        else:
            # Huir del infectado más cercano
            threat = self.agents.find_nearest_infected(agent)
            if threat:
                return self._move_away_from(agent, threat.position)

        # Acción aleatoria si no hay objetivo
        return self._np_random.integers(0, 4)

    def _move_towards(self, agent: Agent, target_pos: Tuple[int, int]) -> int:
        """Determina acción para moverse hacia un objetivo."""
        dx = target_pos[0] - agent.x
        dy = target_pos[1] - agent.y

        # Calcular dirección deseada
        if abs(dx) > abs(dy):
            desired_dir = Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            desired_dir = Direction.DOWN if dy > 0 else Direction.UP

        # Si ya mira en esa dirección, avanzar
        if agent.direction == desired_dir:
            new_x, new_y = agent.move_forward()
            if self._is_valid_move(new_x, new_y, agent.id):
                return 2  # forward
            # Si no puede avanzar, girar aleatoriamente
            return self._np_random.choice([0, 1])

        # Girar hacia la dirección deseada
        diff = (desired_dir - agent.direction) % 4
        if diff == 1:
            return 1  # turn_right
        elif diff == 3:
            return 0  # turn_left
        else:
            # Diferencia de 2, girar cualquier dirección
            return self._np_random.choice([0, 1])

    def _move_away_from(self, agent: Agent, threat_pos: Tuple[int, int]) -> int:
        """Determina acción para alejarse de una amenaza."""
        dx = agent.x - threat_pos[0]
        dy = agent.y - threat_pos[1]

        # Dirección opuesta a la amenaza
        if abs(dx) > abs(dy):
            desired_dir = Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            desired_dir = Direction.DOWN if dy > 0 else Direction.UP

        if agent.direction == desired_dir:
            new_x, new_y = agent.move_forward()
            if self._is_valid_move(new_x, new_y, agent.id):
                return 2  # forward
            return self._np_random.choice([0, 1])

        diff = (desired_dir - agent.direction) % 4
        if diff == 1:
            return 1
        elif diff == 3:
            return 0
        else:
            return self._np_random.choice([0, 1])

    def _get_observation(self, agent: Agent) -> Dict[str, Any]:
        """Genera la observación para un agente."""
        # Vista parcial del grid
        image = self._get_partial_view(agent)

        # Información de agentes cercanos
        nearby = self._get_nearby_agents_info(agent)

        # Posición normalizada
        pos = np.array([
            agent.x / self.config.width,
            agent.y / self.config.height
        ], dtype=np.float32)

        return {
            "image": image,
            "direction": int(agent.direction),
            "state": int(agent.state),
            "position": pos,
            "nearby_agents": nearby,
        }

    def _get_partial_view(self, agent: Agent) -> np.ndarray:
        """
        Genera la vista parcial del agente.

        La vista empieza en la posición del agente y se extiende hacia adelante.
        El agente está en la fila inferior central de su vista.
        El agente puede ver todo lo que tiene delante (los muros NO bloquean la visión).

        Ejemplo (view_size=7):
          La vista es de 7 columnas de ancho x 7 filas de profundidad
          El agente @ está en la posición inferior central

          . . . . . . .   <- 6 celdas adelante
          . . . . . . .
          . . . . . . .
          . . . . . . .
          . . . . . . .
          . . . . . . .
          . . . @ . . .   <- posición del agente (fila 6, columna 3)
        """
        view_size = self.config.view_size
        half_width = view_size // 2

        # Crear vista vacía
        view = np.zeros((view_size, view_size, 3), dtype=np.uint8)

        # Iterar por toda la vista (sin bloqueo por muros)
        for vy in range(view_size):
            for vx in range(view_size):
                rel_x = vx - half_width
                forward_dist = (view_size - 1) - vy

                # Rotar según la dirección del agente
                world_x, world_y = self._rotate_coords_forward(rel_x, forward_dist, agent.direction)
                world_x += agent.x
                world_y += agent.y

                # Obtener tipo de celda
                cell_type = self._get_cell_type(world_x, world_y, agent.id)
                view[vy, vx, 0] = cell_type

                # Si hay un agente, añadir su dirección
                other_agent = self.agents.get_agent_at((world_x, world_y))
                if other_agent and other_agent.id != agent.id:
                    view[vy, vx, 1] = int(other_agent.direction)
                else:
                    view[vy, vx, 1] = 4  # Sin agente

                # Canal 2: distancia normalizada (para depth perception)
                dist = abs(rel_x) + forward_dist
                view[vy, vx, 2] = min(255, int(dist * 25))

        return view

    def _rotate_coords_forward(self, rel_x: int, forward: int, direction: Direction) -> Tuple[int, int]:
        """
        Rota coordenadas de vista hacia adelante según la dirección.

        Args:
            rel_x: desplazamiento lateral (-left, +right desde perspectiva del agente)
            forward: distancia hacia adelante
            direction: dirección del agente

        Returns:
            (world_dx, world_dy) desplazamiento en coordenadas del mundo
        """
        if direction == Direction.UP:
            return (rel_x, -forward)
        elif direction == Direction.DOWN:
            return (-rel_x, forward)
        elif direction == Direction.RIGHT:
            return (forward, rel_x)
        else:  # LEFT
            return (-forward, -rel_x)

    def _rotate_coords(self, x: int, y: int, direction: Direction) -> Tuple[int, int]:
        """Rota coordenadas según la dirección."""
        if direction == Direction.UP:
            return (x, -y)
        elif direction == Direction.RIGHT:
            return (y, x)
        elif direction == Direction.DOWN:
            return (-x, y)
        else:  # LEFT
            return (-y, -x)

    def _get_cell_type(self, x: int, y: int, observer_id: int) -> int:
        """
        Obtiene el tipo de celda para la observación.

        0: Vacío
        1: Muro
        2: Obstáculo
        3: Agente sano
        4: Agente infectado
        5: El propio observador
        """
        # Fuera de límites = muro
        if x < 0 or x >= self.config.width or y < 0 or y >= self.config.height:
            return 1

        # Verificar muro u obstáculo
        if self.grid[y, x] == CellType.WALL.value:
            return 1
        if self.grid[y, x] == CellType.OBSTACLE.value:
            return 2

        # Verificar agentes
        other_agent = self.agents.get_agent_at((x, y))
        if other_agent:
            if other_agent.id == observer_id:
                return 5  # El propio agente
            elif other_agent.is_infected:
                return 4  # Infectado
            else:
                return 3  # Sano

        return 0  # Vacío

    def _get_nearby_agents_info(self, agent: Agent) -> np.ndarray:
        """
        Obtiene información de agentes cercanos.

        Returns:
            Array de shape (max_nearby_agents, 4) con:
            [rel_x_norm, rel_y_norm, is_infected, distance_norm]
        """
        max_agents = self.config.max_nearby_agents
        info = np.zeros((max_agents, 4), dtype=np.float32)

        # Normalización
        max_dist = self.config.width + self.config.height

        # Ordenar otros agentes por distancia
        others = [(a, agent.distance_to(a)) for a in self.agents if a.id != agent.id]
        others.sort(key=lambda x: x[1])

        for i, (other, dist) in enumerate(others[:max_agents]):
            rel_x = (other.x - agent.x) / self.config.width
            rel_y = (other.y - agent.y) / self.config.height
            is_infected = 1.0 if other.is_infected else 0.0
            dist_norm = dist / max_dist

            info[i] = [rel_x, rel_y, is_infected, dist_norm]

        return info

    def _get_info(self) -> Dict[str, Any]:
        """Retorna información adicional del estado."""
        # Calcular métricas estadísticas
        initial_healthy = self.config.num_agents - self.config.initial_infected
        num_healthy = self.agents.num_healthy
        num_infected = self.agents.num_infected
        total_agents = self.config.num_agents

        # Survival rate: proporción de sanos que siguen vivos respecto al inicio
        survival_rate = num_healthy / initial_healthy if initial_healthy > 0 else 0.0

        # Infected percentage: proporción de agentes actualmente infectados
        infected_percentage = num_infected / total_agents if total_agents > 0 else 0.0

        # Infection count: número absoluto de infecciones ocurridas
        infection_count = len(self.infection_events)

        return {
            "step": self.current_step,
            "num_healthy": num_healthy,
            "num_infected": num_infected,
            "infection_events": infection_count,
            "agents": {a.id: a.to_dict() for a in self.agents},
            # Nuevas métricas estadísticas
            "survival_rate": survival_rate,
            "infected_percentage": infected_percentage,
            "infection_count": infection_count,
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
        for y in range(self.config.height):
            line = ""
            for x in range(self.config.width):
                agent = self.agents.get_agent_at((x, y))
                if agent:
                    if agent.is_infected:
                        line += "I"
                    else:
                        line += "H"
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
        """
        Renderiza como imagen RGB.

        Args:
            show_vision: Si True, muestra el campo de visión de cada agente
        """
        cell_size = 20
        width = self.config.width * cell_size
        height = self.config.height * cell_size

        # Crear imagen
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Colores
        colors = {
            "empty": (200, 200, 200),
            "wall": (50, 50, 50),
            "obstacle": (100, 100, 100),
            "healthy": (0, 200, 0),
            "infected": (200, 0, 0),
            "vision_infected": (80, 80, 80),  # Gris opaco para visión de infectados
        }

        # Dibujar grid
        for y in range(self.config.height):
            for x in range(self.config.width):
                x1, y1 = x * cell_size, y * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size

                if self.grid[y, x] == CellType.WALL.value:
                    color = colors["wall"]
                elif self.grid[y, x] == CellType.OBSTACLE.value:
                    color = colors["obstacle"]
                else:
                    color = colors["empty"]

                img[y1:y2, x1:x2] = color

        # Dibujar campo de visión de cada agente (antes de dibujar agentes)
        if show_vision:
            self._draw_vision_overlay(img, cell_size, colors)

        # Dibujar agentes
        for agent in self.agents:
            x1 = agent.x * cell_size + 2
            y1 = agent.y * cell_size + 2
            x2 = x1 + cell_size - 4
            y2 = y1 + cell_size - 4

            color = colors["infected"] if agent.is_infected else colors["healthy"]
            img[y1:y2, x1:x2] = color

            # Indicar dirección con un pequeño triángulo
            cx = agent.x * cell_size + cell_size // 2
            cy = agent.y * cell_size + cell_size // 2
            size = 4

            if agent.direction == Direction.UP:
                points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
            elif agent.direction == Direction.DOWN:
                points = [(cx, cy + size), (cx - size, cy - size), (cx + size, cy - size)]
            elif agent.direction == Direction.LEFT:
                points = [(cx - size, cy), (cx + size, cy - size), (cx + size, cy + size)]
            else:  # RIGHT
                points = [(cx + size, cy), (cx - size, cy - size), (cx - size, cy + size)]

            # Dibujar indicador de dirección (simplificado como punto)
            dir_color = (255, 255, 255)
            if agent.direction == Direction.UP and cy - size >= 0:
                img[cy - size:cy - size + 2, cx - 1:cx + 1] = dir_color
            elif agent.direction == Direction.DOWN and cy + size < height:
                img[cy + size - 2:cy + size, cx - 1:cx + 1] = dir_color
            elif agent.direction == Direction.LEFT and cx - size >= 0:
                img[cy - 1:cy + 1, cx - size:cx - size + 2] = dir_color
            elif agent.direction == Direction.RIGHT and cx + size < width:
                img[cy - 1:cy + 1, cx + size - 2:cx + size] = dir_color

        # Mostrar si es modo human
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

    def _draw_vision_overlay(self, img: np.ndarray, cell_size: int, colors: dict) -> None:
        """
        Dibuja el campo de visión de los agentes infectados como overlay gris opaco.
        Solo muestra la visión de los infectados, sin restricción por muros.
        """
        alpha = 0.5  # Semi-opaco

        for agent in self.agents:
            # Solo dibujar visión de infectados
            if not agent.is_infected:
                continue

            vision_color = colors["vision_infected"]

            # Obtener celdas visibles (sin bloqueo por muros)
            visible_cells = self._get_visible_cells(agent)

            # Dibujar overlay en cada celda visible
            for (world_x, world_y) in visible_cells:
                # Verificar límites
                if 0 <= world_x < self.config.width and 0 <= world_y < self.config.height:
                    x1 = world_x * cell_size
                    y1 = world_y * cell_size
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size

                    # Aplicar color con transparencia (blending)
                    for c in range(3):
                        img[y1:y2, x1:x2, c] = (
                            alpha * vision_color[c] +
                            (1 - alpha) * img[y1:y2, x1:x2, c]
                        ).astype(np.uint8)

    def _get_visible_cells(self, agent: Agent) -> List[Tuple[int, int]]:
        """
        Obtiene las coordenadas del mundo de las celdas visibles para un agente.
        Sin restricción por muros - ve todo lo que tiene delante.

        Returns:
            Lista de tuplas (world_x, world_y) de celdas visibles
        """
        view_size = self.config.view_size
        half_width = view_size // 2
        visible = []

        # Iterar por toda la vista (sin bloqueo)
        for vy in range(view_size):
            for vx in range(view_size):
                rel_x = vx - half_width
                forward_dist = (view_size - 1) - vy

                # Rotar según la dirección del agente
                world_dx, world_dy = self._rotate_coords_forward(rel_x, forward_dist, agent.direction)
                world_x = agent.x + world_dx
                world_y = agent.y + world_dy

                # Verificar límites
                if 0 <= world_x < self.config.width and 0 <= world_y < self.config.height:
                    visible.append((world_x, world_y))

        return visible

    def close(self) -> None:
        """Cierra recursos del entorno."""
        if self._window is not None:
            import pygame
            pygame.quit()
            self._window = None
            self._clock = None

    def get_agent(self, agent_id: int) -> Optional[Agent]:
        """Obtiene un agente por ID."""
        return self.agents.get(agent_id)

    def get_all_agents(self) -> List[Agent]:
        """Retorna lista de todos los agentes."""
        return self.agents.all

    @property
    def num_agents(self) -> int:
        """Número total de agentes."""
        return self.agents.num_total

    @property
    def num_healthy(self) -> int:
        """Número de agentes sanos."""
        return self.agents.num_healthy

    @property
    def num_infected(self) -> int:
        """Número de agentes infectados."""
        return self.agents.num_infected
