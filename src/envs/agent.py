"""
Agent class for Multi-Agent Infection Environment
================================================
Encapsula el estado individual de cada agente.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


class AgentState(IntEnum):
    """Estado de infección del agente."""
    HEALTHY = 0
    INFECTED = 1


class Direction(IntEnum):
    """Dirección del agente (compatible con MiniGrid)."""
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    def to_vec(self) -> Tuple[int, int]:
        """Convierte dirección a vector de movimiento (dx, dy)."""
        vectors = {
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.UP: (0, -1),
        }
        return vectors[self]

    @staticmethod
    def from_vec(dx: int, dy: int) -> "Direction":
        """Convierte vector a dirección."""
        if dx > 0:
            return Direction.RIGHT
        elif dx < 0:
            return Direction.LEFT
        elif dy > 0:
            return Direction.DOWN
        else:
            return Direction.UP

    def turn_left(self) -> "Direction":
        """Retorna la dirección después de girar a la izquierda."""
        return Direction((self - 1) % 4)

    def turn_right(self) -> "Direction":
        """Retorna la dirección después de girar a la derecha."""
        return Direction((self + 1) % 4)


@dataclass
class Agent:
    """
    Representa un agente individual en el entorno.

    Attributes:
        id: Identificador único del agente
        position: Posición (x, y) en el grid
        direction: Dirección actual (0-3)
        state: Estado de infección (HEALTHY/INFECTED)
        infection_time: Step en el que fue infectado (None si sano)
        total_reward: Recompensa acumulada
    """
    id: int
    position: Tuple[int, int]
    direction: Direction = Direction.RIGHT
    state: AgentState = AgentState.HEALTHY
    infection_time: Optional[int] = None
    total_reward: float = 0.0

    # Estadísticas
    steps_survived: int = 0
    agents_infected: int = 0  # Solo para infectados

    @property
    def x(self) -> int:
        """Posición x del agente."""
        return self.position[0]

    @property
    def y(self) -> int:
        """Posición y del agente."""
        return self.position[1]

    @property
    def is_infected(self) -> bool:
        """Retorna True si el agente está infectado."""
        return self.state == AgentState.INFECTED

    @property
    def is_healthy(self) -> bool:
        """Retorna True si el agente está sano."""
        return self.state == AgentState.HEALTHY

    def infect(self, current_step: int) -> None:
        """Infecta al agente."""
        if self.state == AgentState.HEALTHY:
            self.state = AgentState.INFECTED
            self.infection_time = current_step

    def move_forward(self) -> Tuple[int, int]:
        """
        Calcula la nueva posición al avanzar.
        No modifica la posición actual.

        Returns:
            Nueva posición (x, y) después de avanzar
        """
        dx, dy = self.direction.to_vec()
        return (self.x + dx, self.y + dy)

    def set_position(self, x: int, y: int) -> None:
        """Establece una nueva posición."""
        self.position = (x, y)

    def turn_left(self) -> None:
        """Gira el agente 90° a la izquierda."""
        self.direction = self.direction.turn_left()

    def turn_right(self) -> None:
        """Gira el agente 90° a la derecha."""
        self.direction = self.direction.turn_right()

    def distance_to(self, other: "Agent") -> int:
        """
        Calcula la distancia Manhattan a otro agente.

        Args:
            other: Otro agente

        Returns:
            Distancia Manhattan (|dx| + |dy|)
        """
        return abs(self.x - other.x) + abs(self.y - other.y)

    def distance_to_pos(self, pos: Tuple[int, int]) -> int:
        """
        Calcula la distancia Manhattan a una posición.

        Args:
            pos: Posición (x, y)

        Returns:
            Distancia Manhattan
        """
        return abs(self.x - pos[0]) + abs(self.y - pos[1])

    def relative_position(self, other: "Agent") -> Tuple[int, int]:
        """
        Calcula la posición relativa de otro agente.

        Args:
            other: Otro agente

        Returns:
            Posición relativa (dx, dy)
        """
        return (other.x - self.x, other.y - self.y)

    def add_reward(self, reward: float) -> None:
        """Añade recompensa al acumulado."""
        self.total_reward += reward

    def reset_stats(self) -> None:
        """Reinicia estadísticas del episodio."""
        self.steps_survived = 0
        self.agents_infected = 0
        self.total_reward = 0.0

    def to_dict(self) -> dict:
        """Convierte el agente a diccionario para serialización."""
        return {
            "id": self.id,
            "position": self.position,
            "direction": int(self.direction),
            "state": int(self.state),
            "infection_time": self.infection_time,
            "total_reward": self.total_reward,
            "steps_survived": self.steps_survived,
            "agents_infected": self.agents_infected,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        """Crea un agente desde un diccionario."""
        agent = cls(
            id=data["id"],
            position=tuple(data["position"]),
            direction=Direction(data["direction"]),
            state=AgentState(data["state"]),
            infection_time=data.get("infection_time"),
        )
        agent.total_reward = data.get("total_reward", 0.0)
        agent.steps_survived = data.get("steps_survived", 0)
        agent.agents_infected = data.get("agents_infected", 0)
        return agent

    def __repr__(self) -> str:
        state_str = "INFECTED" if self.is_infected else "HEALTHY"
        dir_str = ["→", "↓", "←", "↑"][self.direction]
        return f"Agent(id={self.id}, pos={self.position}, dir={dir_str}, state={state_str})"


class AgentCollection:
    """
    Colección de agentes con métodos de utilidad.
    Facilita la gestión de múltiples agentes.
    """

    def __init__(self):
        self._agents: dict[int, Agent] = {}
        self._next_id: int = 0

    def add(self, position: Tuple[int, int], direction: Direction = Direction.RIGHT,
            state: AgentState = AgentState.HEALTHY) -> Agent:
        """Crea y añade un nuevo agente."""
        agent = Agent(
            id=self._next_id,
            position=position,
            direction=direction,
            state=state,
        )
        self._agents[agent.id] = agent
        self._next_id += 1
        return agent

    def get(self, agent_id: int) -> Optional[Agent]:
        """Obtiene un agente por ID."""
        return self._agents.get(agent_id)

    def remove(self, agent_id: int) -> Optional[Agent]:
        """Elimina y retorna un agente."""
        return self._agents.pop(agent_id, None)

    def clear(self) -> None:
        """Elimina todos los agentes."""
        self._agents.clear()
        self._next_id = 0

    @property
    def all(self) -> list[Agent]:
        """Lista de todos los agentes."""
        return list(self._agents.values())

    @property
    def healthy(self) -> list[Agent]:
        """Lista de agentes sanos."""
        return [a for a in self._agents.values() if a.is_healthy]

    @property
    def infected(self) -> list[Agent]:
        """Lista de agentes infectados."""
        return [a for a in self._agents.values() if a.is_infected]

    @property
    def num_total(self) -> int:
        """Número total de agentes."""
        return len(self._agents)

    @property
    def num_healthy(self) -> int:
        """Número de agentes sanos."""
        return sum(1 for a in self._agents.values() if a.is_healthy)

    @property
    def num_infected(self) -> int:
        """Número de agentes infectados."""
        return sum(1 for a in self._agents.values() if a.is_infected)

    def get_positions(self) -> set[Tuple[int, int]]:
        """Retorna el conjunto de posiciones ocupadas."""
        return {a.position for a in self._agents.values()}

    def get_agent_at(self, position: Tuple[int, int]) -> Optional[Agent]:
        """Retorna el agente en una posición dada, si existe."""
        for agent in self._agents.values():
            if agent.position == position:
                return agent
        return None

    def find_nearest_healthy(self, from_agent: Agent) -> Optional[Agent]:
        """Encuentra el agente sano más cercano."""
        healthy = self.healthy
        if not healthy:
            return None

        min_dist = float('inf')
        nearest = None
        for agent in healthy:
            if agent.id != from_agent.id:
                dist = from_agent.distance_to(agent)
                if dist < min_dist:
                    min_dist = dist
                    nearest = agent
        return nearest

    def find_nearest_infected(self, from_agent: Agent) -> Optional[Agent]:
        """Encuentra el agente infectado más cercano."""
        infected = self.infected
        if not infected:
            return None

        min_dist = float('inf')
        nearest = None
        for agent in infected:
            if agent.id != from_agent.id:
                dist = from_agent.distance_to(agent)
                if dist < min_dist:
                    min_dist = dist
                    nearest = agent
        return nearest

    def __iter__(self):
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)

    def __getitem__(self, agent_id: int) -> Agent:
        return self._agents[agent_id]

    def __contains__(self, agent_id: int) -> bool:
        return agent_id in self._agents
