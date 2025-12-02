"""
Base Agent Class
================
Clase base abstracta para todos los agentes.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .healthy_agent import HealthyAgent
    from .infected_agent import InfectedAgent


class Direction(IntEnum):
    """Dirección del agente."""
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

    def turn_left(self) -> "Direction":
        """Retorna la dirección después de girar a la izquierda."""
        return Direction((self - 1) % 4)

    def turn_right(self) -> "Direction":
        """Retorna la dirección después de girar a la derecha."""
        return Direction((self + 1) % 4)


class BaseAgent(ABC):
    """
    Clase base abstracta para agentes.

    Define la interfaz común para HealthyAgent e InfectedAgent.
    """

    def __init__(
        self,
        agent_id: int,
        position: Tuple[int, int],
        direction: Direction = Direction.RIGHT,
    ):
        self.id = agent_id
        self.position = position
        self.direction = direction
        self.total_reward: float = 0.0
        self.steps_alive: int = 0

    @property
    def x(self) -> int:
        """Posición x del agente."""
        return self.position[0]

    @property
    def y(self) -> int:
        """Posición y del agente."""
        return self.position[1]

    @property
    @abstractmethod
    def is_infected(self) -> bool:
        """Retorna True si el agente está infectado."""
        pass

    @property
    def is_healthy(self) -> bool:
        """Retorna True si el agente está sano."""
        return not self.is_infected

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Retorna el tipo de agente ('healthy' o 'infected')."""
        pass

    def move_forward(self) -> Tuple[int, int]:
        """
        Calcula la nueva posición al avanzar.
        No modifica la posición actual.
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

    def distance_to(self, other: "BaseAgent") -> int:
        """Calcula la distancia Manhattan a otro agente."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def distance_to_pos(self, pos: Tuple[int, int]) -> int:
        """Calcula la distancia Manhattan a una posición."""
        return abs(self.x - pos[0]) + abs(self.y - pos[1])

    def add_reward(self, reward: float) -> None:
        """Añade recompensa al acumulado."""
        self.total_reward += reward

    def step(self) -> None:
        """Llamado cada step del entorno."""
        self.steps_alive += 1

    def to_dict(self) -> dict:
        """Convierte el agente a diccionario para serialización."""
        return {
            "id": self.id,
            "type": self.agent_type,
            "position": self.position,
            "direction": int(self.direction),
            "total_reward": self.total_reward,
            "steps_alive": self.steps_alive,
        }

    def __repr__(self) -> str:
        dir_str = ["→", "↓", "←", "↑"][self.direction]
        return f"{self.__class__.__name__}(id={self.id}, pos={self.position}, dir={dir_str})"
