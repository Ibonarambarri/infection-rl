"""
Infected Agent Class
====================
Agente infectado que persigue a los sanos.
"""

from typing import Tuple, Optional
from .base_agent import BaseAgent, Direction


class InfectedAgent(BaseAgent):
    """
    Agente infectado.

    Objetivo: Infectar a todos los agentes sanos.

    Puede ser creado directamente (infectado inicial) o mediante
    la transformación de un HealthyAgent.
    """

    def __init__(
        self,
        agent_id: int,
        position: Tuple[int, int],
        direction: Direction = Direction.RIGHT,
        infection_time: int = 0,
    ):
        super().__init__(agent_id, position, direction)
        self.infection_time: int = infection_time  # Step en que fue infectado
        self.agents_infected: int = 0  # Número de agentes que ha infectado

    @property
    def is_infected(self) -> bool:
        return True

    @property
    def agent_type(self) -> str:
        return "infected"

    def infect_agent(self) -> None:
        """Registra que este agente infectó a otro."""
        self.agents_infected += 1

    def to_dict(self) -> dict:
        """Convierte el agente a diccionario."""
        data = super().to_dict()
        data["infection_time"] = self.infection_time
        data["agents_infected"] = self.agents_infected
        return data

    def __repr__(self) -> str:
        dir_str = ["→", "↓", "←", "↑"][self.direction]
        return f"InfectedAgent(id={self.id}, pos={self.position}, dir={dir_str}, infected={self.agents_infected})"
