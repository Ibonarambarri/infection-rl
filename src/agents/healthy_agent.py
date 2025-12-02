"""
Healthy Agent Class
===================
Agente sano que huye de los infectados.
"""

from typing import Tuple, TYPE_CHECKING
from .base_agent import BaseAgent, Direction

if TYPE_CHECKING:
    from .infected_agent import InfectedAgent


class HealthyAgent(BaseAgent):
    """
    Agente sano.

    Objetivo: Sobrevivir el mayor tiempo posible evitando a los infectados.

    Cuando es infectado, se transforma en un InfectedAgent.
    """

    def __init__(
        self,
        agent_id: int,
        position: Tuple[int, int],
        direction: Direction = Direction.RIGHT,
    ):
        super().__init__(agent_id, position, direction)
        self.steps_survived: int = 0

    @property
    def is_infected(self) -> bool:
        return False

    @property
    def agent_type(self) -> str:
        return "healthy"

    def step(self) -> None:
        """Llamado cada step - cuenta supervivencia."""
        super().step()
        self.steps_survived += 1

    def to_infected(self, infection_step: int) -> "InfectedAgent":
        """
        Transforma este agente sano en infectado.

        Args:
            infection_step: Step en el que ocurrió la infección

        Returns:
            Nuevo InfectedAgent con los datos de este agente
        """
        from .infected_agent import InfectedAgent

        infected = InfectedAgent(
            agent_id=self.id,
            position=self.position,
            direction=self.direction,
            infection_time=infection_step,
        )
        infected.total_reward = self.total_reward
        infected.steps_alive = self.steps_alive
        return infected

    def to_dict(self) -> dict:
        """Convierte el agente a diccionario."""
        data = super().to_dict()
        data["steps_survived"] = self.steps_survived
        return data

    def __repr__(self) -> str:
        dir_str = ["→", "↓", "←", "↑"][self.direction]
        return f"HealthyAgent(id={self.id}, pos={self.position}, dir={dir_str}, survived={self.steps_survived})"
