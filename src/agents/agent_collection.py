"""
Agent Collection
================
Gestiona la colección de agentes en el entorno.
"""

from typing import Dict, List, Optional, Tuple, Union
from .base_agent import BaseAgent, Direction
from .healthy_agent import HealthyAgent
from .infected_agent import InfectedAgent


class AgentCollection:
    """
    Colección de agentes con métodos de utilidad.

    Gestiona tanto HealthyAgent como InfectedAgent.
    """

    def __init__(self):
        self._agents: Dict[int, BaseAgent] = {}
        self._next_id: int = 0

    def add_healthy(
        self,
        position: Tuple[int, int],
        direction: Direction = Direction.RIGHT,
    ) -> HealthyAgent:
        """Crea y añade un nuevo agente sano."""
        agent = HealthyAgent(
            agent_id=self._next_id,
            position=position,
            direction=direction,
        )
        self._agents[agent.id] = agent
        self._next_id += 1
        return agent

    def add_infected(
        self,
        position: Tuple[int, int],
        direction: Direction = Direction.RIGHT,
        infection_time: int = 0,
    ) -> InfectedAgent:
        """Crea y añade un nuevo agente infectado."""
        agent = InfectedAgent(
            agent_id=self._next_id,
            position=position,
            direction=direction,
            infection_time=infection_time,
        )
        self._agents[agent.id] = agent
        self._next_id += 1
        return agent

    def infect_agent(self, agent_id: int, current_step: int, infected_by: Optional[int] = None) -> Optional[InfectedAgent]:
        """
        Infecta un agente sano, transformándolo en infectado.

        Args:
            agent_id: ID del agente a infectar
            current_step: Step actual del entorno
            infected_by: ID del agente que causó la infección

        Returns:
            El nuevo InfectedAgent, o None si el agente no existía o ya estaba infectado
        """
        agent = self._agents.get(agent_id)

        if agent is None or agent.is_infected:
            return None

        # Transformar HealthyAgent -> InfectedAgent
        healthy_agent: HealthyAgent = agent
        infected_agent = healthy_agent.to_infected(current_step)

        # Reemplazar en la colección
        self._agents[agent_id] = infected_agent

        # Registrar infección en el agente que infectó
        if infected_by is not None:
            infector = self._agents.get(infected_by)
            if infector and isinstance(infector, InfectedAgent):
                infector.infect_agent()

        return infected_agent

    def get(self, agent_id: int) -> Optional[BaseAgent]:
        """Obtiene un agente por ID."""
        return self._agents.get(agent_id)

    def remove(self, agent_id: int) -> Optional[BaseAgent]:
        """Elimina y retorna un agente."""
        return self._agents.pop(agent_id, None)

    def clear(self) -> None:
        """Elimina todos los agentes."""
        self._agents.clear()
        self._next_id = 0

    @property
    def all(self) -> List[BaseAgent]:
        """Lista de todos los agentes."""
        return list(self._agents.values())

    @property
    def healthy(self) -> List[HealthyAgent]:
        """Lista de agentes sanos."""
        return [a for a in self._agents.values() if isinstance(a, HealthyAgent)]

    @property
    def infected(self) -> List[InfectedAgent]:
        """Lista de agentes infectados."""
        return [a for a in self._agents.values() if isinstance(a, InfectedAgent)]

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

    def get_positions(self) -> set:
        """Retorna el conjunto de posiciones ocupadas."""
        return {a.position for a in self._agents.values()}

    def get_agent_at(self, position: Tuple[int, int]) -> Optional[BaseAgent]:
        """Retorna el agente en una posición dada, si existe."""
        for agent in self._agents.values():
            if agent.position == position:
                return agent
        return None

    def find_nearest_healthy(self, from_agent: BaseAgent) -> Optional[HealthyAgent]:
        """Encuentra el agente sano más cercano."""
        healthy_agents = self.healthy
        if not healthy_agents:
            return None

        min_dist = float('inf')
        nearest = None
        for agent in healthy_agents:
            if agent.id != from_agent.id:
                dist = from_agent.distance_to(agent)
                if dist < min_dist:
                    min_dist = dist
                    nearest = agent
        return nearest

    def find_nearest_infected(self, from_agent: BaseAgent) -> Optional[InfectedAgent]:
        """Encuentra el agente infectado más cercano."""
        infected_agents = self.infected
        if not infected_agents:
            return None

        min_dist = float('inf')
        nearest = None
        for agent in infected_agents:
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

    def __getitem__(self, agent_id: int) -> BaseAgent:
        return self._agents[agent_id]

    def __contains__(self, agent_id: int) -> bool:
        return agent_id in self._agents
