"""
Agent Classes
"""

from .base_agent import BaseAgent, Direction
from .healthy_agent import HealthyAgent
from .infected_agent import InfectedAgent
from .agent_collection import AgentCollection

__all__ = [
    "BaseAgent",
    "HealthyAgent",
    "InfectedAgent",
    "AgentCollection",
    "Direction",
]
