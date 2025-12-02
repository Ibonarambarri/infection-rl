"""
Multi-Agent Infection Environment
=================================
Entorno de RL con HealthyAgent e InfectedAgent.
"""

__version__ = "0.1.0"

from .envs import InfectionEnv, EnvConfig, make_infection_env
from .agents import BaseAgent, HealthyAgent, InfectedAgent, AgentCollection, Direction

__all__ = [
    "InfectionEnv",
    "EnvConfig",
    "make_infection_env",
    "BaseAgent",
    "HealthyAgent",
    "InfectedAgent",
    "AgentCollection",
    "Direction",
]
