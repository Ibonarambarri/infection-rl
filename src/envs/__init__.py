"""
Infection Environment
"""

from .environment import InfectionEnv, EnvConfig
from .map_generator import MapGenerator, CellType
from .reward_config import RewardConfig, RewardPreset
from .wrappers import (
    make_infection_env,
    make_vec_env_parameter_sharing,
    SingleAgentWrapper,
    FlattenObservationWrapper,
    DictObservationWrapper,
    MultiAgentToSingleAgentWrapper,
)

__all__ = [
    "InfectionEnv",
    "EnvConfig",
    "MapGenerator",
    "CellType",
    "RewardConfig",
    "RewardPreset",
    "make_infection_env",
    "make_vec_env_parameter_sharing",
    "SingleAgentWrapper",
    "FlattenObservationWrapper",
    "DictObservationWrapper",
    "MultiAgentToSingleAgentWrapper",
]
