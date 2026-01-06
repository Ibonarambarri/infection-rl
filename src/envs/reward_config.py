"""
Configuracion de Rewards Progresivos
====================================
Sistema de rewards que progresa de sparse a dense segun la fase del curriculum.

Curriculum para deployment 8 healthy vs 2 infected:
- SPARSE (Fase 1): Solo victoria/derrota - aprender objetivo final
- INTERMEDIATE (Fase 2): Rewards de progreso - aprender pathfinding
- DENSE (Fase 3): Sistema completo - fine-tuning

IMPORTANTE: Usamos progress_bonus (reduccion de distancia) en lugar de
approach_bonus (proximidad) para evitar que infectados orbiten sin atrapar.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RewardPreset(Enum):
    """Niveles de densidad de rewards para curriculum learning."""
    SPARSE = "sparse"           # Fase 1: Solo victoria/derrota
    INTERMEDIATE = "intermediate"  # Fase 2: Signals basicos (50%)
    DENSE = "dense"             # Fase 3: Sistema completo


@dataclass
class RewardConfig:
    """
    Configuracion de rewards para el entorno de infeccion.

    Soporta 3 presets para curriculum learning:
    - SPARSE: Solo rewards de victoria/derrota (verdaderamente sparse)
    - INTERMEDIATE: Rewards de progreso (progress_bonus para pathfinding)
    - DENSE: Sistema completo de rewards (fine-tuning)

    CAMBIOS v3 (normalizacion y fix de incentivos):
    - SPARSE ahora es verdaderamente sparse (sin rewards por paso)
    - Eliminado approach_bonus (causaba orbiting sin atrapar)
    - Solo usamos progress_bonus (premia REDUCIR distancia, no proximidad)
    - Normalizado reward_infect_agent: 25 -> 10
    - Normalizado reward_all_infected_bonus: 35 -> 15
    """

    # Preset base (define defaults)
    preset: RewardPreset = RewardPreset.DENSE

    # === Rewards para SANOS (HealthyAgent) ===
    reward_survive_step: Optional[float] = None
    reward_distance_bonus: Optional[float] = None
    reward_infected_penalty: Optional[float] = None
    reward_not_moving_penalty: Optional[float] = None
    reward_stuck_penalty: Optional[float] = None
    stuck_threshold: Optional[int] = None
    reward_survive_episode: Optional[float] = None

    # === Rewards para INFECTADOS (InfectedAgent) ===
    reward_infect_agent: Optional[float] = None
    reward_approach_bonus: Optional[float] = None
    reward_step_penalty: Optional[float] = None
    reward_all_infected_bonus: Optional[float] = None
    reward_exploration: Optional[float] = None
    reward_no_progress_penalty: Optional[float] = None  # Penaliza no acercarse
    reward_progress_bonus: Optional[float] = None  # Bonus por reducir distancia BFS

    def __post_init__(self):
        """Aplica valores del preset para campos no especificados."""
        defaults = self._get_preset_defaults()
        for field_name, default_value in defaults.items():
            if getattr(self, field_name) is None:
                setattr(self, field_name, default_value)

    def _get_preset_defaults(self) -> dict:
        """Retorna los valores por defecto segun el preset."""
        if self.preset == RewardPreset.SPARSE:
            return {
                # Sanos - SOLO victoria/derrota (verdaderamente sparse)
                "reward_survive_step": 0.0,
                "reward_distance_bonus": 0.0,
                "reward_infected_penalty": -10.0,
                "reward_not_moving_penalty": 0.0,
                "reward_stuck_penalty": 0.0,
                "stuck_threshold": 3,
                "reward_survive_episode": 10.0,
                # Infectados - SOLO victoria/derrota (verdaderamente sparse)
                "reward_infect_agent": 10.0,  # Reducido de 25 para balance
                "reward_approach_bonus": 0.0,  # Sin rewards por paso (sparse)
                "reward_step_penalty": 0.0,
                "reward_all_infected_bonus": 15.0,  # Reducido de 35 para balance
                "reward_exploration": 0.0,
                "reward_no_progress_penalty": 0.0,  # Sin penalties por paso (sparse)
                "reward_progress_bonus": 0.0,  # Sin rewards por paso (sparse)
            }
        elif self.preset == RewardPreset.INTERMEDIATE:
            return {
                # Sanos - 50% intensidad
                "reward_survive_step": 0.05,
                "reward_distance_bonus": 0.05,
                "reward_infected_penalty": -10.0,
                "reward_not_moving_penalty": -0.025,
                "reward_stuck_penalty": 0.0,
                "stuck_threshold": 3,
                "reward_survive_episode": 7.5,
                # Infectados - Rewards intermedios
                "reward_infect_agent": 10.0,  # Normalizado
                "reward_approach_bonus": 0.0,  # Solo progress, no proximity
                "reward_step_penalty": 0.0,
                "reward_all_infected_bonus": 15.0,  # Normalizado
                "reward_exploration": 0.01,
                "reward_no_progress_penalty": -0.1,   # Aumentado
                "reward_progress_bonus": 0.2,  # Bonus por REDUCIR distancia (aumentado)
            }
        else:  # DENSE (default) - Sistema completo
            return {
                # Sanos
                "reward_survive_step": 0.1,
                "reward_distance_bonus": 0.1,
                "reward_infected_penalty": -10.0,
                "reward_not_moving_penalty": -0.05,
                "reward_stuck_penalty": -0.2,
                "stuck_threshold": 3,
                "reward_survive_episode": 5.0,
                # Infectados - Sistema completo normalizado
                "reward_infect_agent": 10.0,  # Normalizado de 25
                "reward_approach_bonus": 0.0,  # Solo progress, no proximity
                "reward_step_penalty": -0.01,
                "reward_all_infected_bonus": 15.0,  # Normalizado de 35
                "reward_exploration": 0.02,
                "reward_no_progress_penalty": -0.15,  # Penalty por alejarse (aumentado)
                "reward_progress_bonus": 0.4,  # Bonus por REDUCIR distancia (aumentado 2x)
            }

    @classmethod
    def from_preset(cls, preset: RewardPreset) -> "RewardConfig":
        """Crea una configuracion desde un preset."""
        return cls(preset=preset)

    @classmethod
    def sparse(cls) -> "RewardConfig":
        """Atajo para crear config sparse (Fase 1)."""
        return cls.from_preset(RewardPreset.SPARSE)

    @classmethod
    def intermediate(cls) -> "RewardConfig":
        """Atajo para crear config intermediate (Fase 2)."""
        return cls.from_preset(RewardPreset.INTERMEDIATE)

    @classmethod
    def dense(cls) -> "RewardConfig":
        """Atajo para crear config dense (Fase 3)."""
        return cls.from_preset(RewardPreset.DENSE)
