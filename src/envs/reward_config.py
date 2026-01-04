"""
Configuracion de Rewards Progresivos
====================================
Sistema de rewards que progresa de sparse a dense segun la fase del curriculum.

IMPORTANTE: Los infected necesitan rewards mas agresivos porque:
1. Estan en desventaja numerica (2 vs 8)
2. Necesitan coordinar para acorralar presas
3. Los healthy aprenden a huir muy rapido

Solucion: Dar rewards de approach desde el principio y aumentar bonus de infeccion.
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
    - SPARSE: Solo rewards de victoria/derrota (fase 1)
    - INTERMEDIATE: Rewards intermedios al 50% (fase 2)
    - DENSE: Sistema completo de rewards (fase 3)

    CAMBIOS v2 (mejora de infected):
    - Infected reciben approach_bonus desde SPARSE (aprender a perseguir temprano)
    - Aumentado reward_infect_agent: 15 -> 25
    - Aumentado reward_all_infected_bonus: 20 -> 35
    - Aumentado reward_approach_bonus en todas las fases
    - Añadido reward_no_progress_penalty para infected pasivos
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
                # Sanos - Solo victoria/derrota
                "reward_survive_step": 0.0,
                "reward_distance_bonus": 0.0,
                "reward_infected_penalty": -10.0,
                "reward_not_moving_penalty": 0.0,
                "reward_stuck_penalty": 0.0,
                "stuck_threshold": 3,
                "reward_survive_episode": 10.0,
                # Infectados - Victoria + approach basico (CLAVE: aprender a perseguir)
                "reward_infect_agent": 25.0,  # Aumentado de 15
                "reward_approach_bonus": 0.15,  # NUEVO: dar feedback de approach desde fase 1
                "reward_step_penalty": 0.0,
                "reward_all_infected_bonus": 35.0,  # Aumentado de 20
                "reward_exploration": 0.0,
                "reward_no_progress_penalty": -0.05,  # Penalizar no acercarse
                "reward_progress_bonus": 0.05,  # Bonus por acercarse (pathfinding)
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
                # Infectados - Approach mas fuerte
                "reward_infect_agent": 25.0,  # Aumentado de 15
                "reward_approach_bonus": 0.25,  # Aumentado de 0.1
                "reward_step_penalty": 0.0,
                "reward_all_infected_bonus": 35.0,  # Aumentado de 20
                "reward_exploration": 0.01,
                "reward_no_progress_penalty": -0.08,  # Penalizar mas
                "reward_progress_bonus": 0.1,  # Bonus por acercarse (pathfinding)
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
                # Infectados - Maxima agresividad
                "reward_infect_agent": 25.0,  # Aumentado de 15
                "reward_approach_bonus": 0.35,  # Aumentado de 0.2
                "reward_step_penalty": -0.01,
                "reward_all_infected_bonus": 35.0,  # Aumentado de 20
                "reward_exploration": 0.02,
                "reward_no_progress_penalty": -0.1,  # Maxima penalizacion
                "reward_progress_bonus": 0.15,  # Bonus por acercarse (pathfinding)
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
