"""
Map Loader
==========
Carga mapas desde archivos de texto.
"""

from enum import Enum
from typing import Tuple, List, Set, Optional
from dataclasses import dataclass
import numpy as np


class CellType(Enum):
    """Tipos de celdas en el mapa."""
    EMPTY = 0
    WALL = 1
    OBSTACLE = 2


@dataclass
class MapConfig:
    """Configuración del mapa."""
    map_file: Optional[str] = None
    map_data: Optional[str] = None  # String con el contenido del mapa
    seed: Optional[int] = None
    width: int = 0   # Se actualiza al cargar
    height: int = 0  # Se actualiza al cargar


class MapGenerator:
    """
    Cargador de mapas desde archivo o string.

    Formato del mapa:
        # = muro (WALL)
        . = vacío (EMPTY)
        O = obstáculo (OBSTACLE)
        (espacio) = vacío (EMPTY)
    """

    def __init__(self, config: MapConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.grid: np.ndarray = None

        # Cargar mapa desde string o archivo
        if config.map_data is not None:
            self._load_from_string(config.map_data)
        elif config.map_file is not None:
            self._load_from_file(config.map_file)
        else:
            raise ValueError("Debe proporcionar map_file o map_data")

    def _load_from_file(self, filepath: str) -> None:
        """Carga un mapa desde un archivo de texto."""
        with open(filepath, 'r') as f:
            content = f.read().strip()
        self._parse_map_content(content)

    def _load_from_string(self, map_data: str) -> None:
        """Carga un mapa desde un string."""
        self._parse_map_content(map_data.strip())

    def _parse_map_content(self, content: str) -> None:
        """Parsea el contenido del mapa (común para archivo y string)."""
        lines = content.split('\n')

        # Determinar dimensiones
        height = len(lines)
        width = max(len(line) for line in lines)

        # Actualizar configuración
        self.config.width = width
        self.config.height = height

        # Crear grid
        self.grid = np.zeros((height, width), dtype=np.int8)

        # Mapeo de caracteres
        char_map = {
            '#': CellType.WALL.value,
            '.': CellType.EMPTY.value,
            'O': CellType.OBSTACLE.value,
            ' ': CellType.EMPTY.value,
        }

        # Llenar el grid
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if x < width:
                    self.grid[y, x] = char_map.get(char, CellType.EMPTY.value)
            # Rellenar si la línea es más corta
            for x in range(len(line), width):
                self.grid[y, x] = CellType.EMPTY.value

    def get_valid_positions(self) -> List[Tuple[int, int]]:
        """Retorna lista de posiciones vacías para colocar agentes."""
        positions = []
        for y in range(1, self.config.height - 1):
            for x in range(1, self.config.width - 1):
                if self.grid[y, x] == CellType.EMPTY.value:
                    positions.append((x, y))
        return positions

    def get_random_valid_positions(self, n: int, exclude: Set[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """Retorna n posiciones válidas aleatorias sin repetición."""
        valid = self.get_valid_positions()

        if exclude:
            valid = [p for p in valid if p not in exclude]

        if len(valid) < n:
            raise ValueError(f"No hay suficientes posiciones. Requeridas: {n}, Disponibles: {len(valid)}")

        indices = self.rng.choice(len(valid), size=n, replace=False)
        return [valid[i] for i in indices]

    def is_valid_position(self, x: int, y: int) -> bool:
        """Verifica si una posición es válida (dentro del mapa y vacía)."""
        if x < 0 or x >= self.config.width or y < 0 or y >= self.config.height:
            return False
        return self.grid[y, x] == CellType.EMPTY.value

    def is_wall(self, x: int, y: int) -> bool:
        """Verifica si una posición es un muro."""
        if x < 0 or x >= self.config.width or y < 0 or y >= self.config.height:
            return True
        return self.grid[y, x] == CellType.WALL.value

    def is_blocked(self, x: int, y: int) -> bool:
        """Verifica si una posición está bloqueada (muro u obstáculo)."""
        if x < 0 or x >= self.config.width or y < 0 or y >= self.config.height:
            return True
        return self.grid[y, x] != CellType.EMPTY.value
