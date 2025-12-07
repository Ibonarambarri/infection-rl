#!/usr/bin/env python3
"""
Map Generator for Curriculum Learning
======================================
Genera mapas de diferentes tamaños para el curriculum de entrenamiento.

Tipos de mapas:
- small.txt: 20x20 (Entrenamiento rápido, alta densidad)
- medium.txt: 40x40 (Estándar)
- large.txt: 60x60 (Test de generalización)

Los mapas incluyen muros aleatorios y obstáculos, asegurando navegabilidad.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List, Set
from collections import deque
import numpy as np


class CurriculumMapGenerator:
    """
    Generador de mapas para curriculum learning.

    Genera mapas navegables con muros y obstáculos aleatorios.
    Verifica conectividad usando BFS para asegurar navegabilidad.
    """

    # Configuraciones predefinidas para curriculum
    PRESETS = {
        "small": {
            "size": 20,
            "wall_density": 0.08,
            "obstacle_density": 0.05,
            "min_open_ratio": 0.7,
        },
        "medium": {
            "size": 40,
            "wall_density": 0.06,
            "obstacle_density": 0.04,
            "min_open_ratio": 0.75,
        },
        "large": {
            "size": 60,
            "wall_density": 0.05,
            "obstacle_density": 0.03,
            "min_open_ratio": 0.8,
        },
    }

    # Caracteres del mapa
    EMPTY = '.'
    WALL = '#'
    OBSTACLE = 'O'

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        size: int,
        wall_density: float = 0.05,
        obstacle_density: float = 0.03,
        min_open_ratio: float = 0.75,
        max_attempts: int = 100,
    ) -> np.ndarray:
        """
        Genera un mapa navegable.

        Args:
            size: Tamaño del mapa (size x size)
            wall_density: Densidad de muros internos (0-1)
            obstacle_density: Densidad de obstáculos (0-1)
            min_open_ratio: Ratio mínimo de celdas conectadas
            max_attempts: Intentos máximos para generar mapa válido

        Returns:
            Mapa como array 2D de caracteres
        """
        for attempt in range(max_attempts):
            grid = self._create_base_grid(size)
            grid = self._add_internal_walls(grid, wall_density)
            grid = self._add_obstacles(grid, obstacle_density)

            # Verificar navegabilidad
            connected_ratio = self._check_connectivity(grid)

            if connected_ratio >= min_open_ratio:
                return grid

            # Si no es navegable, reducir densidad y reintentar
            wall_density *= 0.9
            obstacle_density *= 0.9

        # Fallback: mapa vacío con solo bordes
        return self._create_base_grid(size)

    def _create_base_grid(self, size: int) -> np.ndarray:
        """Crea grid base con bordes de muros."""
        grid = np.full((size, size), self.EMPTY, dtype='<U1')

        # Bordes
        grid[0, :] = self.WALL
        grid[-1, :] = self.WALL
        grid[:, 0] = self.WALL
        grid[:, -1] = self.WALL

        return grid

    def _add_internal_walls(self, grid: np.ndarray, density: float) -> np.ndarray:
        """Añade muros internos con patrones estructurados."""
        size = grid.shape[0]

        # Número de segmentos de muro a añadir
        num_segments = int((size * size) * density / 5)

        for _ in range(num_segments):
            # Punto de inicio aleatorio (evitando bordes)
            x = self.rng.integers(2, size - 2)
            y = self.rng.integers(2, size - 2)

            # Longitud y dirección del segmento
            length = self.rng.integers(2, min(8, size // 4))
            horizontal = self.rng.random() > 0.5

            if horizontal:
                for i in range(length):
                    nx = x + i
                    if 1 < nx < size - 1:
                        grid[y, nx] = self.WALL
            else:
                for i in range(length):
                    ny = y + i
                    if 1 < ny < size - 1:
                        grid[ny, x] = self.WALL

        return grid

    def _add_obstacles(self, grid: np.ndarray, density: float) -> np.ndarray:
        """Añade obstáculos dispersos."""
        size = grid.shape[0]
        num_obstacles = int((size - 2) ** 2 * density)

        # Posiciones internas disponibles
        available = []
        for y in range(1, size - 1):
            for x in range(1, size - 1):
                if grid[y, x] == self.EMPTY:
                    available.append((x, y))

        if len(available) > num_obstacles:
            indices = self.rng.choice(len(available), size=num_obstacles, replace=False)
            for idx in indices:
                x, y = available[idx]
                grid[y, x] = self.OBSTACLE

        return grid

    def _check_connectivity(self, grid: np.ndarray) -> float:
        """
        Verifica qué proporción de celdas vacías están conectadas.

        Returns:
            Ratio de celdas conectadas (0-1)
        """
        size = grid.shape[0]

        # Encontrar primera celda vacía
        start = None
        empty_cells = set()

        for y in range(1, size - 1):
            for x in range(1, size - 1):
                if grid[y, x] == self.EMPTY:
                    empty_cells.add((x, y))
                    if start is None:
                        start = (x, y)

        if not empty_cells:
            return 0.0

        # BFS para encontrar celdas conectadas
        visited = set()
        queue = deque([start])
        visited.add(start)

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while queue:
            x, y = queue.popleft()

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if (nx, ny) not in visited and (nx, ny) in empty_cells:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return len(visited) / len(empty_cells)

    def generate_preset(self, preset: str) -> np.ndarray:
        """Genera mapa usando configuración predefinida."""
        if preset not in self.PRESETS:
            raise ValueError(f"Preset desconocido: {preset}. Opciones: {list(self.PRESETS.keys())}")

        config = self.PRESETS[preset]
        return self.generate(
            size=config["size"],
            wall_density=config["wall_density"],
            obstacle_density=config["obstacle_density"],
            min_open_ratio=config["min_open_ratio"],
        )

    def save_map(self, grid: np.ndarray, filepath: str) -> None:
        """Guarda el mapa en un archivo de texto."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            for row in grid:
                f.write(''.join(row) + '\n')

        print(f"Mapa guardado: {filepath} ({grid.shape[0]}x{grid.shape[1]})")

    def load_map(self, filepath: str) -> np.ndarray:
        """Carga un mapa desde archivo."""
        with open(filepath, 'r') as f:
            lines = f.read().strip().split('\n')

        height = len(lines)
        width = max(len(line) for line in lines)

        grid = np.full((height, width), self.EMPTY, dtype='<U1')

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                grid[y, x] = char

        return grid

    def get_map_stats(self, grid: np.ndarray) -> dict:
        """Obtiene estadísticas del mapa."""
        total = grid.size
        empty = np.sum(grid == self.EMPTY)
        walls = np.sum(grid == self.WALL)
        obstacles = np.sum(grid == self.OBSTACLE)

        return {
            "size": f"{grid.shape[0]}x{grid.shape[1]}",
            "total_cells": total,
            "empty": empty,
            "walls": walls,
            "obstacles": obstacles,
            "empty_ratio": empty / total,
            "connectivity": self._check_connectivity(grid),
        }


def generate_curriculum_maps(output_dir: str, seed: int = None) -> None:
    """
    Genera los tres mapas del curriculum.

    Args:
        output_dir: Directorio donde guardar los mapas
        seed: Semilla para reproducibilidad
    """
    generator = CurriculumMapGenerator(seed=seed)

    presets = ["small", "medium", "large"]

    print("=" * 50)
    print("Generando mapas para Curriculum Learning")
    print("=" * 50)

    for preset in presets:
        filepath = os.path.join(output_dir, f"{preset}.txt")

        print(f"\nGenerando {preset}...")
        grid = generator.generate_preset(preset)
        generator.save_map(grid, filepath)

        stats = generator.get_map_stats(grid)
        print(f"  Tamaño: {stats['size']}")
        print(f"  Celdas vacías: {stats['empty']} ({stats['empty_ratio']:.1%})")
        print(f"  Muros: {stats['walls']}")
        print(f"  Obstáculos: {stats['obstacles']}")
        print(f"  Conectividad: {stats['connectivity']:.1%}")

    print("\n" + "=" * 50)
    print("Mapas generados exitosamente!")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Generador de mapas para Curriculum Learning"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="maps",
        help="Directorio de salida para los mapas (default: maps)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla para reproducibilidad"
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=["small", "medium", "large", "all"],
        default="all",
        help="Preset a generar (default: all)"
    )

    parser.add_argument(
        "--custom-size",
        type=int,
        default=None,
        help="Tamaño personalizado (ignora preset)"
    )

    parser.add_argument(
        "--wall-density",
        type=float,
        default=0.05,
        help="Densidad de muros para tamaño personalizado"
    )

    parser.add_argument(
        "--obstacle-density",
        type=float,
        default=0.03,
        help="Densidad de obstáculos para tamaño personalizado"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Nombre del archivo de salida (para mapa personalizado)"
    )

    args = parser.parse_args()

    generator = CurriculumMapGenerator(seed=args.seed)

    if args.custom_size:
        # Generar mapa personalizado
        grid = generator.generate(
            size=args.custom_size,
            wall_density=args.wall_density,
            obstacle_density=args.obstacle_density,
        )

        filename = args.output_file or f"custom_{args.custom_size}x{args.custom_size}.txt"
        filepath = os.path.join(args.output_dir, filename)
        generator.save_map(grid, filepath)

        stats = generator.get_map_stats(grid)
        print(f"\nEstadísticas:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.preset == "all":
        generate_curriculum_maps(args.output_dir, args.seed)

    else:
        # Generar preset específico
        grid = generator.generate_preset(args.preset)
        filepath = os.path.join(args.output_dir, f"{args.preset}.txt")
        generator.save_map(grid, filepath)

        stats = generator.get_map_stats(grid)
        print(f"\nEstadísticas:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
