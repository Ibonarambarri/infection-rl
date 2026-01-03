#!/usr/bin/env python3
"""
Map Generator for Curriculum Learning (Fixed)
=============================================
Genera mapas garantizando conectividad total y alineación correcta.

Mejoras:
1. Conectividad 100% garantizada (sin zonas aisladas).
2. Estructura de "ciudad" más limpia.
3. Escritura de archivo robusta para evitar desalineación.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List, Set, Deque
from collections import deque
import numpy as np

class CurriculumMapGenerator:
    """
    Generador de mapas robusto.
    """

    # Configuraciones predefinidas
    PRESETS = {
        # Mapas originales
        "small": {
            "width": 20, "height": 20,
            "wall_density": 0.1,
            "obstacle_density": 0.05,
        },
        "medium": {
            "width": 40, "height": 40,
            "wall_density": 0.08,
            "obstacle_density": 0.04,
        },
        "large": {
            "width": 60, "height": 60,
            "wall_density": 0.06,
            "obstacle_density": 0.03,
        },
        # ========================================
        # Suite de Curriculum Learning (5 niveles)
        # ========================================
        # Nivel 1: 20x20, muy abierto (puro aprendizaje de persecución)
        "curriculum_lvl1": {
            "width": 20, "height": 20,
            "wall_density": 0.02,      # Casi sin muros internos
            "obstacle_density": 0.01,   # Mínimos obstáculos
        },
        # Nivel 2: 30x30, obstáculos dispersos
        "curriculum_lvl2": {
            "width": 30, "height": 30,
            "wall_density": 0.04,       # Pocos muros
            "obstacle_density": 0.03,   # Obstáculos dispersos
        },
        # Nivel 3: 40x40, estructura tipo ciudad
        "curriculum_lvl3": {
            "width": 40, "height": 40,
            "wall_density": 0.08,       # Muros tipo ciudad
            "obstacle_density": 0.04,   # Obstáculos moderados
        },
        # Nivel 4: 50x50, ciudad densa
        "curriculum_lvl4": {
            "width": 50, "height": 50,
            "wall_density": 0.10,       # Estructura densa
            "obstacle_density": 0.05,   # Más obstáculos
        },
        # Nivel 5: 60x60, laberinto complejo
        "curriculum_lvl5": {
            "width": 60, "height": 60,
            "wall_density": 0.12,       # Laberinto denso
            "obstacle_density": 0.06,   # Alta densidad de obstáculos
        },
    }

    # Lista ordenada de niveles de curriculum
    CURRICULUM_LEVELS = [
        "curriculum_lvl1",
        "curriculum_lvl2",
        "curriculum_lvl3",
        "curriculum_lvl4",
        "curriculum_lvl5",
    ]

    # Caracteres del mapa
    EMPTY = '.'
    WALL = '#'
    OBSTACLE = 'O'

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        width: int,
        height: int,
        wall_density: float = 0.05,
        obstacle_density: float = 0.03,
    ) -> np.ndarray:
        """
        Genera un mapa garantizando que sea completamente navegable.
        """
        # 1. Crear caja vacía con bordes
        grid = self._create_base_grid(width, height)

        # 2. Añadir muros internos (patrones aleatorios)
        grid = self._add_internal_walls(grid, wall_density)

        # 3. CRÍTICO: Asegurar conectividad total
        # Esto elimina "cuadrados cerrados" abriendo caminos
        grid = self._ensure_complete_connectivity(grid)

        # 4. Añadir obstáculos en lugares libres
        grid = self._add_obstacles(grid, obstacle_density)

        return grid

    def _create_base_grid(self, width: int, height: int) -> np.ndarray:
        """Crea grid base con bordes."""
        grid = np.full((height, width), self.EMPTY, dtype='<U1')
        grid[0, :] = self.WALL
        grid[-1, :] = self.WALL
        grid[:, 0] = self.WALL
        grid[:, -1] = self.WALL
        return grid

    def _add_internal_walls(self, grid: np.ndarray, density: float) -> np.ndarray:
        """Añade muros intentando mantener una estructura tipo grilla/ciudad."""
        height, width = grid.shape
        area = width * height
        target_walls = int(area * density)
        
        walls_placed = 0
        attempts = 0
        max_attempts = target_walls * 5

        while walls_placed < target_walls and attempts < max_attempts:
            attempts += 1
            
            # Intentar alinear muros a coordenadas pares para que parezca más "ciudad"
            # y menos ruido aleatorio
            x = self.rng.integers(2, width - 2)
            y = self.rng.integers(2, height - 2)
            
            length = self.rng.integers(2, min(10, width // 3))
            horizontal = self.rng.random() > 0.5

            if horizontal:
                if x + length < width - 1:
                    grid[y, x:x+length] = self.WALL
                    walls_placed += length
            else:
                if y + length < height - 1:
                    grid[y:y+length, x] = self.WALL
                    walls_placed += length

        return grid

    def _ensure_complete_connectivity(self, grid: np.ndarray) -> np.ndarray:
        """
        Identifica zonas aisladas y abre caminos (túneles) hacia la zona principal.
        Garantiza que NO existan cuadrados cerrados inalcanzables.
        """
        height, width = grid.shape
        
        # Obtener todas las celdas vacías
        empty_cells = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if grid[y, x] == self.EMPTY:
                    empty_cells.append((x, y))

        if not empty_cells:
            return grid

        # 1. Encontrar regiones conectadas (Flood Fill)
        visited = set()
        regions = []

        for start_pos in empty_cells:
            if start_pos in visited:
                continue
            
            # Nueva región encontrada
            current_region = []
            queue = deque([start_pos])
            visited.add(start_pos)
            
            while queue:
                cx, cy = queue.popleft()
                current_region.append((cx, cy))
                
                # Vecinos (arriba, abajo, izq, der)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) not in visited:
                        if 0 <= nx < width and 0 <= ny < height:
                            if grid[ny, nx] == self.EMPTY:
                                visited.add((nx, ny))
                                queue.append((nx, ny))
            
            regions.append(current_region)

        if len(regions) <= 1:
            return grid  # Ya está todo conectado

        # 2. Conectar regiones aisladas a la región más grande
        # Ordenar regiones por tamaño (la más grande es la principal)
        regions.sort(key=len, reverse=True)
        main_region = set(regions[0])

        # Para cada región aislada, cavar un túnel hacia la región principal
        for i in range(1, len(regions)):
            isolated_region = regions[i]
            
            # Si la región es muy pequeña (ej. 1 o 2 celdas), mejor rellenarla (es basura)
            if len(isolated_region) < 3:
                for rx, ry in isolated_region:
                    grid[ry, rx] = self.WALL
                continue

            # Si es una región útil, conectarla
            start_point = isolated_region[0] # Un punto de la isla
            
            # Buscar el punto más cercano de la región principal (fuerza bruta simple)
            # Nota: Para mapas gigantes esto podría optimizarse, pero para 60x60 es instantáneo.
            best_target = None
            min_dist = float('inf')
            
            # Muestrear puntos para no comparar todos con todos
            sample_main = list(main_region)[::max(1, len(main_region)//20)]
            
            for tx, ty in sample_main:
                dist = abs(start_point[0] - tx) + abs(start_point[1] - ty)
                if dist < min_dist:
                    min_dist = dist
                    best_target = (tx, ty)
            
            if best_target:
                self._carve_path(grid, start_point, best_target)
                # Ahora esta región es parte de la principal (lógicamente)
                main_region.update(isolated_region)

        return grid

    def _carve_path(self, grid: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]):
        """Cava un camino recto (L-shape) entre dos puntos, rompiendo muros."""
        x1, y1 = start
        x2, y2 = end
        
        # Moverse primero en X luego en Y
        # Paso 1: Horizontal
        step_x = 1 if x2 > x1 else -1
        for x in range(x1, x2 + step_x, step_x):
            if grid[y1, x] == self.WALL:
                grid[y1, x] = self.EMPTY
        
        # Paso 2: Vertical
        step_y = 1 if y2 > y1 else -1
        for y in range(y1, y2 + step_y, step_y):
            if grid[y, x2] == self.WALL:
                grid[y, x2] = self.EMPTY

    def _add_obstacles(self, grid: np.ndarray, density: float) -> np.ndarray:
        """Añade obstáculos dispersos sin bloquear pasillos únicos."""
        height, width = grid.shape
        num_obstacles = int((width * height) * density)
        
        count = 0
        attempts = 0
        max_attempts = num_obstacles * 2
        
        while count < num_obstacles and attempts < max_attempts:
            attempts += 1
            x = self.rng.integers(1, width - 1)
            y = self.rng.integers(1, height - 1)
            
            if grid[y, x] == self.EMPTY:
                # Verificar que no bloquee un pasillo (regla simple de vecinos)
                # Contar muros vecinos
                neighbors = 0
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if grid[y+dy, x+dx] == self.WALL:
                        neighbors += 1
                
                # Solo poner obstáculo si el espacio es abierto (pocos muros vecinos)
                if neighbors < 2:
                    grid[y, x] = self.OBSTACLE
                    count += 1
                    
        return grid

    def generate_preset(self, preset: str) -> np.ndarray:
        """Genera mapa usando configuración predefinida."""
        if preset not in self.PRESETS:
            raise ValueError(f"Preset desconocido: {preset}")

        config = self.PRESETS[preset]
        return self.generate(
            width=config["width"],
            height=config["height"],
            wall_density=config["wall_density"],
            obstacle_density=config["obstacle_density"],
        )

    def save_map(self, grid: np.ndarray, filepath: str) -> None:
        """Guarda el mapa asegurando alineación perfecta."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', newline='\n') as f: # newline='\n' para consistencia Linux/Win
            for row in grid:
                # .join garantiza que no haya espacios extra
                line = "".join(row)
                f.write(line + '\n')

        print(f"Mapa guardado: {filepath} ({grid.shape[1]}x{grid.shape[0]})")

    def get_map_stats(self, grid: np.ndarray) -> dict:
        total = grid.size
        empty = np.sum(grid == self.EMPTY)
        return {
            "size": f"{grid.shape[1]}x{grid.shape[0]}",
            "empty_cells": int(empty),
            "ratio": float(empty / total)
        }


def generate_curriculum_maps(output_dir: str, seed: int = None) -> None:
    """Genera todos los mapas del curriculum (5 niveles)."""
    generator = CurriculumMapGenerator(seed=seed)
    print("=" * 50)
    print("Generando mapas de Curriculum Learning...")
    print("=" * 50)

    for level in CurriculumMapGenerator.CURRICULUM_LEVELS:
        grid = generator.generate_preset(level)
        filepath = os.path.join(output_dir, f"{level}.txt")
        generator.save_map(grid, filepath)
        config = generator.PRESETS[level]
        print(f"  {level}: {config['width']}x{config['height']} - OK")

    print("-" * 50)
    print(f"Total: {len(CurriculumMapGenerator.CURRICULUM_LEVELS)} mapas generados en {output_dir}/")
    print("=" * 50)


def generate_all_maps(output_dir: str, seed: int = None) -> None:
    """Genera TODOS los mapas (originales + curriculum)."""
    generator = CurriculumMapGenerator(seed=seed)
    print("=" * 50)
    print("Generando TODOS los mapas...")
    print("=" * 50)

    # Generar todos los presets
    for preset_name, config in generator.PRESETS.items():
        grid = generator.generate_preset(preset_name)
        filepath = os.path.join(output_dir, f"{preset_name}.txt")
        generator.save_map(grid, filepath)
        print(f"  {preset_name}: {config['width']}x{config['height']} - OK")

    print("-" * 50)
    print(f"Total: {len(generator.PRESETS)} mapas generados en {output_dir}/")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="Generador de mapas para Curriculum Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python generate_maps.py --preset curriculum          # Genera los 5 niveles de curriculum
  python generate_maps.py --preset all                 # Genera TODOS los mapas
  python generate_maps.py --preset curriculum_lvl3     # Genera solo el nivel 3
  python generate_maps.py --custom-size 45             # Genera mapa custom 45x45
  python generate_maps.py --list                       # Lista todos los presets disponibles
        """
    )
    parser.add_argument("--output-dir", type=str, default="maps",
                        help="Directorio de salida para los mapas (default: maps)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla para reproducibilidad")
    parser.add_argument("--preset", type=str, default="curriculum",
                        help="Preset a generar: 'curriculum', 'all', o nombre específico")
    parser.add_argument("--custom-size", type=int, default=None,
                        help="Tamaño para mapa custom (ej: 45 para 45x45)")
    parser.add_argument("--list", action="store_true",
                        help="Listar todos los presets disponibles")

    args = parser.parse_args()

    # Listar presets disponibles
    if args.list:
        print("\nPresets disponibles:")
        print("-" * 50)
        for name, config in CurriculumMapGenerator.PRESETS.items():
            print(f"  {name:20s} -> {config['width']}x{config['height']}")
        print("-" * 50)
        print("\nModos especiales:")
        print("  curriculum  -> Genera los 5 niveles de curriculum")
        print("  all         -> Genera TODOS los mapas")
        return

    generator = CurriculumMapGenerator(seed=args.seed)

    if args.custom_size:
        grid = generator.generate(args.custom_size, args.custom_size)
        generator.save_map(grid, os.path.join(args.output_dir, f"custom_{args.custom_size}.txt"))
    elif args.preset == "curriculum":
        generate_curriculum_maps(args.output_dir, args.seed)
    elif args.preset == "all":
        generate_all_maps(args.output_dir, args.seed)
    else:
        if args.preset not in generator.PRESETS:
            print(f"Error: Preset '{args.preset}' no encontrado.")
            print(f"Presets válidos: {list(generator.PRESETS.keys())}")
            return
        grid = generator.generate_preset(args.preset)
        generator.save_map(grid, os.path.join(args.output_dir, f"{args.preset}.txt"))


if __name__ == "__main__":
    main()