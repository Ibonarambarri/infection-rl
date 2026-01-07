#!/usr/bin/env python3
"""
Play - Visualizar UNA partida con modelos entrenados
====================================================

Uso:
    python scripts/play.py -l 1        # Nivel 1 (20x20)
    python scripts/play.py -l 2        # Nivel 2 (30x30)
    python scripts/play.py -l 3        # Nivel 3 (40x40)
    python scripts/play.py -l 3 -m models/run3  # Con modelos custom

Controles:
    SPACE: Pausar/Reanudar
    V: Toggle vision de infectados
    S: Paso a paso (cuando pausado)
    Q: Salir
    1-5: Velocidad
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from stable_baselines3 import PPO

from src.maps import MAP_LVL1, MAP_LVL2, MAP_LVL3
from src.envs import InfectionEnv, EnvConfig, CellType
from src.agents import Direction

# Configuracion de infectados (debe coincidir con train.py)
INFECTED_SPEED = 1  # Misma velocidad que healthy
INFECTED_GLOBAL_VISION = True  # Los infectados ven a todos los healthy


# Configuracion por nivel (ratio 1:1 equilibrado)
LEVEL_CONFIG = {
    1: {"map_data": MAP_LVL1, "num_healthy": 3, "num_infected": 3, "name": "Level 1 (20x20) - 3v3"},
    2: {"map_data": MAP_LVL2, "num_healthy": 4, "num_infected": 4, "name": "Level 2 (30x30) - 4v4"},
    3: {"map_data": MAP_LVL3, "num_healthy": 5, "num_infected": 5, "name": "Level 3 (40x40) - 5v5"},
}


class GamePlayer:
    """Visualizador de una partida con modelos entrenados."""

    def __init__(
        self,
        level: int,
        models_dir: str = None,
        seed: int = 42,
    ):
        self.level = level
        config = LEVEL_CONFIG[level]

        # Configurar entorno (mismos parametros que train.py)
        env_config = EnvConfig(
            map_data=config["map_data"],
            num_agents=config["num_healthy"] + config["num_infected"],
            initial_infected=config["num_infected"],
            max_steps=500,
            seed=seed,
            infected_speed=INFECTED_SPEED,
            infected_global_vision=INFECTED_GLOBAL_VISION,
        )
        self.env = InfectionEnv(env_config)
        self.env.reset()

        # Dimensiones del mapa
        self.width = self.env.width
        self.height = self.env.height

        # Auto-calcular cell_size
        max_screen = 800
        self.cell_size = max(10, min(20, max_screen // max(self.width, self.height)))

        # Cargar modelos
        self.model_healthy = None
        self.model_infected = None

        if models_dir:
            models_path = Path(models_dir)
            healthy_path = models_path / "healthy_final.zip"
            infected_path = models_path / "infected_final.zip"

            if healthy_path.exists():
                print(f"  Cargando modelo healthy: {healthy_path}")
                self.model_healthy = PPO.load(healthy_path)
            else:
                print(f"  No se encontro modelo healthy, usando heuristica")

            if infected_path.exists():
                print(f"  Cargando modelo infected: {infected_path}")
                self.model_infected = PPO.load(infected_path)
            else:
                print(f"  No se encontro modelo infected, usando heuristica")

        # Pygame
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

        # Estado
        self.paused = False
        self.show_vision = True
        self.step_count = 0
        self.game_over = False
        self.winner = None

        # Colores
        self.colors = {
            "bg": (30, 30, 30),
            "empty": (180, 180, 180),
            "wall": (50, 50, 50),
            "obstacle": (100, 100, 100),
            "healthy": (0, 200, 0),
            "infected": (200, 0, 0),
            "vision": (80, 80, 80),
            "text": (255, 255, 255),
            "text_dim": (150, 150, 150),
            "panel_bg": (40, 40, 50),
        }

    def _get_observation_for_agent(self, agent):
        """
        Obtiene observacion en formato MultiInputPolicy para un agente.

        IMPORTANTE: Replica EXACTAMENTE el procesamiento de DictObservationWrapper
        para garantizar consistencia entre entrenamiento y evaluacion.
        """
        raw_obs = self.env._get_observation(agent)

        # Imagen: normalizar a [0, 1] como float32
        image = raw_obs["image"].astype(np.float32) / 255.0

        # Construir vector de features
        parts = []

        # Direction como encoding circular [cos, sin] (2 elementos)
        angle = raw_obs["direction"] * (np.pi / 2)  # 0, pi/2, pi, 3pi/2
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        parts.append(direction)

        # State one-hot (2 elementos: 0=healthy, 1=infected)
        state = np.zeros(2, dtype=np.float32)
        state[raw_obs["state"]] = 1.0
        parts.append(state)

        # Position (ya viene normalizada del environment: x/width, y/height)
        position = raw_obs["position"].astype(np.float32)
        parts.append(position)

        # Nearby agents (5 features por agente)
        nearby = raw_obs["nearby_agents"].flatten().astype(np.float32)
        parts.append(nearby)

        vector = np.concatenate(parts)

        return {
            "image": image,
            "vector": vector,
        }

    def _init_pygame(self):
        """Inicializa pygame."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame no esta instalado")

        pygame.init()

        grid_width = self.width * self.cell_size
        grid_height = self.height * self.cell_size
        panel_width = 280
        window_width = grid_width + panel_width
        window_height = max(grid_height, 400)

        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(f"Infection Game - {LEVEL_CONFIG[self.level]['name']}")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.panel_x = grid_width

    def _render_grid(self):
        """Renderiza el grid."""
        grid = self.env.grid

        for y in range(self.height):
            for x in range(self.width):
                x1 = x * self.cell_size
                y1 = y * self.cell_size

                if grid[y, x] == CellType.WALL.value:
                    color = self.colors["wall"]
                elif grid[y, x] == CellType.OBSTACLE.value:
                    color = self.colors["obstacle"]
                else:
                    color = self.colors["empty"]

                pygame.draw.rect(
                    self.screen, color,
                    (x1, y1, self.cell_size - 1, self.cell_size - 1)
                )

        # Vision de infectados
        if self.show_vision:
            for agent in self.env.agents:
                if not agent.is_infected:
                    continue
                visible = self.env._get_visible_cells(agent)
                for (wx, wy) in visible:
                    if 0 <= wx < self.width and 0 <= wy < self.height:
                        overlay = pygame.Surface((self.cell_size - 1, self.cell_size - 1))
                        overlay.set_alpha(80)
                        overlay.fill(self.colors["vision"])
                        self.screen.blit(overlay, (wx * self.cell_size, wy * self.cell_size))

        # Agentes
        for agent in self.env.agents:
            cx = agent.x * self.cell_size + self.cell_size // 2
            cy = agent.y * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 2 - 2

            color = self.colors["infected"] if agent.is_infected else self.colors["healthy"]
            pygame.draw.circle(self.screen, color, (cx, cy), radius)

            # Direccion
            dir_offset = {
                Direction.RIGHT: (radius - 2, 0),
                Direction.DOWN: (0, radius - 2),
                Direction.LEFT: (-(radius - 2), 0),
                Direction.UP: (0, -(radius - 2)),
            }
            dx, dy = dir_offset[agent.direction]
            pygame.draw.circle(self.screen, (255, 255, 255), (cx + dx, cy + dy), 3)

    def _render_panel(self):
        """Renderiza panel de informacion."""
        pygame.draw.rect(
            self.screen, self.colors["panel_bg"],
            (self.panel_x, 0, 280, self.screen.get_height())
        )

        x = self.panel_x + 15
        y = 20

        # Titulo
        title = self.font.render("INFECTION GAME", True, self.colors["text"])
        self.screen.blit(title, (x, y))
        y += 40

        # Nivel
        level_text = self.small_font.render(LEVEL_CONFIG[self.level]["name"], True, self.colors["text_dim"])
        self.screen.blit(level_text, (x, y))
        y += 35

        pygame.draw.line(self.screen, self.colors["text_dim"], (x, y), (x + 250, y))
        y += 15

        # Estado
        if self.game_over:
            status = f"GAME OVER - {self.winner} WIN!"
            status_color = self.colors["infected"] if self.winner == "INFECTED" else self.colors["healthy"]
        elif self.paused:
            status = "PAUSED"
            status_color = (255, 200, 0)
        else:
            status = "RUNNING"
            status_color = (0, 255, 0)

        text = self.small_font.render(status, True, status_color)
        self.screen.blit(text, (x, y))
        y += 30

        text = self.small_font.render(f"Step: {self.step_count} / 500", True, self.colors["text"])
        self.screen.blit(text, (x, y))
        y += 35

        pygame.draw.line(self.screen, self.colors["text_dim"], (x, y), (x + 250, y))
        y += 15

        # Contadores
        text = self.small_font.render(f"Healthy: {self.env.num_healthy}", True, self.colors["healthy"])
        self.screen.blit(text, (x, y))
        y += 25

        text = self.small_font.render(f"Infected: {self.env.num_infected}", True, self.colors["infected"])
        self.screen.blit(text, (x, y))
        y += 25

        text = self.small_font.render(f"Infections: {len(self.env.infection_events)}", True, self.colors["text_dim"])
        self.screen.blit(text, (x, y))
        y += 35

        pygame.draw.line(self.screen, self.colors["text_dim"], (x, y), (x + 250, y))
        y += 15

        # Modelos
        h_status = "AI" if self.model_healthy else "Heuristic"
        i_status = "AI" if self.model_infected else "Heuristic"

        text = self.small_font.render(f"Healthy: {h_status}", True, self.colors["healthy"])
        self.screen.blit(text, (x, y))
        y += 25

        text = self.small_font.render(f"Infected: {i_status}", True, self.colors["infected"])
        self.screen.blit(text, (x, y))
        y += 35

        pygame.draw.line(self.screen, self.colors["text_dim"], (x, y), (x + 250, y))
        y += 15

        # Controles
        controls = ["SPACE: Pause", "V: Vision", "S: Step", "Q: Quit", "1-5: Speed"]
        for ctrl in controls:
            text = self.small_font.render(ctrl, True, self.colors["text_dim"])
            self.screen.blit(text, (x, y))
            y += 20

    def _get_action(self, agent):
        """Obtiene accion para un agente."""
        if agent.is_infected and self.model_infected:
            obs = self._get_observation_for_agent(agent)
            action, _ = self.model_infected.predict(obs, deterministic=True)
            return int(action)
        elif not agent.is_infected and self.model_healthy:
            obs = self._get_observation_for_agent(agent)
            action, _ = self.model_healthy.predict(obs, deterministic=True)
            return int(action)
        else:
            return self.env._get_other_agent_action(agent)

    def _step(self):
        """Ejecuta un paso del juego."""
        if self.game_over:
            return

        actions = {agent.id: self._get_action(agent) for agent in self.env.agents}
        _, _, terminated, truncated, _ = self.env.step_all(actions)

        self.step_count += 1

        # Verificar fin de partida
        if terminated:  # Todos infectados
            self.game_over = True
            self.winner = "INFECTED"
        elif truncated:  # Max steps alcanzado
            self.game_over = True
            self.winner = "HEALTHY"

    def run(self, fps: int = 15):
        """Ejecuta el visualizador."""
        self._init_pygame()

        running = True
        current_fps = fps

        print("\n" + "=" * 50)
        print("  INFECTION GAME")
        print("=" * 50)
        print(f"  Level: {LEVEL_CONFIG[self.level]['name']}")
        print(f"  Healthy: {self.env.num_healthy} | Infected: {self.env.num_infected}")
        print("=" * 50)
        print("  SPACE: Pause | Q: Quit")
        print("=" * 50 + "\n")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_v:
                        self.show_vision = not self.show_vision
                    elif event.key == pygame.K_s and self.paused:
                        self._step()
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                        speeds = [5, 10, 15, 30, 60]
                        current_fps = speeds[int(event.unicode) - 1]

            if not self.paused and not self.game_over:
                self._step()

            self.screen.fill(self.colors["bg"])
            self._render_grid()
            self._render_panel()

            pygame.display.flip()
            self.clock.tick(current_fps)

        pygame.quit()

        # Resultado final
        print("\n" + "=" * 50)
        if self.game_over:
            print(f"  GAME OVER - {self.winner} WIN!")
            print(f"  Steps: {self.step_count}")
            print(f"  Infections: {len(self.env.infection_events)}")
        else:
            print("  Game interrupted")
        print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualizar una partida de Infection RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/play.py -l 1              # Nivel 1 (20x20)
  python scripts/play.py -l 3 -m models/run3  # Nivel 3 con modelos custom
        """
    )

    parser.add_argument("-l", "--level", type=int, choices=[1, 2, 3], default=3,
                        help="Nivel de mapa: 1 (20x20), 2 (30x30), 3 (40x40). Default: 3")
    parser.add_argument("-m", "--models-dir", type=str, default="models/run3",
                        help="Directorio con modelos (default: models/run3)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames por segundo (default: 15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")

    args = parser.parse_args()

    if not PYGAME_AVAILABLE:
        print("Error: pygame es requerido")
        print("Instalar con: pip install pygame")
        return

    print("\n" + "=" * 50)
    print("  INFECTION GAME - Configuracion")
    print("=" * 50)
    print(f"  Level: {args.level}")
    print(f"  Models: {args.models_dir}")

    player = GamePlayer(
        level=args.level,
        models_dir=args.models_dir,
        seed=args.seed,
    )

    player.run(fps=args.fps)


if __name__ == "__main__":
    main()
