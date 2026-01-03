#!/usr/bin/env python3
"""
Play - Visualizar Partida con Modelos Entrenados
=================================================
Carga modelos entrenados y ejecuta una partida visual infinita.
Flexible para visualizar cualquier enfrentamiento con diferentes mapas y agentes.

Uso:
    # Básico con modelos
    python scripts/play.py --models-dir models/curriculum
    python scripts/play.py --healthy-model path/to/healthy.zip --infected-model path/to/infected.zip

    # Personalizar mapa y agentes
    python scripts/play.py --map-file maps/curriculum_lvl3.txt --num-healthy 8 --num-infected 2
    python scripts/play.py --map-file maps/large.txt --num-healthy 12 --num-infected 3 --fps 20

    # Mezclar modelos con heurística
    python scripts/play.py --healthy-model models/healthy.zip  # Infected usa heurística
    python scripts/play.py --infected-model models/infected.zip  # Healthy usa heurística

    # Sin modelos (heurística vs heurística)
    python scripts/play.py --map-file maps/curriculum_lvl5.txt --num-healthy 10 --num-infected 3

Controles:
    SPACE: Pausar/Reanudar
    R: Reiniciar episodio
    V: Toggle visión de infectados
    S: Paso a paso (cuando pausado)
    Q: Salir
    1-5: Velocidad
"""

import sys
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from stable_baselines3 import PPO

from src.envs import InfectionEnv, EnvConfig, CellType
from src.envs.wrappers import FlattenObservationWrapper, SingleAgentWrapper
from src.agents import Direction


class GamePlayer:
    """Visualizador de partidas con modelos entrenados."""

    def __init__(
        self,
        healthy_model_path: str = None,
        infected_model_path: str = None,
        map_file: str = None,
        num_healthy: int = None,
        num_infected: int = None,
        seed: int = 42,
    ):
        # Configurar entorno con parámetros personalizados
        config_kwargs = {
            "max_steps": 999999999,  # Prácticamente infinito
            "seed": seed,
        }

        # Mapa personalizado
        if map_file is not None:
            config_kwargs["map_file"] = map_file

        # Número de agentes personalizado
        if num_healthy is not None and num_infected is not None:
            config_kwargs["num_agents"] = num_healthy + num_infected
            config_kwargs["initial_infected"] = num_infected
        elif num_healthy is not None:
            # Solo se especificó healthy, usar 1 infected por defecto
            config_kwargs["num_agents"] = num_healthy + 1
            config_kwargs["initial_infected"] = 1
        elif num_infected is not None:
            # Solo se especificó infected, usar valor por defecto de healthy
            default_healthy = 14  # num_agents default es 15
            config_kwargs["num_agents"] = default_healthy + num_infected
            config_kwargs["initial_infected"] = num_infected

        self.config = EnvConfig(**config_kwargs)
        self.env = InfectionEnv(self.config)

        # Reset to load map and get correct dimensions
        self.env.reset()

        # Guardar configuración para mostrar en panel
        self.map_file = map_file or self.config.map_file
        self.num_healthy_config = num_healthy
        self.num_infected_config = num_infected

        # Dimensiones del mapa (after reset so map is loaded)
        self.width = self.env.width
        self.height = self.env.height

        # Auto-calcular cell_size para que quepa en pantalla
        max_screen = 900
        self.cell_size = max(10, min(25, max_screen // max(self.width, self.height)))

        # Cargar modelos
        self.model_healthy = None
        self.model_infected = None

        if healthy_model_path:
            print(f"Loading healthy model: {healthy_model_path}")
            self.model_healthy = PPO.load(healthy_model_path)

        if infected_model_path:
            print(f"Loading infected model: {infected_model_path}")
            self.model_infected = PPO.load(infected_model_path)

        # Crear wrappers para obtener observaciones correctas
        self._setup_observation_wrappers()

        # Pygame
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

        # Estado
        self.paused = False
        self.show_vision = True
        self.step_count = 0
        self.episode_count = 0

        # Estadísticas
        self.stats = {
            "total_infections": 0,
            "episodes_played": 0,
            "healthy_wins": 0,
            "infected_wins": 0,
        }

        # Colores
        self.colors = {
            "bg": (30, 30, 30),
            "empty": (180, 180, 180),
            "wall": (50, 50, 50),
            "obstacle": (100, 100, 100),
            "healthy": (0, 200, 0),
            "infected": (200, 0, 0),
            "vision_infected": (80, 80, 80),
            "text": (255, 255, 255),
            "text_dim": (150, 150, 150),
            "panel_bg": (40, 40, 50),
        }

    def _setup_observation_wrappers(self):
        """Crea entornos con wrappers para obtener observaciones."""
        # Para obtener observaciones de agentes healthy
        self.obs_env_healthy = InfectionEnv(self.config)
        self.obs_env_healthy = SingleAgentWrapper(self.obs_env_healthy, force_role="healthy")
        self.obs_env_healthy = FlattenObservationWrapper(self.obs_env_healthy)

        # Para obtener observaciones de agentes infected
        self.obs_env_infected = InfectionEnv(self.config)
        self.obs_env_infected = SingleAgentWrapper(self.obs_env_infected, force_role="infected")
        self.obs_env_infected = FlattenObservationWrapper(self.obs_env_infected)

    def _get_observation_for_agent(self, agent):
        """Obtiene observación aplanada para un agente."""
        obs_dict = self.env._get_observation(agent)

        # Aplanar manualmente
        parts = []

        # Imagen normalizada
        image = obs_dict["image"].astype(np.float32) / 255.0
        parts.append(image.flatten())

        # Dirección one-hot
        direction = np.zeros(4, dtype=np.float32)
        direction[obs_dict["direction"]] = 1.0
        parts.append(direction)

        # Estado one-hot
        state = np.zeros(2, dtype=np.float32)
        state[obs_dict["state"]] = 1.0
        parts.append(state)

        # Posición
        parts.append(obs_dict["position"])

        # Agentes cercanos
        parts.append(obs_dict["nearby_agents"].flatten())

        return np.concatenate(parts)

    def _init_pygame(self):
        """Inicializa pygame."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame no está instalado")

        pygame.init()

        # Tamaño de ventana
        grid_width = self.width * self.cell_size
        grid_height = self.height * self.cell_size
        panel_width = 320
        window_width = grid_width + panel_width
        window_height = max(grid_height, 500)

        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Infection Game - AI vs AI")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
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

        # Dibujar visión de infectados
        if self.show_vision:
            self._render_vision()

        # Dibujar agentes
        for agent in self.env.agents:
            cx = agent.x * self.cell_size + self.cell_size // 2
            cy = agent.y * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 2 - 3

            color = self.colors["infected"] if agent.is_infected else self.colors["healthy"]
            pygame.draw.circle(self.screen, color, (cx, cy), radius)

            # Indicador de dirección
            dir_vectors = {
                Direction.RIGHT: (radius - 2, 0),
                Direction.DOWN: (0, radius - 2),
                Direction.LEFT: (-(radius - 2), 0),
                Direction.UP: (0, -(radius - 2)),
            }
            dx, dy = dir_vectors[agent.direction]
            pygame.draw.circle(self.screen, (255, 255, 255), (cx + dx, cy + dy), 4)

    def _render_vision(self):
        """Renderiza campo de visión de infectados."""
        alpha = 100

        for agent in self.env.agents:
            if not agent.is_infected:
                continue

            visible_cells = self.env._get_visible_cells(agent)

            for (wx, wy) in visible_cells:
                if 0 <= wx < self.width and 0 <= wy < self.height:
                    x1 = wx * self.cell_size
                    y1 = wy * self.cell_size

                    overlay = pygame.Surface((self.cell_size - 1, self.cell_size - 1))
                    overlay.set_alpha(alpha)
                    overlay.fill(self.colors["vision_infected"])
                    self.screen.blit(overlay, (x1, y1))

    def _render_panel(self):
        """Renderiza panel de información."""
        pygame.draw.rect(
            self.screen, self.colors["panel_bg"],
            (self.panel_x, 0, 320, self.screen.get_height())
        )

        x = self.panel_x + 15
        y = 20

        # Título
        title = self.font.render("INFECTION GAME", True, self.colors["text"])
        self.screen.blit(title, (x, y))
        y += 45

        # Estado
        status = "PAUSED" if self.paused else "RUNNING"
        status_color = (255, 200, 0) if self.paused else (0, 255, 0)
        text = self.small_font.render(f"Status: {status}", True, status_color)
        self.screen.blit(text, (x, y))
        y += 35

        # Información del episodio
        pygame.draw.line(self.screen, self.colors["text_dim"], (x, y), (x + 290, y))
        y += 15

        text = self.small_font.render(f"Episode: {self.episode_count + 1}", True, self.colors["text"])
        self.screen.blit(text, (x, y))
        y += 25

        text = self.small_font.render(f"Step: {self.step_count}", True, self.colors["text"])
        self.screen.blit(text, (x, y))
        y += 35

        # Estado actual
        pygame.draw.line(self.screen, self.colors["text_dim"], (x, y), (x + 290, y))
        y += 15

        text = self.small_font.render("Current Game:", True, self.colors["text"])
        self.screen.blit(text, (x, y))
        y += 25

        healthy_count = self.env.num_healthy
        infected_count = self.env.num_infected

        text = self.small_font.render(f"  Healthy: {healthy_count}", True, self.colors["healthy"])
        self.screen.blit(text, (x, y))
        y += 22

        text = self.small_font.render(f"  Infected: {infected_count}", True, self.colors["infected"])
        self.screen.blit(text, (x, y))
        y += 22

        text = self.small_font.render(f"  Infections: {len(self.env.infection_events)}", True, self.colors["text_dim"])
        self.screen.blit(text, (x, y))
        y += 35

        # Estadísticas globales
        pygame.draw.line(self.screen, self.colors["text_dim"], (x, y), (x + 290, y))
        y += 15

        text = self.small_font.render("Statistics:", True, self.colors["text"])
        self.screen.blit(text, (x, y))
        y += 25

        text = self.small_font.render(f"  Episodes: {self.stats['episodes_played']}", True, self.colors["text_dim"])
        self.screen.blit(text, (x, y))
        y += 22

        text = self.small_font.render(f"  Healthy wins: {self.stats['healthy_wins']}", True, self.colors["healthy"])
        self.screen.blit(text, (x, y))
        y += 22

        text = self.small_font.render(f"  Infected wins: {self.stats['infected_wins']}", True, self.colors["infected"])
        self.screen.blit(text, (x, y))
        y += 22

        text = self.small_font.render(f"  Total infections: {self.stats['total_infections']}", True, self.colors["text_dim"])
        self.screen.blit(text, (x, y))
        y += 35

        # Configuración del entorno
        pygame.draw.line(self.screen, self.colors["text_dim"], (x, y), (x + 290, y))
        y += 15

        text = self.small_font.render("Environment:", True, self.colors["text"])
        self.screen.blit(text, (x, y))
        y += 25

        # Mostrar mapa (nombre del archivo)
        map_name = Path(self.map_file).stem if self.map_file else "default"
        text = self.small_font.render(f"  Map: {map_name}", True, self.colors["text_dim"])
        self.screen.blit(text, (x, y))
        y += 22

        text = self.small_font.render(f"  Size: {self.width}x{self.height}", True, self.colors["text_dim"])
        self.screen.blit(text, (x, y))
        y += 35

        # Modelos
        pygame.draw.line(self.screen, self.colors["text_dim"], (x, y), (x + 290, y))
        y += 15

        text = self.small_font.render("Models:", True, self.colors["text"])
        self.screen.blit(text, (x, y))
        y += 25

        h_status = "AI" if self.model_healthy else "Heuristic"
        i_status = "AI" if self.model_infected else "Heuristic"

        text = self.small_font.render(f"  Healthy: {h_status}", True, self.colors["healthy"])
        self.screen.blit(text, (x, y))
        y += 22

        text = self.small_font.render(f"  Infected: {i_status}", True, self.colors["infected"])
        self.screen.blit(text, (x, y))
        y += 35

        # Controles
        pygame.draw.line(self.screen, self.colors["text_dim"], (x, y), (x + 290, y))
        y += 15

        text = self.small_font.render("Controls:", True, self.colors["text"])
        self.screen.blit(text, (x, y))
        y += 25

        controls = [
            "SPACE: Pause/Resume",
            "R: Reset episode",
            "V: Toggle vision",
            "S: Step (when paused)",
            "Q: Quit",
            "1-5: Speed",
        ]
        for ctrl in controls:
            text = self.small_font.render(f"  {ctrl}", True, self.colors["text_dim"])
            self.screen.blit(text, (x, y))
            y += 20

    def _get_action(self, agent):
        """Obtiene acción para un agente usando el modelo o heurística."""
        if agent.is_infected and self.model_infected:
            obs = self._get_observation_for_agent(agent)
            action, _ = self.model_infected.predict(obs, deterministic=True)
            return int(action)
        elif not agent.is_infected and self.model_healthy:
            obs = self._get_observation_for_agent(agent)
            action, _ = self.model_healthy.predict(obs, deterministic=True)
            return int(action)
        else:
            # Usar heurística del entorno
            return self.env._get_other_agent_action(agent)

    def _step(self):
        """Ejecuta un paso del juego."""
        # Obtener acciones para todos los agentes
        actions = {}
        for agent in self.env.agents:
            actions[agent.id] = self._get_action(agent)

        # Ejecutar paso (ignoramos terminated/truncated - queremos partida infinita)
        self.env.step_all(actions)

        self.step_count += 1
        self.stats["total_infections"] = len(self.env.infection_events)

    def _reset(self):
        """Reinicia el episodio."""
        self.env.reset()
        self.step_count = 0
        self.episode_count += 1

    def run(self, fps: int = 15):
        """Ejecuta el visualizador."""
        self._init_pygame()
        self.env.reset()

        running = True
        current_fps = fps

        print("\n" + "=" * 50)
        print("  INFECTION GAME")
        print("=" * 50)
        print("  Press SPACE to pause/resume")
        print("  Press Q to quit")
        print("=" * 50 + "\n")

        while running:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self._reset()
                    elif event.key == pygame.K_v:
                        self.show_vision = not self.show_vision
                    elif event.key == pygame.K_s and self.paused:
                        # Step-by-step cuando pausado
                        self._step()
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                        speeds = [5, 10, 15, 30, 60]
                        current_fps = speeds[int(event.unicode) - 1]

            # Ejecutar paso si no está pausado
            if not self.paused:
                self._step()

            # Renderizar
            self.screen.fill(self.colors["bg"])
            self._render_grid()
            self._render_panel()

            pygame.display.flip()
            self.clock.tick(current_fps)

        pygame.quit()

        print(f"\n  Final Statistics:")
        print(f"  - Episodes: {self.stats['episodes_played']}")
        print(f"  - Healthy wins: {self.stats['healthy_wins']}")
        print(f"  - Infected wins: {self.stats['infected_wins']}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualizar partidas de Infection RL con modelos entrenados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Con modelos del curriculum
  python scripts/play.py --models-dir models/curriculum

  # Personalizar mapa y agentes
  python scripts/play.py --map-file maps/curriculum_lvl3.txt --num-healthy 8 --num-infected 2

  # Mezclar modelo con heurística
  python scripts/play.py --healthy-model models/healthy.zip --map-file maps/large.txt

  # Solo heurística (sin modelos)
  python scripts/play.py --map-file maps/curriculum_lvl5.txt --num-healthy 10 --num-infected 3
        """
    )

    # Argumentos de modelos
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Directorio con modelos (healthy_final.zip y infected_final.zip)")
    parser.add_argument("--healthy-model", type=str, default=None,
                        help="Ruta al modelo healthy (.zip)")
    parser.add_argument("--infected-model", type=str, default=None,
                        help="Ruta al modelo infected (.zip)")

    # Argumentos de entorno
    parser.add_argument("--map-file", type=str, default=None,
                        help="Ruta al archivo de mapa (.txt)")
    parser.add_argument("--num-healthy", type=int, default=None,
                        help="Número de agentes sanos")
    parser.add_argument("--num-infected", type=int, default=None,
                        help="Número de agentes infectados")

    # Argumentos de visualización
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames por segundo (default: 15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")

    args = parser.parse_args()

    if not PYGAME_AVAILABLE:
        print("Error: pygame is required")
        print("Install with: pip install pygame")
        return

    # Determinar paths de modelos
    healthy_path = args.healthy_model
    infected_path = args.infected_model

    if args.models_dir:
        models_dir = Path(args.models_dir)
        if not healthy_path:
            # Buscar en orden de prioridad
            for name in ["healthy_final.zip", "best_healthy_model.zip"]:
                candidate = models_dir / name
                if candidate.exists():
                    healthy_path = candidate
                    break
        if not infected_path:
            for name in ["infected_final.zip", "best_infected_model.zip"]:
                candidate = models_dir / name
                if candidate.exists():
                    infected_path = candidate
                    break

    # Convertir a string
    healthy_path = str(healthy_path) if healthy_path else None
    infected_path = str(infected_path) if infected_path else None

    # Mostrar configuración
    print("\n" + "=" * 50)
    print("  INFECTION GAME - Configuración")
    print("=" * 50)
    if args.map_file:
        print(f"  Mapa: {args.map_file}")
    else:
        print("  Mapa: default")
    if args.num_healthy:
        print(f"  Healthy: {args.num_healthy}")
    if args.num_infected:
        print(f"  Infected: {args.num_infected}")
    print(f"  Modelo Healthy: {'AI' if healthy_path else 'Heurística'}")
    print(f"  Modelo Infected: {'AI' if infected_path else 'Heurística'}")
    print("=" * 50)

    if not healthy_path and not infected_path:
        print("  (Ambos bandos usan políticas heurísticas)")

    player = GamePlayer(
        healthy_model_path=healthy_path,
        infected_model_path=infected_path,
        map_file=args.map_file,
        num_healthy=args.num_healthy,
        num_infected=args.num_infected,
        seed=args.seed,
    )

    player.run(fps=args.fps)


if __name__ == "__main__":
    main()
