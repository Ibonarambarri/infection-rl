"""
Evaluation Renderer
===================
Visualizador para evaluaciones durante entrenamiento.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class EvaluationRenderer:
    """
    Renderizador para visualizar evaluaciones en tiempo real.

    Controles:
        1-5: Velocidad (5, 10, 20, 40, 60 FPS)
        SPACE: Pausar/continuar
        Q/ESC: Saltar evaluacion actual
    """

    # Velocidades disponibles (FPS)
    SPEED_FPS = [5, 10, 20, 40, 60]

    # Colores
    COLORS = {
        "bg": (30, 30, 35),
        "panel_bg": (40, 40, 50),
        "text": (220, 220, 220),
        "text_dim": (140, 140, 150),
        "healthy": (50, 205, 50),
        "infected": (220, 60, 60),
        "wall": (60, 60, 70),
        "obstacle": (100, 100, 110),
        "empty": (180, 180, 190),
        "vision": (80, 80, 90),
        "header": (100, 150, 255),
        "paused": (255, 200, 50),
        "running": (100, 255, 100),
    }

    def __init__(self, width: int, height: int, cell_size: int = None):
        """
        Inicializa el renderer.

        Args:
            width: Ancho del mapa en celdas
            height: Alto del mapa en celdas
            cell_size: Tamano de cada celda en pixeles (auto si None)
        """
        self.map_width = width
        self.map_height = height

        # Calcular cell_size para que quepa en pantalla
        max_screen = 800
        if cell_size is None:
            self.cell_size = max(8, min(20, max_screen // max(width, height)))
        else:
            self.cell_size = cell_size

        # Dimensiones
        self.panel_width = 280
        self.min_height = 400  # Altura minima para que quepa el panel
        self.grid_width = self.map_width * self.cell_size
        self.grid_height = self.map_height * self.cell_size
        self.screen_width = self.grid_width + self.panel_width
        self.screen_height = max(self.grid_height, self.min_height)

        # Estado
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        self.initialized = False

        # Control
        self.paused = False
        self.speed_level = 2  # Indice 0-4, default 20 FPS
        self.skip_requested = False

        # Contexto
        self.phase_id = 0
        self.episode = 0
        self.total_episodes = 0

        # Referencia al entorno
        self.env = None

    def init_pygame(self) -> bool:
        """
        Inicializa pygame y crea la ventana.

        Returns:
            True si se inicializo correctamente, False si pygame no disponible
        """
        if not PYGAME_AVAILABLE:
            print("Warning: pygame no disponible, visualizacion desactivada")
            return False

        if self.initialized:
            return True

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Infection RL - Evaluation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)
        self.initialized = True

        return True

    def set_env(self, env) -> None:
        """Establece el entorno a renderizar y actualiza dimensiones si es necesario."""
        self.env = env

        # Usar dimensiones del grid real si esta disponible (mas robusto)
        if hasattr(env, 'grid') and env.grid is not None:
            new_height, new_width = env.grid.shape
        else:
            new_width = env.width
            new_height = env.height

        if new_width != self.map_width or new_height != self.map_height:
            self._resize(new_width, new_height)

    def _resize(self, new_width: int, new_height: int) -> None:
        """Redimensiona el renderer para un nuevo tamaño de mapa."""
        self.map_width = new_width
        self.map_height = new_height

        # Recalcular cell_size para que quepa en pantalla
        max_screen = 800
        self.cell_size = max(8, min(20, max_screen // max(new_width, new_height)))

        # Recalcular dimensiones
        self.grid_width = self.map_width * self.cell_size
        self.grid_height = self.map_height * self.cell_size
        self.screen_width = self.grid_width + self.panel_width
        self.screen_height = max(self.grid_height, self.min_height)

        # Redimensionar ventana si ya está inicializada
        if self.initialized and PYGAME_AVAILABLE:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

    def set_context(self, phase_id: int, episode: int, total_episodes: int) -> None:
        """
        Establece el contexto de la evaluacion actual.

        Args:
            phase_id: ID de la fase actual
            episode: Numero de episodio actual
            total_episodes: Total de episodios en la evaluacion
        """
        self.phase_id = phase_id
        self.episode = episode
        self.total_episodes = total_episodes
        self.skip_requested = False

    def handle_events(self) -> bool:
        """
        Procesa eventos de teclado.

        Returns:
            True para continuar, False para saltar esta evaluacion
        """
        if not self.initialized:
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.skip_requested = True
                return False

            if event.type == pygame.KEYDOWN:
                # Q o ESC: saltar
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.skip_requested = True
                    return False

                # SPACE: pausar/continuar
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                # 1-5: velocidad
                if pygame.K_1 <= event.key <= pygame.K_5:
                    self.speed_level = event.key - pygame.K_1

        # Si esta pausado, esperar sin avanzar
        while self.paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.skip_requested = True
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.skip_requested = True
                        return False
                    if event.key == pygame.K_SPACE:
                        self.paused = False
                        break
                    if pygame.K_1 <= event.key <= pygame.K_5:
                        self.speed_level = event.key - pygame.K_1

            # Redibujar mientras pausado
            if self.env is not None:
                self._render_all()
            self.clock.tick(10)

        return True

    def render_frame(self, step: int, healthy_count: int, infected_count: int) -> None:
        """
        Renderiza un frame completo.

        Args:
            step: Paso actual del episodio
            healthy_count: Numero de agentes sanos
            infected_count: Numero de agentes infectados
        """
        if not self.initialized or self.env is None:
            return

        self.current_step = step
        self.healthy_count = healthy_count
        self.infected_count = infected_count

        self._render_all()

    def _render_all(self) -> None:
        """Renderiza todo el frame."""
        self.screen.fill(self.COLORS["bg"])
        self._render_grid()
        self._render_panel()
        pygame.display.flip()

    def _render_grid(self) -> None:
        """Renderiza el grid del entorno."""
        if self.env is None:
            return

        env = self.env

        # Usar dimensiones reales del grid para evitar IndexError
        actual_height, actual_width = env.grid.shape
        if actual_width != self.map_width or actual_height != self.map_height:
            self._resize(actual_width, actual_height)

        cs = self.cell_size

        # Dibujar celdas del mapa
        for y in range(actual_height):
            for x in range(actual_width):
                rect = pygame.Rect(x * cs, y * cs, cs, cs)

                # Tipo de celda
                from src.envs.map_generator import CellType
                cell = env.grid[y, x]

                if cell == CellType.WALL.value:
                    color = self.COLORS["wall"]
                elif cell == CellType.OBSTACLE.value:
                    color = self.COLORS["obstacle"]
                else:
                    color = self.COLORS["empty"]

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLORS["bg"], rect, 1)

        # Dibujar campo de vision de infectados
        self._render_vision()

        # Dibujar agentes
        for agent in env.agents:
            x, y = agent.position
            rect = pygame.Rect(x * cs + 2, y * cs + 2, cs - 4, cs - 4)

            color = self.COLORS["infected"] if agent.is_infected else self.COLORS["healthy"]
            pygame.draw.rect(self.screen, color, rect)

            # Indicador de direccion
            cx = x * cs + cs // 2
            cy = y * cs + cs // 2
            size = cs // 4

            from src.agents import Direction
            if agent.direction == Direction.UP:
                end = (cx, cy - size)
            elif agent.direction == Direction.DOWN:
                end = (cx, cy + size)
            elif agent.direction == Direction.LEFT:
                end = (cx - size, cy)
            else:
                end = (cx + size, cy)

            pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), end, 2)

    def _render_vision(self) -> None:
        """Renderiza el campo de vision de los infectados."""
        if self.env is None:
            return

        env = self.env
        cs = self.cell_size

        # Usar dimensiones reales del grid
        actual_height, actual_width = env.grid.shape

        # Crear superficie semi-transparente
        vision_surface = pygame.Surface((self.grid_width, self.grid_height), pygame.SRCALPHA)

        for agent in env.agents:
            if not agent.is_infected:
                continue

            # Obtener celdas visibles
            visible = env._get_visible_cells(agent)

            for (vx, vy) in visible:
                if 0 <= vx < actual_width and 0 <= vy < actual_height:
                    rect = pygame.Rect(vx * cs, vy * cs, cs, cs)
                    pygame.draw.rect(vision_surface, (80, 80, 100, 80), rect)

        self.screen.blit(vision_surface, (0, 0))

    def _render_panel(self) -> None:
        """Renderiza el panel de informacion."""
        panel_x = self.grid_width
        panel_rect = pygame.Rect(panel_x, 0, self.panel_width, self.screen_height)
        pygame.draw.rect(self.screen, self.COLORS["panel_bg"], panel_rect)

        y = 20

        # Titulo
        self._draw_text("EVALUATION", panel_x + 20, y, self.COLORS["header"], self.font)
        y += 40

        # Linea separadora
        pygame.draw.line(self.screen, self.COLORS["text_dim"],
                        (panel_x + 15, y), (panel_x + self.panel_width - 15, y))
        y += 20

        # Info de contexto
        phase_text = f"Phase: {self.phase_id}" if self.phase_id < 100 else f"Refinement: {self.phase_id - 100}"
        self._draw_text(phase_text, panel_x + 20, y, self.COLORS["text"])
        y += 30

        self._draw_text(f"Episode: {self.episode} / {self.total_episodes}",
                       panel_x + 20, y, self.COLORS["text"])
        y += 30

        step_text = f"Step: {getattr(self, 'current_step', 0)}"
        self._draw_text(step_text, panel_x + 20, y, self.COLORS["text"])
        y += 40

        # Linea separadora
        pygame.draw.line(self.screen, self.COLORS["text_dim"],
                        (panel_x + 15, y), (panel_x + self.panel_width - 15, y))
        y += 20

        # Contadores de agentes
        healthy = getattr(self, 'healthy_count', 0)
        infected = getattr(self, 'infected_count', 0)

        self._draw_text(f"Healthy: {healthy}", panel_x + 20, y, self.COLORS["healthy"])
        y += 30
        self._draw_text(f"Infected: {infected}", panel_x + 20, y, self.COLORS["infected"])
        y += 40

        # Linea separadora
        pygame.draw.line(self.screen, self.COLORS["text_dim"],
                        (panel_x + 15, y), (panel_x + self.panel_width - 15, y))
        y += 20

        # Controles
        self._draw_text("Controls:", panel_x + 20, y, self.COLORS["text_dim"], self.font_small)
        y += 25
        self._draw_text("1-5: Speed", panel_x + 30, y, self.COLORS["text_dim"], self.font_small)
        y += 22
        self._draw_text("SPACE: Pause", panel_x + 30, y, self.COLORS["text_dim"], self.font_small)
        y += 22
        self._draw_text("Q/ESC: Skip", panel_x + 30, y, self.COLORS["text_dim"], self.font_small)
        y += 35

        # Estado
        pygame.draw.line(self.screen, self.COLORS["text_dim"],
                        (panel_x + 15, y), (panel_x + self.panel_width - 15, y))
        y += 20

        fps = self.SPEED_FPS[self.speed_level]
        self._draw_text(f"Speed: {self.speed_level + 1} ({fps} FPS)",
                       panel_x + 20, y, self.COLORS["text"])
        y += 30

        status = "PAUSED" if self.paused else "RUNNING"
        status_color = self.COLORS["paused"] if self.paused else self.COLORS["running"]
        self._draw_text(f"Status: {status}", panel_x + 20, y, status_color)

    def _draw_text(self, text: str, x: int, y: int, color: Tuple, font=None) -> None:
        """Dibuja texto en la pantalla."""
        if font is None:
            font = self.font
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def wait_frame(self) -> None:
        """Espera el tiempo adecuado segun la velocidad actual."""
        if not self.initialized:
            return
        fps = self.SPEED_FPS[self.speed_level]
        self.clock.tick(fps)

    def close(self) -> None:
        """Cierra pygame."""
        if self.initialized:
            pygame.quit()
            self.initialized = False
            self.screen = None
            self.clock = None
