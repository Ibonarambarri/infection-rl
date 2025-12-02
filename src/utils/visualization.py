"""
Visualization utilities for the Infection Environment.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def render_episode(
    env,
    model=None,
    max_steps: int = 200,
    deterministic: bool = True,
    render: bool = True,
) -> Dict[str, any]:
    """
    Renderiza un episodio completo y retorna estadísticas.

    Args:
        env: Entorno de infección
        model: Modelo entrenado (None para política random)
        max_steps: Máximo de pasos
        deterministic: Si usar política determinista
        render: Si renderizar visualmente

    Returns:
        Diccionario con estadísticas del episodio
    """
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0
    positions_history = []
    rewards_history = []

    while not done and step < max_steps:
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step += 1
        rewards_history.append(reward)

        # Guardar posiciones de agentes
        if hasattr(env.unwrapped, "agents"):
            positions = {
                a.id: (a.x, a.y, a.is_infected)
                for a in env.unwrapped.agents
            }
            positions_history.append(positions)

        if render:
            env.render()

    return {
        "total_reward": total_reward,
        "episode_length": step,
        "terminated": terminated,
        "info": info,
        "positions_history": positions_history,
        "rewards_history": rewards_history,
    }


def create_heatmap(
    positions_history: List[Dict],
    grid_size: Tuple[int, int],
    agent_type: str = "all",  # "all", "healthy", "infected"
    title: str = "Agent Positions Heatmap",
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Crea un heatmap de posiciones de agentes.

    Args:
        positions_history: Historial de posiciones
        grid_size: Tamaño del grid (width, height)
        agent_type: Tipo de agentes a incluir
        title: Título del gráfico
        save_path: Ruta para guardar imagen

    Returns:
        Array del heatmap
    """
    width, height = grid_size
    heatmap = np.zeros((height, width))

    for positions in positions_history:
        for agent_id, (x, y, is_infected) in positions.items():
            include = (
                agent_type == "all" or
                (agent_type == "healthy" and not is_infected) or
                (agent_type == "infected" and is_infected)
            )
            if include and 0 <= x < width and 0 <= y < height:
                heatmap[y, x] += 1

    # Normalizar
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Visualizar
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, label="Frecuencia (normalizada)")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()

    return heatmap


def plot_training_curves(
    log_dir: str,
    save_path: Optional[str] = None,
    window_size: int = 100,
):
    """
    Visualiza curvas de entrenamiento desde logs de TensorBoard.

    Args:
        log_dir: Directorio con logs de TensorBoard
        save_path: Ruta para guardar imagen
        window_size: Tamaño de ventana para suavizado
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()

        # Obtener métricas disponibles
        tags = ea.Tags()["scalars"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics_to_plot = [
            ("rollout/ep_rew_mean", "Episode Reward Mean"),
            ("rollout/ep_len_mean", "Episode Length Mean"),
            ("train/loss", "Training Loss"),
            ("train/entropy_loss", "Entropy Loss"),
        ]

        for ax, (tag, title) in zip(axes.flat, metrics_to_plot):
            if tag in tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                # Suavizar
                if len(values) > window_size:
                    smoothed = np.convolve(
                        values,
                        np.ones(window_size) / window_size,
                        mode="valid"
                    )
                    steps = steps[window_size - 1:]
                    values = smoothed

                ax.plot(steps, values)
                ax.set_title(title)
                ax.set_xlabel("Steps")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f"No data for {tag}",
                       ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    except ImportError:
        print("TensorBoard no instalado. Instala con: pip install tensorboard")


def save_episode_video(
    env,
    model=None,
    filepath: str = "episode.mp4",
    max_steps: int = 200,
    fps: int = 10,
    deterministic: bool = True,
):
    """
    Guarda un video de un episodio.

    Args:
        env: Entorno
        model: Modelo entrenado
        filepath: Ruta del archivo de video
        max_steps: Máximo de pasos
        fps: Frames por segundo
        deterministic: Si usar política determinista
    """
    try:
        import imageio

        frames = []
        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            # Obtener frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            # Acción
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = env.action_space.sample()

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1

        # Guardar video
        if frames:
            imageio.mimsave(filepath, frames, fps=fps)
            print(f"Video guardado: {filepath}")
        else:
            print("No se capturaron frames")

    except ImportError:
        print("imageio no instalado. Instala con: pip install imageio imageio-ffmpeg")


def plot_reward_distribution(
    rewards: List[float],
    title: str = "Reward Distribution",
    save_path: Optional[str] = None,
):
    """
    Visualiza la distribución de recompensas.

    Args:
        rewards: Lista de recompensas
        title: Título
        save_path: Ruta para guardar
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma
    axes[0].hist(rewards, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"{title} - Histogram")
    axes[0].set_xlabel("Reward")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(np.mean(rewards), color="red", linestyle="--",
                    label=f"Mean: {np.mean(rewards):.2f}")
    axes[0].legend()

    # Serie temporal
    axes[1].plot(rewards, alpha=0.7)
    axes[1].set_title(f"{title} - Over Time")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Reward")
    axes[1].axhline(np.mean(rewards), color="red", linestyle="--",
                    label=f"Mean: {np.mean(rewards):.2f}")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def compare_agents(
    results: Dict[str, Dict],
    metric: str = "total_reward",
    title: str = "Agent Comparison",
    save_path: Optional[str] = None,
):
    """
    Compara rendimiento de diferentes agentes.

    Args:
        results: Dict con {agent_name: {metric: [values], ...}}
        metric: Métrica a comparar
        title: Título
        save_path: Ruta para guardar
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    means = []
    stds = []

    for name in names:
        values = results[name].get(metric, [])
        means.append(np.mean(values) if values else 0)
        stds.append(np.std(values) if values else 0)

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)

    ax.set_xlabel("Agent")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")

    # Añadir valores sobre las barras
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(f"{mean:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
