"""Simplex trajectory visualization for 3-strategy games.

The 2-simplex (triangle) is the natural space for probability
distributions over three actions.  We map barycentric coordinates
to Cartesian coordinates for plotting.

Vertices of the equilateral triangle (side length 1):
    e1 = (0, 0)          -- action 0
    e2 = (1, 0)          -- action 1
    e3 = (0.5, sqrt(3)/2) -- action 2

A mixed strategy (p0, p1, p2) maps to:
    x = p1 + 0.5 * p2
    y = (sqrt(3) / 2) * p2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.figure import Figure
from typing import List, Optional, Tuple

# Simplex vertices (equilateral triangle)
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, np.sqrt(3) / 2])


def barycentric_to_cartesian(p: NDArray) -> NDArray:
    """Convert barycentric coordinates on the 2-simplex to 2D Cartesian.

    Parameters
    ----------
    p : (..., 3) array
        Barycentric coordinates (must sum to 1 along last axis).

    Returns
    -------
    (..., 2) array
        Cartesian (x, y) coordinates.
    """
    p = np.asarray(p, dtype=np.float64)
    x = p[..., 1] + 0.5 * p[..., 2]
    y = (np.sqrt(3) / 2) * p[..., 2]
    return np.stack([x, y], axis=-1)


def plot_simplex_trajectory(
    strategies: List[NDArray],
    ax: Optional[plt.Axes] = None,
    label: str = "Empirical strategy",
    color: str = "steelblue",
    alpha: float = 0.7,
    show_start_end: bool = True,
    vertex_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot a trajectory of 3-dimensional probability vectors on the simplex.

    Parameters
    ----------
    strategies : list of (3,) arrays
        Sequence of mixed strategies.
    ax : matplotlib Axes, optional
        Axes to plot on.  If ``None``, a new figure is created.
    label : str
        Legend label.
    color : str
        Line / marker color.
    alpha : float
        Transparency.
    show_start_end : bool
        Highlight the start (green) and end (red) points.
    vertex_labels : list of str, optional
        Labels for the three vertices.  Defaults to ["Action 0", "Action 1",
        "Action 2"].
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))

    if vertex_labels is None:
        vertex_labels = ["Action 0", "Action 1", "Action 2"]

    # Draw the simplex triangle
    triangle = plt.Polygon(
        [_V0, _V1, _V2],
        fill=False,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(triangle)

    # Vertex labels
    offset = 0.04
    ax.text(_V0[0] - offset, _V0[1] - offset, vertex_labels[0],
            ha="right", va="top", fontsize=10, fontweight="bold")
    ax.text(_V1[0] + offset, _V1[1] - offset, vertex_labels[1],
            ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(_V2[0], _V2[1] + offset, vertex_labels[2],
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Convert trajectory
    pts = np.array([barycentric_to_cartesian(s) for s in strategies])

    # Plot trajectory
    ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha,
            linewidth=0.8, label=label)

    if show_start_end and len(pts) > 0:
        ax.plot(pts[0, 0], pts[0, 1], "o", color="green",
                markersize=8, zorder=5, label="Start")
        ax.plot(pts[-1, 0], pts[-1, 1], "s", color="red",
                markersize=8, zorder=5, label="End")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)

    return ax


def plot_dual_simplex(
    row_strategies: List[NDArray],
    col_strategies: List[NDArray],
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    title: str = "Simplex Trajectories",
) -> Figure:
    """Plot both players' trajectories side by side.

    Parameters
    ----------
    row_strategies : list of (3,) arrays
    col_strategies : list of (3,) arrays
    row_labels, col_labels : list of str, optional
    title : str

    Returns
    -------
    matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    plot_simplex_trajectory(
        row_strategies, ax=ax1, label="Row player",
        color="steelblue", vertex_labels=row_labels,
        title="Row Player",
    )
    plot_simplex_trajectory(
        col_strategies, ax=ax2, label="Column player",
        color="coral", vertex_labels=col_labels,
        title="Column Player",
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_convergence(
    distances: NDArray,
    exploitabilities: Optional[NDArray] = None,
    title: str = "Convergence to Nash Equilibrium",
) -> Figure:
    """Plot log-distance to NE (and optionally exploitability) over time.

    Parameters
    ----------
    distances : (T,) array
        L2 distance to nearest NE at each round.
    exploitabilities : (T,) array, optional
        Exploitability at each round.
    title : str

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    T = len(distances)
    t = np.arange(1, T + 1)

    mask = distances > 1e-15
    if mask.any():
        ax.semilogy(t[mask], distances[mask], label="L2 dist to NE",
                    color="steelblue", linewidth=1)

    if exploitabilities is not None:
        mask_e = exploitabilities > 1e-15
        if mask_e.any():
            ax.semilogy(t[mask_e], exploitabilities[mask_e],
                        label="Exploitability", color="coral", linewidth=1)

    # Reference 1/t line
    ref = distances[0] * (1.0 / t) if distances[0] > 0 else 1.0 / t
    ax.semilogy(t, ref, "--", color="gray", alpha=0.5, label="O(1/t) reference")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_belief_evolution(
    strategies: List[NDArray],
    action_names: Optional[List[str]] = None,
    title: str = "Belief Evolution",
    max_steps: int = 200,
) -> Figure:
    """Plot the evolution of empirical strategy components over time.

    Parameters
    ----------
    strategies : list of (k,) arrays
    action_names : list of str, optional
    title : str
    max_steps : int
        Downsample to at most this many time steps for clarity.

    Returns
    -------
    matplotlib Figure
    """
    strats = np.array(strategies)
    T, k = strats.shape

    if action_names is None:
        action_names = [f"Action {i}" for i in range(k)]

    # Downsample if needed
    if T > max_steps:
        indices = np.linspace(0, T - 1, max_steps, dtype=int)
        strats = strats[indices]
        t = indices + 1
    else:
        t = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, k))

    for i in range(k):
        ax.plot(t, strats[:, i], label=action_names[i], color=colors[i],
                linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
