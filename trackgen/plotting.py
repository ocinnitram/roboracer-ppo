"""Draw occupancy maps in metres and gate plots on Enter."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def new_axes(figsize: tuple[float, float] = (12, 12)) -> tuple[Figure, Axes]:
    """Create a single-axes figure."""
    import matplotlib.pyplot as plt

    fig: Figure = plt.figure(figsize=figsize)
    ax: Axes = fig.add_subplot(1, 1, 1)
    return fig, ax


def draw_map(
    ax: Axes,
    image: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    alpha: float = 1.0,
) -> None:
    """Draw an occupancy image on ax in world metres, y up."""
    height, width = image.shape
    ax.imshow(
        np.flipud(image),
        cmap="gray",
        origin="lower",
        extent=(origin[0], origin[0] + width * resolution, origin[1], origin[1] + height * resolution),
        alpha=alpha,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def show_plot(fig: Any, plot: bool) -> None:
    """Display fig and wait for Enter, or close it when plots are off."""
    import matplotlib.pyplot as plt

    if not plot:
        plt.close(fig)
        return
    plt.show(block=False)
    try:
        input("Enter to continue, Ctrl-C to abort... ")
    except EOFError:
        pass
    except KeyboardInterrupt:
        plt.close(fig)
        raise SystemExit(130)
    plt.close(fig)
