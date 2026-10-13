"""Research visualizers and diagnostics for one, two, or three coordinates."""

from .fields import Grid, PotentialResult, evaluate_field, recover_potential
from .interactive import interactive_field
from .plots import (
    draw_walls,
    estimate_density,
    plot_dataset,
    plot_density,
    plot_potential,
    plot_trajectories,
    plot_vector_field,
    save_denoising_gif,
    save_figure,
)

__all__ = [
    "Grid",
    "PotentialResult",
    "evaluate_field",
    "recover_potential",
    "interactive_field",
    "draw_walls",
    "estimate_density",
    "plot_dataset",
    "plot_density",
    "plot_potential",
    "plot_trajectories",
    "plot_vector_field",
    "save_denoising_gif",
    "save_figure",
]
