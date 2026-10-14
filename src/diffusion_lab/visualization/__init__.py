"""Research visualizers and diagnostics for one, two, or three coordinates."""

from .animation import plot_rainbow_trajectories, save_trajectory_gif
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
from .widgets import notebook_field_explorer, notebook_sampling_explorer

__all__ = [
    "Grid",
    "PotentialResult",
    "evaluate_field",
    "recover_potential",
    "interactive_field",
    "notebook_field_explorer",
    "notebook_sampling_explorer",
    "draw_walls",
    "estimate_density",
    "plot_dataset",
    "plot_density",
    "plot_potential",
    "plot_trajectories",
    "plot_rainbow_trajectories",
    "plot_vector_field",
    "save_denoising_gif",
    "save_trajectory_gif",
    "save_figure",
]
