"""Diffusion Lab: components are importable and independently replaceable."""

from .interval_model import IntervalPointDenoiser
from .models import PointDenoiser
from .notebook import ExperimentOutput, ResearchExperiment, create_experiment
from .pipeline import ResearchDiffusionPipeline
from .schedulers import make_scheduler
from .training import DiffusionTrainer, TrainerConfig

__all__ = [
    "PointDenoiser",
    "IntervalPointDenoiser",
    "ResearchDiffusionPipeline",
    "make_scheduler",
    "DiffusionTrainer",
    "TrainerConfig",
    "create_experiment",
    "ResearchExperiment",
    "ExperimentOutput",
]
