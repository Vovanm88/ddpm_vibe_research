"""Diffusion Lab: components are importable and independently replaceable."""

from .models import PointDenoiser
from .pipeline import ResearchDiffusionPipeline
from .schedulers import make_scheduler
from .training import DiffusionTrainer, TrainerConfig

__all__ = [
    "PointDenoiser",
    "ResearchDiffusionPipeline",
    "make_scheduler",
    "DiffusionTrainer",
    "TrainerConfig",
]
