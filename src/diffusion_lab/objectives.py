"""Swappable training methods. Register a factory with the same prepare contract."""

from dataclasses import dataclass

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from .schedulers import prediction_to_x0

OBJECTIVES = {}


def register_objective(name):
    def decorate(cls):
        if name in OBJECTIVES:
            raise ValueError(f"Objective {name} already exists")
        OBJECTIVES[name] = cls
        return cls

    return decorate


@dataclass
class TrainingPair:
    noisy: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor
    weights: torch.Tensor


@register_objective("diffusion")
class DiffusionObjective:
    def __init__(self, scheduler, snr_gamma=None):
        if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            raise ValueError("diffusion objective requires a DDPM/DDIM scheduler")
        if snr_gamma is not None and snr_gamma <= 0:
            raise ValueError("snr_gamma must be positive")
        self.scheduler, self.snr_gamma = scheduler, snr_gamma

    def prepare(self, clean, generator):
        noise = randn_tensor(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        time = torch.randint(
            self.scheduler.config.num_train_timesteps,
            (len(clean),),
            device=generator.device,
            generator=generator,
        ).to(clean.device)
        noisy = self.scheduler.add_noise(clean, noise, time)
        kind = self.scheduler.config.prediction_type
        target = {
            "epsilon": lambda: noise,
            "sample": lambda: clean,
            "v_prediction": lambda: self.scheduler.get_velocity(clean, noise, time),
        }[kind]()
        weights = torch.ones(len(clean), device=clean.device)
        if self.snr_gamma is not None:
            a = self.scheduler.alphas_cumprod.to(clean)[time]
            snr = a / (1 - a)
            clipped = snr.clamp(max=self.snr_gamma)
            weights = (
                clipped / snr
                if kind == "epsilon"
                else clipped / (snr + 1)
                if kind == "v_prediction"
                else clipped
            )
        return TrainingPair(noisy, target, time, weights)

    def clean_prediction(self, pair, prediction):
        return prediction_to_x0(self.scheduler, pair.noisy, prediction, pair.timesteps)

    def regularization_weights(self, pair):
        # High-noise x0 estimates are ill-conditioned; regularize only lower-noise half.
        return (pair.timesteps < self.scheduler.config.num_train_timesteps / 2).to(pair.noisy.dtype)


@register_objective("rectified_flow")
class RectifiedFlowObjective:
    """Linear bridge x_sigma=(1-sigma)x0+sigma*z; target v=z-x0."""

    def __init__(self, scheduler, snr_gamma=None):
        if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            raise ValueError("rectified_flow requires scheduler.kind=flow")
        if snr_gamma is not None:
            raise ValueError("Min-SNR weighting applies to diffusion, not rectified_flow")
        self.scheduler = scheduler

    def prepare(self, clean, generator):
        noise = randn_tensor(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        sigma = torch.rand(len(clean), device=generator.device, generator=generator).to(
            clean.device
        )
        expanded = sigma.reshape(-1, *([1] * (clean.ndim - 1)))
        return TrainingPair(
            (1 - expanded) * clean + expanded * noise,
            noise - clean,
            sigma * self.scheduler.config.num_train_timesteps,
            torch.ones_like(sigma),
        )

    def clean_prediction(self, pair, prediction):
        sigma = (pair.timesteps / self.scheduler.config.num_train_timesteps).reshape(
            -1, *([1] * (prediction.ndim - 1))
        )
        return pair.noisy - sigma * prediction

    def regularization_weights(self, pair):
        return torch.ones(len(pair.noisy), device=pair.noisy.device)
