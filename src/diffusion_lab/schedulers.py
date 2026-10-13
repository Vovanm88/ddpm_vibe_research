"""Official Diffusers schedules; no hand-written reverse diffusion indexing."""

import torch
from diffusers import DDIMScheduler, DDPMScheduler, FlowMatchEulerDiscreteScheduler


def make_scheduler(
    kind="ddpm",
    num_train_timesteps=1000,
    prediction_type="epsilon",
    beta_schedule="squaredcos_cap_v2",
    beta_start=1e-4,
    beta_end=0.02,
):
    if num_train_timesteps < 2:
        raise ValueError("num_train_timesteps must be >= 2")
    if kind == "flow":
        return FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)
    classes = {"ddpm": DDPMScheduler, "ddim": DDIMScheduler}
    if kind not in classes:
        raise ValueError(f"Unknown scheduler: {kind}")
    if prediction_type not in ("epsilon", "sample", "v_prediction"):
        raise ValueError(f"Unknown prediction_type: {prediction_type}")
    return classes[kind](
        num_train_timesteps=num_train_timesteps,
        prediction_type=prediction_type,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False,
        timestep_spacing="trailing",
    )


def alpha_bar(scheduler, timesteps, sample):
    time = timesteps.to(device=sample.device)
    if time.dtype not in (torch.int32, torch.int64):
        raise ValueError("Diffusion timesteps must be integer indices")
    if (
        time.ndim != 1
        or len(time) != len(sample)
        or time.min() < 0
        or time.max() >= scheduler.config.num_train_timesteps
    ):
        raise ValueError("Diffusion timestep index out of range or incompatible batch")
    return scheduler.alphas_cumprod.to(sample)[time.long()].reshape(-1, *([1] * (sample.ndim - 1)))


def prediction_to_x0(scheduler, noisy, prediction, timesteps):
    a = alpha_bar(scheduler, timesteps, noisy)
    kind = scheduler.config.prediction_type
    if kind == "epsilon":
        return (noisy - (1 - a).sqrt() * prediction) / a.sqrt()
    if kind == "sample":
        return prediction
    if kind == "v_prediction":
        return a.sqrt() * noisy - (1 - a).sqrt() * prediction
    raise ValueError(f"Unsupported prediction_type: {kind}")
