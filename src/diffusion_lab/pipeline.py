"""Point / trajectory sampling, CFG, inpainting and interpretable vector fields."""

from dataclasses import dataclass

import torch
from diffusers import DDIMScheduler, DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from .schedulers import alpha_bar, prediction_to_x0


@dataclass
class ResearchPipelineOutput(BaseOutput):
    samples: torch.Tensor
    history: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None


class ResearchDiffusionPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    def _conditions(self, conditions, batch_size, device):
        if conditions is None:
            return None
        result = torch.as_tensor(conditions, device=device)
        continuous = self.model.config.condition_dim > 0 and self.model.config.conditioner not in (
            "none",
            "one_hot",
            "embedding",
        )
        if continuous and result.ndim == 1:
            result = result[None]
        if result.ndim == 0:
            result = result.expand(batch_size)
        elif len(result) == 1:
            result = result.expand(batch_size, *result.shape[1:])
        if len(result) != batch_size:
            raise ValueError("Conditions must have one row or batch_size rows")
        return result

    def _prediction(self, samples, timestep, conditions, guidance_scale):
        conditioned = self.model(samples, timestep, conditions=conditions).sample
        if conditions is None or guidance_scale == 1:
            return conditioned
        unconditional = self.model(samples, timestep, conditions=None).sample
        return unconditional + guidance_scale * (conditioned - unconditional)

    @torch.no_grad()
    def __call__(
        self,
        batch_size=128,
        conditions=None,
        num_inference_steps=100,
        guidance_scale=1.0,
        generator=None,
        record_history=False,
        eta=0.0,
        known_values=None,
        known_mask=None,
        return_dict=True,
    ):
        """Sample; known values use the same noise realization at each noise level.

        For paths, mask the first / last waypoint and set known_values to A / B.
        This fixes endpoints only. Collision-free generated routes are not guaranteed.
        History contains initial noise plus *actual* scheduler steps, without interpolation.
        """
        if (
            batch_size < 1
            or not 1 <= num_inference_steps <= self.scheduler.config.num_train_timesteps
        ):
            raise ValueError("Invalid batch_size or num_inference_steps")
        if not torch.isfinite(torch.tensor(guidance_scale)) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if (
            not torch.isfinite(torch.tensor(eta))
            or eta < 0
            or eta > 1
            or (eta and not isinstance(self.scheduler, DDIMScheduler))
        ):
            raise ValueError("eta in [0,1] is only supported with DDIM")
        device, dtype = self.model.device, self.model.dtype
        conditions = self._conditions(conditions, batch_size, device)
        shape = (batch_size, *self.model.sample_shape)
        samples = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        samples = samples * getattr(self.scheduler, "init_noise_sigma", 1.0)
        flow = isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler)
        if (known_values is None) != (known_mask is None):
            raise ValueError("Provide both known_values and known_mask")
        if known_values is not None:
            known_values = torch.as_tensor(known_values, device=device, dtype=dtype).expand(shape)
            known_mask = torch.as_tensor(known_mask, device=device, dtype=torch.bool).expand(shape)
            constraint_noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        def constrain(values, timestep):
            if known_values is None:
                return values
            if timestep is None:
                target = known_values
            elif flow:
                sigma = timestep.to(dtype) / self.scheduler.config.num_train_timesteps
                target = (1 - sigma) * known_values + sigma * constraint_noise
            else:
                times = timestep.long().expand(batch_size)
                target = self.scheduler.add_noise(known_values, constraint_noise, times)
            return torch.where(known_mask, target, values)

        samples = constrain(samples, self.scheduler.timesteps[0])
        history = [samples.cpu().clone()] if record_history else None
        was_training = self.model.training
        self.model.eval()
        try:
            for index, timestep in enumerate(self.scheduler.timesteps):
                model_input = (
                    samples if flow else self.scheduler.scale_model_input(samples, timestep)
                )
                prediction = self._prediction(model_input, timestep, conditions, guidance_scale)
                kwargs = {}
                if isinstance(self.scheduler, DDIMScheduler):
                    kwargs = {"eta": eta, "generator": generator}
                elif not flow:
                    kwargs = {"generator": generator}
                samples = self.scheduler.step(prediction, timestep, samples, **kwargs).prev_sample
                next_time = (
                    self.scheduler.timesteps[index + 1]
                    if index + 1 < len(self.scheduler.timesteps)
                    else None
                )
                samples = constrain(samples, next_time)
                if history is not None:
                    history.append(samples.cpu().clone())
        finally:
            self.model.train(was_training)
        output = ResearchPipelineOutput(
            samples=samples,
            history=torch.stack(history) if history is not None else None,
            timesteps=self.scheduler.timesteps.cpu().clone(),
        )
        return output if return_dict else (output.samples, output.history, output.timesteps)

    @torch.no_grad()
    def vector_field(self, points, timestep, conditions=None, kind="score", guidance_scale=1.0):
        """No forward noise is added. Query points already represent x_t.

        score = grad log p_t(x); denoising = predicted x_0 - x_t.
        Neither is an arbitrary one-step sampler displacement. For rectified flow,
        velocity is dx/dsigma (noise direction); denoising is its negative.
        """
        samples = torch.as_tensor(points, device=self.model.device, dtype=self.model.dtype)
        if len(self.model.sample_shape) != 1:
            raise ValueError("Spatial field queries require a point model, not a path model")
        time = torch.as_tensor(timestep, device=samples.device)
        maximum = self.scheduler.config.num_train_timesteps
        flow = isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler)
        if (
            time.numel() != 1
            or not torch.isfinite(time).all()
            or time < 0
            or (time > maximum if flow else time >= maximum)
        ):
            raise ValueError("timestep must be a scalar in the training range")
        conditions = self._conditions(conditions, len(samples), samples.device)
        was_training = self.model.training
        self.model.eval()
        try:
            prediction = self._prediction(samples, time, conditions, guidance_scale)
        finally:
            self.model.train(was_training)
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            if kind == "velocity":
                return prediction
            if kind == "denoising":
                return -prediction
            raise ValueError("Flow fields support velocity or denoising; score is undefined")
        if time.dtype not in (torch.int32, torch.int64):
            raise ValueError("Diffusion timestep must be an integer index")
        times = time.expand(len(samples))
        clean = prediction_to_x0(self.scheduler, samples, prediction, times)
        a = alpha_bar(self.scheduler, times, samples)
        epsilon = (samples - a.sqrt() * clean) / (1 - a).sqrt()
        if kind == "epsilon":
            return epsilon
        if kind == "denoising":
            return clean - samples
        if kind == "score":
            return -epsilon / (1 - a).sqrt()
        raise ValueError(f"Unknown field kind: {kind}")
