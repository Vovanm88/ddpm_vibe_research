"""Train editable PyTorch / Diffusers component swaps.

Run after installing the package::

    python examples/notebook_components.py --epochs 3 --samples 256

The default is a CPU smoke run, not a convergence benchmark. The standalone
notebook 04_change_components.ipynb shows the same loop and objects in cells.
"""

from __future__ import annotations

import argparse

import torch
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from torch.utils.data import DataLoader

from diffusion_lab import PointDenoiser, ResearchDiffusionPipeline
from diffusion_lab.conditioning import CONDITIONERS, register_conditioner
from diffusion_lab.environments import SampleDataset, make_environment
from diffusion_lab.experiment import endpoint_constraints
from diffusion_lab.losses import bounds, collision, endpoint, smoothness
from diffusion_lab.metrics import evaluate_samples


def train_loop(
    model,
    scheduler,
    loader,
    optimizer,
    loss_fn,
    epochs=3,
    condition_dropout=0.15,
    regularizers=None,
    metadata=None,
    grad_clip=1.0,
    generator=None,
    initial_epoch=0,
    callback=None,
):
    """Plain PyTorch optimization; every object and loss callable can be replaced.

    Loss may be scalar or unreduced [B]. Regularizers are
    {name: (function(clean_prediction, conditions, metadata), weight)}.
    A CPU generator also works with CUDA/MPS through Diffusers' randn_tensor.
    """
    history = []
    flow = isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    for epoch in range(initial_epoch, initial_epoch + epochs):
        model.train()
        sums, count = {}, 0
        for batch in loader:
            clean = batch["samples"].to(device=model.device, dtype=model.dtype)
            conditions = batch.get("conditions")
            conditions = conditions.to(model.device) if conditions is not None else None
            noise = randn_tensor(
                clean.shape, generator=generator, device=model.device, dtype=model.dtype
            )
            if flow:
                sigma = torch.rand(len(clean), generator=generator).to(model.device)
                expanded_sigma = sigma.reshape(-1, *([1] * (clean.ndim - 1)))
                noisy = (1 - expanded_sigma) * clean + expanded_sigma * noise
                timesteps = sigma * scheduler.config.num_train_timesteps
                target = noise - clean  # dx_sigma / d_sigma
            else:
                timesteps = torch.randint(
                    scheduler.config.num_train_timesteps, (len(clean),), generator=generator
                ).to(model.device)
                noisy = scheduler.add_noise(clean, noise, timesteps)
                prediction_type = scheduler.config.prediction_type
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = scheduler.get_velocity(clean, noise, timesteps)
                elif prediction_type == "sample":
                    target = clean
                else:
                    raise ValueError(f"Unsupported prediction_type: {prediction_type}")

            keep = (torch.rand(len(clean), generator=generator) >= condition_dropout).to(
                model.device
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy, timesteps, conditions=conditions, condition_mask=keep).sample
            regression = loss_fn(prediction, target).mean()
            loss = regression
            values = {"regression": regression.detach().item()}
            if regularizers:
                if flow:
                    clean_prediction = noisy - expanded_sigma * prediction
                    active = torch.ones(len(clean), dtype=torch.bool, device=model.device)
                else:
                    alpha = scheduler.alphas_cumprod.to(clean)[timesteps].reshape(
                        -1, *([1] * (clean.ndim - 1))
                    )
                    clean_prediction = (
                        (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
                        if prediction_type == "epsilon"
                        else alpha.sqrt() * noisy - (1 - alpha).sqrt() * prediction
                        if prediction_type == "v_prediction"
                        else prediction
                    )
                    active = timesteps < scheduler.config.num_train_timesteps // 2
                if active.any():
                    selected_conditions = conditions[active] if conditions is not None else None
                    for name, (penalty_fn, weight) in regularizers.items():
                        penalty = penalty_fn(
                            clean_prediction[active], selected_conditions, metadata or {}
                        )
                        loss = loss + weight * penalty
                        values[name] = penalty.detach().item()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            optimizer.step()
            values["loss"] = loss.detach().item()
            for name, value in values.items():
                sums[name] = sums.get(name, 0.0) + len(clean) * value
            count += len(clean)
        row = {name: value / count for name, value in sums.items()}
        row.update(epoch=epoch + 1, step=(epoch + 1) * len(loader))
        history.append(row)
        if callback is not None:
            callback(row)
    return history


def notebook_charbonnier(prediction, target):
    return ((prediction - target).square() + 1e-6).sqrt().flatten(1).mean(1)


class FourierConditioner(nn.Module):
    def __init__(self, hidden_dim, num_classes, condition_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(3 * condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, conditions):
        value = conditions.to(self.projection[0].weight.dtype)
        features = torch.cat((value, (torch.pi * value).sin(), (torch.pi * value).cos()), -1)
        return self.projection(features)


def radius_penalty(clean_prediction, conditions, metadata):
    return torch.relu(clean_prediction.norm(dim=-1) - metadata.get("radius", 1.0)).square().mean()


class AdapterDenoiser(PointDenoiser):
    """Add a learned residual input adapter while retaining the model contract."""

    @register_to_config
    def __init__(
        self,
        sample_shape=(2,),
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        backbone="mlp",
        conditioner="embedding",
        num_classes=4,
        condition_dim=0,
        adapter_scale=0.1,
    ):
        super().__init__(
            sample_shape=sample_shape,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            backbone=backbone,
            conditioner=conditioner,
            num_classes=num_classes,
            condition_dim=condition_dim,
        )
        if len(sample_shape) != 1:
            raise ValueError("This example adapter handles points [B,D]")
        self.adapter = nn.Sequential(
            nn.Linear(sample_shape[0], hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, sample_shape[0]),
        )
        self.adapter_scale = adapter_scale

    def forward(self, sample, timestep, **kwargs):
        return super().forward(
            sample + self.adapter_scale * self.adapter(sample), timestep, **kwargs
        )


def run_examples(n_samples=256, epochs=3, device="cpu", seed=42):
    """Return actual dataset/model/scheduler/pipeline/optimizer objects for each run."""
    if "notebook_fourier" not in CONDITIONERS:
        register_conditioner("notebook_fourier")(FourierConditioner)
    rays = make_environment("rotating_rays", n_samples=n_samples, seed=seed, n_classes=4)
    torch.set_num_threads(4)
    cases = {}

    def run(
        name,
        *,
        dataset=rays,
        environment="rotating_rays",
        model=None,
        scheduler=None,
        loss_fn=None,
        regularizers=None,
        environment_kwargs=None,
        optimizer_factory=torch.optim.AdamW,
        training_epochs=epochs,
    ):
        torch.manual_seed(seed)
        if dataset is None:
            dataset = make_environment(
                environment,
                n_samples=min(n_samples, 32) if environment == "navigation_2d" else n_samples,
                seed=seed,
                **(environment_kwargs or {}),
            )
        model_kwargs = {
            "sample_shape": dataset.sample_shape,
            "hidden_dim": 32,
            "num_layers": 1,
            "conditioner": {"categorical": "embedding", "continuous": "mlp", "none": "none"}[
                dataset.condition_kind
            ],
            "num_classes": dataset.num_classes,
            "condition_dim": dataset.condition_dim,
        }
        denoiser = (
            model
            if isinstance(model, nn.Module)
            else PointDenoiser(
                **{
                    **model_kwargs,
                    **(model or {}),
                }
            )
        )
        schedule = scheduler or DDIMScheduler(
            num_train_timesteps=64,
            prediction_type="v_prediction",
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            timestep_spacing="trailing",
        )
        pipeline = ResearchDiffusionPipeline(denoiser, schedule).to(device)
        loader = DataLoader(
            dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed + 1)
        )
        optimizer = optimizer_factory(denoiser.parameters(), lr=2e-3, weight_decay=1e-4)
        history = train_loop(
            denoiser,
            schedule,
            loader,
            optimizer,
            loss_fn if loss_fn is not None else nn.MSELoss(),
            epochs=training_epochs,
            regularizers=regularizers,
            metadata=dataset.metadata,
            generator=torch.Generator().manual_seed(seed + 2),
        )
        indices = torch.randint(
            len(dataset), (64,), generator=torch.Generator().manual_seed(seed + 3)
        )
        conditions = dataset.conditions[indices] if dataset.conditions is not None else None
        constraints = (
            endpoint_constraints(denoiser, conditions) if hasattr(dataset, "environment") else {}
        )
        output = pipeline(
            batch_size=64,
            conditions=conditions,
            num_inference_steps=24,
            generator=torch.Generator().manual_seed(seed + 3),
            **constraints,
        )
        cases[name] = {
            "dataset": dataset,
            "model": denoiser,
            "scheduler": schedule,
            "pipeline": pipeline,
            "loader": loader,
            "optimizer": optimizer,
            "history": history,
            "conditions": conditions,
            "output": output,
            "metrics": evaluate_samples(dataset, output.samples, conditions, seed=seed),
        }

    run("MLP + MSE + embedding")
    run("Transformer", model={"backbone": "transformer"})
    torch.manual_seed(seed)
    run(
        "Custom model",
        model=AdapterDenoiser(sample_shape=rays.sample_shape, num_classes=rays.num_classes),
    )
    run("Huber", loss_fn=nn.SmoothL1Loss())
    run("Charbonnier", loss_fn=notebook_charbonnier)
    run("One-hot", model={"conditioner": "one_hot"})
    run("Adam", optimizer_factory=torch.optim.Adam)
    for prediction_type in ("epsilon", "v_prediction"):
        run(
            f"{prediction_type} (linear 1000)",
            scheduler=DDIMScheduler(
                num_train_timesteps=1000,
                prediction_type=prediction_type,
                beta_schedule="linear",
                clip_sample=False,
                timestep_spacing="trailing",
            ),
        )
    run("Rectified flow", scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=64))
    run("Radius penalty", regularizers={"radius": (radius_penalty, 0.1)})

    generator = torch.Generator().manual_seed(seed)
    context = torch.linspace(-1, 1, n_samples)[:, None]
    samples = torch.cat((context, context.square()), dim=1)
    samples += 0.02 * torch.randn(samples.shape, generator=generator)
    custom_data = SampleDataset(samples, context, metadata={"name": "conditioned_parabola"})
    for conditioner in ("mlp", "notebook_fourier"):
        run(f"Continuous {conditioner}", dataset=custom_data, model={"conditioner": conditioner})
    for environment in ("gaussian_mixture_1d", "helix_3d"):
        run(environment, dataset=None, environment=environment)
    run(
        "Navigation",
        dataset=None,
        environment="navigation_2d",
        environment_kwargs={"path_length": 16, "grid_resolution": 17},
        model={"backbone": "transformer"},
        loss_fn=nn.SmoothL1Loss(),
        regularizers={
            "smoothness": (smoothness, 0.05),
            "collision": (collision, 0.2),
            "endpoint": (endpoint, 0.1),
            "bounds": (bounds, 0.1),
        },
        training_epochs=min(epochs, 2),
    )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for name, result in run_examples(args.samples, args.epochs, args.device).items():
        metrics = result["metrics"]
        key = "sliced_wasserstein" if "sliced_wasserstein" in metrics else "collision_free_fraction"
        print(f"{name}: loss={result['history'][-1]['loss']:.5g}; {key}={metrics[key]:.5g}")


if __name__ == "__main__":
    main()
