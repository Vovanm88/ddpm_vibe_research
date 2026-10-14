"""Small notebook interface; all underlying research components remain accessible."""

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ModelMixin
from diffusers.utils import BaseOutput
from safetensors.torch import load_file, save_file

from .environments import SampleDataset, WallNavigation, make_environment
from .experiment import endpoint_constraints, resolve_device, save_run, write_json
from .metrics import evaluate_samples, sliced_wasserstein
from .models import PointDenoiser
from .pipeline import ResearchDiffusionPipeline
from .schedulers import make_scheduler
from .training import DiffusionTrainer, TrainerConfig


@dataclass
class ExperimentOutput(BaseOutput):
    samples: torch.Tensor
    conditions: torch.Tensor | None = None
    history: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None


class ResearchExperiment:
    """A dataset, pipeline and trainer assembled for interactive research.

    ``fit(epochs=5)`` trains five ADDITIONAL epochs each time it is called.
    Sampling/fields use EMA if enabled and restore training components afterward.
    """

    def __init__(self, dataset, pipeline, trainer, config, default_epochs=20):
        self.dataset, self.pipeline, self.trainer = dataset, pipeline, trainer
        self.config = copy.deepcopy(config)
        self.default_epochs = default_epochs

    @property
    def is_navigation(self):
        return hasattr(self.dataset, "environment") and isinstance(
            self.dataset.environment, WallNavigation
        )

    def fit(self, epochs=None, callback=None, checkpoint_dir=None):
        additional = self.default_epochs if epochs is None else epochs
        if not isinstance(additional, int) or isinstance(additional, bool) or additional < 1:
            raise ValueError("epochs must be a positive integer number of additional epochs")
        start = len(self.trainer.history)
        self.trainer.config.epochs = self.trainer.completed_epochs + additional
        self.trainer.fit(self.dataset, callback=callback, checkpoint_dir=checkpoint_dir)
        return self.trainer.history[start:]

    def _condition_batch(self, condition, batch_size):
        if condition is None:
            return None
        values = torch.as_tensor(condition, device=self.trainer.device)
        if self.dataset.condition_kind == "continuous":
            if values.ndim == 1:
                values = values[None]
        elif self.dataset.condition_kind == "categorical" and values.ndim == 0:
            values = values[None]
        if values.ndim < 1:
            raise ValueError("Conditions must be a class ID, context vector or batch")
        if len(values) == 1:
            values = values.expand(batch_size, *values.shape[1:])
        if len(values) != batch_size:
            raise ValueError("Condition batch must contain one row or num_samples rows")
        return values

    def sample(
        self,
        num_samples=256,
        condition=None,
        steps=50,
        guidance_scale=1.0,
        seed=42,
        record_history=False,
        pin_endpoints=True,
        unconditional=False,
    ):
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if unconditional and condition is not None:
            raise ValueError("unconditional=True cannot be combined with an explicit condition")
        if unconditional:
            conditions = None
        elif condition is not None:
            conditions = self._condition_batch(condition, num_samples)
        elif self.dataset.conditions is not None:
            indices = torch.randint(
                len(self.dataset), (num_samples,), generator=torch.Generator().manual_seed(seed)
            )
            conditions = self.dataset.conditions[indices.to(self.dataset.conditions.device)].to(
                self.trainer.device
            )
        else:
            conditions = None
        constraints = (
            endpoint_constraints(self.trainer.model, conditions)
            if pin_endpoints and self.is_navigation
            else {}
        )
        original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = copy.deepcopy(original_scheduler)
        try:
            with self.trainer.ema_weights():
                output = self.pipeline(
                    batch_size=num_samples,
                    conditions=conditions,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(seed),
                    record_history=record_history,
                    **constraints,
                )
        finally:
            self.pipeline.scheduler = original_scheduler
        return ExperimentOutput(output.samples, conditions, output.history, output.timesteps)

    def evaluate(self, output):
        if len(self.dataset.sample_shape) == 1 or self.is_navigation:
            return evaluate_samples(
                self.dataset, output.samples, output.conditions, seed=self.config["seed"]
            )
        return {
            "sliced_wasserstein_flat": sliced_wasserstein(
                self.dataset.samples.flatten(1).detach().cpu().numpy(),
                output.samples.flatten(1).detach().cpu().numpy(),
                seed=self.config["seed"],
            )
        }

    def plot_samples(self, output=None, ax=None, title=None):
        from .visualization import plot_dataset, plot_trajectories

        values = self.dataset.samples if output is None else output.samples
        title = title or ("Reference data" if output is None else "Generated samples")
        if len(self.dataset.sample_shape) == 2:
            return plot_trajectories(
                values.detach().cpu().numpy(),
                walls=self.dataset.metadata.get("walls", []),
                ax=ax,
                title=title,
            )
        labels = (
            self.dataset.labels
            if output is None
            else output.conditions
            if self.dataset.condition_kind == "categorical"
            else None
        )
        return plot_dataset(values, labels, ax=ax, title=title)

    def vector_field(self, points, timestep, condition=None, kind="score", guidance_scale=1.0):
        conditions = self._condition_batch(condition, len(points))
        with self.trainer.ema_weights():
            return self.pipeline.vector_field(
                points, timestep, conditions=conditions, kind=kind, guidance_scale=guidance_scale
            )

    def save(self, path):
        """Save data + model + optimizer/EMA/RNG for portable continued training."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(self.config)
        config["training"] = asdict(self.trainer.config)
        self.trainer.save_checkpoint(path / "checkpoint.pt")
        save_run(self.trainer, self.dataset, config, path)
        tensors = {"samples": self.dataset.samples.detach().cpu().contiguous()}
        if self.dataset.conditions is not None:
            tensors["conditions"] = self.dataset.conditions.detach().cpu().contiguous()
        if self.dataset.labels is not None:
            # Clone because categorical conditions and labels may share storage.
            tensors["labels"] = self.dataset.labels.detach().cpu().clone().contiguous()
        save_file(tensors, path / "dataset.safetensors")
        write_json(
            path / "notebook_state.json",
            {
                "metadata": self.dataset.metadata,
                "num_classes": self.dataset.num_classes,
                "default_epochs": self.default_epochs,
            },
        )
        return path

    @classmethod
    def load(cls, path, *, device=None, dataset=None, model=None):
        """Custom model/conditioner modules must be imported before loading.

        For a model class defined only in a notebook, pass a constructed model.
        """
        import yaml

        path = Path(path)
        config = yaml.safe_load((path / "config.yaml").read_text())
        torch.set_num_threads(config.get("num_threads", 4))
        state = json.loads((path / "notebook_state.json").read_text())
        if dataset is None:
            tensors = load_file(path / "dataset.safetensors")
            dataset = SampleDataset(
                tensors["samples"],
                tensors.get("conditions"),
                labels=tensors.get("labels"),
                metadata=state["metadata"],
                num_classes=state["num_classes"] or None,
            )
            metadata = dataset.metadata
            if str(metadata.get("name", "")).startswith("navigation_"):
                dataset.environment = WallNavigation(
                    dim=metadata["dim"],
                    walls=metadata["walls"],
                    bounds=metadata["bounds"],
                    path_length=metadata["path_length"],
                    grid_resolution=metadata["grid_resolution"],
                    margin=metadata["margin"],
                )
        kwargs = {"model": model} if model is not None else {}
        pipeline = ResearchDiffusionPipeline.from_pretrained(
            path / "pipeline", local_files_only=True, low_cpu_mem_usage=False, **kwargs
        )
        chosen_device = resolve_device(device or config.get("device", "cpu"))
        config["device"] = chosen_device
        trainer = DiffusionTrainer(
            pipeline,
            TrainerConfig(**config["training"]),
            device=chosen_device,
            metadata=dataset.metadata,
        )
        trainer.load_checkpoint(path / "checkpoint.pt")
        return cls(dataset, pipeline, trainer, config, state["default_epochs"])


def create_experiment(
    environment="rotating_rays",
    *,
    dataset=None,
    n_samples=None,
    environment_kwargs=None,
    model=None,
    scheduler=None,
    training=None,
    device="cpu",
    seed=42,
    num_threads=4,
):
    """Assemble points/paths and inferred conditions without YAML or local files.

    ``model`` accepts constructor overrides or a compatible ModelMixin instance.
    ``scheduler``/``training`` are override dicts; flow derives its training objective.
    A custom SampleDataset can be passed instead of a named environment.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    if isinstance(environment, SampleDataset):
        if dataset is not None:
            raise ValueError("Provide custom dataset once")
        dataset = environment
    if dataset is None:
        env_config = (
            copy.deepcopy(environment) if isinstance(environment, dict) else {"name": environment}
        )
        env_config.update(copy.deepcopy(environment_kwargs or {}))
        if n_samples is not None:
            env_config["n_samples"] = n_samples
        env_config.setdefault("n_samples", 2048)
        env_config.setdefault("seed", seed)
        dataset = make_environment(**env_config)
    else:
        env_config = {"name": dataset.metadata.get("name", "custom"), "n_samples": len(dataset)}
    model_kwargs = {
        "sample_shape": dataset.sample_shape,
        "hidden_dim": 64,
        "num_layers": 3,
        "conditioner": {"categorical": "embedding", "continuous": "mlp", "none": "none"}[
            dataset.condition_kind
        ],
        "num_classes": dataset.num_classes,
        "condition_dim": dataset.condition_dim,
    }
    if isinstance(model, ModelMixin):
        denoiser = model
        if tuple(denoiser.sample_shape) != dataset.sample_shape:
            raise ValueError("Provided model sample_shape differs from dataset")
    else:
        model_kwargs.update(copy.deepcopy(model or {}))
        denoiser = PointDenoiser(**model_kwargs)
    scheduler_kwargs = {
        "kind": "ddim",
        "num_train_timesteps": 1000,
        "prediction_type": "v_prediction",
    }
    scheduler_kwargs.update(copy.deepcopy(scheduler or {}))
    schedule = make_scheduler(**scheduler_kwargs)
    training_kwargs = {
        "epochs": 20,
        "ema_decay": 0.99,
        "seed": seed,
        "objective": "rectified_flow"
        if isinstance(schedule, FlowMatchEulerDiscreteScheduler)
        else "diffusion",
    }
    training_kwargs.update(copy.deepcopy(training or {}))
    trainer_config = TrainerConfig(**training_kwargs)
    pipeline = ResearchDiffusionPipeline(denoiser, schedule)
    chosen_device = resolve_device(device)
    trainer = DiffusionTrainer(
        pipeline, trainer_config, device=chosen_device, metadata=dataset.metadata
    )
    config = {
        "seed": seed,
        "device": chosen_device,
        "num_threads": num_threads,
        "environment": env_config,
        "model": {
            key: value for key, value in dict(denoiser.config).items() if not key.startswith("_")
        },
        "scheduler": scheduler_kwargs,
        "training": asdict(trainer_config),
    }
    return ResearchExperiment(dataset, pipeline, trainer, config, trainer_config.epochs)
