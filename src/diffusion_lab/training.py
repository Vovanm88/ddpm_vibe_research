"""Reproducible trainer with EMA and epoch-boundary checkpoint/resume."""

import copy
import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .losses import LOSSES, REGULARIZERS
from .objectives import OBJECTIVES


@dataclass
class TrainerConfig:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    condition_dropout: float = 0.15
    grad_clip: float = 1.0
    loss: str = "mse"
    objective: str = "diffusion"
    snr_gamma: float | None = None
    ema_decay: float | None = 0.995
    seed: int = 42
    regularizers: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, learning_rate must be positive")
        if not 0 <= self.condition_dropout <= 1 or self.grad_clip <= 0:
            raise ValueError("Invalid condition_dropout or grad_clip")
        if self.ema_decay is not None and not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        if self.loss not in LOSSES or self.objective not in OBJECTIVES:
            raise ValueError("Unknown loss or objective; register custom implementation first")
        if any(
            name not in REGULARIZERS or weight < 0 for name, weight in self.regularizers.items()
        ):
            raise ValueError("Unknown regularizer or negative weight")


class DiffusionTrainer:
    def __init__(self, pipeline, config=None, device="cpu", metadata=None):
        self.pipeline = pipeline
        self.config = config or TrainerConfig()
        self.device = torch.device(device)
        self.pipeline.to(self.device)
        self.model = self.pipeline.model
        self.metadata = metadata or {}
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.objective = OBJECTIVES[self.config.objective](
            pipeline.scheduler, self.config.snr_gamma
        )
        # MPS draws through a seeded CPU generator and copies to the accelerator.
        self.generator = torch.Generator(
            device="cpu" if self.device.type == "mps" else self.device
        ).manual_seed(self.config.seed)
        self.loader_generator = torch.Generator().manual_seed(self.config.seed + 1)
        self.ema = (
            copy.deepcopy(self.model).eval().requires_grad_(False)
            if self.config.ema_decay is not None
            else None
        )
        self.global_step = 0
        self.completed_epochs = 0
        self.history = []
        self.dataset_signature = None

    def train_step(self, batch):
        self.model.train()
        clean = batch["samples"].to(self.device, dtype=self.model.dtype)
        conditions = batch.get("conditions")
        if conditions is not None:
            conditions = conditions.to(self.device)
        pair = self.objective.prepare(clean, self.generator)
        keep = (
            torch.rand(len(clean), device=self.generator.device, generator=self.generator)
            >= self.config.condition_dropout
        ).to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(
            pair.noisy, pair.timesteps, conditions=conditions, condition_mask=keep
        ).sample
        per_sample = LOSSES[self.config.loss](prediction, pair.target)
        if per_sample.shape != (len(clean),):
            raise ValueError("Registered loss must return one value per example")
        regression = (per_sample * pair.weights).mean()
        total = regression
        metrics = {"regression": float(regression.detach())}
        active = self.objective.regularization_weights(pair).bool()
        if self.config.regularizers and active.any():
            predicted_clean = self.objective.clean_prediction(pair, prediction)[active]
            selected_conditions = conditions[active] if conditions is not None else None
            for name, weight in self.config.regularizers.items():
                if weight:
                    penalty = REGULARIZERS[name](
                        predicted_clean, selected_conditions, self.metadata
                    )
                    total = total + weight * penalty
                    metrics[name] = float(penalty.detach())
        if not torch.isfinite(total):
            raise FloatingPointError("Training loss is not finite; inspect data and configuration")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        if self.ema is not None:
            with torch.no_grad():
                for averaged, live in zip(
                    self.ema.parameters(), self.model.parameters(), strict=True
                ):
                    averaged.lerp_(live, 1 - self.config.ema_decay)
                for averaged, live in zip(self.ema.buffers(), self.model.buffers(), strict=True):
                    averaged.copy_(live)
        self.global_step += 1
        metrics.update(loss=float(total.detach()), grad_norm=float(norm), step=self.global_step)
        return metrics

    def fit(self, dataset, callback=None, checkpoint_dir=None):
        """config.epochs is the total target, including already-completed epochs."""
        if tuple(dataset.sample_shape) != self.model.sample_shape:
            raise ValueError("Dataset shape does not match model")
        signature = hashlib.sha256(dataset.samples.detach().cpu().numpy().tobytes())
        if dataset.conditions is not None:
            signature.update(dataset.conditions.detach().cpu().numpy().tobytes())
        signature = signature.hexdigest()
        if self.dataset_signature is not None and signature != self.dataset_signature:
            raise ValueError("Resume dataset differs from checkpoint")
        self.dataset_signature = signature
        self.metadata = dataset.metadata
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=self.loader_generator,
            num_workers=0,
        )
        for epoch in range(self.completed_epochs, self.config.epochs):
            sums, counts = {}, {}
            for batch in loader:
                metrics = self.train_step(batch)
                weight = len(batch["samples"])
                for name, value in metrics.items():
                    if name != "step":
                        sums[name] = sums.get(name, 0.0) + value * weight
                        counts[name] = counts.get(name, 0) + weight
            result = {name: total / counts[name] for name, total in sums.items()}
            self.completed_epochs = epoch + 1
            result.update(epoch=self.completed_epochs, step=self.global_step)
            self.history.append(result)
            if callback is not None:
                callback(result)
            if checkpoint_dir is not None:
                self.save_checkpoint(Path(checkpoint_dir) / "checkpoint.pt")
        return self.history

    @contextmanager
    def ema_weights(self):
        if self.ema is None:
            yield
            return
        original = self.pipeline.model
        self.pipeline.model = self.ema
        try:
            yield
        finally:
            self.pipeline.model = original

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "config": asdict(self.config),
            "metadata": self.metadata,
            "model_config": _semantic_config(self.model.config),
            "scheduler_class": type(self.pipeline.scheduler).__name__,
            "scheduler_config": _semantic_config(self.pipeline.scheduler.config),
            "dataset_signature": self.dataset_signature,
            "generator": self.generator.get_state(),
            "loader_generator": self.loader_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "global_step": self.global_step,
            "completed_epochs": self.completed_epochs,
            "history": self.history,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if _semantic_config(checkpoint["model_config"]) != _semantic_config(self.model.config):
            raise ValueError("Resume model config differs from checkpoint")
        if checkpoint["scheduler_class"] != type(
            self.pipeline.scheduler
        ).__name__ or _semantic_config(checkpoint["scheduler_config"]) != _semantic_config(
            self.pipeline.scheduler.config
        ):
            raise ValueError("Resume scheduler config differs from checkpoint")
        saved = checkpoint["config"].copy()
        current = asdict(self.config)
        # Epoch target alone can be extended on resume.
        saved.pop("epochs")
        current.pop("epochs")
        if saved != current:
            raise ValueError("Resume config differs from checkpoint (only epochs may change)")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        self.metadata = checkpoint["metadata"]
        self.generator.set_state(checkpoint["generator"])
        self.loader_generator.set_state(checkpoint["loader_generator"])
        torch.set_rng_state(checkpoint["torch_rng"])
        self.global_step, self.completed_epochs = (
            checkpoint["global_step"],
            checkpoint["completed_epochs"],
        )
        self.history = checkpoint["history"]
        self.dataset_signature = checkpoint["dataset_signature"]


def _semantic_config(config):
    # Diffusers save_pretrained changes tuples to JSON lists. Also canonicalize
    # older checkpoints when comparing so portable reload preserves semantics.
    semantic = {key: value for key, value in dict(config).items() if not key.startswith("_")}
    return json.loads(json.dumps(semantic))
