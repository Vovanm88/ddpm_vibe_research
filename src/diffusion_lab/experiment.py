"""Config-driven construction and reproducible run artifacts."""

import copy
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from diffusers import DDIMScheduler, DDPMScheduler, FlowMatchEulerDiscreteScheduler

from .environments import make_environment
from .models import PointDenoiser
from .pipeline import ResearchDiffusionPipeline
from .schedulers import make_scheduler
from .training import DiffusionTrainer, TrainerConfig


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a mapping")
    unknown = set(config) - {
        "seed",
        "device",
        "num_threads",
        "environment",
        "model",
        "scheduler",
        "training",
        "sampling",
    }
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    return config


def build_experiment(config):
    config = copy.deepcopy(config)
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("num_threads", 4))
    environment_config = config.get("environment", {"name": "rotating_rays"}).copy()
    environment_config.setdefault("seed", seed)
    dataset = make_environment(**environment_config)
    model_config = config.get("model", {}).copy()
    model_config.setdefault("sample_shape", dataset.sample_shape)
    model_config.setdefault(
        "conditioner",
        {"categorical": "embedding", "continuous": "mlp", "none": "none"}[dataset.condition_kind],
    )
    model_config.setdefault("num_classes", dataset.num_classes)
    model_config.setdefault("condition_dim", dataset.condition_dim)
    model = PointDenoiser(**model_config)
    scheduler = make_scheduler(**config.get("scheduler", {}))
    pipeline = ResearchDiffusionPipeline(model, scheduler)
    training_config = config.get("training", {}).copy()
    training_config.setdefault("seed", seed)
    trainer = DiffusionTrainer(
        pipeline,
        TrainerConfig(**training_config),
        device=resolve_device(config.get("device", "cpu")),
        metadata=dataset.metadata,
    )
    return dataset, pipeline, trainer


def resolve_device(device):
    if device == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return device


def switch_scheduler(pipeline, kind):
    if isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler):
        if kind != "flow":
            raise ValueError("A flow model requires a flow scheduler")
        return
    classes = {"ddpm": DDPMScheduler, "ddim": DDIMScheduler}
    if kind not in classes:
        raise ValueError("A diffusion model requires DDPM or DDIM")
    pipeline.scheduler = classes[kind].from_config(pipeline.scheduler.config)


def endpoint_constraints(model, conditions):
    if len(model.sample_shape) != 2 or conditions is None or conditions.ndim != 2:
        return {}
    batch, (length, dim) = len(conditions), model.sample_shape
    if conditions.shape[1] < 2 * dim:
        return {}
    values = torch.zeros(batch, length, dim, device=model.device, dtype=model.dtype)
    mask = torch.zeros_like(values, dtype=torch.bool)
    values[:, 0], values[:, -1] = conditions[:, :dim], conditions[:, dim : 2 * dim]
    mask[:, 0] = mask[:, -1] = True
    return {"known_values": values, "known_mask": mask}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_run(trainer, dataset, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    write_json(output / "history.json", trainer.history)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffusers", "numpy", "scipy", "matplotlib")
        },
        "model": dict(trainer.model.config),
        "training": asdict(trainer.config),
        "environment": dataset.metadata,
        "dataset_signature": trainer.dataset_signature,
        "exported_weights": "ema" if trainer.ema is not None else "live",
    }
    write_json(output / "manifest.json", manifest)
    with trainer.ema_weights():
        trainer.pipeline.save_pretrained(output / "pipeline", safe_serialization=True)
    write_json(
        output / "pipeline" / "research_metadata.json",
        {
            "environment_config": config.get("environment", {"name": "rotating_rays"}),
            "seed": config.get("seed", 42),
            "environment": dataset.metadata,
        },
    )
