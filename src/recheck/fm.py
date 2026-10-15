"""Plain Torch FM training, deterministic Euler sampling, and trusted restore."""

import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import config_hash
from .data import load_mnist_split, make_toy_dataset
from .models import FMMLP, SmallUNetModel, architecture_manifest
from .runtime import (
    ResourceBudget,
    RNGStreams,
    algorithm_hash,
    atomic_json,
    choose_device,
    environment,
    file_hash,
    model_hash,
    require_rays_gate,
    save_checkpoint,
    tensor_hash,
)
from .schedules import call_velocity


def benchmark_data(config, split="train"):
    data = config["data"]
    if data["name"] == "mnist":
        require_rays_gate(config)
        manifest = Path(config["output_dir"]) / "data" / "mnist_split_manifest.json"
        return load_mnist_split(data["root"], split, manifest, allow_test=split == "test")[0]
    count = data[{"train": "n_train", "validation": "n_validation", "test": "n_test"}[split]]
    kwargs = {"n_classes": data["n_classes"]} if data["name"] == "rotating_rays" else {}
    return make_toy_dataset(data["name"], count, data["seed"], split=split, **kwargs)


def build_model(config):
    m = config["model"]
    if m["kind"] == "mlp":
        return FMMLP(
            input_dim=2,
            hidden_dim=m["hidden_dim"],
            num_layers=m["num_layers"],
            time_embedding_dim=m["time_embedding_dim"],
        )
    if m["kind"] == "unet":
        return SmallUNetModel(
            base_channels=m["base_channels"], time_embedding_dim=m["time_embedding_dim"]
        )
    raise ValueError("Unknown model kind")


def load_model(path, device="cpu"):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    cls = {"FMMLP": FMMLP, "SmallUNetModel": SmallUNetModel}[config["_class_name"]]
    return cls.from_pretrained(path, local_files_only=True, low_cpu_mem_usage=False).to(device)


def learning_rate(update, total, warmup, peak):
    if warmup and update < warmup:
        return peak * (update + 1) / warmup
    fraction = (update - warmup) / max(1, total - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, fraction)))))


@torch.no_grad()
def update_ema(ema, model, decay):
    for average, parameter in zip(ema.parameters(), model.parameters()):
        average.lerp_(parameter.detach(), 1 - decay)
    for average, buffer in zip(ema.buffers(), model.buffers()):
        average.copy_(buffer)


@torch.no_grad()
def sample_euler(
    model, initial_noise, times=None, *, steps=32, record_history=True, batch_size=None
):
    """Input noise is supplied explicitly; changing NFE never changes its seed."""
    if times is None:
        times = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    times = torch.as_tensor(times, dtype=torch.float64).cpu()
    if len(times) < 2 or times[0] != 1 or times[-1] != 0 or not (times[:-1] > times[1:]).all():
        raise ValueError("Euler grid must strictly decrease from 1 to 0")
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else initial_noise.device
    dtype = parameter.dtype if parameter is not None else initial_noise.dtype
    was_training = model.training
    model.eval()
    chunks, histories, calls = [], [], 0
    try:
        for noise in initial_noise.split(batch_size or len(initial_noise)):
            state = noise.to(device=device, dtype=dtype).clone()
            history = [state.cpu().clone()] if record_history else []
            for t, width in zip(times[:-1], times[:-1] - times[1:]):
                state = state - width.to(device=device, dtype=dtype) * call_velocity(
                    model, state, float(t)
                )
                calls += 1
                if not torch.isfinite(state).all():
                    raise FloatingPointError(
                        "Nonfinite Euler state; corresponding series must stop"
                    )
                if record_history:
                    history.append(state.cpu().clone())
            chunks.append(state.cpu())
            if record_history:
                histories.append(torch.stack(history))
    finally:
        model.train(was_training)
    return {
        "samples": torch.cat(chunks),
        "history": torch.cat(histories, dim=1) if histories else None,
        "times": times,
        "nfe": len(times) - 1,
        "total_model_calls": calls,
        "solver_steps": len(times) - 1,
        "initial_noise_sha256": tensor_hash(initial_noise),
    }


def make_manifest(config, role, **extra):
    fingerprints = Path("replication/source/fingerprints.json")
    return {
        "run_id": f"{config['protocol_id']}/{role}",
        "role": role,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "config": config,
        "config_sha256": config_hash(config),
        "algorithm_sha256": algorithm_hash(),
        "environment": environment(),
        "sources": json.loads(fingerprints.read_text()) if fingerprints.exists() else {},
        "precision": "float32_network",
        **extra,
    }


def training_state(model, ema, optimizer, streams, completed, history, contract):
    return {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "streams": streams.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "updates_completed": completed,
        "history": history,
        "contract": contract,
    }


def restore_training(path, model, ema, optimizer, streams, contract):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["contract"] != contract:
        raise ValueError("Resume contract changed: use a new run directory for another experiment")
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    streams.load_state_dict(state["streams"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if torch.backends.mps.is_available() and state.get("mps_rng") is not None:
        torch.mps.set_rng_state(state["mps_rng"])
    return state["updates_completed"], state["history"]


def train_teacher(config, *, samples=None, resume=False, dry_run=False, stop_after=None):
    settings = config["teacher"]
    directory = Path(config["output_dir"]) / "teacher"
    if dry_run:
        return {
            "stage": "teacher",
            "updates": settings["updates"],
            "batch_size": settings["batch_size"],
            "output": str(directory),
            "device": config["device"],
        }
    require_rays_gate(config)
    device = choose_device(config["device"])
    if config["data"]["name"] == "mnist" and device.type not in {"cuda", "mps"}:
        raise RuntimeError(
            "Full MNIST training requires CUDA or MPS; CPU is reserved for unit tests/rays pilot"
        )
    budget = ResourceBudget(config, device)
    try:
        torch.set_num_threads(config["threads"])
        samples = benchmark_data(config) if samples is None else samples.cpu()
        streams = RNGStreams(settings["seed"])
        with torch.random.fork_rng():
            torch.manual_seed(settings["seed"])
            model = build_model(config).to(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings["lr"],
            betas=(0.9, 0.999),
            weight_decay=settings["weight_decay"],
        )
        contract = {
            "config_sha256": config_hash(config),
            "data_sha256": tensor_hash(samples),
            "stage": "teacher",
        }
        checkpoint = directory / "training.pt"
        completed, history = 0, []
        if directory.exists() and any(directory.iterdir()) and not resume:
            raise FileExistsError(
                f"Existing run {directory}; use explicit --resume or another output_dir"
            )
        directory.mkdir(parents=True, exist_ok=True)
        if resume:
            completed, history = restore_training(
                checkpoint, model, ema, optimizer, streams, contract
            )
        manifest = make_manifest(
            config,
            "teacher",
            model=architecture_manifest(model),
            data_sha256=tensor_hash(samples),
            rng_seeds=streams.seeds,
            updates_completed=completed,
        )
        atomic_json(directory / "manifest.json", manifest)
        save_checkpoint(
            checkpoint, training_state(model, ema, optimizer, streams, completed, history, contract)
        )
        started = time.perf_counter()
        previous_seconds = history[-1]["training_seconds"] if history else 0.0
    except BaseException:
        try:
            budget.check()
        finally:
            budget.close()
        raise
    try:
        for update in range(completed, min(settings["updates"], stop_after or settings["updates"])):
            indices = torch.randint(
                len(samples), (settings["batch_size"],), generator=streams["data"]
            )
            clean = samples[indices].to(device)
            noise = torch.randn(clean.shape, generator=streams["noise"]).to(device)
            t = torch.rand(len(clean), generator=streams["interval"]).to(device)
            time_shape = (len(clean),) + (1,) * (clean.ndim - 1)
            noisy = (1 - t.reshape(time_shape)) * clean + t.reshape(time_shape) * noise
            lr = learning_rate(update, settings["updates"], settings["warmup"], settings["lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(call_velocity(model, noisy, t), noise - clean)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite FM loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["grad_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            update_ema(ema, model, settings["ema_decay"])
            completed = update + 1
            history.append(
                {
                    "update": completed,
                    "fm_loss": float(loss.detach()),
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "training_seconds": previous_seconds + time.perf_counter() - started,
                }
            )
            if completed == 1 or completed % 100 == 0:
                print(
                    f"teacher {completed}/{settings['updates']}: FM={float(loss.detach()):.6f}",
                    flush=True,
                )
            if completed % 100 == 0:
                budget.check()
            if completed % settings["checkpoint_every"] == 0:
                budget.check()
                save_checkpoint(
                    checkpoint,
                    training_state(model, ema, optimizer, streams, completed, history, contract),
                )
        budget.check()
        save_checkpoint(
            checkpoint, training_state(model, ema, optimizer, streams, completed, history, contract)
        )
        ema.save_pretrained(directory / "ema_model", safe_serialization=True)
        model.save_pretrained(directory / "live_model", safe_serialization=True)
        manifest.update(
            status="completed" if completed == settings["updates"] else "interrupted",
            updates_completed=completed,
            teacher_sha256=model_hash(ema),
            checkpoint_sha256=file_hash(checkpoint),
            training_seconds=history[-1]["training_seconds"],
        )
        atomic_json(directory / "history.json", history)
        atomic_json(directory / "manifest.json", manifest)
    except BaseException as error:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            updates_completed=saved["updates_completed"],
            error=str(error),
            recovery="Resume last fully committed checkpoint; incomplete update is discarded",
        )
        atomic_json(directory / "manifest.json", manifest)
        raise
    finally:
        try:
            budget.check()
        finally:
            budget.close()
    return {"directory": str(directory), "model": ema, "manifest": manifest}
