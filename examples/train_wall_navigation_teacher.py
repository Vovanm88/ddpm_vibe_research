"""Plain Torch conditional FM teacher for the multimodal A→B wall pilot.

The model predicts only the 30 interior points. Start/goal are reconstructed
from the eight continuous conditions, so exact endpoints are a construction
contract, never evidence of a learned ability. No route ID, obstacle loss,
projection, clipping, or classifier-free dropout is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from diffusion_lab import PointDenoiser
from diffusion_lab.environments.multimodal_navigation import make_multimodal_navigation
from recheck.fm import learning_rate
from recheck.runtime import (
    RNGStreams,
    atomic_json,
    choose_device,
    model_hash,
    save_checkpoint,
    tensor_hash,
)

PROTOCOL = "wall_navigation_conditional_v1"
ROOT = Path(__file__).resolve().parents[1]
PROFILE = {
    "protocol_id": PROTOCOL,
    "scope": "one_seed_exploratory_multimodal_navigation_not_recheck_replication",
    "data": {"n_scenes": 128, "samples_per_scene": 128, "path_length": 32,
             "seed": 61, "split": "train", "jitter": .025, "clearance": .06, "margin": .035},
    "validation": {"n_scenes": 32, "samples_per_scene": 128, "path_length": 32,
                   "seed": 10061, "split": "validation", "jitter": .025, "clearance": .06, "margin": .035},
    "model": {"sample_shape": [30, 2], "hidden_dim": 128, "num_layers": 3,
              "backbone": "mlp", "conditioner": "mlp", "condition_dim": 8, "num_classes": 0},
    "teacher": {"seed": 61, "updates": 20000, "batch_size": 256, "lr": .001,
                "weight_decay": .0001, "warmup": 1000, "ema_decay": .999,
                "grad_clip": 1., "condition_dropout": 0.},
    "time_scale": 1000,
    "threads": 4,
    "endpoint_clamped_contract": True,
    "endpoint_reconstruction": "prepend condition[:2], append condition[2:4]; model predicts interior only",
    "condition_fields": ["start_x", "start_y", "goal_x", "goal_y", "wall_low_x", "wall_low_y", "wall_high_x", "wall_high_y"],
    "route_ids_used_by_model": False,
    "obstacle_penalty": False,
    "training_objective": "pure_conditional_FM: ((v(x_t,t,condition)-(noise-clean_interior))**2).mean()",
}


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def make_training_components(device, dataset):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(PROFILE["teacher"]["seed"])
        model = PointDenoiser(**PROFILE["model"]).to(device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROFILE["teacher"]["lr"],
                                  betas=(.9, .999), weight_decay=PROFILE["teacher"]["weight_decay"])
    return model, ema, optimizer, RNGStreams(PROFILE["teacher"]["seed"])


def update_teacher(model, ema, optimizer, streams, dataset, update):
    settings = PROFILE["teacher"]
    device = model.device
    indices = torch.randint(len(dataset), (settings["batch_size"],), generator=streams["data"])
    clean = dataset.samples[indices, 1:-1].to(device)
    conditions = dataset.conditions[indices].to(device)
    noise = torch.randn(clean.shape, generator=streams["noise"]).to(device)
    t = torch.rand(len(clean), generator=streams["interval"]).to(device)
    anchor = (1 - t[:, None, None]) * clean + t[:, None, None] * noise
    target = noise - clean
    prediction = model(anchor, t * PROFILE["time_scale"], conditions=conditions).sample
    loss = (prediction - target).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Navigation teacher loss is nonfinite")
    lr = learning_rate(update, settings["updates"], settings["warmup"], settings["lr"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        for average, parameter in zip(ema.parameters(), model.parameters()):
            average.lerp_(parameter.detach(), 1 - settings["ema_decay"])
        for average, buffer in zip(ema.buffers(), model.buffers()):
            average.copy_(buffer)
    return {"update": update + 1, "loss": float(loss.detach()), "lr": lr, "grad_norm": float(norm)}


def benchmark(dataset, device, output):
    model, ema, optimizer, streams = make_training_components(device, dataset)
    for index in range(100):
        update_teacher(model, ema, optimizer, streams, dataset, index)
    synchronize(device)
    started = time.perf_counter()
    for index in range(100, 200):
        update_teacher(model, ema, optimizer, streams, dataset, index)
    synchronize(device)
    elapsed = time.perf_counter() - started
    result = {"device": str(device), "torch_threads": torch.get_num_threads(),
              "warmup_updates": 100, "measured_updates": 100,
              "seconds": elapsed, "seconds_per_update": elapsed / 100,
              "forecast_20000_updates_seconds": elapsed * 200,
              "profile": PROFILE, "train_sha256": tensor_hash(dataset.samples),
              "conditions_sha256": tensor_hash(dataset.conditions), "source_sha256": digest_file(__file__),
              "includes": ["CPU train gather and transfer", "conditional FM forward/backward", "AdamW", "clipping", "EMA"],
              "timing_context": "two independent rays training cohorts may run concurrently; operational device choice, not latency publication"}
    atomic_json(output / f"benchmark_{device.type}.json", result)
    return result


def train(dataset, device, output, resume=False):
    directory = output / "teacher"
    checkpoint = directory / "training.pt"
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise FileExistsError("Existing wall teacher directory; use explicit --resume")
    directory.mkdir(parents=True, exist_ok=True)
    model, ema, optimizer, streams = make_training_components(device, dataset)
    source_sha = digest_file(__file__)
    contract = {"profile": PROFILE, "device": str(device), "source_sha256": source_sha,
                "train_sha256": tensor_hash(dataset.samples),
                "conditions_sha256": tensor_hash(dataset.conditions)}
    completed, history = 0, []
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["contract"] != contract:
            raise ValueError("Navigation teacher resume contract differs")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(device)
        streams.load_state_dict(state["streams"])
        completed, history = state["updates_completed"], state["history"]
    manifest = {"status": "running", "config": PROFILE, "device": str(device),
                "data_sha256": contract["train_sha256"],
                "conditions_sha256": contract["conditions_sha256"], "source_sha256": source_sha,
                "updates_completed": completed, "endpoint_clamped_contract": True,
                "interior_shape": [30, 2], "full_path_shape": [32, 2],
                "route_ids_used_by_model": False, "condition_dropout": 0.,
                "parameter_count": sum(p.numel() for p in model.parameters())}

    def commit(status):
        save_checkpoint(checkpoint, {"contract": contract, "model": model.state_dict(),
            "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "streams": streams.state_dict(),
            "updates_completed": completed, "history": history, "status": status})
        manifest.update(status=status, updates_completed=completed)
        atomic_json(directory / "manifest.json", manifest)

    commit("running")
    start_time = time.perf_counter()
    previous_seconds = history[-1].get("training_seconds", 0.) if history else 0.
    try:
        for update in range(completed, PROFILE["teacher"]["updates"]):
            row = update_teacher(model, ema, optimizer, streams, dataset, update)
            completed = update + 1
            row["training_seconds"] = previous_seconds + time.perf_counter() - start_time
            history.append(row)
            if completed == 1 or completed % 500 == 0:
                synchronize(device)
                print(f"wall teacher {completed}/{PROFILE['teacher']['updates']} loss={row['loss']:.7f} "
                      f"elapsed={row['training_seconds']:.1f}s device={device}", flush=True)
            if completed % 1000 == 0:
                commit("running")
        synchronize(device)
        ema.save_pretrained(directory / "ema_model", safe_serialization=True)
        manifest.update(teacher_sha256=model_hash(ema), training_seconds=history[-1]["training_seconds"])
        commit("completed")
        atomic_json(directory / "history.json", history)
        return {"status": "completed", "directory": str(directory),
                "teacher_sha256": manifest["teacher_sha256"], "updates_completed": completed,
                "training_seconds": history[-1]["training_seconds"]}
    except BaseException as error:
        # Last fully saved checkpoint remains the exact resume point.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        updates_completed=saved["updates_completed"], error=str(error))
        atomic_json(directory / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "train"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "replication/artifacts" / PROTOCOL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(PROFILE["threads"])
    device = choose_device(args.device)
    dataset = make_multimodal_navigation(**PROFILE["data"])
    if dataset.samples.shape != (16384, 32, 2) or dataset.conditions.shape != (16384, 8):
        raise ValueError("Dataset changed the registered navigation tensor contract")
    if not torch.isfinite(dataset.samples).all() or not torch.isfinite(dataset.conditions).all():
        raise ValueError("Nonfinite navigation training data")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "benchmark":
        result = benchmark(dataset, device, args.output)
    else:
        registration = args.output / "registration.json"
        if not registration.exists():
            atomic_json(registration, {"profile": PROFILE, "device": str(device),
                        "source_sha256": digest_file(__file__),
                        "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "choice_criterion": "warmed throughput only, before training and any quality evaluation"})
        result = train(dataset, device, args.output, resume=args.resume)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
