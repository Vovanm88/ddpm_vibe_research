"""One paired Torch loop for local multi-step MSE, OT/SWD, and tube ablations."""

import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from .calibration import calibrate_interval_features
from .config import config_hash, method_name
from .fm import (
    learning_rate,
    make_manifest,
    restore_training,
    training_state,
    update_ema,
)
from .losses import (
    features,
    pair_loss,
    sinkhorn_divergence,
    sliced_wasserstein_loss,
)
from .runtime import (
    ResourceBudget,
    RNGStreams,
    atomic_json,
    choose_device,
    file_hash,
    model_hash,
    require_rays_gate,
    save_checkpoint,
    stable_seed,
    synchronize,
    tensor_hash,
)
from .schedules import make_schedule, student_endpoint, teacher_target


def experiment_schedule(config):
    d = config["distillation"]
    return make_schedule(d["teacher_steps"], d["student_steps"], d["groups"])


@torch.no_grad()
def make_anchor(teacher, clean, noise, t, interval, schedule, source):
    if source == "interpolant":
        return (1 - t) * clean + t * noise, 0
    state = noise.clone()
    for j in range(interval):
        state = teacher_target(
            teacher, state, schedule.times[j], schedule.widths[j], schedule.groups[j]
        )
    return state.detach(), sum(schedule.groups[:interval])


@torch.no_grad()
def calibrate(config, teacher, samples, *, output_path=None):
    schedule = experiment_schedule(config)
    d = config["distillation"]
    device = next(teacher.parameters()).device
    teacher.eval().requires_grad_(False)
    rng = RNGStreams(stable_seed(config["teacher"]["seed"], "calibration"))
    intervals = []
    for j, (t, width, substeps) in enumerate(
        zip(schedule.times[:-1], schedule.widths, schedule.groups)
    ):
        indices = torch.randint(len(samples), (d["calibration_size"],), generator=rng["data"])
        clean = samples[indices].to(device)
        noise = torch.randn(clean.shape, generator=rng["noise"]).to(device)
        anchor, _ = make_anchor(teacher, clean, noise, float(t), j, schedule, d["anchor_source"])
        endpoint = teacher_target(teacher, anchor, t, width, substeps)
        record = {
            "interval": j,
            "t": float(t),
            "H": float(width),
            "teacher_substeps": substeps,
            "indices_sha256": tensor_hash(indices),
            "noise_sha256": tensor_hash(noise),
            "representations": {},
        }
        for representation in ("endpoint", "displacement", "joint"):
            record["representations"][representation] = calibrate_interval_features(
                features(anchor, endpoint, representation, gamma=d["gamma"]),
                alpha=d["epsilon_alpha"],
            )
        intervals.append(record)
    degenerate = sum(
        x["representations"][rep]["degenerate"]
        for x in intervals
        for rep in ("endpoint", "displacement", "joint")
    )
    if degenerate > len(intervals):
        raise RuntimeError(
            "Widespread calibration degeneracy; inspect teacher before fitting students"
        )
    result = {
        "teacher_sha256": model_hash(teacher),
        "train_data_sha256": tensor_hash(samples),
        "calibration_settings": {
            key: d[key] for key in ("gamma", "epsilon_alpha", "calibration_size")
        },
        "schedule": schedule.to_dict(),
        "anchor_source": d["anchor_source"],
        "tube_rho": 0.0,
        "student_independent": True,
        "intervals": intervals,
        "degenerate_count": degenerate,
    }
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    if output_path is not None:
        atomic_json(output_path, result)
    return result


def distribution_loss(
    loss_name, anchor, predicted, target, epsilon, settings, generator, *, diagnostic=False
):
    if loss_name == "mse":
        return predicted.new_zeros(()), None
    family, representation = loss_name.split("_", 1)
    student_features = features(anchor, predicted, representation, gamma=settings["gamma"])
    teacher_features = features(anchor, target.detach(), representation, gamma=settings["gamma"])
    if family == "swd":
        return sliced_wasserstein_loss(
            student_features,
            teacher_features,
            generator=generator,
            num_directions=settings["swd_train_projections"],
        ), None
    if settings["ot_backend"] == "geomloss_approximate":
        raise ValueError(
            "Unconverged GeomLoss annealing is verification-only, not a registered production backend"
        )
    result = sinkhorn_divergence(
        student_features, teacher_features, epsilon, return_diagnostics=diagnostic
    )
    return result if diagnostic else (result, None)


def gradient_diagnostics(pair, distribution, model, weight):
    parameters = [p for p in model.parameters() if p.requires_grad]

    def vector(loss):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        return torch.cat(
            [
                (g if g is not None else torch.zeros_like(p)).flatten()
                for g, p in zip(gradients, parameters)
            ]
        )

    pair_gradient = vector(pair)
    result = {
        "pair_grad_norm": float(pair_gradient.norm()),
        "dist_grad_norm": None,
        "grad_cosine": None,
        "weighted_dist_pair_ratio": None,
    }
    if distribution.requires_grad:
        dist_gradient = vector(distribution)
        pn, dn = pair_gradient.norm(), dist_gradient.norm()
        result.update(
            dist_grad_norm=float(dn),
            grad_cosine=float((pair_gradient @ dist_gradient) / (pn * dn).clamp_min(1e-30)),
            weighted_dist_pair_ratio=float(weight * dn / pn.clamp_min(1e-30)),
        )
    return result


def train_student(
    config,
    teacher,
    samples,
    calibration,
    *,
    loss="mse",
    rho=0.0,
    seed=101,
    resume=False,
    dry_run=False,
    stop_after=None,
):
    settings, d = config["student"], config["distillation"]
    name = method_name(loss, rho)
    directory = Path(config["output_dir"]) / "students" / f"seed{seed}_{name}"
    if dry_run:
        return {
            "method": name,
            "seed": seed,
            "loss": loss,
            "rho": rho,
            "updates": settings["updates"],
            "micro_batch": settings["batch_size"],
            "optimizer_effective_batch": settings["batch_size"],
            "ot_support_size": settings["batch_size"],
            "directory": str(directory),
        }
    require_rays_gate(config)
    device = choose_device(config["device"])
    torch.set_num_threads(config["threads"])
    if config["data"]["name"] == "mnist" and device.type not in {"cuda", "mps"}:
        raise RuntimeError("Full MNIST distillation requires CUDA or MPS; CPU is for tests/rays")
    budget = ResourceBudget(config, device)
    try:
        teacher.eval().requires_grad_(False)
        parent_digest = model_hash(teacher)
        if calibration["teacher_sha256"] != parent_digest or calibration[
            "train_data_sha256"
        ] != tensor_hash(samples):
            raise ValueError("Calibration teacher or train split differs")
        expected_settings = {key: d[key] for key in ("gamma", "epsilon_alpha", "calibration_size")}
        if calibration.get("calibration_settings") != expected_settings:
            raise ValueError("Calibration gamma/alpha/support changed")
        unsigned = {key: value for key, value in calibration.items() if key != "sha256"}
        if (
            hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
            != calibration["sha256"]
        ):
            raise ValueError("Calibration SHA256 mismatch")
        schedule = experiment_schedule(config)
        if (
            calibration["schedule"] != schedule.to_dict()
            or calibration["anchor_source"] != d["anchor_source"]
        ):
            raise ValueError("Calibration schedule or anchor source differs")
        student = copy.deepcopy(teacher).to(device).train().requires_grad_(True)
        if model_hash(student) != parent_digest:
            raise AssertionError("Student must initialize as an exact teacher copy")
        ema = copy.deepcopy(student).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=settings["lr"],
            betas=(0.9, 0.999),
            weight_decay=settings["weight_decay"],
        )
        streams = RNGStreams(seed)
        contract = {
            "config_sha256": config_hash(config),
            "teacher_sha256": parent_digest,
            "calibration_sha256": calibration["sha256"],
            "loss": loss,
            "rho": rho,
            "seed": seed,
        }
        checkpoint = directory / "training.pt"
        completed, history = 0, []
        if directory.exists() and any(directory.iterdir()) and not resume:
            raise FileExistsError(
                f"Existing student run {directory}; use --resume or another output_dir"
            )
        directory.mkdir(parents=True, exist_ok=True)
        if resume:
            completed, history = restore_training(
                checkpoint, student, ema, optimizer, streams, contract
            )
        manifest = make_manifest(
            config,
            f"seed{seed}_{name}",
            teacher_sha256=parent_digest,
            student_init_sha256=parent_digest,
            seed=seed,
            rng_seeds=streams.seeds,
            loss_family=loss,
            rho=rho,
            lambda_w=d["lambda_w"],
            gamma=d["gamma"],
            schedule=schedule.to_dict(),
            calibration=calibration,
            micro_batch=settings["batch_size"],
            optimizer_effective_batch=settings["batch_size"],
            ot_support_size=settings["batch_size"],
            ot_dtype="float64",
            updates_completed=completed,
        )
        atomic_json(directory / "manifest.json", manifest)
        save_checkpoint(
            checkpoint,
            training_state(student, ema, optimizer, streams, completed, history, contract),
        )
        started = time.perf_counter()
        previous_seconds = history[-1]["training_seconds"] if history else 0.0
        teacher_calls = sum(row["teacher_calls"] for row in history)
    except BaseException:
        try:
            budget.check()
        finally:
            budget.close()
        raise
    try:
        for update in range(completed, min(settings["updates"], stop_after or settings["updates"])):
            interval = int(torch.randint(schedule.student_steps, (), generator=streams["interval"]))
            t, width, microsteps = (
                float(schedule.times[interval]),
                float(schedule.widths[interval]),
                schedule.groups[interval],
            )
            indices = torch.randint(
                len(samples), (settings["batch_size"],), generator=streams["data"]
            )
            clean = samples[indices].to(device)
            noise = torch.randn(clean.shape, generator=streams["noise"]).to(device)
            eta = torch.randn(clean.shape, generator=streams["tube"]).to(device)
            anchor, anchor_calls = make_anchor(
                teacher, clean, noise, t, interval, schedule, d["anchor_source"]
            )
            used_anchor = (anchor + rho * width * eta).detach()
            # Retarget from THIS perturbed anchor; never reuse the clean target.
            target = teacher_target(teacher, used_anchor, t, width, microsteps)
            predicted = student_endpoint(student, used_anchor, t, width)
            pair = pair_loss(predicted, target)
            representation = loss.split("_", 1)[1] if loss != "mse" else "endpoint"
            epsilon = calibration["intervals"][interval]["representations"][representation][
                "epsilon"
            ]
            diagnostic = update == 0 or (update + 1) % d["plan_diagnostics_every"] == 0
            synchronize(device)
            ot_started = time.perf_counter()
            dist, plan = distribution_loss(
                loss if d["lambda_w"] else "mse",
                used_anchor,
                predicted,
                target,
                epsilon,
                d,
                streams["swd"],
                diagnostic=diagnostic,
            )
            synchronize(device)
            ot_seconds = time.perf_counter() - ot_started
            total = pair + d["lambda_w"] * dist
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite distillation loss; series stopped")
            gradients = {}
            if update == 0 or (update + 1) % d["gradient_diagnostics_every"] == 0:
                gradients = gradient_diagnostics(pair, dist, student, d["lambda_w"])
            lr = learning_rate(update, settings["updates"], settings["warmup"], settings["lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            before = [p.detach().clone() for p in student.parameters()]
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), settings["grad_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            update_ema(ema, student, settings["ema_decay"])
            update_norm = (
                torch.stack(
                    [
                        (p.detach() - old).square().sum()
                        for p, old in zip(student.parameters(), before)
                    ]
                )
                .sum()
                .sqrt()
            )
            completed = update + 1
            teacher_calls += microsteps + anchor_calls
            row = {
                "update": completed,
                "loss": float(total.detach()),
                "pair_loss": float(pair.detach()),
                "dist_loss": float(dist.detach()),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "parameter_update_norm": float(update_norm),
                "ot_seconds": ot_seconds,
                "teacher_calls": microsteps + anchor_calls,
                "interval": interval,
                "t": t,
                "H": width,
                "epsilon": epsilon,
                "training_seconds": previous_seconds + time.perf_counter() - started,
                **gradients,
            }
            if diagnostic or completed % 200 == 0:
                row["common_rng_trace"] = {
                    "data": tensor_hash(indices),
                    "noise": tensor_hash(noise),
                    "tube": tensor_hash(eta),
                    "interval": interval,
                }
            if plan is not None:
                row["ot_plan"] = plan
            history.append(row)
            if completed == 1 or completed % 100 == 0:
                print(
                    f"{name} seed{seed} {completed}/{settings['updates']}: pair={float(pair.detach()):.7g} dist={float(dist.detach()):.7g}",
                    flush=True,
                )
            if completed % 100 == 0:
                budget.check()
            if completed % settings["checkpoint_every"] == 0:
                budget.check()
                save_checkpoint(
                    checkpoint,
                    training_state(student, ema, optimizer, streams, completed, history, contract),
                )
        budget.check()
        save_checkpoint(
            checkpoint,
            training_state(student, ema, optimizer, streams, completed, history, contract),
        )
        if model_hash(teacher) != parent_digest or any(
            p.grad is not None for p in teacher.parameters()
        ):
            raise AssertionError("Frozen teacher changed or accumulated gradients")
        ema.save_pretrained(directory / "ema_model", safe_serialization=True)
        student.save_pretrained(directory / "live_model", safe_serialization=True)
        manifest.update(
            status="completed" if completed == settings["updates"] else "interrupted",
            updates_completed=completed,
            student_ema_sha256=model_hash(ema),
            teacher_sha256_after=model_hash(teacher),
            teacher_calls=teacher_calls,
            checkpoint_sha256=file_hash(checkpoint),
            training_seconds=history[-1]["training_seconds"],
            ot_seconds=sum(row["ot_seconds"] for row in history),
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
