"""Strict, resolved experiment settings. Final evaluation noise is preregistered."""

import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

FINAL_SAMPLE_SEED = 730001
DEFAULTS = {
    "schema_version": 1,
    "protocol_id": "rays_cpu_pilot_v1",
    "scope": "cpu_pilot",
    "output_dir": "replication/artifacts/rays_cpu_pilot_v1",
    "device": "cpu",
    "threads": 4,
    "data": {
        "name": "rotating_rays",
        "n_train": 8192,
        "n_validation": 4096,
        "n_test": 4096,
        "seed": 41,
        "root": "replication/data",
        "n_classes": 4,
    },
    "model": {
        "kind": "mlp",
        "hidden_dim": 128,
        "num_layers": 3,
        "time_embedding_dim": 32,
        "base_channels": 32,
    },
    "teacher": {
        "seed": 41,
        "updates": 6000,
        "batch_size": 512,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "warmup": 500,
        "ema_decay": 0.999,
        "grad_clip": 1.0,
        "checkpoint_every": 1000,
    },
    "student": {
        "updates": 500,
        "batch_size": 128,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "warmup": 50,
        "ema_decay": 0.99,
        "grad_clip": 1.0,
        "checkpoint_every": 250,
    },
    "distillation": {
        "teacher_steps": 32,
        "student_steps": 16,
        "groups": None,
        "anchor_source": "interpolant",
        "seeds": [101, 202, 303],
        "losses": ["mse", "sinkhorn_endpoint"],
        "tube_rhos": [0.0, 0.1],
        "lambda_w": 1.0,
        "gamma": 1.0,
        "epsilon_alpha": 0.05,
        "calibration_size": 256,
        "swd_train_projections": 128,
        "ot_backend": "reference",
        "sinkhorn_scaling": 0.95,
        "gradient_diagnostics_every": 200,
        "plan_diagnostics_every": 1000,
    },
    "evaluation": {
        "final_seed": FINAL_SAMPLE_SEED,
        "validation_seed": 730002,
        "reference_seed": 730003,
        "samples": 1024,
        "validation_samples": 512,
        "teacher_nfe": [8, 16, 24, 32, 64, 128],
        "batch_size": 128,
        "swd_projections": 1024,
        "mmd_block_size": 256,
        "diagnostic_samples": 128,
        "gif_particles": 16,
        "gif_fps": 10,
        "timing_repeats": 3,
        "sensitivity_swd_margin": 0.005,
        "rho_eval": [0.0, 0.05, 0.1, 0.2],
    },
    "evaluator": {
        "seed": 41,
        "epochs": 5,
        "batch_size": 128,
        "lr": 0.001,
        "weight_decay": 0.0,
        "accuracy_threshold": 0.99,
        "feature_batch_size": 512,
        "bandwidth_samples": 2000,
    },
    "resources": {
        "max_gpu_hours": 6.0,
        "max_artifact_gib": 8.0,
        "rays_gate": "replication/reports/rays_gate.json",
    },
}
LOSSES = {
    "mse",
    "sinkhorn_endpoint",
    "sinkhorn_displacement",
    "sinkhorn_joint",
    "swd_endpoint",
    "swd_displacement",
    "swd_joint",
}


def _merge(base, supplied, prefix=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    unknown = set(supplied) - set(base)
    if unknown:
        raise ValueError(f"Unknown config keys at {prefix or 'root'}: {sorted(unknown)}")
    for key, value in supplied.items():
        if isinstance(base[key], dict):
            _merge(base[key], value, prefix + key + ".")
        else:
            base[key] = value


def resolve_config(supplied=None):
    config = copy.deepcopy(DEFAULTS)
    _merge(config, supplied or {})

    def finite_tree(value):
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, list):
            return all(finite_tree(item) for item in value)
        return not isinstance(value, float) or math.isfinite(value)

    if not finite_tree(config):
        raise ValueError("Nonfinite numeric configuration value")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    if config["data"]["name"] not in {"rotating_rays", "gmm", "mnist"}:
        raise ValueError("Unknown benchmark")
    if config["evaluation"]["final_seed"] != FINAL_SAMPLE_SEED:
        raise ValueError(
            "Final noise seed is preregistered; a change requires a new protocol version"
        )
    d = config["distillation"]
    if set(d["losses"]) - LOSSES or not d["losses"]:
        raise ValueError("Unknown or empty distillation loss family")
    if d["anchor_source"] not in {"interpolant", "teacher_rollout"}:
        raise ValueError("anchor_source must be interpolant or teacher_rollout")
    if d["ot_backend"] != "reference":
        raise ValueError("The registered production OT backend is the converged reference solver")
    if not d["seeds"] or len(set(d["seeds"])) != len(d["seeds"]):
        raise ValueError("Student seeds must be nonempty and unique")
    if any(r < 0 for r in d["tube_rhos"]) or not d["tube_rhos"]:
        raise ValueError("tube_rhos must be nonnegative")
    if d["lambda_w"] < 0 or d["gamma"] < 0 or d["epsilon_alpha"] <= 0:
        raise ValueError("Invalid distribution loss scaling")
    for section in ("teacher", "student"):
        for key in ("updates", "batch_size", "checkpoint_every"):
            if not isinstance(config[section][key], int) or config[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
        if not 0 <= config[section]["ema_decay"] < 1:
            raise ValueError("EMA decay must lie in [0,1)")
    if config["threads"] < 1 or d["calibration_size"] < 2:
        raise ValueError("Invalid thread/calibration count")
    if config["data"]["name"] == "mnist" and config["model"]["kind"] != "unet":
        raise ValueError("MNIST requires the registered unconditional UNet")
    return config


def load_config(path):
    return resolve_config(yaml.safe_load(Path(path).read_text()))


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def method_name(loss, rho):
    if loss == "mse" and rho in (0, 0.1):
        return "M0" if rho == 0 else "M1"
    if loss == "sinkhorn_endpoint" and rho in (0, 0.1):
        return "E0" if rho == 0 else "E1"
    return f"{loss}_rho{rho:g}"
