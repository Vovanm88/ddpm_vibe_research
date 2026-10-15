"""Gated MNIST preparation, real-only evaluator and immutable quality context.

All dataset access is explicit and follows the rotating-rays gate. Validation
contexts do not open test. A test context may record the classifier's one-time
test accuracy after fixation; later contexts reuse that recorded measurement.
Generator/teacher/student training and evaluation scheduling live in workflow.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .data import load_mnist_split, prepare_mnist
from .evaluate import FeatureReference, extract_mnist_features, mnist_quality
from .mnist import finalize_evaluator_test, train_evaluator
from .models import EvaluatorCNN, state_dict_sha256
from .runtime import ResourceBudget, choose_device, file_hash, require_rays_gate, tensor_hash


def _paths(config):
    if config["data"]["name"] != "mnist":
        raise ValueError("MNIST workflow requires data.name='mnist'")
    root = Path(config["output_dir"])
    return root, root / "data" / "mnist_split_manifest.json", root / "evaluator"


def prepare_stage(config, dry_run=False):
    root, split_manifest, _ = _paths(config)
    plan = {
        "stage": "prepare_mnist",
        "manifest_path": str(split_manifest),
        "indices_path": str(split_manifest.parent / "mnist_split_indices.pt"),
        "data_root": str(config["data"]["root"]),
        "test_accessed": False,
    }
    if dry_run:
        return plan | {"status": "dry_run"}
    require_rays_gate(config)  # MUST precede even constructing a dataset factory.
    data = config["data"]
    manifest = prepare_mnist(
        data["root"],
        split_manifest.parent,
        seed=data["seed"],
        n_validation=data["n_validation"],
        download=True,
        expected_train_size=data["n_train"] + data["n_validation"],
        expected_test_size=data["n_test"],
    )
    if not split_manifest.is_file():
        raise RuntimeError("MNIST preparation did not persist its split manifest")
    return plan | {
        "status": "completed",
        "manifest": manifest,
        "manifest_sha256": file_hash(split_manifest),
        "output_dir": str(root),
    }


def evaluator_stage(config, resume=False, dry_run=False):
    _, split_manifest, evaluator_dir = _paths(config)
    arguments = {
        "output_dir": evaluator_dir,
        "data_root": config["data"]["root"],
        "split_manifest": split_manifest,
        "resume": resume,
    }
    if dry_run:
        return train_evaluator(config, **arguments, dry_run=True) | {
            "manifest_path": str(evaluator_dir / "manifest.json"),
            "split_manifest_path": str(split_manifest),
        }
    require_rays_gate(config)
    device = choose_device(config["device"])
    # There is exactly one budget/accelerator lock, scoped to the full run root.
    budget = ResourceBudget(config, device)
    try:
        resolved = dict(config) | {"device": str(device)}
        result = train_evaluator(resolved, **arguments, progress_callback=budget.check)
        return result | {
            "manifest_path": str(evaluator_dir / "manifest.json"),
            "split_manifest_path": str(split_manifest),
            "resolved_device": str(device),
        }
    finally:
        try:
            budget.check()
        finally:
            budget.close()


def _qualified_manifest(path, config):
    manifest = json.loads(path.read_text())
    threshold = max(0.99, float(config["evaluator"]["accuracy_threshold"]))
    if manifest.get("status") != "completed" or not manifest.get("evaluator_gate_passed"):
        raise RuntimeError("MNIST evaluator is not frozen and validation-qualified")
    accuracy = float(manifest.get("best_validation_accuracy", -1))
    if not torch.isfinite(torch.tensor(accuracy)) or accuracy < threshold:
        raise RuntimeError(f"MNIST evaluator real validation accuracy must be >= {threshold:.3f}")
    return manifest


def build_quality_context(config, split="validation"):
    """Load a frozen evaluator/reference and real features for an explicit split.

    Returned tensors and model are intentionally in-memory research components;
    ``metadata`` contains only JSON-friendly fingerprints for a run manifest.
    All 55k real train features are retained for the nearest-train diagnostic.
    The train-only mean/std/bandwidth are read from their original fixed artifact.
    """
    _, split_manifest, evaluator_dir = _paths(config)
    if split not in ("validation", "test"):
        raise ValueError("quality context split must be validation or explicit final test")
    require_rays_gate(config)
    evaluator_manifest_path = evaluator_dir / "manifest.json"
    manifest = _qualified_manifest(evaluator_manifest_path, config)
    if file_hash(split_manifest) != manifest["split_manifest_sha256"]:
        raise ValueError("MNIST split manifest differs from the evaluator's registered data")
    reference_path = evaluator_dir / "feature_reference.pt"
    if file_hash(reference_path) != manifest.get("feature_reference_sha256"):
        raise ValueError("train-only feature reference failed its registered hash check")
    saved_reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    mean = torch.as_tensor(saved_reference["mean"], dtype=torch.float64)
    std = torch.as_tensor(saved_reference["std"], dtype=torch.float64)
    ell2 = float(saved_reference["bandwidth_squared"])
    indices = torch.as_tensor(saved_reference["bandwidth_indices"], dtype=torch.long)
    if (
        mean.shape != (128,)
        or std.shape != (128,)
        or not torch.isfinite(mean).all()
        or (not torch.isfinite(std).all() or (std < 1e-6).any())
    ):
        raise ValueError("invalid fixed FC128 mean/std reference")
    if (
        not torch.isfinite(torch.tensor(ell2))
        or ell2 <= 0
        or saved_reference.get("bandwidth_degenerate")
    ):
        raise ValueError("degenerate/invalid fixed train-only kernel bandwidth")
    reference = FeatureReference(
        mean=mean,
        std=std,
        ell2=ell2,
        indices=indices,
        seed=int(saved_reference.get("seed", config["evaluator"]["seed"] + 2)),
        bandwidth_degenerate=False,
    )
    split_record = json.loads(split_manifest.read_text())
    split_indices = torch.load(
        split_manifest.parent / split_record["indices_file"], map_location="cpu", weights_only=True
    )
    if (
        indices.ndim != 1
        or len(indices) < 2
        or len(torch.unique(indices)) != len(indices)
        or (not torch.isin(indices, split_indices["train"]).all())
    ):
        raise ValueError("kernel bandwidth indices must be unique registered train indices")
    device = choose_device(config["device"])
    evaluator = (
        EvaluatorCNN.from_pretrained(
            evaluator_dir / "evaluator", local_files_only=True, low_cpu_mem_usage=False
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    if state_dict_sha256(evaluator) != manifest["evaluator_sha256"]:
        raise ValueError("frozen evaluator weights failed their registered hash check")

    test_accuracy = manifest.get("test_accuracy")
    measured_test_here = False
    if split == "test":
        previously_accessed = bool(manifest.get("test_accessed"))
        if previously_accessed != (test_accuracy is not None):
            raise RuntimeError("inconsistent evaluator one-time test record; do not reopen test")
        if not previously_accessed:
            finalize_evaluator_test(
                evaluator_dir,
                data_root=config["data"]["root"],
                split_manifest=split_manifest,
                batch_size=config["evaluation"]["batch_size"],
            )
            manifest = _qualified_manifest(evaluator_manifest_path, config)
            test_accuracy = manifest.get("test_accuracy")
            measured_test_here = True
            if not manifest.get("test_accessed") or test_accuracy is None:
                raise RuntimeError("classifier test measurement was not safely recorded")
        if (
            not torch.isfinite(torch.tensor(float(test_accuracy)))
            or not 0 <= float(test_accuracy) <= 1
        ):
            raise RuntimeError("invalid cached classifier test accuracy")
    batch_size = config["evaluation"]["batch_size"]
    train_samples, _ = load_mnist_split(config["data"]["root"], "train", split_manifest)
    real_samples, real_labels = load_mnist_split(
        config["data"]["root"], split, split_manifest, allow_test=split == "test"
    )
    train_features, _, _ = extract_mnist_features(evaluator, train_samples, batch_size=batch_size)
    real_features, real_logits, _ = extract_mnist_features(
        evaluator, real_samples, batch_size=batch_size
    )
    metadata = {
        "reference_split": split,
        "evaluator_sha256": manifest["evaluator_sha256"],
        "evaluator_manifest_sha256": file_hash(evaluator_manifest_path),
        "feature_reference_sha256": file_hash(reference_path),
        "split_manifest_sha256": file_hash(split_manifest),
        "train_samples_sha256": tensor_hash(train_samples),
        "real_samples_sha256": tensor_hash(real_samples),
        "train_feature_count": len(train_features),
        "real_feature_count": len(real_features),
        "feature_layer": "real-only evaluator FC128 after ReLU",
        "preprocessing": "(raw_fm+1)/2 then clamp(0,1); raw diagnostics before clamp",
        "normalization_fit_split": "train_only",
        "reference_seed": reference.seed,
        "classifier_test_accuracy": test_accuracy,
        "classifier_test_accuracy_recomputed": False,
        "classifier_test_measurement_created_here": measured_test_here,
        "resolved_device": str(device),
    }
    return {
        "evaluator": evaluator,
        "reference": reference,
        "real_features": real_features,
        "real_logits": real_logits,
        "real_samples": real_samples,
        "real_labels": real_labels,
        "train_features": train_features,
        "evaluator_manifest": manifest,
        "test_accuracy": test_accuracy,
        "split": split,
        "metadata": metadata,
    }


def quality(config, samples, context):
    """Independent generated-to-real quality with the frozen, shared preprocessing."""
    return mnist_quality(
        samples,
        context["real_features"],
        context["real_logits"],
        context["train_features"],
        context["evaluator"],
        context["reference"],
        batch_size=config["evaluation"]["batch_size"],
        block_size=config["evaluation"]["mmd_block_size"],
    )
