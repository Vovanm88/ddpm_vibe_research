"""Explicit real-only evaluator training and frozen train feature calibration.

This module is inert on import. It neither downloads MNIST nor opens the test
split. The CLI must enforce the rotating-rays gate before calling preparation
or this trainer. Generator training lives in fm.py/distill.py, independently of
the evaluator's classification objective.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import evaluator_preprocess, load_mnist_split
from .models import EvaluatorCNN, architecture_manifest, state_dict_sha256
from .runtime import choose_device


def _atomic_checkpoint(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def extract_features(
    evaluator, raw_images, batch_size=512, mean=None, std=None, progress_callback=None
):
    """Return fixed FC128 features, logits and identical pre-clamp diagnostics."""
    if batch_size < 1 or len(raw_images) < 1:
        raise ValueError("nonempty images and positive batch_size required")
    if (mean is None) != (std is None):
        raise ValueError("supply both mean/std or neither")
    pixels, diagnostics = evaluator_preprocess(raw_images)
    device = next(evaluator.parameters()).device
    previous_mode = evaluator.training
    evaluator.eval()
    feature_batches, logit_batches = [], []
    try:
        for batch in pixels.split(batch_size):
            logits, features = evaluator(batch.to(device), return_features=True)
            feature_batches.append(features.cpu())
            logit_batches.append(logits.cpu())
            if progress_callback is not None:
                progress_callback()
    finally:
        evaluator.train(previous_mode)
    features = torch.cat(feature_batches)
    if mean is not None:
        features = (features - torch.as_tensor(mean)) / torch.as_tensor(std)
    diagnostics["features_standardized"] = mean is not None
    return features, torch.cat(logit_batches), diagnostics


def fit_feature_statistics(
    evaluator,
    train_images,
    train_indices,
    *,
    seed=41,
    bandwidth_samples=2000,
    batch_size=512,
    progress_callback=None,
):
    """Train-only mean/std and median bandwidth; never recalibrate on generations."""
    if len(train_indices) != len(train_images):
        raise ValueError("train indices must identify exactly the provided train images")
    if not 2 <= bandwidth_samples <= len(train_images):
        raise ValueError("bandwidth sample size must lie between 2 and the train size")
    features, _, _ = extract_features(
        evaluator, train_images, batch_size=batch_size, progress_callback=progress_callback
    )
    # The evaluation module owns the canonical kernel/reference definition.
    from .evaluate import fit_feature_reference

    if progress_callback is not None:
        progress_callback()
    reference = fit_feature_reference(
        features,
        seed=seed,
        bandwidth_samples=bandwidth_samples,
        indices=torch.as_tensor(train_indices, dtype=torch.long),
    )
    if progress_callback is not None:
        progress_callback()
    return {
        "mean": reference.mean,
        "std": reference.std,
        "bandwidth_squared": reference.ell2,
        "bandwidth_indices": reference.indices,
        "bandwidth_degenerate": reference.bandwidth_degenerate,
        "std_floor": 1e-6,
        "source": "real train only; post-activation FC128",
    }


@torch.no_grad()
def classifier_accuracy(evaluator, raw_images, labels, batch_size=512):
    pixels, _ = evaluator_preprocess(raw_images)
    device = next(evaluator.parameters()).device
    previous_mode = evaluator.training
    evaluator.eval()
    correct = 0
    loss_sum = 0.0
    try:
        for start in range(0, len(pixels), batch_size):
            target = labels[start : start + batch_size].to(device)
            logits = evaluator(pixels[start : start + batch_size].to(device))
            correct += int((logits.argmax(1) == target).sum())
            loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
    finally:
        evaluator.train(previous_mode)
    return {
        "accuracy": correct / len(labels),
        "cross_entropy": loss_sum / len(labels),
        "n_examples": len(labels),
    }


def train_evaluator(
    config,
    *,
    output_dir,
    data_root,
    split_manifest,
    resume=False,
    dry_run=False,
    dataset_factory=None,
    progress_callback=None,
):
    """Train only on real train; select best real validation; retain exact resume state.

    ``config`` may be the evaluator subsection or the resolved global dictionary.
    Supported fields are seed, epochs, batch_size, lr, weight_decay,
    accuracy_threshold, feature_batch_size, bandwidth_samples, device,
    checkpoint_every and grad_clip. Optional progress_callback checks a caller-owned
    resource budget after committed updates/epochs and during feature calibration.
    No test-set argument exists on this API.
    """
    options = dict(config.get("evaluator", config))
    if "evaluator" in config:
        # A subsection may override it, otherwise respect the global accelerator.
        options.setdefault("device", config.get("device", "cpu"))
    defaults = dict(
        seed=41,
        epochs=5,
        batch_size=128,
        lr=1e-3,
        weight_decay=0.0,
        accuracy_threshold=0.99,
        feature_batch_size=512,
        bandwidth_samples=2000,
        device="cpu",
        checkpoint_every=100,
        grad_clip=1.0,
    )
    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(f"unknown evaluator options: {sorted(unknown)}")
    options = defaults | options
    if min(options["epochs"], options["batch_size"], options["checkpoint_every"]) < 1:
        raise ValueError("epochs, batch_size and checkpoint_every must be positive")
    if not 0 <= options["accuracy_threshold"] <= 1:
        raise ValueError("accuracy threshold must lie in [0, 1]")
    plan = {
        "stage": "train_evaluator",
        "config": options,
        "data_source": "registered real MNIST train/validation only",
        "selection": "highest real validation accuracy, earliest on ties",
        "test_accessed": False,
    }
    if dry_run:
        return plan | {"status": "dry_run"}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "evaluator_checkpoint.pt"
    manifest_path = output_dir / "manifest.json"
    if checkpoint_path.exists() and not resume:
        raise ValueError("evaluator checkpoint exists; use resume=True or a new run directory")
    if resume and not checkpoint_path.exists():
        raise ValueError("resume requires an existing evaluator checkpoint")
    if (
        resume
        and manifest_path.exists()
        and json.loads(manifest_path.read_text()).get("test_accessed")
    ):
        raise ValueError("the evaluator already accessed test; its training/selection is frozen")
    split_manifest = Path(split_manifest)
    split = json.loads(split_manifest.read_text())
    split_hash = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    contract = {"options": options, "split_manifest_sha256": split_hash}
    train, train_labels = load_mnist_split(
        data_root, "train", split_manifest, dataset_factory=dataset_factory
    )
    validation, validation_labels = load_mnist_split(
        data_root, "validation", split_manifest, dataset_factory=dataset_factory
    )
    device = choose_device(options["device"])
    # Initialization does not perturb any caller's training/noise RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(options["seed"])
        model = EvaluatorCNN()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"]
    )
    order_rng = torch.Generator().manual_seed(options["seed"] + 1)
    epoch, offset, update, running_loss, running_count = 0, 0, 0, 0.0, 0
    order = None
    best_accuracy, best_epoch, best_state = -1.0, None, None
    history = []
    previous_seconds = 0.0
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["contract"] != contract:
            raise ValueError("evaluator resume contract/config/split differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        order_rng.set_state(saved["order_rng"])
        torch.random.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        epoch, offset, update = saved["epoch"], saved["offset"], saved["update"]
        order, history = saved["order"], saved["history"]
        running_loss, running_count = saved["running_loss"], saved["running_count"]
        best_accuracy, best_epoch = saved["best_accuracy"], saved["best_epoch"]
        best_state = saved["best_state"]
        previous_seconds = saved["training_seconds"]
    started = time.perf_counter()

    def save(status):
        nonlocal previous_seconds
        elapsed = previous_seconds + time.perf_counter() - started
        _atomic_checkpoint(
            {
                "contract": contract,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "offset": offset,
                "update": update,
                "order": order,
                "order_rng": order_rng.get_state(),
                "torch_rng": torch.random.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "history": history,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "running_loss": running_loss,
                "running_count": running_count,
                "training_seconds": elapsed,
                "status": status,
            },
            checkpoint_path,
        )
        _write_json(
            manifest_path,
            plan
            | {
                "status": status,
                "split_manifest_sha256": split_hash,
                "architecture": architecture_manifest(model),
                "updates": update,
                "epochs_completed": epoch,
                "best_validation_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "training_seconds": elapsed,
                "evaluator_gate_passed": best_accuracy >= options["accuracy_threshold"],
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            },
        )

    progress_interrupted = False

    def check_progress():
        nonlocal progress_interrupted
        if progress_callback is None:
            return
        # Commit sampler/optimizer/RNG BEFORE a budget callback may stop this stage.
        save("running")
        try:
            progress_callback()
        except BaseException:
            progress_interrupted = True
            raise

    save("running")
    try:
        check_progress()
        while epoch < options["epochs"]:
            model.train()
            if order is None:
                order = torch.randperm(len(train), generator=order_rng)
            while offset < len(order):
                indices = order[offset : offset + options["batch_size"]]
                pixels, _ = evaluator_preprocess(train[indices])
                logits = model(pixels.to(device))
                loss = F.cross_entropy(logits, train_labels[indices].to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite evaluator training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), options["grad_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                offset += len(indices)
                update += 1
                running_loss += float(loss.detach()) * len(indices)
                running_count += len(indices)
                if update % options["checkpoint_every"] == 0:
                    save("running")
                if update % 100 == 0:
                    check_progress()
            check_progress()
            measured = classifier_accuracy(
                model, validation, validation_labels, options["feature_batch_size"]
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "updates": update,
                    "train_cross_entropy": running_loss / running_count,
                    "validation_accuracy": measured["accuracy"],
                    "validation_cross_entropy": measured["cross_entropy"],
                }
            )
            if measured["accuracy"] > best_accuracy:
                best_accuracy, best_epoch = measured["accuracy"], epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epoch += 1
            order, offset, running_loss, running_count = None, 0, 0.0, 0
            save("running")
            check_progress()
        # Keep the optimizer-owned model aligned with its last training checkpoint.
        # A separate selected copy is exported/calibrated, including on budget resume.
        status = "completed" if best_accuracy >= options["accuracy_threshold"] else "failed_gate"
        save(status)
        selected = copy.deepcopy(model).eval().requires_grad_(False)
        selected.load_state_dict(best_state)
        selected.save_pretrained(output_dir / "evaluator")
        evaluator_hash = state_dict_sha256(selected)
        _write_json(output_dir / "history.json", history)
        final_manifest = json.loads(manifest_path.read_text()) | {
            "evaluator_sha256": evaluator_hash,
            "preprocessing": "(raw_fm+1)/2 then clamp(0,1)",
            "feature_layer": "FC128 after ReLU, before logits",
            "test_accuracy": None,
        }
        if status == "completed":
            split_indices = torch.load(
                split_manifest.parent / split["indices_file"], weights_only=True
            )
            stats = fit_feature_statistics(
                selected,
                train,
                split_indices["train"],
                seed=options["seed"] + 2,
                bandwidth_samples=options["bandwidth_samples"],
                batch_size=options["feature_batch_size"],
                progress_callback=check_progress,
            )
            _atomic_checkpoint(stats, output_dir / "feature_reference.pt")
            final_manifest["feature_reference_sha256"] = hashlib.sha256(
                (output_dir / "feature_reference.pt").read_bytes()
            ).hexdigest()
        check_progress()
        save(status)
        final_manifest |= json.loads(manifest_path.read_text())
        _write_json(manifest_path, final_manifest)
        return final_manifest
    except KeyboardInterrupt:
        save("interrupted")
        raise
    except Exception:
        save("interrupted" if progress_interrupted else "failed")
        raise


def finalize_evaluator_test(
    output_dir, *, data_root, split_manifest, dataset_factory=None, batch_size=512
):
    """One explicit test measurement after evaluator selection; refuses a second look."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("evaluator_gate_passed") or manifest["status"] != "completed":
        raise ValueError("freeze a validation-qualified evaluator before opening test")
    if manifest.get("test_accuracy") is not None or manifest.get("test_accessed"):
        raise ValueError("classifier test was already measured; use the recorded result")
    if (
        hashlib.sha256(Path(split_manifest).read_bytes()).hexdigest()
        != manifest["split_manifest_sha256"]
    ):
        raise ValueError("test evaluation split differs from the evaluator's registered data")
    model = EvaluatorCNN.from_pretrained(output_dir / "evaluator").eval().requires_grad_(False)
    if state_dict_sha256(model) != manifest["evaluator_sha256"]:
        raise ValueError("frozen evaluator hash differs from the selected checkpoint")
    test, labels = load_mnist_split(
        data_root, "test", split_manifest, allow_test=True, dataset_factory=dataset_factory
    )
    measured = classifier_accuracy(model, test, labels, batch_size)
    manifest |= {
        "test_accuracy": measured["accuracy"],
        "test_accessed": True,
        "test_evaluation": measured,
    }
    _write_json(manifest_path, manifest)
    return measured
