"""Auditable CLI stages; the research loop is also exposed in notebook 10."""

import copy
import json
from pathlib import Path

import torch

from .config import config_hash, method_name
from .diagnostics import (
    diagnostics_numerical_report,
    local_map_jacobian_diagnostics,
    local_rollout_diagnostics,
)
from .distill import calibrate, experiment_schedule, train_student
from .evaluate import (
    evaluation_numerical_report,
    fit_feature_reference,
    fixed_projection_bank,
    paired_fidelity,
    rays_quality,
    timed_generation,
    toy_quality,
)
from .fm import benchmark_data, build_model, load_model, sample_euler, train_teacher
from .losses import verify_loss_contracts
from .oracle import oracle_numerical_report
from .reporting import build_report
from .runtime import (
    algorithm_hash,
    atomic_json,
    choose_device,
    environment,
    file_hash,
    model_hash,
    require_rays_gate,
    tensor_hash,
)
from .schedules import student_endpoint, teacher_target


def doctor():
    result = environment()
    result["mps_access_note"] = (
        "macOS sandbox may hide Metal; check the same Python outside the sandbox"
    )
    result["recommended_device"] = "cuda" if result["cuda"] else "mps" if result["mps"] else "cpu"
    return result


def verify_losses(output="replication/reports/tests_report.json"):
    reports = {
        "loss_contracts": verify_loss_contracts(),
        "evaluation": evaluation_numerical_report(),
        "diagnostics": diagnostics_numerical_report(),
        "oracle": oracle_numerical_report(),
    }
    reports["passed"] = all(value.get("passed", True) for value in reports.values())
    atomic_json(output, reports)
    if not reports["passed"]:
        raise RuntimeError("Mathematical verification failed; training series must not continue")
    return reports


def prepare_mnist_stage(config, dry_run=False):
    from .mnist_workflow import prepare_stage

    return prepare_stage(config, dry_run=dry_run)


def train_evaluator_stage(config, resume=False, dry_run=False):
    from .mnist_workflow import evaluator_stage

    return evaluator_stage(config, resume=resume, dry_run=dry_run)


def train_teacher_stage(config, resume=False, dry_run=False):
    result = train_teacher(config, resume=resume, dry_run=dry_run)
    return {key: value for key, value in result.items() if key != "model"}


def _teacher(config):
    root = Path(config["output_dir"]) / "teacher"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise RuntimeError("Teacher training is not completed; resume it before distillation")
    teacher = (
        load_model(root / "ema_model", choose_device(config["device"])).eval().requires_grad_(False)
    )
    if model_hash(teacher) != manifest["teacher_sha256"]:
        raise ValueError("Teacher EMA digest differs from its fixed manifest")
    return teacher, manifest


def _context(config, split):
    """Load an explicitly selected real split and fixed train-only calibration."""
    if split not in {"validation", "test"}:
        raise ValueError("Quality contexts accept validation or explicit final test only")
    if config["data"]["name"] == "mnist":
        from .mnist_workflow import build_quality_context

        context = build_quality_context(config, split=split)
        if context.get("split", split) != split:
            raise ValueError("MNIST quality context returned the wrong real split")
        return context
    train = benchmark_data(config)
    real = benchmark_data(config, split)
    reference = fit_feature_reference(train, seed=config["data"]["seed"], standardize=False)
    bank = fixed_projection_bank(
        2, config["evaluation"]["swd_projections"], config["evaluation"]["reference_seed"]
    )
    return {
        "train": train,
        "real_samples": real,
        "reference": reference,
        "projection_bank": bank,
        "split": split,
        "metadata": {
            "reference_split": split,
            "train_samples_sha256": tensor_hash(train),
            "real_samples_sha256": tensor_hash(real),
            "projection_bank_sha256": tensor_hash(bank),
            "kernel_calibration": reference.to_dict(),
            "normalization_fit_split": "train_only",
        },
    }


def _quality(config, samples, context):
    """Independent quality to real; reject nonfinite results before persistence."""
    if not torch.isfinite(torch.as_tensor(samples)).all():
        raise FloatingPointError("Nonfinite generated samples invalidate this evaluation")
    if config["data"]["name"] == "mnist":
        from .mnist_workflow import quality

        result = quality(config, samples, context)
    elif config["data"]["name"] == "gmm":
        result = toy_quality(
            samples,
            context["real_samples"],
            context["reference"],
            seed=config["evaluation"]["reference_seed"],
            subsample_size=512,
            block_size=config["evaluation"]["mmd_block_size"],
        )
    else:
        result = rays_quality(
            samples,
            context["real_samples"],
            context["reference"],
            context["projection_bank"],
            block_size=config["evaluation"]["mmd_block_size"],
        )

    def finite(value):
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        if isinstance(value, (int, float)):
            return bool(torch.isfinite(torch.tensor(value, dtype=torch.float64)))
        return True

    if not finite(result):
        raise FloatingPointError("Nonfinite independent quality metric invalidates this evaluation")
    return result


def _initial_noise(config, model, validation=False):
    e = config["evaluation"]
    seed = e["validation_seed"] if validation else e["final_seed"]
    count = e["validation_samples"] if validation else e["samples"]
    return torch.randn((count, *model.sample_shape), generator=torch.Generator().manual_seed(seed))


def teacher_gate(config, dry_run=False):
    """Validation only: learned quality, finite samples, and numerical stabilization."""
    from .runtime import ResourceBudget, stable_seed
    from .schedules import call_velocity

    if dry_run:
        return {
            "stage": "teacher-gate",
            "nfe": config["evaluation"]["teacher_nfe"],
            "split": "validation",
        }
    budget = ResourceBudget(config, choose_device(config["device"]))
    try:
        require_rays_gate(config)
        teacher, manifest = _teacher(config)
        teacher_digest = model_hash(teacher)
        context = _context(config, "validation")
        # Independent held-out CFM probe: never reuse generation/tuning/final RNG.
        # The final EMA remains fixed; this records a diagnostic, not model selection.
        cfm_seeds = {
            stream: stable_seed(config["teacher"]["seed"], "heldout_cfm/" + stream)
            for stream in ("data", "noise", "interval")
        }
        cfm_rng = {
            stream: torch.Generator(device="cpu").manual_seed(seed)
            for stream, seed in cfm_seeds.items()
        }
        count = min(2048, len(context["real_samples"]))
        if count < 1:
            raise ValueError("Held-out CFM diagnostic requires nonempty validation data")
        indices = torch.randperm(len(context["real_samples"]), generator=cfm_rng["data"])[:count]
        clean = context["real_samples"][indices].detach().cpu().float()
        cfm_noise = torch.randn(clean.shape, generator=cfm_rng["noise"], dtype=clean.dtype)
        cfm_times = torch.rand(count, generator=cfm_rng["interval"], dtype=clean.dtype)
        parameter = next(teacher.parameters())
        error_sum, target_energy_sum = 0.0, 0.0
        with torch.no_grad():
            for offset in range(0, count, 128):
                data = clean[offset : offset + 128]
                endpoint_noise = cfm_noise[offset : offset + 128]
                t = cfm_times[offset : offset + 128]
                broadcast_t = t.reshape(len(t), *([1] * (data.ndim - 1)))
                anchor = (1 - broadcast_t) * data + broadcast_t * endpoint_noise
                target = endpoint_noise - data
                prediction = call_velocity(
                    teacher,
                    anchor.to(dtype=parameter.dtype).to(parameter.device),
                    t.to(dtype=parameter.dtype).to(parameter.device),
                )
                if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                    raise FloatingPointError(
                        "Nonfinite held-out teacher velocity invalidates its quality gate"
                    )
                error_sum += float(
                    (prediction.detach().cpu().double() - target.double()).square().sum()
                )
                target_energy_sum += float(target.double().square().sum())
        cfm_diagnostic = {
            "raw_cfm_mse": error_sum / clean.numel(),
            "zero_velocity_cfm_mse": target_energy_sum / clean.numel(),
            "samples": count,
            "batch_size": 128,
            "teacher_forward_calls": (count + 127) // 128,
            "split": "validation",
            "rng_seeds": cfm_seeds,
            "indices_sha256": tensor_hash(indices),
            "noise_sha256": tensor_hash(cfm_noise),
            "times_sha256": tensor_hash(cfm_times),
            "checkpoint_rule": "fixed final EMA; diagnostic only, no HPO or checkpoint selection",
        }
        noise = _initial_noise(config, teacher, validation=True)
        outputs, rows = {}, []
        steps_to_check = sorted(
            set(config["evaluation"]["teacher_nfe"])
            | {32, 64, 128, config["distillation"]["teacher_steps"]}
        )
        for steps in steps_to_check:
            generated = sample_euler(
                teacher,
                noise,
                steps=steps,
                record_history=False,
                batch_size=config["evaluation"]["batch_size"],
            )
            if generated["initial_noise_sha256"] != tensor_hash(noise):
                raise AssertionError("Teacher gate changed the preregistered validation noise")
            outputs[steps] = generated["samples"]
            rows.append(
                {
                    "nfe": steps,
                    "quality": _quality(config, generated["samples"], context),
                    "total_model_calls": generated["total_model_calls"],
                    "initial_noise_sha256": generated["initial_noise_sha256"],
                }
            )
        noise_quality = _quality(config, noise, context)
        main = next(row for row in rows if row["nfe"] == config["distillation"]["teacher_steps"])
        metric = "mmd2_mnist_features_real" if config["data"]["name"] == "mnist" else "mmd2_real"
        learned = main["quality"][metric] < noise_quality[metric]
        mse32 = float((outputs[32].double() - outputs[128].double()).square().mean())
        mse64 = float((outputs[64].double() - outputs[128].double()).square().mean())
        # Even a constant FP32 field accumulates roundoff across 128 additions.
        # A dtype/step-count floor distinguishes that from unresolved ODE error.
        reference_energy = max(1.0, float(outputs[128].double().square().mean()))
        roundoff_floor = max(
            1e-12, (128 * torch.finfo(outputs[128].dtype).eps) ** 2 * reference_energy
        )
        stabilized = mse64 <= mse32 + roundoff_floor
        result = {
            "passed": bool(learned and stabilized),
            "teacher_sha256": manifest["teacher_sha256"],
            "split": "validation",
            "test_accessed": False,
            "rows": rows,
            "noise_baseline": noise_quality,
            "beats_noise_baseline": bool(learned),
            "validation_cfm": cfm_diagnostic,
            "stabilization": {
                "mse_32_to_128": mse32,
                "mse_64_to_128": mse64,
                "converging": bool(stabilized),
                "roundoff_mse_floor": roundoff_floor,
                "criterion": "paired MSE(64,128) <= MSE(32,128)+max(1e-12,(128*dtype_eps)^2*max(1,reference_energy)); quality need not be monotonic",
            },
            "initial_noise_sha256": tensor_hash(noise),
            "evaluator_is_independent": True,
            "quality_context": context.get("metadata", {}),
            "config_sha256": config_hash(config),
            "algorithm_sha256": algorithm_hash(),
            "assessment_source_sha256": file_hash(Path(__file__)),
        }
        if config["data"]["name"] == "mnist":
            frequencies = main["quality"].get("predicted_digit_frequencies", [])
            result["recognizability_and_coverage"] = {
                "predicted_digit_count": sum(frequency > 0 for frequency in frequencies),
                "predicted_digit_frequencies": frequencies,
                "mean_confidence": main["quality"].get("mean_confidence"),
                "fixed_validation_examples_saved": True,
                "interpretation": "Confidence and occupancy are diagnostics, not sufficient evidence of realistic digits; inspect fixed examples.",
            }
            torch.save(
                {
                    "noise": noise[:16],
                    "teacher": outputs[config["distillation"]["teacher_steps"]][:16],
                },
                Path(config["output_dir"]) / "teacher" / "validation_gate_previews.pt",
            )
        if config["data"]["name"] == "rotating_rays":
            swd = {row["nfe"]: row["quality"]["sliced_wasserstein"] for row in rows}
            result["low_sensitivity_16_vs_32"] = (
                abs(swd[16] - swd[32]) < config["evaluation"]["sensitivity_swd_margin"]
                if 16 in swd
                else None
            )
        if model_hash(teacher) != teacher_digest:
            raise AssertionError("Teacher weights changed during its validation gate")
        atomic_json(Path(config["output_dir"]) / "teacher_gate.json", result)
        if not result["passed"]:
            raise RuntimeError(
                "Teacher gate failed: require quality better than independent noise and stabilized numerical trajectories"
            )
        return result

    finally:
        try:
            budget.check()
        finally:
            budget.close()


def calibrate_stage(config, dry_run=False):
    if dry_run:
        return {
            "stage": "calibrate",
            "support": config["distillation"]["calibration_size"],
            "representations": ["endpoint", "displacement", "joint"],
            "split": "train",
        }
    require_rays_gate(config)
    teacher, _ = _teacher(config)
    gate = json.loads((Path(config["output_dir"]) / "teacher_gate.json").read_text())
    if not gate["passed"] or gate["teacher_sha256"] != model_hash(teacher):
        raise RuntimeError("Teacher gate is missing/stale")
    return calibrate(
        config,
        teacher,
        benchmark_data(config),
        output_path=Path(config["output_dir"]) / "calibration.json",
    )


def distill_suite(config, resume=False, dry_run=False):
    d = config["distillation"]
    if dry_run:
        return {
            "stage": "distill-suite",
            "runs": [
                train_student(config, None, None, None, loss=loss, rho=rho, seed=seed, dry_run=True)
                for seed in d["seeds"]
                for loss in d["losses"]
                for rho in d["tube_rhos"]
            ],
        }
    require_rays_gate(config)
    teacher, _ = _teacher(config)
    teacher_check = json.loads((Path(config["output_dir"]) / "teacher_gate.json").read_text())
    if not teacher_check["passed"] or teacher_check["teacher_sha256"] != model_hash(teacher):
        raise RuntimeError("Teacher quality gate is failed or stale")
    samples = benchmark_data(config)
    calibration = json.loads((Path(config["output_dir"]) / "calibration.json").read_text())
    results = []
    traces = {}
    for seed in d["seeds"]:
        for loss in d["losses"]:
            for rho in d["tube_rhos"]:
                checkpoint = (
                    Path(config["output_dir"])
                    / "students"
                    / f"seed{seed}_{method_name(loss, rho)}"
                    / "training.pt"
                )
                result = train_student(
                    config,
                    teacher,
                    samples,
                    calibration,
                    loss=loss,
                    rho=rho,
                    seed=seed,
                    resume=resume and checkpoint.exists(),
                )
                history = json.loads((Path(result["directory"]) / "history.json").read_text())
                trace = [row["common_rng_trace"] for row in history if "common_rng_trace" in row]
                if seed in traces and traces[seed] != trace:
                    raise AssertionError("Paired RNG traces diverged between compared methods")
                traces[seed] = trace
                results.append({key: value for key, value in result.items() if key != "model"})
    atomic_json(
        Path(config["output_dir"]) / "suite_status.json",
        {"status": "completed", "runs": results, "common_rng_traces_equal": True},
    )
    return {"status": "completed", "runs": len(results), "common_rng_traces_equal": True}


def _fidelity(
    generated,
    teacher_output,
    student_times,
    teacher_times,
    *,
    batch_size=128,
    student_logits=None,
    teacher_logits=None,
):
    """Bounded-memory raw paired metrics; SSIM never materializes a 10k batch."""
    import math

    samples, target = generated["samples"], teacher_output["samples"]
    if samples.shape != target.shape or len(samples) < 1 or batch_size < 1:
        raise ValueError(
            "Fidelity requires equally ordered nonempty samples and positive batch size"
        )
    result = {"paired_examples": len(samples)}
    for offset in range(0, len(samples), batch_size):
        if (
            not torch.isfinite(samples[offset : offset + batch_size]).all()
            or not torch.isfinite(target[offset : offset + batch_size]).all()
        ):
            raise FloatingPointError("Nonfinite raw samples invalidate paired fidelity")
    if generated is teacher_output:
        result["paired_endpoint_mse"] = 0.0
        if samples.ndim == 4:
            result.update(pixel_ssim=1.0, ssim_data_range=2.0, psnr_db="+inf", psnr_infinite=True)
    else:
        mse_sum, ssim_sum = 0.0, 0.0
        for offset in range(0, len(samples), batch_size):
            chunk = samples[offset : offset + batch_size]
            raw = paired_fidelity(chunk, target[offset : offset + batch_size], data_range=2.0)
            mse_sum += len(chunk) * raw["paired_endpoint_mse"]
            if samples.ndim == 4:
                ssim_sum += len(chunk) * raw["pixel_ssim"]
        mse = mse_sum / len(samples)
        result["paired_endpoint_mse"] = mse
        if samples.ndim == 4:
            result.update(
                pixel_ssim=ssim_sum / len(samples),
                ssim_data_range=2.0,
                psnr_db="+inf" if mse == 0 else 10 * math.log10(4 / mse),
                psnr_infinite=mse == 0,
            )
    if student_logits is not None or teacher_logits is not None:
        if (
            student_logits is None
            or teacher_logits is None
            or len(student_logits) != len(samples)
            or len(teacher_logits) != len(samples)
        ):
            raise ValueError(
                "Paired class agreement requires both full, ordered classifier-logit banks"
            )
        sl = torch.as_tensor(student_logits).detach().cpu()
        tl = torch.as_tensor(teacher_logits).detach().cpu()
        if not torch.isfinite(sl).all() or not torch.isfinite(tl).all():
            raise FloatingPointError("Nonfinite classifier logits invalidate paired fidelity")
        result["class_agreement"] = float((sl.argmax(1) == tl.argmax(1)).double().mean())
    if generated.get("rollout_mse_per_node") is not None:
        result["rollout_mse_per_node"] = generated["rollout_mse_per_node"]
        result["rollout_times"] = generated["rollout_times"]
        result["rollout_scope"] = "all paired evaluation examples; common grid nodes"
    else:
        sh, th = generated.get("history"), teacher_output.get("history")
        # Preview-only histories must never be mislabeled as the full evaluation.
        if (
            sh is not None
            and th is not None
            and sh.shape[1] == len(samples)
            and th.shape[1] == len(target)
        ):
            st = torch.as_tensor(student_times).detach().cpu().double()
            tt = torch.as_tensor(teacher_times).detach().cpu().double()
            curves, common_times = [], []
            for index, time in enumerate(st):
                matches = torch.where(torch.isclose(tt, time, atol=1e-9, rtol=0))[0]
                if len(matches) != 1:
                    continue
                teacher_index = int(matches[0])
                if generated is teacher_output and index == teacher_index:
                    curves.append(0.0)
                else:
                    total = 0.0
                    for offset in range(0, len(samples), batch_size):
                        first = sh[index, offset : offset + batch_size].detach().cpu().double()
                        second = (
                            th[teacher_index, offset : offset + batch_size].detach().cpu().double()
                        )
                        total += float((first - second).square().sum())
                    curves.append(total / samples.numel())
                common_times.append(float(time))
            result.update(
                rollout_mse_per_node=curves,
                rollout_times=common_times,
                rollout_scope="all paired evaluation examples; common grid nodes",
            )
    return result


def evaluate_suite(config, dry_run=False):
    """Final independent quality with one shared noise bank and bounded histories."""
    import hashlib

    from .evaluate import extract_mnist_features
    from .runtime import ResourceBudget, save_checkpoint

    if dry_run:
        return {
            "stage": "evaluate-suite",
            "samples": config["evaluation"]["samples"],
            "split": "test",
            "noise_seed": config["evaluation"]["final_seed"],
            "main_metric": "mmd2_mnist_features_real"
            if config["data"]["name"] == "mnist"
            else "mmd2_real",
        }
    require_rays_gate(config)
    root, e, d = Path(config["output_dir"]), config["evaluation"], config["distillation"]
    directory = root / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    teacher, teacher_manifest = _teacher(config)
    teacher_digest = model_hash(teacher)
    student_manifests = {}
    # Check completion/fixation BEFORE loading the closed test reference.
    for seed in d["seeds"]:
        for loss in d["losses"]:
            for rho in d["tube_rhos"]:
                name = method_name(loss, rho)
                path = root / "students" / f"seed{seed}_{name}"
                manifest = json.loads((path / "manifest.json").read_text())
                if manifest.get("status") != "completed":
                    raise RuntimeError(
                        f"Student run {path} is not completed; final test remains closed"
                    )
                if manifest.get("updates_completed") != config["student"]["updates"]:
                    raise RuntimeError(
                        f"Student run {path} has the wrong update budget; final test remains closed"
                    )
                if (
                    manifest.get("teacher_sha256", teacher_digest) != teacher_digest
                    or manifest.get("student_init_sha256", teacher_digest) != teacher_digest
                ):
                    raise ValueError(
                        "Student parent/initialization differs from the frozen teacher"
                    )
                if manifest.get("config_sha256", config_hash(config)) != config_hash(config):
                    raise ValueError(
                        "Student final configuration differs from the registered evaluation"
                    )
                exported = load_model(path / "ema_model", "cpu").eval()
                if model_hash(exported) != manifest["student_ema_sha256"]:
                    raise ValueError(
                        "Student export failed fixation hash; final test remains closed"
                    )
                del exported
                student_manifests[seed, name] = manifest
    budget = ResourceBudget(config, next(teacher.parameters()).device)
    try:
        context = _context(config, "test")
        noise = _initial_noise(config, teacher)
        noise_digest = tensor_hash(noise)
        schedule = experiment_schedule(config)
        teacher_times = torch.linspace(1, 0, d["teacher_steps"] + 1, dtype=torch.float64)
        is_mnist = config["data"]["name"] == "mnist"
        metadata = {
            "config": config,
            "config_sha256": config_hash(config),
            "teacher_sha256": teacher_digest,
            "initial_noise_sha256": noise_digest,
            "student_checkpoint_hashes": {
                f"seed{seed}_{name}": manifest["student_ema_sha256"]
                for (seed, name), manifest in student_manifests.items()
            },
            "evaluation_split": "test",
            "scope": config["scope"],
            "status": "running",
            "algorithm_sha256": algorithm_hash(),
            "assessment_source_sha256": file_hash(Path(__file__)),
            "quality_source_sha256": file_hash(Path(__file__).with_name("evaluate.py")),
            "quality_context": context.get("metadata", {}),
            "student_seed_selection": "first registered seed, not best quality",
            "raw_fidelity_preprocessing": "none; raw FM coordinates; SSIM data_range=2",
            "diagnostic_split": "test, accessed only after all registered student weights/budgets/exports were fixed",
            "history_retention": "only frozen main-teacher full CPU float32 history; other methods retain first fixed preview states and stream full-bank rollout MSE",
        }
        evaluation_contract = copy.deepcopy(metadata)
        # This event flag records whether a cached classifier-test measurement
        # was created on this invocation; it does not change fixed weights/data.
        evaluation_contract["quality_context"].pop("classifier_test_measurement_created_here", None)
        metadata["evaluation_sha256"] = hashlib.sha256(
            json.dumps(evaluation_contract, sort_keys=True).encode()
        ).hexdigest()
        previous = directory / "results.json"
        if previous.is_file():
            old = json.loads(previous.read_text()).get("metadata", {})
            if old.get("evaluation_sha256") not in (None, metadata["evaluation_sha256"]):
                raise ValueError(
                    "Existing evaluation has a different source/config/noise/reference hash; use a new artifact directory"
                )
        rows, trajectories, diagnostics = [], {}, []

        def persist():
            atomic_json(directory / "results.json", {"rows": rows, "metadata": metadata})

        persist()
    except BaseException:
        try:
            budget.check()
        finally:
            budget.close()
        raise
    try:
        # Keeping this ONE CPU32 reference permits exact full-bank rollout
        # comparisons without recomputing teacher trajectories for each student.
        reference = sample_euler(
            teacher, noise, times=teacher_times, batch_size=e["batch_size"], record_history=True
        )
        if reference["initial_noise_sha256"] != noise_digest:
            raise AssertionError("Reference changed the common initial noise")
        metadata["full_reference_history_bytes"] = (
            reference["history"].numel() * reference["history"].element_size()
        )
        teacher_logits = (
            extract_mnist_features(
                context["evaluator"], reference["samples"], batch_size=e["batch_size"]
            )[1]
            if is_mnist
            else None
        )

        def generated_output(model, times, *, record_common, preview):
            chunks, preview_history = [], None
            calls, curve_totals = 0, None
            common = []
            for index, time in enumerate(times):
                matches = torch.where(torch.isclose(teacher_times, time, atol=1e-9, rtol=0))[0]
                if len(matches) == 1:
                    common.append((index, int(matches[0]), float(time)))
            for offset in range(0, len(noise), e["batch_size"]):
                initial = noise[offset : offset + e["batch_size"]]
                chunk = sample_euler(
                    model,
                    initial,
                    times=times,
                    batch_size=e["batch_size"],
                    record_history=record_common or (preview and offset == 0),
                )
                if chunk["initial_noise_sha256"] != tensor_hash(initial):
                    raise AssertionError("Method changed the common initial noise or sample order")
                if not torch.isfinite(chunk["samples"]).all():
                    raise FloatingPointError(
                        "Nonfinite generated endpoint invalidates this evaluation"
                    )
                chunks.append(chunk["samples"])
                calls += chunk["total_model_calls"]
                history = chunk.get("history")
                if preview and offset == 0 and history is not None:
                    preview_history = history[:, : e["gif_particles"]].clone()
                if record_common and history is not None:
                    if curve_totals is None:
                        curve_totals = [0.0] * len(common)
                    for cursor, (si, ti, _) in enumerate(common):
                        first = history[si].detach().cpu().double()
                        second = (
                            reference["history"][ti, offset : offset + len(initial)]
                            .detach()
                            .cpu()
                            .double()
                        )
                        curve_totals[cursor] += float((first - second).square().sum())
                budget.check()
            result = {
                "samples": torch.cat(chunks),
                "history": preview_history,
                "history_scope": "first registered fixed preview only",
                "total_model_calls": calls,
                "initial_noise_sha256": noise_digest,
            }
            if curve_totals is not None:
                result["rollout_mse_per_node"] = [value / noise.numel() for value in curve_totals]
                result["rollout_times"] = [time for _, _, time in common]
            return result

        def measure(model, name, seed, times, manifest):
            preview = (
                seed is None and len(times) - 1 in {d["teacher_steps"], d["student_steps"]}
            ) or seed == d["seeds"][0]
            is_reference = name == f"Original N{d['teacher_steps']}"
            output = (
                reference
                if is_reference
                else generated_output(
                    model,
                    times,
                    record_common=(seed is not None or len(times) - 1 == d["student_steps"]),
                    preview=preview,
                )
            )
            quality = _quality(config, output["samples"], context)
            logits = (
                teacher_logits
                if is_reference
                else extract_mnist_features(
                    context["evaluator"], output["samples"], batch_size=e["batch_size"]
                )[1]
                if is_mnist
                else None
            )
            fidelity = _fidelity(
                output,
                reference,
                times,
                teacher_times,
                batch_size=e["batch_size"],
                student_logits=logits,
                teacher_logits=teacher_logits,
            )
            timing = timed_generation(
                lambda: sample_euler(
                    model,
                    noise[: e["batch_size"]],
                    times=times,
                    record_history=False,
                    batch_size=e["batch_size"],
                ),
                repeats=e["timing_repeats"],
                warmup=1,
                device=next(model.parameters()).device,
            )
            single = timed_generation(
                lambda: sample_euler(model, noise[:1], times=times, record_history=False),
                repeats=e["timing_repeats"],
                warmup=1,
                device=next(model.parameters()).device,
            )
            row = {
                "run_id": manifest.get("run_id", name),
                "method": name,
                "student_seed": seed,
                "nfe": len(times) - 1,
                "quality": quality,
                "fidelity": fidelity,
                "initial_noise_sha256": noise_digest,
                "evaluation_sha256": metadata["evaluation_sha256"],
                "model_sha256": model_hash(model),
                "cost": {
                    "training_seconds": manifest.get("training_seconds", 0.0),
                    "ot_seconds": manifest.get("ot_seconds", 0.0),
                    "teacher_calls_training": manifest.get("teacher_calls", 0),
                    "median_sampling_seconds": timing["median_seconds"],
                    "p95_sampling_seconds": timing["p95_seconds"],
                    "sampling_batch_size": e["batch_size"],
                    "batch1_median_seconds": single["median_seconds"],
                    "total_model_calls": output["total_model_calls"],
                    "generated_samples": len(noise),
                    "peak_allocated_bytes": timing["peak_allocated_bytes"],
                    "peak_reserved_bytes": timing["peak_reserved_bytes"],
                },
            }
            rows.append(row)
            if preview and output.get("history") is not None:
                trajectories[name] = output["history"][:, : e["gif_particles"]].clone()
            if seed == d["seeds"][0]:
                device = next(model.parameters()).device
                n = min(e["diagnostic_samples"], len(noise), len(context["real_samples"]))
                diagnostic = local_rollout_diagnostics(
                    teacher,
                    model,
                    context["real_samples"][:n].to(device),
                    noise[:n].to(device),
                    schedule.times,
                    schedule.groups,
                    seed=e["reference_seed"],
                    rhos=e["rho_eval"],
                )
                anchors = context["real_samples"][: min(n, 64)].to(device)
                diagnostic["jacobian"] = local_map_jacobian_diagnostics(
                    lambda x: teacher_target(
                        teacher, x, 0.5, 1 / d["student_steps"], schedule.groups[0]
                    ),
                    lambda x: student_endpoint(model, x, 0.5, 1 / d["student_steps"]),
                    anchors,
                    seed=e["reference_seed"],
                )
                diagnostics.append({"method": name, "student_seed": seed, **diagnostic})
            persist()
            budget.check()
            print(f"eval {name} seed={seed}: {quality}", flush=True)

        baseline_steps = sorted(set(e["teacher_nfe"]) | {d["student_steps"], d["teacher_steps"]})
        for steps in baseline_steps:
            measure(
                teacher,
                f"Original N{steps}",
                None,
                torch.linspace(1, 0, steps + 1, dtype=torch.float64),
                teacher_manifest,
            )
        if d["groups"] is not None:
            measure(
                teacher, "Original student grouped grid", None, schedule.times, teacher_manifest
            )
        for seed in d["seeds"]:
            for loss in d["losses"]:
                for rho in d["tube_rhos"]:
                    name = method_name(loss, rho)
                    path = root / "students" / f"seed{seed}_{name}"
                    manifest = student_manifests[seed, name]
                    model = (
                        load_model(path / "ema_model", choose_device(config["device"]))
                        .eval()
                        .requires_grad_(False)
                    )
                    if model_hash(model) != manifest["student_ema_sha256"]:
                        raise ValueError("Student export digest changed")
                    measure(model, name, seed, schedule.times, manifest)
                    del model
        if model_hash(teacher) != metadata["teacher_sha256"]:
            raise AssertionError("Teacher weights changed during final evaluation")
        atomic_json(directory / "diagnostics.json", diagnostics)
        save_checkpoint(directory / "trajectories.pt", trajectories)
        datasets = (
            {"test": context["real_samples"][:64], "preview_only": True}
            if is_mnist
            else {
                "train": benchmark_data(config),
                "validation": benchmark_data(config, "validation"),
                "test": context["real_samples"],
            }
        )
        save_checkpoint(root / "datasets.pt", datasets)
        metadata.update(
            status="completed",
            trajectories_sha256=file_hash(directory / "trajectories.pt"),
            diagnostics_sha256=file_hash(directory / "diagnostics.json"),
        )
        persist()
        return {
            "status": "completed",
            "rows": len(rows),
            "directory": str(directory),
            "evaluation_sha256": metadata["evaluation_sha256"],
        }
    except BaseException as error:
        metadata.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
        persist()
        raise
    finally:
        try:
            budget.check()
        finally:
            budget.close()


def report_stage(run_root, report_dir, config=None):
    if config is None:
        config = json.loads((Path(run_root) / "evaluation/results.json").read_text())["metadata"][
            "config"
        ]
    return build_report(run_root, report_dir, config)


def pilot(config, resume=False, dry_run=False):
    if dry_run:
        return {
            "teacher": train_teacher_stage(config, dry_run=True),
            "students": distill_suite(config, dry_run=True),
            "evaluation": evaluate_suite(config, dry_run=True),
        }
    verify_losses()
    train_teacher_stage(config, resume=resume)
    teacher_gate(config)
    calibrate_stage(config)
    distill_suite(config, resume=resume)
    evaluate_suite(config)
    summary = report_stage(config["output_dir"], "replication/reports", config)
    if config["data"]["name"] == "rotating_rays":
        gate = {
            "passed": True,
            "algorithm_sha256": algorithm_hash(),
            "all_mvp_methods_completed": True,
            "config_sha256": config_hash(config),
            "animations": summary["animations"],
            "report": summary["report"],
            "scope": "rotating_rays_CPU_pilot",
        }
        atomic_json(config["resources"]["rays_gate"], gate)
    return summary


def benchmark_stage(config, dry_run=False):
    if dry_run:
        return {
            "stage": "benchmark",
            "warmup_updates": 20,
            "measured_updates": 100,
            "final_seeds_used": False,
            "forecast_hours": None,
        }
    import hashlib
    import time

    from .planning import (
        check_registered_resources,
        merge_benchmark_ledger,
        propose_equal_budget,
        read_resource_ledger,
    )
    from .runtime import stable_seed

    require_rays_gate(config)
    root = Path(config["output_dir"])
    device = choose_device(config["device"])
    started = time.monotonic()
    torch.set_num_threads(config["threads"])
    check_registered_resources(config)
    teacher_manifest_path = root / "teacher/manifest.json"
    teacher_manifest = (
        json.loads(teacher_manifest_path.read_text()) if teacher_manifest_path.is_file() else {}
    )
    completed_teacher_updates = int(teacher_manifest.get("updates_completed", 0))
    if (
        teacher_manifest.get("status") == "completed"
        and (root / "teacher/ema_model/config.json").is_file()
    ):
        teacher, _ = _teacher(config)
        teacher_source = "fixed_completed_teacher"
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stable_seed(919189, "benchmark_initialization"))
            teacher = build_model(config).to(device).eval().requires_grad_(False)
        teacher_source = "untrained_architecture_probe_rebenchmark_after_teacher_is_fixed"
    samples = benchmark_data(config)
    parent_digest = model_hash(teacher)
    contract = {
        "caller_config_sha256": config_hash(config),
        "teacher_sha256": parent_digest,
        "train_data_sha256": tensor_hash(samples),
        "algorithm_sha256": algorithm_hash(),
        "device": str(device),
        "warmup_updates": 20,
        "measured_updates": 100,
    }
    cache_digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    directory = (
        root / "benchmarks" / f"{parent_digest[:12]}_{config_hash(config)[:12]}_{cache_digest[:12]}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path = directory / "resource_ledger.json"
    destination = directory / "benchmark.json"
    counter_key = (
        "gpu_training_seconds" if device.type in {"cuda", "mps"} else "cpu_training_wall_seconds"
    )
    prior = read_resource_ledger(directory)
    last_counter, last_accounted_time = float(prior.get(counter_key, 0.0)), started

    def account():
        # Training checks already record most wall time. Credit only the gap
        # (calibration, setup, serialization, or an uncommitted interrupted tail).
        nonlocal last_counter, last_accounted_time
        now = time.monotonic()
        ledger = read_resource_ledger(directory)
        current = float(ledger.get(counter_key, 0.0))
        if current < last_counter:
            raise RuntimeError("Benchmark resource counter decreased")
        overhead = max(0.0, now - last_accounted_time - (current - last_counter))
        ledger[counter_key] = current + overhead
        ledger["benchmark_overhead_seconds"] = (
            ledger.get("benchmark_overhead_seconds", 0.0) + overhead
        )
        atomic_json(ledger_path, ledger)
        last_counter, last_accounted_time = ledger[counter_key], now
        merge_benchmark_ledger(root, ledger_path)
        return check_registered_resources(config)

    failure = None
    try:
        parent_ledger = account()
        if destination.is_file():
            cached = json.loads(destination.read_text())
            if cached.get("cache_contract") != contract:
                raise ValueError(
                    "Benchmark cache contract differs; preserve it and register a new probe"
                )
            return cached
        registration_path = directory / "registration.json"
        if registration_path.is_file():
            registration = json.loads(registration_path.read_text())
            if registration["cache_contract"] != contract:
                raise ValueError("Benchmark registration changed")
            b = registration["benchmark_config"]
            foreign_spent = float(parent_ledger.get("gpu_training_seconds", 0.0)) - float(
                read_resource_ledger(directory).get("gpu_training_seconds", 0.0)
            )
            if foreign_spent > registration["parent_gpu_seconds_excluding_probe"] + 1e-6:
                raise RuntimeError(
                    "Other accelerator work changed the remaining budget; register a new probe before resuming"
                )
        else:
            b = copy.deepcopy(config)
            b["output_dir"] = str(directory)
            b["protocol_id"] = f"{config['protocol_id']}/timing_probe"
            b["scope"] = "timing_probe_only"
            b["teacher"].update(
                seed=stable_seed(919190, "benchmark_teacher"),
                updates=120,
                warmup=0,
                checkpoint_every=120,
            )
            b["student"].update(updates=120, warmup=0, checkpoint_every=120)
            b["distillation"].update(
                seeds=[stable_seed(919191, "benchmark_student")],
                losses=["mse", "sinkhorn_endpoint"],
                tube_rhos=[0.0],
            )
            foreign_spent = float(parent_ledger.get("gpu_training_seconds", 0.0)) - float(
                read_resource_ledger(directory).get("gpu_training_seconds", 0.0)
            )
            b["resources"]["max_gpu_hours"] = (
                config["resources"]["max_gpu_hours"] - foreign_spent / 3600
            )
            registration = {
                "cache_contract": contract,
                "benchmark_config": b,
                "parent_gpu_seconds_excluding_probe": foreign_spent,
            }
            atomic_json(registration_path, registration)
        calibration = calibrate(b, teacher, samples)
        account()

        def run_case(kind):
            # The three calls are sequential. Each training function releases
            # its accelerator lock before parent accounting or the next call.
            account()
            if kind == "teacher":
                case_directory = directory / "teacher"
            else:
                case_directory = (
                    directory
                    / "students"
                    / f"seed{b['distillation']['seeds'][0]}_{method_name(kind, 0)}"
                )
            manifest_path = case_directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            if manifest.get("status") != "completed" or manifest.get("updates_completed") != 120:
                resume = (case_directory / "training.pt").is_file()
                if kind == "teacher":
                    train_teacher(b, samples=samples, resume=resume)
                else:
                    train_student(
                        b,
                        teacher,
                        samples,
                        calibration,
                        loss=kind,
                        seed=b["distillation"]["seeds"][0],
                        resume=resume,
                    )
            account()
            history = json.loads((case_directory / "history.json").read_text())
            if len(history) != 120 or history[-1]["update"] != 120:
                raise ValueError(
                    "Timing probe must complete exactly 20 warmup plus 100 measured updates"
                )
            timing = (history[-1]["training_seconds"] - history[19]["training_seconds"]) / 100
            if not 0 < timing < float("inf"):
                raise ValueError("Nonfinite or nonpositive measured update timing")
            return timing

        timings = {kind: run_case(kind) for kind in ("teacher", "mse", "sinkhorn_endpoint")}
        if model_hash(teacher) != parent_digest:
            raise AssertionError("Timing probe changed its frozen parent teacher")
        parent_ledger = account()
        proposal = propose_equal_budget(
            config,
            timings,
            spent_seconds=parent_ledger.get("gpu_training_seconds", 0.0),
            teacher_completed_updates=completed_teacher_updates,
        )
        result = {
            "warmup_updates": 20,
            "measured_updates": 100,
            "seconds_per_update": timings,
            "training_forecast_hours": proposal["full_training_forecast_seconds"] / 3600,
            "conservative_total_forecast_hours": proposal["full_total_with_spent_seconds"] / 3600,
            "forecast_assumptions": proposal["assumptions"],
            "fits_registered_budget": proposal["fits_registered_budget"],
            "budget_proposal": proposal,
            "cache_contract": contract,
            "benchmark_teacher_sha256": parent_digest,
            "teacher_source": teacher_source,
            "device": str(device),
            "final_student_seeds_used": False,
            "test_or_validation_data_used": False,
            "benchmark_config": b,
            "parent_resource_ledger": str(root / "resource_ledger.json"),
            "child_resource_ledger": str(ledger_path),
            "forecast_reference": "Snapshot after timing; recompute propose_equal_budget after any later spending",
        }
        atomic_json(destination, result)
        atomic_json(root / "benchmark.json", result)
        return result
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            account()
        except Exception as accounting_error:
            if failure is None:
                raise
            failure.add_note(f"Benchmark accounting also failed: {accounting_error}")
