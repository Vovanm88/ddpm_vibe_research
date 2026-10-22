"""Evaluate cached method-zoo samples without a hidden scalar winner.

Primary comparisons are within task and actual network NFE1/2/4. Quality,
parent fidelity and training cost remain separate. Three paired training seeds
are the statistical units; generated paths are never substituted for seeds.
The registered final bank must remain closed until ALL48 runs and every
budget-specific export have been fixed and the coordinator has given GO.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from diffusion_lab.conditional_metrics import conditional_routing_metrics
from diffusion_lab.environments.multimodal_navigation import evaluate_navigation
from recheck.evaluate import paired_statistics, sliced_wasserstein_fixed
from recheck.rays_analysis import analyse_rays
from recheck.runtime import atomic_json, tensor_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_SHA256 = "e420ea63a08389faf1d2f803f4b86437168fe703755a8e6da5a032490c511c05"
EVALUATION = {
    "rays": {"projection_count": 1024, "projection_seed": 1717, "extra_noise_seed": 945002},
    "wall": {"projection_count": 128, "projection_seed": 861003, "extra_noise_seed": 615002},
}
SOURCE_PATHS = (
    "examples/evaluate_method_zoo.py",
    "examples/train_method_zoo.py",
    "examples/run_method_zoo_evaluation.py",
    "notebooks/15_conditional_method_zoo.ipynb",
    "src/diffusion_lab/interval_model.py",
    "src/diffusion_lab/models.py",
    "src/diffusion_lab/conditional_metrics.py",
    "src/diffusion_lab/environments/multimodal_navigation.py",
    "src/diffusion_lab/environments/point_clouds.py",
    "src/recheck/evaluate.py",
    "src/recheck/rays_analysis.py",
    "src/recheck/runtime.py",
)

METHODS = (
    "cfm",
    "ot_cfm",
    "endpoint",
    "progressive",
    "reflow",
    "consistency",
    "shortcut",
    "meanflow",
)
SEEDS = (161, 262, 363)
BUDGETS = (1, 2, 4)
TASKS = ("rays", "wall")
METRICS = {
    "rays": {
        "conditional_swd": "min",
        "routing_accuracy_outer": "max",
        "routing_accuracy_all": "max",
        "correct_outer_fraction_all": "max",
        "outer_mass_gap": "min",
        "radial_w1": "min",
        "phase_distance_w1_sector": "min",
        "requested_phase_mae_outer_sector": "min",
    },
    "wall": {
        "success_fraction": "max",
        "collision_free_fraction": "max",
        "both_modes_all_conditions": "max",
        "valid_mode_balance_tv": "min",
        "conditional_swd": "min",
        "valid_path_length": "min",
    },
}
PARETO = {
    "rays": (
        ("routing_accuracy_outer", "max"),
        ("outer_mass_gap", "min"),
        ("conditional_swd", "min"),
    ),
    "wall": (
        ("success_fraction", "max"),
        ("both_modes_all_conditions", "max"),
        ("conditional_swd", "min"),
    ),
}
PRIMARY = {"rays": ("conditional_swd", "min"), "wall": ("success_fraction", "max")}


def _read(path):
    return json.loads(Path(path).read_text())


def _path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _relative(path):
    path = _path(path)
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def _file_hash(path):
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ref(path):
    return {"path": _relative(path), "sha256": _file_hash(path)}


def _verify_file_ref(record, label):
    if (
        not isinstance(record, dict)
        or not {"path", "sha256"} <= record.keys()
        or _file_hash(record["path"]) != record["sha256"]
    ):
        raise RuntimeError(f"Closed final bank: {label} file provenance mismatch")
    return _path(record["path"])


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def _training_api():
    spec = importlib.util.spec_from_file_location(
        "method_zoo_sampling_api", PROJECT_ROOT / "examples/train_method_zoo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_record(path, sha256, **metadata):
    path = _path(path)
    return {
        "path": _relative(path),
        "sha256": sha256,
        "file_sha256": _file_hash(path / "diffusion_pytorch_model.safetensors"),
        "config_file_sha256": _file_hash(path / "config.json"),
        **metadata,
    }


def _verify_native_file(record, *, task):
    directory = _path(record["path"])
    if (
        _file_hash(directory / "diffusion_pytorch_model.safetensors") != record["file_sha256"]
        or _file_hash(directory / "config.json") != record["config_file_sha256"]
    ):
        raise RuntimeError("Closed final bank: a native export changed after fixation")
    config = _read(directory / "config.json")
    if config.get("sample_shape") != ([2] if task == "rays" else [30, 2]) or config.get(
        "conditioner"
    ) != ("embedding" if task == "rays" else "mlp"):
        raise RuntimeError(
            "Closed final bank: native shape/conditioner differs from registered task"
        )
    if (task == "rays" and config.get("num_classes") != 16) or (
        task == "wall" and config.get("condition_dim") != 8
    ):
        raise RuntimeError("Closed final bank: categorical16/continuous8 conditions required")
    if config.get("_class_name") not in ("PointDenoiser", "IntervalPointDenoiser"):
        raise RuntimeError("Closed final bank: unsupported native class")


def _expected_runs():
    return {(task, method, seed) for task in TASKS for method in METHODS for seed in SEEDS}


def _expected_historical():
    return {
        (task, method, seed, nfe)
        for task in TASKS
        for method in ("M0", "E0")
        for seed in ((101, 202, 303) if task == "rays" else (161,))
        for nfe in (2, 4)
    }


def _registration(record):
    path = _verify_file_ref(record, "registration")
    if record["sha256"] != REGISTRATION_SHA256:
        raise RuntimeError("Closed final bank: pre-training registration SHA differs")
    value = _read(path)
    if (
        value.get("methods") != list(METHODS)
        or value.get("training_seeds") != list(SEEDS)
        or value.get("tasks") != list(TASKS)
        or value.get("evaluation_network_nfe") != list(BUDGETS)
        or value.get("new_method_runs") != 48
    ):
        raise RuntimeError("Closed final bank: registration cohort/budget mismatch")
    return value


def _training_configuration(gate, registration):
    config = _read(_verify_file_ref(gate.get("training_config"), "training-config"))
    expected = {
        "updates": 2000,
        "batch_size": 128,
        "lr": 0.0001,
        "warmup": 100,
        "weight_decay": 0.0001,
        "ema_decay": 0.99,
        "grad_clip": 1.0,
        "initialization": "parent",
        "teacher_steps": 32,
        "pairbank_size": 8192,
        "pairbank_seed": 816001,
        "progressive_steps": [16, 8, 4, 2, 1],
        "progressive_stage_warmup": 20,
        "consistency_step": 1 / 32,
        "consistency_huber": 0.03,
        "shortcut_fm_fraction": 0.75,
        "shortcut_widths": [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0],
        "shortcut_smallest_half": 1 / 32,
        "meanflow_fm_fraction": 0.75,
        "meanflow_logit_mean": -0.4,
        "meanflow_logit_std": 1.0,
        "meanflow_adaptive_p": 1.0,
        "meanflow_adaptive_c": 0.001,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"Closed final bank: training recipe {key} differs from preregistration"
            )
    for key, value in (("methods", METHODS), ("tasks", TASKS), ("seeds", SEEDS)):
        if config.get(key) != list(value):
            raise RuntimeError(
                "Closed final bank: resolved run config must name the complete registered cohort"
            )
    return config, hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _verify_training_sources(freeze, amendment_ref=None):
    """Allow exactly the documented GIF figure.clear/new-axis UI replacement."""
    refs = {ref["path"]: ref for ref in freeze["source_refs"]}
    notebook_path = "notebooks/15_conditional_method_zoo.ipynb"
    original = refs[notebook_path]
    training_sha = original["sha256"]
    for path, ref in refs.items():
        if path != notebook_path:
            _verify_file_ref(ref, "pre-training source")
    if _file_hash(notebook_path) == training_sha:
        return refs, training_sha
    amendment = _read(_verify_file_ref(amendment_ref, "notebook-UI-amendment"))
    old_path = _verify_file_ref(amendment.get("training_notebook"), "archived training notebook")
    new_path = _verify_file_ref(amendment.get("current_notebook"), "current UI notebook")
    if (
        amendment["training_notebook"]["sha256"] != training_sha
        or new_path != _path(notebook_path)
        or amendment.get("kind") != "ui_only_gif_colorbar_lifecycle_fix"
    ):
        raise RuntimeError(
            "Closed final bank: notebook amendment is not the frozen-training UI fix"
        )
    old, new = _read(old_path), _read(new_path)
    if len(old["cells"]) != len(new["cells"]):
        raise RuntimeError("Closed final bank: notebook UI amendment changed cell structure")
    changed = []
    for index, (before, after) in enumerate(zip(old["cells"], new["cells"])):
        tags = before.get("metadata", {}).get("tags", [])
        if before["cell_type"] != after["cell_type"] or tags != after.get("metadata", {}).get(
            "tags", []
        ):
            raise RuntimeError("Closed final bank: notebook UI amendment changed cell type/tags")
        first, second = "".join(before.get("source", [])), "".join(after.get("source", []))
        if first == second:
            continue
        if before["cell_type"] != "code" or "cached_gif_export" not in tags or "zoo-api" in tags:
            raise RuntimeError("Closed final bank: a numeric or unapproved notebook cell changed")
        old_line = "        axis.clear()"
        replacement = "        figure.clear()\n        axis = figure.add_subplot(111)"
        if first.count(old_line) != 1 or first.replace(old_line, replacement, 1) != second:
            raise RuntimeError(
                "Closed final bank: notebook UI callback changed beyond the exact axis lifecycle fix"
            )
        changed.append(index)
    if (
        len(changed) != 1
        or amendment.get("changed_cell_indices") != changed
        or amendment.get("changed_cell_tags") != ["cached_gif_export"]
        or amendment.get("zoo_api_source_sha256_before")
        != amendment.get("zoo_api_source_sha256_after")
    ):
        raise RuntimeError("Closed final bank: invalid notebook UI equivalence proof")
    refs[notebook_path] = amendment["current_notebook"]
    return refs, training_sha


def verify_final_gate(gate_path, *, verify_native=True, api=None):
    """ALL metadata/144 exports precede model loading; no data factory is called."""
    gate = _read(gate_path)
    registration = _registration(gate.get("registration"))
    config, config_digest = _training_configuration(gate, registration)
    freeze = _read(
        _verify_file_ref(gate.get("training_source_freeze"), "pre-training-source-freeze")
    )
    if (
        freeze.get("registration") != gate["registration"]
        or freeze.get("config", {}).get("resolved_sha256") != config_digest
    ):
        raise RuntimeError("Closed final bank: pre-training source/config freeze differs")
    training_refs, training_notebook_sha = _verify_training_sources(
        freeze, gate.get("notebook_ui_amendment")
    )
    frozen_config = freeze.get("config", {})
    if _file_hash(frozen_config["path"]) != frozen_config.get("file_sha256"):
        raise RuntimeError("Closed final bank: originally registered config file changed")
    if gate.get("evaluation") != EVALUATION:
        raise RuntimeError("Closed final bank: fixed projection/extra-noise protocol differs")
    sources = gate.get("source_refs", [])
    source_map = {_relative(_verify_file_ref(ref, "source")): ref["sha256"] for ref in sources}
    if not set(SOURCE_PATHS) <= set(source_map):
        raise RuntimeError("Closed final bank: incomplete source freeze")
    selection = _read(_verify_file_ref(gate.get("validation_selection"), "validation-selection"))
    if (
        selection.get("policy") != "fixed_final_EMA_no_search"
        or selection.get("registration") != gate["registration"]
        or selection.get("source_freeze") != gate["training_source_freeze"]
        or selection.get("checkpoint_selection_on_validation") is not False
        or selection.get("hyperparameter_search") is not False
    ):
        raise RuntimeError("Closed final bank: validation cannot select a better checkpoint")
    fixed_exports = _read(_verify_file_ref(gate.get("fixed_exports"), "fixed-inference-exports"))
    if (
        fixed_exports.get("policy") != "fixed_final_EMA_no_search"
        or fixed_exports.get("registration_sha256") != REGISTRATION_SHA256
    ):
        raise RuntimeError("Closed final bank: fixed export mapping differs")
    runs = gate.get("runs", [])
    keys = [(row.get("task"), row.get("method"), row.get("seed")) for row in runs]
    if len(keys) != len(set(keys)) or set(keys) != _expected_runs():
        raise RuntimeError(
            "Closed final bank: exactly ALL48 distinct Cartesian task/method/seed runs required"
        )
    run_root = _path(gate["run_root"])
    all_exports, manifests, selected_exports = [], {}, []
    for path, ref in training_refs.items():
        if source_map.get(path) != ref["sha256"]:
            raise RuntimeError(
                "Closed final bank: current source refs differ from validated training/UI sources"
            )
    notebook_sha = training_notebook_sha
    for row in runs:
        task, method, seed = row["task"], row["method"], row["seed"]
        directory = _path(row["directory"])
        if directory != run_root / task / f"seed{seed}" / method:
            raise RuntimeError(
                "Closed final bank: explicit run directory does not match its cohort key"
            )
        manifest_path = directory / "manifest.json"
        if _file_hash(manifest_path) != row["manifest_sha256"]:
            raise RuntimeError("Closed final bank: run manifest changed")
        manifest = _read(manifest_path)
        expected = {
            "task": task,
            "method": method,
            "seed": seed,
            "status": "completed",
            "updates_planned": 2000,
            "updates_completed": 2000,
            "source_sha256": notebook_sha,
            "teacher_sha256": registration["parents"][task]["sha256"],
            "common_ancestor_sha256": registration["parents"][task]["sha256"],
            "initialization": "parent",
            "cfg": False,
            "condition_dropout": 0.0,
            "whole_support_single_condition": True,
            "route_id_input": False,
            "endpoint_clamped_contract": task == "wall",
            "state_dimension": 2 if task == "rays" else 60,
            "config_sha256": config_digest,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"Closed final bank: {task}/{method}/{seed} incomplete or mismatched {key}"
                )
        expected_kind = "interval" if method in ("shortcut", "meanflow") else "velocity"
        expected_class = "IntervalPointDenoiser" if expected_kind == "interval" else "PointDenoiser"
        if (
            manifest.get("model_kind") != expected_kind
            or manifest.get("model_class") != expected_class
        ):
            raise RuntimeError("Closed final bank: wrong registered native model class")
        for key in ("inference_exports",):
            if set(row.get(key, {})) != {"1", "2", "4"} or set(manifest.get(key, {})) != {
                "1",
                "2",
                "4",
            }:
                raise RuntimeError(
                    "Closed final bank: each run needs ALL three fixed inference exports"
                )
        export_paths = []
        for nfe in BUDGETS:
            export, declared = (
                row["inference_exports"][str(nfe)],
                manifest["inference_exports"][str(nfe)],
            )
            native_path = (directory / declared["path"]).resolve()
            if not native_path.is_relative_to(directory) or _path(export["path"]) != native_path:
                raise RuntimeError(
                    "Closed final bank: inference export points outside its registered run"
                )
            for key in (
                "sha256",
                "trained_nfe",
                "native_or_off_target",
                "cumulative_updates_at_export",
            ):
                if export.get(key) != declared.get(key):
                    raise RuntimeError(
                        "Closed final bank: inference export metadata differs from manifest"
                    )
            expected_updates = {4: 1200, 2: 1600, 1: 2000}[nfe] if method == "progressive" else 2000
            trained_nfe = (
                nfe
                if method == "progressive"
                else (1 if method in ("endpoint", "consistency") else None)
            )
            native_kind = "off_target" if method == "endpoint" and nfe > 1 else "native"
            if (
                export.get("cumulative_updates_at_export") != expected_updates
                or export.get("trained_nfe") != trained_nfe
                or export.get("native_or_off_target") != native_kind
            ):
                raise RuntimeError(
                    "Closed final bank: stage/native training budget differs from preregistration"
                )
            if method == "progressive":
                stage = _read(native_path.parent / "manifest.json")
                if (
                    stage.get("status") != "completed"
                    or stage.get("ema_sha256") != export["sha256"]
                    or stage.get("trained_nfe") != nfe
                    or stage.get("cumulative_updates_at_export") != expected_updates
                ):
                    raise RuntimeError(
                        "Closed final bank: progressive stage incomplete or mismatched"
                    )
            _verify_native_file(export, task=task)
            if _read(native_path / "config.json").get("_class_name") != expected_class:
                raise RuntimeError("Closed final bank: export class differs from registered method")
            all_exports.append(export)
            export_paths.append(native_path)
            selected_exports.append(
                {
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "nfe": nfe,
                    "path": export["path"],
                    "sha256": export["sha256"],
                }
            )
        if method == "progressive" and len(set(export_paths)) != 3:
            raise RuntimeError(
                "Closed final bank: progressive requires distinct native stage4/2/1 exports"
            )
        checkpoint_path = directory / "training.pt"
        if _file_hash(checkpoint_path) != row["checkpoint_file_sha256"]:
            raise RuntimeError("Closed final bank: final training checkpoint changed")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            saved.get("status") != "completed"
            or saved.get("completed") != 2000
            or _state_hash(saved["ema"]) != manifest.get("ema_sha256")
        ):
            raise RuntimeError("Closed final bank: checkpoint actual final EMA/updates differ")
        contract = saved.get("contract", {})
        for key in (
            "updates",
            "batch_size",
            "lr",
            "warmup",
            "weight_decay",
            "ema_decay",
            "grad_clip",
            "initialization",
            "teacher_steps",
            "pairbank_size",
            "pairbank_seed",
            "progressive_stage_warmup",
        ):
            if contract.get("config", {}).get(key) != config[key]:
                raise RuntimeError("Closed final bank: actual checkpoint training recipe differs")
        for key in (
            "task",
            "method",
            "seed",
            "teacher_sha256",
            "source_sha256",
            "train_data_sha256",
            "train_state_sha256",
            "train_conditions_sha256",
            "pairbank_sha256",
        ):
            if contract.get(key) != manifest.get(key):
                raise RuntimeError(
                    "Closed final bank: training checkpoint contract differs from manifest"
                )
        manifests[(task, method, seed)] = manifest

    def selected_order(row):
        return row["task"], row["method"], row["seed"], row["nfe"]

    if sorted(fixed_exports.get("chosen_exports", []), key=selected_order) != sorted(
        selected_exports, key=selected_order
    ):
        raise RuntimeError(
            "Closed final bank: immutable fixed-EMA selection record differs from ALL144 exports"
        )
    for task in TASKS:
        task_manifests = [manifest for (kind, _, _), manifest in manifests.items() if kind == task]
        for key in (
            "config_sha256",
            "train_data_sha256",
            "train_state_sha256",
            "train_conditions_sha256",
            "pairbank_sha256",
        ):
            if (
                len({manifest.get(key) for manifest in task_manifests}) != 1
                or task_manifests[0].get(key) is None
            ):
                raise RuntimeError(f"Closed final bank: shared {task} training {key} differs")
        pair_record = freeze.get("pairbanks", {}).get(task, {})
        if _file_hash(pair_record["path"]) != pair_record.get("file_sha256"):
            raise RuntimeError("Closed final bank: shared train-only pairbank manifest changed")
        pairbank = _read(_path(pair_record["path"]))
        if (
            pairbank.get("status") != "completed"
            or pairbank.get("quality_filtering") is not False
            or pairbank.get("contract", {}).get("split") != "train"
            or pairbank.get("sha256") != pair_record.get("tensor_contract_sha256")
            or pairbank.get("sha256") != task_manifests[0]["pairbank_sha256"]
        ):
            raise RuntimeError("Closed final bank: wrong or quality-filtered teacher pairbank")
        for seed in SEEDS:
            for method in METHODS:
                if (
                    method not in ("shortcut", "meanflow")
                    and manifests[(task, method, seed)].get("init_sha256")
                    != registration["parents"][task]["sha256"]
                ):
                    raise RuntimeError(
                        "Closed final bank: velocity methods must start from the same actual parent"
                    )
            if manifests[(task, "shortcut", seed)].get("init_sha256") != manifests[
                (task, "meanflow", seed)
            ].get("init_sha256"):
                raise RuntimeError(
                    "Closed final bank: paired interval methods require the same neutral branch initialization"
                )
            if (
                len(
                    {
                        manifests[(task, method, seed)]["raw_batch_trace_sha256"]
                        for method in ("cfm", "ot_cfm", "shortcut", "meanflow")
                    }
                )
                != 1
            ):
                raise RuntimeError(
                    "Closed final bank: comparable data-method raw batch traces differ"
                )
            if (
                len(
                    {
                        manifests[(task, method, seed)]["pair_indices_trace_sha256"]
                        for method in ("endpoint", "reflow")
                    }
                )
                != 1
            ):
                raise RuntimeError(
                    "Closed final bank: comparable endpoint/reflow pair traces differ"
                )
    parents = gate.get("parents", {})
    if set(parents) != set(TASKS):
        raise RuntimeError("Closed final bank: both common parents required")
    for task, parent in parents.items():
        if parent["sha256"] != registration["parents"][task]["sha256"]:
            raise RuntimeError("Closed final bank: common parent tensor SHA differs")
        _verify_native_file(parent, task=task)
        all_exports.append(parent)
    historical = gate.get("historical", [])
    history_keys = [
        (row.get("task"), row.get("method"), row.get("seed"), row.get("nfe")) for row in historical
    ]
    if len(history_keys) != len(set(history_keys)) or set(history_keys) != _expected_historical():
        raise RuntimeError("Closed final bank: all16 historical native controls must be fixed too")
    for row in historical:
        if row.get("updates") != 1000:
            raise RuntimeError(
                "Closed final bank: historical controls have unequal disclosed1000-update budget"
            )
        _verify_native_file(row, task=row["task"])
        all_exports.append(row)
    if verify_native:
        api = _training_api() if api is None else api
        seen = set()
        for export in all_exports:
            key = (str(_path(export["path"])), export["sha256"])
            if key in seen:
                continue
            model = api.load_zoo_model(_path(export["path"]), device="cpu")
            if _state_hash(model.state_dict()) != export["sha256"]:
                raise RuntimeError("Closed final bank: actual native tensor SHA differs")
            seen.add(key)
            del model
    return {
        "gate": gate,
        "registration": registration,
        "training_config": config,
        "manifests": manifests,
        "all48_metadata_fixed": True,
        "all144_exports_fixed": True,
        "actual_native_hashes_verified": verify_native,
    }


def _save_immutable(path, payload):
    path = _path(path)
    if path.exists():
        if _read(path) != payload:
            raise FileExistsError(f"Preserve the existing frozen record: {path}")
    else:
        atomic_json(path, payload)


def build_final_gate(run_root, registration_path, output_path):
    """Build explicit ALL48/144 fixation, without creating any evaluation bank.

    Registration is never reset or amended here. The selection policy is the
    predeclared final EMA/matched progressive stage, without quality inspection.
    Incomplete cohorts fail before writing a selection or loading native models.
    """
    run_root, registration_path, output_path = (
        _path(run_root),
        _path(registration_path),
        _path(output_path),
    )
    registration_ref = _file_ref(registration_path)
    registration = _registration(registration_ref)
    manifests = {}
    for task, method, seed in sorted(_expected_runs()):
        directory = run_root / task / f"seed{seed}" / method
        manifest = _read(directory / "manifest.json")
        if (
            manifest.get("status") != "completed"
            or manifest.get("updates_completed") != 2000
            or manifest.get("updates_planned") != 2000
        ):
            raise RuntimeError(
                f"Closed final bank: all48 must complete2000 before fixation ({task}/{method}/{seed})"
            )
        manifests[(task, method, seed)] = manifest
    runs, chosen = [], []
    for (task, method, seed), manifest in sorted(manifests.items()):
        directory = run_root / task / f"seed{seed}" / method
        exports = {}
        for nfe in BUDGETS:
            declared = manifest["inference_exports"][str(nfe)]
            exports[str(nfe)] = _native_record(
                directory / declared["path"],
                declared["sha256"],
                **{
                    key: declared[key]
                    for key in (
                        "trained_nfe",
                        "native_or_off_target",
                        "cumulative_updates_at_export",
                    )
                },
            )
            chosen.append(
                {
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "nfe": nfe,
                    "path": exports[str(nfe)]["path"],
                    "sha256": declared["sha256"],
                }
            )
        runs.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "directory": _relative(directory),
                "manifest_sha256": _file_hash(directory / "manifest.json"),
                "checkpoint_file_sha256": _file_hash(directory / "training.pt"),
                "inference_exports": exports,
            }
        )
    selection_path = output_path.with_name("validation_selection.json")
    fixed_path = output_path.with_name("fixed_inference_exports.json")
    fixed_exports = {
        "policy": "fixed_final_EMA_no_search",
        "registration_sha256": REGISTRATION_SHA256,
        "validation_used_for_checkpoint_search": False,
        "chosen_exports": chosen,
    }
    parents = {
        task: _native_record(value["native_path"], value["sha256"])
        for task, value in registration["parents"].items()
    }
    historical = []
    for task, method, seed, nfe in sorted(_expected_historical()):
        if task == "rays":
            directory = (
                PROJECT_ROOT
                / f"replication/demo/rays16_conditional_low_nfe_v1/nfe{nfe}/seed{seed}/{method}"
            )
        else:
            directory = (
                PROJECT_ROOT
                / f"replication/demo/wall_navigation_conditional_v1/students_continuation_nfe{nfe}/{method}"
            )
        manifest = _read(directory / "manifest.json")
        if (
            manifest.get("status") != "completed"
            or manifest.get("updates_completed") != 1000
            or manifest.get("teacher_sha256") != parents[task]["sha256"]
        ):
            raise RuntimeError("Historical controls must be completed1000 with the same parent")
        historical.append(
            _native_record(
                directory / "ema_model",
                manifest["ema_sha256"],
                task=task,
                method=method,
                seed=seed,
                nfe=nfe,
                updates=1000,
            )
        )
    freeze_path = output_path.with_name("source_freeze.json")
    freeze = _read(freeze_path)
    original_notebook = next(
        ref
        for ref in freeze["source_refs"]
        if ref["path"] == "notebooks/15_conditional_method_zoo.ipynb"
    )
    amendment_ref = (
        _file_ref(output_path.with_name("notebook_ui_amendment.json"))
        if _file_hash(original_notebook["path"]) != original_notebook["sha256"]
        else None
    )
    source_refs, _ = _verify_training_sources(freeze, amendment_ref)
    for path in SOURCE_PATHS:
        if path not in source_refs:
            source_refs[path] = _file_ref(path)
    _save_immutable(fixed_path, fixed_exports)
    gate = {
        "schema_version": 1,
        "scope": "all48_fixed_before_any_new_final_bank",
        "registration": registration_ref,
        "training_config": _file_ref(run_root / "run_config.json"),
        "run_root": _relative(run_root),
        "source_refs": [source_refs[path] for path in sorted(source_refs)],
        "training_source_freeze": _file_ref(freeze_path),
        "runs": runs,
        "parents": parents,
        "historical": historical,
        "evaluation": EVALUATION,
        "validation_selection": _file_ref(selection_path),
        "fixed_exports": _file_ref(fixed_path),
    }
    if amendment_ref is not None:
        gate["notebook_ui_amendment"] = amendment_ref
    # atomic_json's serialization owns the actual digest, so do not infer it
    # from another serializer. The immutable selection is written once.
    pending = output_path.with_suffix(output_path.suffix + ".pending")
    if pending.exists():
        raise FileExistsError(f"Review an existing unfinished gate attempt: {pending}")
    atomic_json(pending, gate)
    try:
        verify_final_gate(pending, verify_native=True)
        _save_immutable(output_path, gate)
    finally:
        pending.unlink(missing_ok=True)
    return gate


def _cpu(value):
    # Separate transport/cast also preserves actual MPS values on affected
    # Torch builds; never combine device='cpu' and dtype=float64 in one .to.
    return torch.as_tensor(value).detach().cpu()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _inputs(samples, conditions, reference, reference_conditions):
    samples, conditions = _cpu(samples), _cpu(conditions)
    reference, reference_conditions = _cpu(reference), _cpu(reference_conditions)
    if (
        len(samples) == 0
        or len(reference) == 0
        or len(samples) != len(conditions)
        or len(reference) != len(reference_conditions)
    ):
        raise ValueError("Nonempty samples/reference must align with their own conditions")
    if not torch.isfinite(samples).all() or not torch.isfinite(reference).all():
        raise ValueError(
            "Nonfinite outputs must be recorded as failed runs, never silently dropped"
        )
    return samples, conditions, reference, reference_conditions


def conditional_swd(samples, conditions, reference, reference_conditions, projections):
    """Raw-space sliced W1 per EXACT condition, preserving condition semantics."""
    samples, conditions, reference, reference_conditions = _inputs(
        samples, conditions, reference, reference_conditions
    )
    if samples.shape[1:] != reference.shape[1:]:
        raise ValueError("Generated/reference state shapes differ")
    if (
        conditions.ndim != reference_conditions.ndim
        or conditions.shape[1:] != reference_conditions.shape[1:]
    ):
        raise ValueError("Condition layouts differ")
    categories = conditions[:, None] if conditions.ndim == 1 else conditions
    real_categories = (
        reference_conditions[:, None] if reference_conditions.ndim == 1 else reference_conditions
    )
    unique = torch.unique(categories, dim=0, sorted=True)
    if not torch.equal(unique, torch.unique(real_categories, dim=0, sorted=True)):
        raise ValueError("Every requested condition needs an independent real reference")
    values = []
    for condition in unique:
        selected = (categories == condition).all(1)
        real_selected = (real_categories == condition).all(1)
        values.append(
            sliced_wasserstein_fixed(
                samples[selected].flatten(1), reference[real_selected].flatten(1), projections
            )
        )
    return {
        "mean": float(np.mean(values)),
        "per_condition": values,
        "conditions": unique.tolist(),
        "marginal": sliced_wasserstein_fixed(samples.flatten(1), reference.flatten(1), projections),
        "projection_bank_sha256": tensor_hash(projections),
        "limitation": "Marginal mixing can hide condition swaps; the per-condition mean is primary.",
    }


def rays_quality(
    samples,
    targets,
    reference,
    reference_targets,
    projections,
    *,
    initial_noise=None,
    n_classes=16,
    radius=10 / 11,
):
    """Requested routing, real-relative geometry and conditional2D distribution."""
    samples, targets, reference, reference_targets = _inputs(
        samples, targets, reference, reference_targets
    )
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("Rays samples must be[N,2]")
    requested = torch.unique(targets, sorted=True)
    if (
        targets.ndim != 1
        or targets.dtype not in (torch.int32, torch.int64)
        or not torch.equal(requested, torch.arange(n_classes))
    ):
        raise ValueError("Every categorical ray target must be requested explicitly")
    routing = conditional_routing_metrics(
        samples, targets, initial_noise, n_classes=n_classes, radius=radius
    )
    real_routing = conditional_routing_metrics(
        reference, reference_targets, n_classes=n_classes, radius=radius
    )
    geometry = analyse_rays(samples, reference, n_rays=n_classes, radius=radius)
    distribution = conditional_swd(samples, targets, reference, reference_targets, projections)
    # Origin collapse yields undefined outer accuracy, not a perfect phase
    # score. Pareto/rank functions explicitly exclude incomplete objectives.
    flat = {
        "conditional_swd": distribution["mean"],
        "routing_accuracy_outer": routing["accuracy_outer"],
        "routing_accuracy_all": routing["accuracy_all"],
        "correct_outer_fraction_all": routing["correct_outer_fraction_of_all"],
        "outer_coverage": routing["outer_coverage"],
        "real_outer_coverage": real_routing["outer_coverage"],
        "outer_mass_gap": abs(routing["outer_coverage"] - real_routing["outer_coverage"]),
        "radial_w1": geometry["metrics"]["radial_w1_raw"],
        "phase_distance_w1_sector": geometry["metrics"][
            "absolute_phase_distance_w1_sector_fraction"
        ],
        "requested_phase_mae_outer_sector": routing["requested_phase_error"][
            "mean_absolute_sector_fraction_outer"
        ],
        "outside_nominal_radius_fraction": geometry["generated"]["radial"][
            "outside_nominal_radius_fraction"
        ],
    }
    return {
        "flat": flat,
        "routing": routing,
        "real_routing": real_routing,
        "geometry": geometry,
        "distribution": distribution,
        "finite_fraction": 1.0,
    }


def wall_quality(
    samples, conditions, reference, reference_conditions, projections, *, margin=0.035
):
    """Exact full-segment geometry; distribution uses ALL raw interior paths."""
    samples, conditions, reference, reference_conditions = _inputs(
        samples, conditions, reference, reference_conditions
    )
    quality = evaluate_navigation(samples, conditions, margin=margin)
    distribution = conditional_swd(
        samples[:, 1:-1], conditions, reference[:, 1:-1], reference_conditions, projections
    )
    flat = {
        "success_fraction": quality["success_fraction"],
        "collision_free_fraction": quality["collision_free_fraction"],
        "both_modes_all_conditions": quality["mode_coverage"]["both_modes_fraction_all_conditions"],
        "valid_mode_balance_tv": quality["mode_coverage"][
            "mean_mode_balance_tv_successful_assigned"
        ],
        "conditional_swd": distribution["mean"],
        "valid_path_length": quality["mean_path_length_successful"],
    }
    per_sample = quality.pop("per_sample")
    return {
        "flat": flat,
        "geometry": quality,
        "distribution": distribution,
        "finite_fraction": 1.0,
        "per_sample": per_sample,
        "endpoint_contract": "Endpoints reconstructed from A/B; not evidence of learned endpoint accuracy.",
        "pathlength_limitation": "Length is only reported for valid successful routes; never ranked alone as safety.",
    }


@contextmanager
def count_network_evaluations(model):
    """Count actual sample-evaluations, including doubled CFG batches.

    One forward on2B is2 NFE per original sample, not1. Model-call totals
    separately reflect inference batching and do not stand in for per-sample NFE.
    """
    counts = {"model_calls": 0, "sample_evaluations": 0, "batch_sizes": []}

    def before(module, args, kwargs):
        sample = args[0] if args else kwargs.get("sample")
        if not torch.is_tensor(sample) or sample.ndim < 2:
            raise ValueError("Cannot verify network NFE without the actual sample batch")
        counts["model_calls"] += 1
        counts["sample_evaluations"] += len(sample)
        counts["batch_sizes"].append(len(sample))

    handle = model.register_forward_pre_hook(before, with_kwargs=True)
    try:
        yield counts
    finally:
        handle.remove()


def verified_sample(
    sample_fn, model, noise, conditions, *, nfe, batch_size=128, extra_noise=None, reference=False
):
    """Call the public sampler and independently verify actual network cost."""
    if nfe not in BUDGETS and not (reference and nfe in (32, 64)):
        raise ValueError("Primary method-zoo NFE must be1/2/4")
    initial_before = tensor_hash(noise)
    conditions_before = tensor_hash(conditions)
    kwargs = {"nfe": nfe, "batch_size": batch_size, "record_history": True}
    if extra_noise is not None:
        kwargs["extra_noise"] = extra_noise
    with count_network_evaluations(model) as counts:
        result = sample_fn(model, noise, conditions, **kwargs)
    if counts["sample_evaluations"] != len(noise) * nfe:
        raise RuntimeError(
            f"Actual network sample-evaluations {counts['sample_evaluations']} differ from registered {len(noise) * nfe}"
        )
    if result.get("nfe") != nfe:
        raise RuntimeError("Sampler-reported NFE differs from independently measured budget")
    if tensor_hash(noise) != initial_before or tensor_hash(conditions) != conditions_before:
        raise RuntimeError("Sampler mutated the shared initial noise/condition bank")
    samples = _cpu(result["samples"])
    if len(samples) != len(noise) or not torch.isfinite(samples).all():
        raise FloatingPointError("Sampler returned nonfinite or incomplete final states")
    result["actual_network_cost"] = {
        "nfe_per_sample": counts["sample_evaluations"] / len(noise),
        "total_model_calls": counts["model_calls"],
        "total_sample_evaluations": counts["sample_evaluations"],
    }
    return result


def sampling_events(result, *, method):
    """Preserve actual CM input→output0→renoised input times without smoothing."""
    if method == "consistency":
        inputs = _cpu(result["actual_inputs"])
        outputs = _cpu(result["postcall_outputs"])
        input_times = _cpu(result["input_times"]).double()
        output_times = _cpu(result["output_times"]).double()
        if (
            inputs.shape != outputs.shape
            or len(inputs) != result["nfe"]
            or input_times.shape != (len(inputs),)
            or output_times.shape != input_times.shape
        ):
            raise ValueError("Consistency sampling must expose every actual network input/output")
        if not (input_times > 0).all() or not torch.equal(
            output_times, torch.zeros_like(output_times)
        ):
            raise ValueError("Consistency model calls denoise positive-time inputs to t0")
        events = torch.stack((inputs, outputs), dim=1).flatten(0, 1)
        times = torch.stack((input_times, output_times), dim=1).flatten()
        kinds = [kind for _ in inputs for kind in ("network_input_or_renoise", "denoised_output")]
        calls = [count for index in range(len(inputs)) for count in (index, index + 1)]
    else:
        events = _cpu(result["history"])
        times = _cpu(result["times"]).double()
        if (
            len(events) != result["nfe"] + 1
            or times.shape != (len(events),)
            or not (times[:-1] >= times[1:]).all()
        ):
            raise ValueError("Euler history must expose its actual decreasing-time solver nodes")
        kinds = ["initial_input"] + ["euler_update"] * (len(events) - 1)
        calls = list(range(len(events)))
    if (
        not torch.isfinite(events).all()
        or not torch.isfinite(times).all()
        or not torch.equal(events[-1], _cpu(result["samples"]))
    ):
        raise ValueError("Actual event history must be finite and end in the measured raw samples")
    return {
        "states": events,
        "times": times,
        "kinds": kinds,
        "network_calls_used": calls,
        "time_caveat": "CM denoise/re-noise event times are nonmonotonic; these are sampling events, never physical particle time.",
    }


def pareto_layers(rows, objectives):
    """Descriptive point-estimate fronts; undefined metrics never become winners."""
    remaining, result = [], {}
    for row in rows:
        values = [row["metrics"].get(key) for key, _ in objectives]
        if row.get("status", "completed") != "completed" or not all(_finite(v) for v in values):
            result[row["method"]] = None
        else:
            remaining.append(
                (
                    row["method"],
                    np.array(
                        [
                            value if direction == "min" else -value
                            for value, (_, direction) in zip(values, objectives)
                        ]
                    ),
                )
            )
    layer = 1
    while remaining:
        front = []
        for name, values in remaining:
            dominated = any(
                other != name and np.all(other_values <= values) and np.any(other_values < values)
                for other, other_values in remaining
            )
            if not dominated:
                front.append(name)
        for name in front:
            result[name] = layer
        remaining = [(name, values) for name, values in remaining if name not in front]
        layer += 1
    return result


def aggregate_rankings(rows, *, expected_seeds=SEEDS):
    """Pair all NEW methods by training seed; keep historical/baselines separate."""
    summaries, contrasts = [], []
    primary = [row for row in rows if row.get("league") == "primary"]
    for task in TASKS:
        for nfe in BUDGETS:
            group = [row for row in primary if row["task"] == task and row["nfe"] == nfe]
            method_rows, metric_arrays = [], {}
            for method in METHODS:
                selected = [row for row in group if row["method"] == method]
                by_seed = {row["seed"]: row for row in selected}
                complete = (
                    len(by_seed) == len(selected) == len(expected_seeds)
                    and set(by_seed) == set(expected_seeds)
                    and all(row.get("status") == "completed" for row in selected)
                )
                metrics, statistics = {}, {}
                for key in METRICS[task]:
                    values = [
                        by_seed.get(seed, {}).get("quality", {}).get("flat", {}).get(key)
                        for seed in expected_seeds
                    ]
                    valid = complete and all(_finite(value) for value in values)
                    statistics[key] = (
                        paired_statistics(values)
                        if valid
                        else {
                            "n_student_seeds": sum(_finite(v) for v in values),
                            "values": values,
                            "mean": None,
                            "sample_std": None,
                            "paired_t_ci": None,
                            "status": "incomplete_or_undefined; never drop a seed",
                        }
                    )
                    metrics[key] = statistics[key]["mean"]
                    metric_arrays[(method, key)] = values if valid else None
                method_rows.append(
                    {
                        "task": task,
                        "nfe": nfe,
                        "method": method,
                        "status": "completed" if complete else "incomplete",
                        "metrics": metrics,
                        "seed_statistics": statistics,
                    }
                )
            layers = pareto_layers(method_rows, PARETO[task])
            metric, direction = PRIMARY[task]
            for summary in method_rows:
                value = summary["metrics"][metric]
                summary["pareto_layer"] = layers[summary["method"]]
                summary["primary_metric"] = metric
                summary["primary_point_rank"] = (
                    (
                        1
                        + sum(
                            _finite(other["metrics"][metric])
                            and (
                                (other["metrics"][metric] < value)
                                if direction == "min"
                                else (other["metrics"][metric] > value)
                            )
                            for other in method_rows
                        )
                    )
                    if _finite(value)
                    and summary["status"] == "completed"
                    and summary["pareto_layer"] is not None
                    else None
                )
                summaries.append(summary)
            for method in METHODS:
                if method == "cfm":
                    continue
                for key, direction in METRICS[task].items():
                    first, second = metric_arrays[(method, key)], metric_arrays[("cfm", key)]
                    if first is not None and second is not None:
                        contrasts.append(
                            {
                                "task": task,
                                "nfe": nfe,
                                "method": method,
                                "control": "cfm",
                                "metric": key,
                                "direction": direction,
                                "training_seeds": list(expected_seeds),
                                **paired_statistics(first, second),
                            }
                        )
    return {
        "summaries": summaries,
        "paired_vs_warm_cfm": contrasts,
        "universal_winner": None,
        "ranking_scope": "Descriptive per-task, fixed ACTUAL network NFE; no weighted task/metric score.",
        "uncertainty": "Pointwise approximate paired t95 intervals across3 independent training seeds, conditional on one common parent and fixed shared banks. Not familywise-adjusted; n3 is exploratory, not universal superiority evidence.",
        "historical_policy": "Historical1000-update models and frozen parent baselines are not included in the2000-update primary league or paired new-vs-old seed contrasts.",
    }


def reference_summaries(rows):
    """Historical/parent descriptive means, never paired to new training seeds."""
    groups = {}
    for row in rows:
        if row.get("league") == "primary":
            continue
        key = (row["task"], row["nfe"], row["method"], row["league"])
        groups.setdefault(key, []).append(row)
    results = []
    for (task, nfe, method, league), group in groups.items():
        metrics = {}
        for key in METRICS[task]:
            values = [row.get("quality", {}).get("flat", {}).get(key) for row in group]
            metrics[key] = (
                float(np.mean(values)) if all(_finite(value) for value in values) else None
            )
        results.append(
            {
                "task": task,
                "nfe": nfe,
                "method": method,
                "league": league,
                "training_seed_count": len(group),
                "training_seeds": [row.get("seed") for row in group],
                "training_updates": 1000 if league == "historical_reference" else 20000,
                "metrics": metrics,
                "uncertainty": "Descriptive historical/reference league; not a paired new-vs-old contrast.",
            }
        )
    return results


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def save_ranking_artifacts(rows, output):
    """Compact public tables/plots; raw per-sample arrays belong in private cache."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ranking = aggregate_rankings(rows)
    ranking["references"] = reference_summaries(rows)
    (output / "ranking.json").write_text(json.dumps(ranking, indent=2, allow_nan=False) + "\n")
    summary_rows = []
    for row in ranking["summaries"]:
        item = {
            key: row[key]
            for key in (
                "task",
                "nfe",
                "method",
                "status",
                "pareto_layer",
                "primary_metric",
                "primary_point_rank",
            )
        }
        for metric, stats in row["seed_statistics"].items():
            item[metric] = stats["mean"]
            item[f"{metric}_seed_std"] = stats["sample_std"]
            item[f"{metric}_t95"] = stats["paired_t_ci"]
        summary_rows.append(item)
    _write_csv(output / "primary_summary.csv", summary_rows)
    _write_csv(output / "paired_vs_warm_cfm.csv", ranking["paired_vs_warm_cfm"])
    _write_csv(
        output / "historical_and_parent.csv",
        [
            {**{key: value for key, value in row.items() if key != "metrics"}, **row["metrics"]}
            for row in ranking["references"]
        ],
    )
    _write_csv(
        output / "per_seed.csv",
        [
            {
                **{
                    key: row.get(key)
                    for key in (
                        "task",
                        "method",
                        "seed",
                        "nfe",
                        "league",
                        "status",
                        "training_updates",
                        "export_cumulative_updates",
                    )
                },
                **row.get("quality", {}).get("flat", {}),
            }
            for row in rows
        ],
    )
    plot_quality_pareto(ranking, output)
    plot_paired_intervals(ranking, output)
    return ranking


def plot_quality_pareto(ranking, output):
    """Per-task/budget quality fronts with historical controls labeled by budget."""
    import matplotlib.pyplot as plt

    output = Path(output)
    palette = {method: plt.get_cmap("tab10")(index) for index, method in enumerate(METHODS)}
    for task in TASKS:
        figure, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained", squeeze=False)
        for axis, nfe in zip(axes.flat, BUDGETS):
            selected = [
                row for row in ranking["summaries"] if row["task"] == task and row["nfe"] == nfe
            ]
            key = "routing_accuracy_outer" if task == "rays" else "success_fraction"
            for row in selected:
                x, y = row["metrics"]["conditional_swd"], row["metrics"][key]
                if not _finite(x) or not _finite(y):
                    continue
                axis.scatter(
                    x,
                    y,
                    c=[palette[row["method"]]],
                    marker="o",
                    s=60,
                    edgecolors="black" if row["pareto_layer"] == 1 else "none",
                    lw=1.2,
                    label=f"{row['method']} ·2000 updates",
                )
                axis.annotate(
                    row["method"], (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points"
                )
            for row in ranking.get("references", []):
                if row["task"] != task or row["nfe"] != nfe:
                    continue
                x, y = row["metrics"]["conditional_swd"], row["metrics"][key]
                if not _finite(x) or not _finite(y):
                    continue
                historical = row["league"] == "historical_reference"
                label = f"{row['method']} ·{row['training_updates']} updates"
                axis.scatter(
                    x,
                    y,
                    marker="D" if historical else "*",
                    s=60 if historical else 130,
                    facecolors="none" if historical else "black",
                    edgecolors="#64748b" if historical else "black",
                    label=label,
                )
                axis.annotate(
                    label,
                    (x, y),
                    fontsize=7,
                    xytext=(4, -11),
                    textcoords="offset points",
                    color="#475569",
                )
            axis.set_title(f"Actual network NFE {nfe}")
            axis.set_xlabel("Mean conditional raw-space SWD ↓")
            axis.set_ylabel(
                "Outer requested-ray accuracy ↑"
                if task == "rays"
                else "Exact all-segment success fraction ↑"
            )
            axis.set_ylim(-0.03, 1.03)
            axis.grid(alpha=0.2)
        figure.suptitle(
            f"{task} · NEW three-seed means; hollow diamonds are historical1000-update controls\nBlack outlines indicate point-estimate Pareto layer1; coverage also enters rays Pareto, not this2D view",
            fontsize=11,
        )
        for extension in ("png", "svg"):
            figure.savefig(output / f"{task}_quality_pareto.{extension}", dpi=150)
        plt.close(figure)
        figure, axes = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
        for axis, field, title in zip(
            axes,
            ("primary_point_rank", "pareto_layer"),
            ("Primary-metric point rank", "Multi-metric Pareto layer"),
        ):
            grid = np.full((len(METHODS), len(BUDGETS)), np.nan)
            for row in ranking["summaries"]:
                if row["task"] == task and row[field] is not None:
                    grid[METHODS.index(row["method"]), BUDGETS.index(row["nfe"])] = row[field]
            axis.imshow(
                np.ma.masked_invalid(grid), cmap="Blues_r", vmin=1, vmax=len(METHODS), aspect="auto"
            )
            for (r, c), value in np.ndenumerate(grid):
                axis.text(
                    c,
                    r,
                    "—" if not math.isfinite(value) else str(int(value)),
                    ha="center",
                    va="center",
                    color="black" if not math.isfinite(value) or value > 3 else "white",
                )
            axis.set_xticks(range(len(BUDGETS)), [str(nfe) for nfe in BUDGETS])
            axis.set_yticks(range(len(METHODS)), METHODS)
            axis.set(xlabel="Actual network NFE", title=title)
        figure.suptitle(
            f"{task}: descriptive ranks, not statistical proof or a universal winner", fontsize=11
        )
        for extension in ("png", "svg"):
            figure.savefig(output / f"{task}_ranks.{extension}", dpi=150)
        plt.close(figure)


def plot_paired_intervals(ranking, output):
    """Paired three-training-seed differences, preserving zero and wide CIs."""
    import matplotlib.pyplot as plt

    output = Path(output)
    for task in TASKS:
        metric = PRIMARY[task][0]
        figure, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained", squeeze=False)
        for axis, nfe in zip(axes.flat, BUDGETS):
            selected = [
                row
                for row in ranking["paired_vs_warm_cfm"]
                if row["task"] == task and row["nfe"] == nfe and row["metric"] == metric
            ]
            for index, row in enumerate(selected):
                lo, hi = row["paired_t_ci"]
                axis.plot([lo, hi], [index, index], color="#475569", lw=1.4)
                axis.scatter([row["mean"]], [index], color="#2563eb", s=35)
            axis.axvline(0, color="black", lw=1, alpha=0.6)
            axis.set_yticks(range(len(selected)), [row["method"] for row in selected])
            axis.set(title=f"Actual NFE {nfe}", xlabel=f"{metric}: method − warm CFM")
            axis.grid(axis="x", alpha=0.2)
        figure.suptitle(
            f"{task} · paired t95 over3 TRAINING seeds\nPointwise exploratory intervals; no multiplicity adjustment or generated-sample pseudo-replication",
            fontsize=11,
        )
        for extension in ("png", "svg"):
            figure.savefig(output / f"{task}_paired_intervals.{extension}", dpi=150)
        plt.close(figure)


def _preview_payload(value):
    return (
        torch.load(value, map_location="cpu", weights_only=True)
        if isinstance(value, (str, Path))
        else value
    )


def plot_method_zoo_preview(
    payload, *, nfe=2, frame_index=None, noise_index=0, condition_id=0, method=None
):
    """Cached actual events only; CM re-noise clocks remain explicitly nonmonotone."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from diffusion_lab.navigation_visualization import _RAINBOW
    from diffusion_lab.visualization.plots import draw_walls

    payload = _preview_payload(payload)
    task = payload["task"]
    conditions = _cpu(payload["conditions"])
    ids = _cpu(payload["condition_ids"])
    selected = torch.nonzero(ids == condition_id).flatten()
    if len(selected) == 0 or not 0 <= noise_index < len(selected):
        raise ValueError("Choose a cached condition and noise index; no resampling occurs")
    point = int(selected[noise_index])
    runs = [
        run
        for run in payload["runs"]
        if run["nfe"] == nfe
        and (run.get("league") != "primary" or run.get("seed") == SEEDS[0])
        and (method is None or run["method"] == method)
    ]
    order = {name: index for index, name in enumerate(METHODS)}
    runs.sort(key=lambda run: order.get(run["method"], len(METHODS)))
    if not runs:
        raise ValueError("No cached run for this method/NFE")
    first = _cpu(runs[0]["events"]["states"])[0]
    if any(not torch.equal(_cpu(run["events"]["states"])[0], first) for run in runs):
        raise ValueError("Preview models must use the exact same initial state/conditions")
    count = max(len(run["events"]["states"]) for run in runs)
    frame_index = count - 1 if frame_index is None else frame_index
    if not isinstance(frame_index, int) or not 0 <= frame_index < count:
        raise ValueError("frame_index must address an actual cached event")
    fraction = frame_index / max(1, count - 1)
    columns = min(3, len(runs))
    rows = (len(runs) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.3 * columns, 4.0 * rows + 0.9), squeeze=False
    )
    figure.subplots_adjust(
        top=0.91 if rows > 1 else 0.78, bottom=0.1 if rows > 1 else 0.22, hspace=0.5, wspace=0.3
    )
    real = _cpu(payload["reference_samples"])
    real_conditions = _cpu(payload["reference_conditions"])
    if task == "rays":
        selected_bounds = torch.cat(
            [_cpu(run["events"]["states"])[:, point].reshape(-1, 2) for run in runs] + [real]
        )
        bound = max(1.1, float(selected_bounds.abs().max()) * 1.08)
    for axis, run in zip(axes.flat, runs):
        events = run["events"]
        states = _cpu(events["states"])
        index = min(int(math.floor(fraction * (len(states) - 1) + 1e-10)), len(states) - 1)
        kind = events["kinds"][index]
        clock = float(events["times"][index])
        calls = events["network_calls_used"][index]
        if task == "rays":
            mask = real_conditions == conditions[point]
            axis.scatter(real[:, 0], real[:, 1], s=2, color="#cbd5e1", alpha=0.18)
            axis.scatter(real[mask, 0], real[mask, 1], s=4, color="#087f8c", alpha=0.35)
            path = states[:, point]
            axis.plot(path[:, 0], path[:, 1], color="#94a3b8", lw=0.8, alpha=0.6)
            axis.scatter(
                path[:, 0], path[:, 1], c=_RAINBOW(np.linspace(0, 1, len(path))), s=23, zorder=5
            )
            current = states[index, point]
            axis.scatter(current[0], current[1], color="black", s=55, zorder=8)
            cloud = states[-1, selected]
            axis.scatter(cloud[:, 0], cloud[:, 1], color="#475569", alpha=0.3, s=9)
            axis.set(xlim=(-bound, bound), ylim=(-bound, bound), aspect="equal")
        else:
            context = conditions[point]
            draw_walls(axis, context[4:].reshape(1, 2, 2), alpha=0.55)
            for path in states[index, selected]:
                axis.plot(path[:, 0], path[:, 1], color="#64748b", alpha=0.2, lw=0.8, zorder=5)
            path = states[index, point].numpy()
            segments = np.stack((path[:-1], path[1:]), axis=1)
            line = LineCollection(segments, cmap=_RAINBOW, linewidths=2.1, zorder=6)
            line.set_array(np.linspace(0, 1, len(segments)))
            line.set_clim(0, 1)
            axis.add_collection(line)
            axis.scatter(context[0], context[1], color=_RAINBOW(0.0), marker="o", s=45, zorder=9)
            axis.scatter(context[2], context[3], color=_RAINBOW(1.0), marker="*", s=100, zorder=9)
            axis.annotate("A", context[:2], xytext=(5, 5), textcoords="offset points")
            axis.annotate("B", context[2:4], xytext=(5, 5), textcoords="offset points")
            axis.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal")
            outside = int((states[index, selected].abs() > 1.1).any(-1).sum())
            axis.text(
                0.01,
                0.01,
                f"{outside} raw waypoints outside view",
                transform=axis.transAxes,
                fontsize=7,
            )
        name = "Original" if run.get("league") == "parent_baseline" else run["method"]
        axis.set_title(
            f"{name} · actual event {index}/{len(states) - 1}\n{kind}; t={clock:.3f}; network calls {calls}/{nfe}",
            fontsize=9,
        )
        axis.grid(alpha=0.18)
        axis.set(xlabel="x", ylabel="y")
    for axis in list(axes.flat)[len(runs) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"{task} · requested condition {condition_id} · fixed noise {noise_index} · ACTUAL NFE {nfe}\nSampling events: CM denoise→re-noise time is nonmonotone; no event interpolation",
        fontsize=11,
        y=0.985,
    )
    caveat = (
        "Red→violet follows stored sampling-event order; black dot is the selected actual event."
    )
    if task == "wall":
        caveat = "ENTIRE path states, not particle motion. Red A→violet B encodes SPATIAL waypoint progress."
    figure.text(
        0.5,
        0.015,
        caveat + "\nPanels align by cached event index, not by a shared physical time.",
        ha="center",
        fontsize=8,
    )
    return figure, axes


def render_method_zoo_previews(payload, output, *, fps=3):
    """Public first-seed/condition0/noise0 previews; data only, never a sampler."""
    import matplotlib.pyplot as plt
    from PIL import Image

    from diffusion_lab.navigation_visualization import save_navigation_particle_gif

    payload = _preview_payload(payload)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for nfe in BUDGETS:
        selected = [
            run
            for run in payload["runs"]
            if run["nfe"] == nfe and (run.get("league") != "primary" or run.get("seed") == SEEDS[0])
        ]
        if not selected:
            continue
        figure, _ = plot_method_zoo_preview(payload, nfe=nfe)
        for extension in ("png", "svg"):
            path = output / f"{payload['task']}_nfe{nfe}_final.{extension}"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            paths.append(path)
        plt.close(figure)
        frames = []
        for frame in range(max(len(run["events"]["states"]) for run in selected)):
            figure, _ = plot_method_zoo_preview(payload, nfe=nfe, frame_index=frame)
            figure.canvas.draw()
            frame_image = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(Image.fromarray(frame_image))
            plt.close(figure)
        path = output / f"{payload['task']}_nfe{nfe}_sampling_events.gif"
        frames[0].save(
            path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0
        )
        paths.append(path)
    if payload["task"] == "wall":
        context = _cpu(payload["conditions"])[0]
        for run in payload["runs"]:
            if run["nfe"] != 2 or (run.get("league") == "primary" and run.get("seed") != SEEDS[0]):
                continue
            name = "original" if run.get("league") == "parent_baseline" else run["method"]
            path = output / f"wall_{name}_nfe2_spatial_particle.gif"
            save_navigation_particle_gif(
                _cpu(run["events"]["states"])[-1, :4],
                path,
                walls=context[4:].reshape(1, 2, 2),
                start=context[:2],
                goal=context[2:4],
                fps=10,
                substeps=1,
            )
            paths.append(path)
    return paths


def sampling_preview_widget(payload):
    """Step1 cached event playback; changing a control never calls a model."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display

    payload = _preview_payload(payload)
    methods = list(dict.fromkeys(run["method"] for run in payload["runs"]))
    choose_method = widgets.Dropdown(options=methods, description="Method")
    choose_nfe = widgets.Dropdown(options=list(BUDGETS), value=2, description="NFE")
    choose_condition = widgets.Dropdown(
        options=torch.unique(_cpu(payload["condition_ids"])).tolist(), description="Condition"
    )
    noise = widgets.IntSlider(
        value=0,
        min=0,
        max=int((_cpu(payload["condition_ids"]) == choose_condition.value).sum()) - 1,
        step=1,
        description="Noise",
        continuous_update=False,
    )
    frame = widgets.IntSlider(
        value=0, min=0, max=3, step=1, description="Event", continuous_update=False
    )
    play = widgets.Play(value=0, min=0, max=3, step=1, interval=350)
    widgets.jslink((play, "value"), (frame, "value"))
    output = widgets.Output()

    def show(change=None):
        candidates = [
            run
            for run in payload["runs"]
            if run["method"] == choose_method.value and run["nfe"] == choose_nfe.value
        ]
        if not candidates:
            return
        limit = len(candidates[0]["events"]["states"]) - 1
        frame.max = play.max = limit
        frame.value = min(frame.value, limit)
        noise.max = int((_cpu(payload["condition_ids"]) == choose_condition.value).sum()) - 1
        with output:
            clear_output(wait=True)
            figure, _ = plot_method_zoo_preview(
                payload,
                method=choose_method.value,
                nfe=choose_nfe.value,
                frame_index=frame.value,
                condition_id=choose_condition.value,
                noise_index=noise.value,
            )
            display(figure)
            plt.close(figure)

    for control in (choose_method, choose_nfe, choose_condition, noise, frame):
        control.observe(show, names="value")
    show()
    return widgets.VBox(
        [
            widgets.HBox([choose_method, choose_nfe, choose_condition, noise]),
            widgets.HBox([play, frame]),
            output,
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cached-rows",
        type=Path,
        help="Already measured rows JSON; never constructs data or samples models",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cached_rows:
        payload = json.loads(args.cached_rows.read_text())
        ranking = aggregate_rankings(payload["rows"])
        if args.output:
            save_ranking_artifacts(payload["rows"], args.output)
        else:
            print(json.dumps(ranking, indent=2, allow_nan=False))
    else:
        print(
            json.dumps(
                {
                    "status": "prepared_not_executed",
                    "methods": METHODS,
                    "seeds": SEEDS,
                    "actual_network_nfe": BUDGETS,
                    "tasks": TASKS,
                    "gate": "ALL48fixed+allstageexports+validationpolicy+ROOTGO before any NEW final bank",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
