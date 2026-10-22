"""Execute notebook15's visible objectives, sampling clocks and portable state."""

import ast
import importlib.util
import json
import shutil
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nbformat
import pytest
import torch

from diffusion_lab import PointDenoiser
from diffusion_lab.environments import make_environment
from diffusion_lab.environments.multimodal_navigation import make_multimodal_navigation
from recheck.runtime import model_hash, tensor_hash

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "notebooks/15_conditional_method_zoo.ipynb"


def runner_module():
    spec = importlib.util.spec_from_file_location(
        "method_zoo_test_runner", ROOT / "examples/train_method_zoo.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def api():
    return runner_module().notebook_api()


def tiny_data(task):
    if task == "rays":
        dataset = make_environment("rotating_rays", n_samples=64, n_classes=16, seed=47)
        conditions = dataset.conditions
        groups = conditions
        states = dataset.samples
        spec = {"kind": "categorical", "num_classes": 16}
    else:
        dataset = make_multimodal_navigation(n_scenes=2, samples_per_scene=8, seed=61)
        conditions = dataset.conditions
        groups = dataset.scene_indices
        states = dataset.samples[:, 1:-1]
        spec = {"kind": "continuous", "dimension": 8, "route_id_input": False}
    return {
        "task": task,
        "split": "train",
        "samples": states.contiguous(),
        "raw_samples": dataset.samples,
        "conditions": conditions,
        "groups": groups,
        "pools": [torch.where(groups == i)[0] for i in range(int(groups.max()) + 1)],
        "sample_shape": tuple(states.shape[1:]),
        "condition_spec": spec,
        "train_data_sha256": tensor_hash(dataset.samples),
        "train_state_sha256": tensor_hash(states),
        "train_conditions_sha256": tensor_hash(conditions),
        "endpoint_clamped_contract": task == "wall",
    }


def tiny_parent(task):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(90)
        parent = PointDenoiser(
            sample_shape=(2,) if task == "rays" else (30, 2),
            hidden_dim=16,
            num_layers=1,
            conditioner="embedding" if task == "rays" else "mlp",
            num_classes=16 if task == "rays" else 0,
            condition_dim=0 if task == "rays" else 8,
        )
    parent.zoo_task = task
    parent.zoo_method = "teacher"
    return parent.eval().requires_grad_(False)


def config(api, directory, methods=None):
    return api["resolved_config"](
        {
            "updates": 10,
            "batch_size": 8,
            "warmup": 1,
            "progressive_stage_warmup": 0,
            "pairbank_size": 32,
            "pairbank_batch_size": 16,
            "checkpoint_every": 2,
            "progress": False,
            "output_dir": str(directory),
            "methods": list(methods or api["METHODS"]),
            "threads": 2,
        }
    )


def setup(api, directory, task="rays", methods=None):
    data, parent = tiny_data(task), tiny_parent(task)
    cfg = config(api, directory, methods)
    bank, metadata = api["prepare_pairbank"](data, parent, cfg, directory / "pairbank")
    return data, parent, cfg, bank, metadata


def test_notebook_compiles_and_keeps_every_objective_visible():
    book = nbformat.read(BOOK, 4)
    nbformat.validate(book)
    tags = set()
    for index, cell in enumerate(book.cells):
        tags.update(cell.metadata.get("tags", []))
        if cell.cell_type == "code":
            assert not cell.outputs and cell.execution_count is None
            compile(ast.parse(cell.source), f"zoo-cell{index}", "exec")
    assert {
        "cfm_training",
        "endpoint_training",
        "progressive_training",
        "reflow_training",
        "consistency_training",
        "shortcut_training",
        "meanflow_training",
    } <= tags
    source = "\n".join(cell.source for cell in book.cells)
    assert "torch.func.jvp" in source and "torch.ones_like(dd)" in source
    assert ".square().sum(dim=1)" in source
    assert "loss.backward()" in source and "ctx.optimizer.step()" in source
    assert "linear_sum_assignment" in source and "call_velocity(ctx.stage_teacher" in source


@pytest.mark.parametrize("task", ["rays", "wall"])
def test_all_eight_actual_loops_train_freeze_parent_and_export_portable_weights(
    api, tmp_path, task
):
    data, parent, cfg, bank, metadata = setup(api, tmp_path, task)
    digest = model_hash(parent)
    manifests, checkpoints = {}, {}
    for method in api["METHODS"]:
        directory = tmp_path / method
        manifests[method] = api["train_method"](
            method,
            data,
            parent,
            161,
            directory,
            cfg,
            bank,
            metadata,
            source_sha256="tiny-visible-notebook-test",
        )
        manifest = manifests[method]
        checkpoints[method] = torch.load(directory / "training.pt", weights_only=True)
        assert manifest["status"] == "completed" and manifest["updates_completed"] == 10
        assert manifest["whole_support_single_condition"] and not manifest["route_id_input"]
        assert manifest["state_dimension"] == (2 if task == "rays" else 60)
        assert len(checkpoints[method]["history"]) == 10
        assert all(
            torch.isfinite(torch.tensor(row["loss"])) for row in checkpoints[method]["history"]
        )
        for nfe in (1, 2, 4):
            export = manifest["inference_exports"][str(nfe)]
            restored = api["load_zoo_model"](directory / export["path"])
            assert model_hash(restored) == export["sha256"]
            assert restored.zoo_method == method and restored.zoo_task == task
        assert model_hash(parent) == digest
        assert all(p.grad is None and not p.requires_grad for p in parent.parameters())
    for method in ("ot_cfm", "shortcut", "meanflow"):
        assert checkpoints["cfm"]["raw_trace"] == checkpoints[method]["raw_trace"]
    assert checkpoints["endpoint"]["pair_trace"] == checkpoints["reflow"]["pair_trace"]
    assert checkpoints["consistency"]["pair_trace"] == []
    for nfe, cumulative in ((4, 6), (2, 8), (1, 10)):
        assert (
            manifests["progressive"]["inference_exports"][str(nfe)]["cumulative_updates_at_export"]
            == cumulative
        )
    costs = manifests["progressive"]["training_cost"]
    assert costs.get("teacher_sample_evals", 0) == 0
    assert costs["student_sample_evals"] == 80
    assert costs["stage_teacher_sample_evals"] == 160
    assert manifests["meanflow"]["training_cost"]["student_sample_evals"] == 100


@pytest.mark.parametrize(
    "method",
    ["cfm", "ot_cfm", "endpoint", "progressive", "reflow", "consistency", "shortcut", "meanflow"],
)
def test_stop_and_resume_reproduces_exact_final_ema_and_rng(api, tmp_path, method):
    data, parent, cfg, bank, metadata = setup(api, tmp_path, methods=[method])
    first = api["train_method"](
        method,
        data,
        parent,
        161,
        tmp_path / "full",
        cfg,
        bank,
        metadata,
        source_sha256="resume-fixture",
    )
    stopped = api["train_method"](
        method,
        data,
        parent,
        161,
        tmp_path / "split",
        cfg,
        bank,
        metadata,
        source_sha256="resume-fixture",
        stop_after=3,
    )
    assert stopped["status"] == "interrupted" and stopped["updates_completed"] == 3
    resumed = api["train_method"](
        method,
        data,
        parent,
        161,
        tmp_path / "split",
        cfg,
        bank,
        metadata,
        source_sha256="resume-fixture",
        resume=True,
    )
    assert first["ema_sha256"] == resumed["ema_sha256"]
    full = torch.load(tmp_path / "full/training.pt", weights_only=True)
    split = torch.load(tmp_path / "split/training.pt", weights_only=True)
    assert full["raw_trace"] == split["raw_trace"] and full["pair_trace"] == split["pair_trace"]
    assert full["history"] == split["history"]
    assert full["costs"] == split["costs"]
    for name in full["rng"]:
        torch.testing.assert_close(full["rng"][name], split["rng"][name], rtol=0, atol=0)


@pytest.mark.parametrize("task", ["rays", "wall"])
@pytest.mark.parametrize("method", ["cfm", "consistency", "shortcut"])
def test_sampling_actual_nfe_event_clock_conditions_and_fixed_endpoints(api, task, method):
    parent = tiny_parent(task)
    model = api["initialize_interval_from_parent"](parent) if method == "shortcut" else parent
    model.zoo_method = method
    model.zoo_task = task
    data = tiny_data(task)
    conditions = data["conditions"][:8]
    initial = torch.randn(8, *data["sample_shape"])
    extra = torch.randn(3, *initial.shape)
    calls = []

    def count(module, args, kwargs):
        calls.append((len(args[0]), kwargs["conditions"].detach().clone()))

    hook = model.register_forward_pre_hook(count, with_kwargs=True)
    result = api["sample_zoo"](model, initial, conditions, nfe=4, batch_size=3, extra_noise=extra)
    hook.remove()
    assert sum(n for n, _ in calls) == 4 * len(initial)
    assert result["nfe"] == 4 and result["model_sample_evals"] == 32
    assert result["sampler"] == ("denoise_renoise" if method == "consistency" else "euler")
    if method == "consistency":
        assert result["history"].shape[0] == 8
        torch.testing.assert_close(
            result["times"], torch.tensor([1.0, 0.0, 0.75, 0.0, 0.5, 0.0, 0.25, 0.0])
        )
        assert result["actual_inputs"].shape[0] == result["postcall_outputs"].shape[0] == 4
        assert result["event_kinds"] == ["model_input", "denoised_endpoint"] * 4
    else:
        assert result["history"].shape[0] == 5
    if task == "wall":
        history = result["history"]
        torch.testing.assert_close(
            history[:, :, 0], conditions[None, :, :2].expand(len(history), -1, -1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            history[:, :, -1], conditions[None, :, 2:4].expand(len(history), -1, -1), rtol=0, atol=0
        )
    again = api["sample_zoo"](model, initial, conditions, nfe=4, batch_size=8, extra_noise=extra)
    torch.testing.assert_close(result["samples"], again["samples"], rtol=1e-5, atol=1e-6)


def test_pairbank_is_immutable_train_only_independent_noise_and_corruption_refused(api, tmp_path):
    data, parent, cfg, bank, metadata = setup(api, tmp_path)
    original = {p.name: p.read_bytes() for p in (tmp_path / "pairbank").iterdir() if p.is_file()}
    second, second_metadata = api["prepare_pairbank"](data, parent, cfg, tmp_path / "pairbank")
    assert second_metadata == metadata
    assert torch.unique(bank["noise"], dim=0).shape[0] == cfg["pairbank_size"]
    for name, raw in original.items():
        assert (tmp_path / "pairbank" / name).read_bytes() == raw
    torch.testing.assert_close(second["history"], bank["history"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="train/validation only"):
        api["load_task"]("wall", "test")
    metadata_path = tmp_path / "pairbank/manifest.json"
    corrupt = json.loads(metadata_path.read_text())
    corrupt["hashes"]["history"] = "bad"
    metadata_path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="tensor hash differs"):
        api["prepare_pairbank"](data, parent, cfg, tmp_path / "pairbank")


def test_cli_import_dry_run_and_config_validation_have_no_data_side_effects(tmp_path):
    module = runner_module()
    assert module._API is None
    api = module.notebook_api()
    assert not (tmp_path / "out").exists()
    report = module.run_zoo({"output_dir": str(tmp_path / "out")}, dry_run=True)
    assert report["status"] == "dry_run" and not report["data_loaded"]
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="Unknown method-zoo config"):
        module.resolved_config({"misspelled_lr": 1})
    with pytest.raises(ValueError, match="must be finite"):
        module.resolved_config({"lr": float("nan")})
    with pytest.raises(ValueError, match="Scratch mode"):
        module.resolved_config({"initialization": "scratch"})
    data, parent = tiny_data("rays"), tiny_parent("rays")
    cfg = config(api, tmp_path / "scratch", ["cfm"])
    cfg["initialization"] = "scratch"
    bank, metadata = api["prepare_pairbank"](data, parent, cfg, tmp_path / "scratch/pairbank")
    ctx = api["start_context"](
        "cfm", data, parent, 161, cfg, tmp_path / "scratch/cfm", "scratch-source", bank, metadata
    )
    assert model_hash(ctx.model) != model_hash(parent)
    for hook in ctx.hooks:
        hook.remove()


def test_progressive_recovers_checkpoint_export_interruption_window(api, tmp_path):
    data, parent, cfg, bank, metadata = setup(api, tmp_path, methods=["progressive"])
    expected = api["train_method"](
        "progressive",
        data,
        parent,
        161,
        tmp_path / "full",
        cfg,
        bank,
        metadata,
        source_sha256="boundary-recovery",
    )
    api["train_method"](
        "progressive",
        data,
        parent,
        161,
        tmp_path / "split",
        cfg,
        bank,
        metadata,
        source_sha256="boundary-recovery",
        stop_after=2,
    )
    shutil.rmtree(tmp_path / "split/stage16")
    actual = api["train_method"](
        "progressive",
        data,
        parent,
        161,
        tmp_path / "split",
        cfg,
        bank,
        metadata,
        source_sha256="boundary-recovery",
        resume=True,
    )
    assert expected["ema_sha256"] == actual["ema_sha256"]
    assert expected["training_cost"] == actual["training_cost"]
    assert (tmp_path / "split/stage16/ema_model/config.json").exists()


@pytest.mark.parametrize("task", ["rays", "wall"])
def test_cached_widgets_change_nfe_target_noise_and_true_event_without_model_calls(
    api, task, monkeypatch
):
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    from diffusion_lab.navigation_visualization import plot_navigation_paths

    data = tiny_data(task)
    per_condition = 2
    condition_values = torch.cat([data["conditions"][p[:1]] for p in data["pools"]])
    conditions = condition_values.repeat_interleave(per_condition, dim=0)
    noise = torch.randn(len(conditions), *data["sample_shape"])
    extra = torch.randn(3, *noise.shape)
    counts = []
    generations = {}
    for method in ("cfm", "consistency"):
        model = tiny_parent(task)
        model.zoo_method = method
        hook = model.register_forward_pre_hook(lambda m, a: counts.append(len(a[0])))
        for nfe in (1, 2, 4):
            generations[(method, nfe)] = api["sample_zoo"](
                model, noise, conditions, nfe=nfe, extra_noise=extra[: nfe - 1]
            )
        hook.remove()
    before = list(counts)
    cell = next(
        c.source
        for c in nbformat.read(BOOK, 4).cells
        if "validation_visualization" in c.metadata.get("tags", [])
    )
    controls = textwrap.dedent(cell[cell.index("    method_control = ") :])
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {
        "widgets": widgets,
        "plt": plt,
        "torch": torch,
        "CONFIG": {"methods": ["cfm", "consistency"]},
        "EVAL_NFES": (1, 2, 4),
        "MAX_NFE": 4,
        "TASK": task,
        "PREVIEW_PER_CONDITION": per_condition,
        "condition_values": condition_values,
        "validation": data,
        "generations": generations,
        "plot_navigation_paths": plot_navigation_paths,
        "display": lambda *_: None,
        "clear_output": lambda **_: None,
    }
    exec(controls, namespace)
    namespace["condition_control"].value = len(condition_values) - 1
    namespace["noise_control"].value = 1
    namespace["method_control"].value = "consistency"
    namespace["nfe_control"].value = 2
    namespace["frame_control"].value = 3
    assert namespace["frame_control"].max == namespace["play"].max == 3
    assert namespace["frame_control"].step == namespace["play"].step == 1
    assert counts == before
    plt.close("all")


def test_parallel_new_root_setup_serializes_atomic_config_write(api, tmp_path, monkeypatch):
    import time

    root = tmp_path / "parallel"
    cfg = config(api, root, ["cfm"])
    cfg["tasks"] = ["rays", "wall"]
    data = {task: tiny_data(task) for task in ("rays", "wall")}
    parent = {task: tiny_parent(task) for task in ("rays", "wall")}
    original_atomic = api["atomic_json"]
    original_calls = []

    def slow_atomic(path, value):
        if Path(path).name == "run_config.json":
            original_calls.append(str(path))
            time.sleep(0.05)
        return original_atomic(path, value)

    monkeypatch.setitem(api, "atomic_json", slow_atomic)
    monkeypatch.setitem(api, "load_task", lambda task, split: data[task])
    monkeypatch.setitem(api, "load_parent", lambda task, *args: (parent[task], {}))
    monkeypatch.setitem(
        api, "prepare_pairbank", lambda *args: ({}, {"sha256": "fake-prepared-bank"})
    )
    monkeypatch.setitem(api, "notebook_source_hash", lambda *_: "source")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(api["run_zoo"], cfg, repo_root=ROOT, tasks=[task], prepare_only=True)
            for task in ("rays", "wall")
        ]
        assert all(future.result()["status"] == "completed" for future in futures)
    assert original_calls == [str(root / "run_config.json")]
    assert json.loads((root / "run_config.json").read_text()) == cfg
