"""Execute notebook12's visible code on tiny CPU C16 data, without downloads."""

import copy
import json
from pathlib import Path

import nbformat
import pytest
import test_recheck_notebook as original
import torch

from recheck.config import load_config
from recheck.data import make_toy_dataset
from recheck.fourier_model import FourierFMMLP
from recheck.models import FMMLP
from recheck.runtime import model_hash

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/12_many_rays_wasserstein_tube.ipynb"


def cell_source(tag):
    book = json.loads(NOTEBOOK.read_text())
    return next(
        "".join(cell["source"])
        for cell in book["cells"]
        if tag in cell.get("metadata", {}).get("tags", [])
    )


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def notebook_namespace(output_dir, model_kind="mlp"):
    namespace = {}
    exec(cell_source("imports"), namespace)
    exec(cell_source("parameters"), namespace)
    # An injected tiny parent verifies the loop only. It does not impersonate a
    # completed 100k teacher, bypass the production locator, or run native setup.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model_class = FourierFMMLP if model_kind == "fourier_mlp" else FMMLP
        teacher = model_class(hidden_dim=8, num_layers=2).eval().requires_grad_(False)
    namespace.update(
        teacher=teacher,
        teacher_hash=model_hash(teacher),
        device=torch.device("cpu"),
        OUTPUT_DIR=output_dir,
        train=make_toy_dataset("rotating_rays", 64, 41, split="train", n_classes=16),
        validation=make_toy_dataset(
            "rotating_rays", 64, 41, split="validation", n_classes=16
        ),
        UPDATES=2,
        BATCH_SIZE=16,
        CALIBRATION_SAMPLES=16,
        CHECKPOINT_EVERY=1,
        WARMUP=1,
        MICROSTEPS=2,
        H=1 / 16,
        EVAL_SAMPLES=64,
        MAKE_GIFS=False,
        SHOW_WIDGETS=False,
    )
    output_dir.mkdir(parents=True)
    exec(cell_source("transition"), namespace)
    exec(cell_source("calibration"), namespace)
    return namespace


def test_source_clean_compilable_and_all_training_operations_visible(monkeypatch):
    # Reuse the stronger original notebook structural contract, on the new file.
    monkeypatch.setattr(original, "NOTEBOOK", NOTEBOOK)
    original.test_notebook_source_is_clean_small_compilable_and_training_is_visible()
    original.test_actual_notebook_target_has_correct_sign_and_retargets_tube()
    assert 'MIN_VERSION = "0.4.2"' in cell_source("install")
    assert "rays_cpu_pilot_v1" not in NOTEBOOK.read_text()
    for tag in ("teacher_locator", "teacher_geometry_qc", "geometry_comparison"):
        assert cell_source(tag)
    tags = [cell.get("metadata", {}).get("tags", [None])[0] for cell in nbformat.read(NOTEBOOK, 4).cells]
    assert tags.index("teacher_locator") < tags.index("setup")
    assert tags.index("setup") < tags.index("teacher_geometry_qc") < tags.index("calibration")
    assert tags.index("evaluation") < tags.index("geometry_comparison")


def test_default_exploratory_run_is_separate_from_full_registration():
    namespace = {"Path": Path}
    exec(cell_source("parameters"), namespace)
    registered = load_config(ROOT / namespace["CONFIG_PATH"])
    assert namespace["N_CLASSES"] == registered["data"]["n_classes"] == 16
    assert namespace["N_TRAIN"] == registered["data"]["n_train"] == 65_536
    assert namespace["UPDATES"] == 200
    assert registered["student"]["updates"] == 2500
    assert registered["distillation"]["seeds"] == [101, 202, 303]
    assert registered["teacher"]["updates"] == 100_000
    assert registered["model"]["kind"] == "fourier_mlp"
    assert namespace["SEED"] == 101
    assert namespace["CALIBRATION_SEED"] != namespace["SEED"]
    assert "exploratory" in str(namespace["OUTPUT_DIR"])
    assert str(namespace["PUBLISHED_TEACHER_PATH"]) == (
        "replication/demo/rays16_fourier_teacher_v1/teacher/ema_model"
    )
    assert cell_source("setup").index('registered_config["output_dir"]') < cell_source("setup").index(
        "repo_dir / PUBLISHED_TEACHER_PATH"
    )


@pytest.fixture
def native_parent(tmp_path, request):
    kind = getattr(request, "param", "mlp")
    config_name = "many_rays_fourier_cpu.yaml" if kind == "fourier_mlp" else "many_rays_cpu.yaml"
    registered = load_config(ROOT / "replication/configs" / config_name)
    namespace = {}
    exec(cell_source("imports"), namespace)
    exec(cell_source("teacher_locator"), namespace)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model_class = FourierFMMLP if kind == "fourier_mlp" else FMMLP
        model = model_class(hidden_dim=256, num_layers=4, time_embedding_dim=32)
    directory = tmp_path / "native" / "teacher"
    model.save_pretrained(directory / "ema_model", safe_serialization=True)
    manifest = {
        "status": "completed",
        "updates_completed": 100_000,
        "teacher_sha256": model_hash(model),
        "data_sha256": "a" * 64,
        "config": copy.deepcopy(registered),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return namespace, registered, directory / "ema_model", manifest


def test_no_native_parent_stops_before_calibration(tmp_path):
    namespace = {"Path": Path, "json": json}
    exec(cell_source("teacher_locator"), namespace)
    registered = load_config(ROOT / "replication/configs/many_rays_cpu.yaml")
    with pytest.raises(FileNotFoundError, match="Calibration and student training have not started"):
        namespace["select_completed_teacher"]([tmp_path / "missing"], registered)


def test_auto_selects_fourier_config_when_present_without_parent_fallback(tmp_path):
    namespace = {"Path": Path, "json": json, "load_config": load_config}
    exec(cell_source("teacher_locator"), namespace)
    fourier, plain = Path("fourier.yaml"), Path("plain.yaml")
    (tmp_path / plain).write_bytes((ROOT / "replication/configs/many_rays_cpu.yaml").read_bytes())
    resolve = namespace["resolve_teacher_profile"]
    assert resolve(tmp_path, "auto", fourier, plain)[0] == "plain"
    with pytest.raises(FileNotFoundError):
        resolve(tmp_path, "fourier", fourier, plain)
    (tmp_path / fourier).write_bytes(
        (ROOT / "replication/configs/many_rays_fourier_cpu.yaml").read_bytes()
    )
    selected, _, registered = resolve(tmp_path, "auto", fourier, plain)
    assert selected == "fourier" and registered["model"]["kind"] == "fourier_mlp"
    assert resolve(tmp_path, "plain", fourier, plain)[0] == "plain"
    # Config presence chooses the preset. Missing native Fourier weights must
    # stop, rather than switching to the available plain profile.
    with pytest.raises(FileNotFoundError, match="Calibration and student training have not started"):
        namespace["select_completed_teacher"]([tmp_path / "missing_fourier"], registered)


@pytest.mark.parametrize("failure", ["four_rays", "interrupted", "short", "architecture"])
def test_teacher_locator_rejects_wrong_or_incomplete_parent(native_parent, failure):
    namespace, registered, path, manifest = native_parent
    if failure == "four_rays":
        manifest["config"]["data"]["n_classes"] = 4
    elif failure == "interrupted":
        manifest["status"] = "interrupted"
    elif failure == "short":
        manifest["updates_completed"] = 99_999
    else:
        architecture = json.loads((path / "config.json").read_text())
        architecture["hidden_dim"] = 128
        (path / "config.json").write_text(json.dumps(architecture))
    (path.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="completed 100000-update 16-ray teacher"):
        namespace["select_completed_teacher"]([path], registered)


@pytest.mark.parametrize("native_parent", ["mlp", "fourier_mlp"], indirect=True)
def test_native_loading_verifies_weights_and_keeps_parent_frozen(native_parent):
    namespace, registered, path, manifest = native_parent
    teacher, selected, found_manifest = namespace["load_verified_teacher"](
        [path], registered, "cpu"
    )
    assert selected == path
    assert model_hash(teacher) == found_manifest["teacher_sha256"]
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    manifest["teacher_sha256"] = "b" * 64
    (path.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="weights do not match"):
        namespace["load_verified_teacher"]([path], registered, "cpu")


@pytest.mark.parametrize("native_parent", ["fourier_mlp"], indirect=True)
@pytest.mark.parametrize("failure", ["class", "frequencies", "polar", "epsilon", "manifest_kind"])
def test_fourier_guard_checks_native_class_and_exact_fixed_feature_profile(native_parent, failure):
    namespace, registered, path, manifest = native_parent
    architecture = json.loads((path / "config.json").read_text())
    if failure == "class":
        architecture["_class_name"] = "FMMLP"
    elif failure == "frequencies":
        architecture["spatial_frequencies"] = [1, 2]
    elif failure == "polar":
        architecture["polar_features"] = False
    elif failure == "epsilon":
        architecture["epsilon"] = 0.01
    else:
        manifest["config"]["model"]["kind"] = "mlp"
    (path / "config.json").write_text(json.dumps(architecture))
    (path.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="completed 100000-update 16-ray teacher"):
        namespace["select_completed_teacher"]([path], registered)


def test_optional_stress_copy_is_inert_then_preserves_the_exact_parent(native_parent, tmp_path):
    # The shipped cell cannot copy anything until the user explicitly changes its flag.
    exec(cell_source("prepare_stress_parent"), {})
    namespace, registered, path, manifest = native_parent
    teacher, _, _ = namespace["load_verified_teacher"]([path], registered, "cpu")
    stress_root = tmp_path / "stress"
    namespace.update(
        teacher=teacher,
        teacher_hash=model_hash(teacher),
        teacher_path=path,
        teacher_manifest=manifest,
        registered_config=registered,
        STUDY_ID=registered["protocol_id"],
        repo_dir=tmp_path,
        load_config=lambda _: {
            "output_dir": str(stress_root),
            "protocol_id": "rays16_long_teacher_stress8_v1",
        },
    )
    source = cell_source("prepare_stress_parent").replace(
        "PREPARE_REGISTERED_STRESS = False", "PREPARE_REGISTERED_STRESS = True"
    )
    exec(source, namespace)
    assert (stress_root / "teacher/manifest.json").read_bytes() == (
        path.parent / "manifest.json"
    ).read_bytes()
    provenance = json.loads((stress_root / "shared_teacher_provenance.json").read_text())
    assert provenance["teacher_sha256"] == model_hash(teacher)
    assert provenance["source_study"] == registered["protocol_id"]
    assert model_hash(namespace["copied_parent"]) == model_hash(teacher)
    with pytest.raises(FileExistsError, match="already exists"):
        exec(source, namespace)


@pytest.mark.parametrize("model_kind", ["mlp", "fourier_mlp"])
def test_actual_two_update_c16_four_method_cpu_training_and_native_reload(tmp_path, model_kind):
    namespace = notebook_namespace(tmp_path / "c16", model_kind)
    parent_hash = model_hash(namespace["teacher"])
    exec(cell_source("training"), namespace)
    assert set(namespace["trained"]) == {"M0", "M1", "E0", "E1"}
    traces = list(namespace["pairing_traces"].values())
    assert len(traces[0]) == 2 and all(trace == traces[0] for trace in traces)
    for name, model in namespace["trained"].items():
        assert model_hash(model) != parent_hash
        state = torch.load(namespace["OUTPUT_DIR"] / name / "training.pt", weights_only=True)
        assert state["status"] == "completed" and state["updates_completed"] == 2
        assert state["contract"]["n_classes"] == 16
        assert state["contract"]["scope"] == "exploratory_visible_loop"
        assert state["contract"]["teacher_sha256"] == parent_hash
        assert all(torch.isfinite(torch.tensor(row["total_loss"])) for row in state["history"])
    assert model_hash(namespace["teacher"]) == parent_hash
    assert all(p.grad is None for p in namespace["teacher"].parameters())
    exec(cell_source("evaluation"), namespace)
    exec(cell_source("reload"), namespace)
    assert len({row["initial_noise_sha256"] for row in namespace["metrics"]}) == 1
    geometry = namespace["analyse_rays"](
        namespace["generations"]["E1 N16"]["samples"], namespace["validation"], n_rays=16
    )
    assert "nearest_arm_occupancy_tv_outer" in geometry["metrics"]
    assert len(geometry["reference"]["nearest_arm_occupancy_outer"]) == 16
    before = {name: model_hash(model) for name, model in namespace["trained"].items()}
    namespace["RESUME"] = True
    exec(cell_source("training"), namespace)
    assert before == {name: model_hash(model) for name, model in namespace["trained"].items()}


def test_calibration_stays_independent_of_exploratory_student_seed(tmp_path):
    first = notebook_namespace(tmp_path / "seed101")
    second = notebook_namespace(tmp_path / "seed202")
    second["SEED"] = 202
    exec(cell_source("calibration"), second)
    assert first["calibration"] == second["calibration"]
    assert first["calibration_trace"] == second["calibration_trace"]
