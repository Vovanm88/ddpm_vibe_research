import copy
import json
import os
import subprocess
import sys

import pytest
import torch

from recheck.config import DEFAULTS, FINAL_SAMPLE_SEED, config_hash, load_config, resolve_config
from recheck.fm import sample_euler
from recheck.losses import sliced_wasserstein_loss
from recheck.runtime import (
    ResourceBudget,
    RNGStreams,
    algorithm_hash,
    atomic_json,
    choose_device,
    model_hash,
    require_rays_gate,
    stable_seed,
    tensor_hash,
)


@pytest.mark.parametrize(
    "supplied",
    [
        {"unknown": 1},
        {"teacher": {"unkown_lr": 0.1}},
        {"data": {"use_test_for_calibration": True}},
        {"teacher": 42},
    ],
)
def test_unknown_and_malformed_config_keys_are_rejected(supplied):
    with pytest.raises(ValueError):
        resolve_config(supplied)


def test_final_evaluation_seed_cannot_be_changed_via_config(tmp_path):
    with pytest.raises(ValueError, match="preregistered"):
        resolve_config({"evaluation": {"final_seed": FINAL_SAMPLE_SEED + 1}})
    path = tmp_path / "config.yaml"
    path.write_text(f"evaluation:\n  final_seed: {FINAL_SAMPLE_SEED + 2}\n")
    with pytest.raises(ValueError, match="preregistered"):
        load_config(path)


def test_config_resolution_is_isolated_and_hash_records_changes():
    first = resolve_config()
    original_hash = config_hash(first)
    first["student"]["lr"] *= 2
    assert config_hash(first) != original_hash
    assert resolve_config()["student"]["lr"] == DEFAULTS["student"]["lr"]


@pytest.mark.parametrize(
    "supplied",
    [
        {"distillation": {"lambda_w": float("nan")}},
        {"distillation": {"epsilon_alpha": float("inf")}},
        {"distillation": {"tube_rhos": [0.0, float("nan")]}},
        {"teacher": {"lr": float("nan")}},
        {"resources": {"max_gpu_hours": float("inf")}},
    ],
)
def test_nonfinite_config_values_are_rejected_before_training(supplied):
    with pytest.raises(ValueError):
        resolve_config(supplied)


def test_stable_rng_seed_is_independent_of_python_hash_randomization():
    code = "from recheck.runtime import stable_seed; print(stable_seed(101, 'noise'))"
    values = []
    for hash_seed in ("17", "932"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        values.append(
            subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
        )
    assert values[0] == values[1] == str(stable_seed(101, "noise"))
    assert stable_seed(101, "noise") != stable_seed(101, "tube")
    assert stable_seed(101, "noise") != stable_seed(202, "noise")


def _training_draws(streams):
    return (
        torch.randint(1000, (16,), generator=streams["data"]),
        torch.randint(16, (), generator=streams["interval"]),
        torch.randn(16, 2, generator=streams["noise"]),
        torch.randn(16, 2, generator=streams["tube"]),
    )


def test_swd_previews_and_logging_do_not_change_training_rng(tmp_path):
    plain, observed = RNGStreams(101), RNGStreams(101)
    values = torch.tensor([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]])
    for update in range(4):
        expected = _training_draws(plain)
        # SWD consumes only its explicitly registered independent generator.
        swd = sliced_wasserstein_loss(values, values + 0.2, generator=observed["swd"])
        torch.randn(8, 2, generator=observed["preview"])
        torch.randint(10, (32,), generator=observed["bootstrap"])
        atomic_json(tmp_path / "log.json", {"update": update, "swd": float(swd)})
        actual = _training_draws(observed)
        for reference, measured in zip(expected, actual):
            assert torch.equal(reference, measured)


def test_rng_stream_state_roundtrip_covers_every_registered_stream(tmp_path):
    streams = RNGStreams(202)
    for name in streams.names:
        torch.rand(3, generator=streams[name])
    checkpoint = tmp_path / "rng.pt"
    torch.save(streams.state_dict(), checkpoint)
    expected = {name: torch.rand(5, generator=streams[name]) for name in streams.names}
    restored = RNGStreams(999)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    for name in streams.names:
        assert torch.equal(expected[name], torch.rand(5, generator=restored[name]))


def test_initial_sampling_noise_is_identical_across_nfe_and_not_mutated():
    class LinearField(torch.nn.Module):
        def forward(self, x, t):
            return 0.3 * x + t[:, None]

    model = LinearField().train()
    noise = torch.randn(9, 2, generator=torch.Generator().manual_seed(FINAL_SAMPLE_SEED))
    original = noise.clone()
    hashes = []
    for steps in (8, 16, 24, 32, 64, 128):
        result = sample_euler(model, noise, steps=steps, batch_size=4)
        hashes.append(result["initial_noise_sha256"])
        assert result["nfe"] == result["solver_steps"] == steps
        assert result["total_model_calls"] == 3 * steps
        assert torch.equal(result["history"][0], original)
        assert torch.equal(noise, original)
        assert model.training
    assert len(set(hashes)) == 1
    assert hashes[0] == tensor_hash(original)


def test_model_and_tensor_hashes_capture_values_shape_and_dtype():
    x = torch.arange(4, dtype=torch.float32)
    assert tensor_hash(x) == tensor_hash(x.clone())
    assert tensor_hash(x) != tensor_hash(x.reshape(2, 2))
    assert tensor_hash(x) != tensor_hash(x.double())
    model = torch.nn.Linear(2, 2)
    cloned = copy.deepcopy(model)
    assert model_hash(model) == model_hash(cloned)
    with torch.no_grad():
        cloned.weight[0, 0] += 0.1
    assert model_hash(model) != model_hash(cloned)


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_auto_device_detects_metal_when_cuda_is_unavailable(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert choose_device("auto").type == expected


def test_unavailable_metal_reports_process_access_constraint(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="process.*MPS"):
        choose_device("mps")


def test_rays_gate_requires_results_animation_and_current_algorithm(tmp_path):
    path = tmp_path / "rays_gate.json"
    config = resolve_config(
        {
            "data": {"name": "mnist"},
            "model": {"kind": "unet"},
            "resources": {"rays_gate": str(path)},
        }
    )
    with pytest.raises(RuntimeError, match="missing"):
        require_rays_gate(config)
    atomic_json(
        path,
        {
            "passed": True,
            "algorithm_sha256": "stale",
            "animations": ["trajectory.gif"],
            "all_mvp_methods_completed": True,
        },
    )
    with pytest.raises(RuntimeError, match="stale"):
        require_rays_gate(config)
    atomic_json(
        path,
        {
            "passed": True,
            "algorithm_sha256": algorithm_hash(),
            "animations": [],
            "all_mvp_methods_completed": True,
        },
    )
    with pytest.raises(RuntimeError, match="animations"):
        require_rays_gate(config)
    atomic_json(
        path,
        {
            "passed": True,
            "algorithm_sha256": algorithm_hash(),
            "animations": ["trajectory.gif"],
            "all_mvp_methods_completed": True,
        },
    )
    require_rays_gate(config)


def test_disk_budget_stops_before_an_unbounded_artifact_run(tmp_path):
    config = resolve_config({"output_dir": str(tmp_path), "resources": {"max_artifact_gib": 1e-6}})
    budget = ResourceBudget(config, torch.device("cpu"))
    budget.check()
    (tmp_path / "oversized.bin").write_bytes(b"x" * 2048)
    with pytest.raises(RuntimeError, match="disk budget"):
        budget.check()


def test_accelerator_budget_is_cumulative_across_training_commands(tmp_path, monkeypatch):
    import recheck.runtime as implementation

    fcntl = pytest.importorskip("fcntl")
    # Exercise only accounting: no GPU context/computation or shared real lock.
    monkeypatch.setattr(fcntl, "flock", lambda *arguments: None)
    clock = [0.0]
    monkeypatch.setattr(implementation.time, "monotonic", lambda: clock[0])
    config = resolve_config({"output_dir": str(tmp_path), "resources": {"max_gpu_hours": 0.01}})
    first = ResourceBudget(config, torch.device("cuda"))
    try:
        clock[0] = 20.0
        first.check()
    finally:
        first.close()
    clock[0] = 1000.0  # Idle time between commands consumes no accelerator budget.
    second = ResourceBudget(config, torch.device("cuda"))
    try:
        clock[0] = 1017.0
        with pytest.raises(RuntimeError, match="[Cc]umulative"):
            second.check()
    finally:
        second.close()
    ledger = json.loads((tmp_path / "resource_ledger.json").read_text())
    assert ledger["gpu_training_seconds"] == 37.0


def test_second_accelerator_process_is_rejected_without_gpu_execution(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")

    def already_locked(*arguments):
        raise BlockingIOError

    monkeypatch.setattr(fcntl, "flock", already_locked)
    config = resolve_config({"output_dir": str(tmp_path)})
    with pytest.raises(RuntimeError, match="process"):
        ResourceBudget(config, torch.device("cuda"))


def test_constructor_quota_failure_closes_actual_lock_descriptor_before_gc(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    descriptors = []
    monkeypatch.setattr(fcntl, "flock", lambda descriptor, flags: descriptors.append(descriptor))
    config = resolve_config({"output_dir": str(tmp_path), "resources": {"max_gpu_hours": 1.0}})
    atomic_json(tmp_path / "resource_ledger.json", {"gpu_training_seconds": 3600.0})
    # Keep the instance before its initializer raises, so __del__/GC cannot be
    # the mechanism that releases its actual open lock-file descriptor.
    retained = ResourceBudget.__new__(ResourceBudget)
    with pytest.raises(RuntimeError, match="[Cc]umulative"):
        ResourceBudget.__init__(retained, config, torch.device("cuda"))
    assert len(descriptors) == 1
    assert retained.closed and retained.lock is None
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_failed_nonfinite_json_write_preserves_committed_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_json(path, {"status": "interrupted", "updates_completed": 2})
    with pytest.raises(ValueError):
        atomic_json(path, {"loss": float("nan")})
    assert json.loads(path.read_text()) == {"status": "interrupted", "updates_completed": 2}
