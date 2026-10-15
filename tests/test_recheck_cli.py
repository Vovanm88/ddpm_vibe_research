import json
import sys
from types import SimpleNamespace

import pytest

from recheck import cli
from recheck.config import FINAL_SAMPLE_SEED


def test_importing_cli_does_not_import_workflow_or_torchvision_in_fresh_process():
    import subprocess

    code = "import sys; import recheck.cli; print('recheck.workflow' in sys.modules, 'torchvision' in sys.modules)"
    output = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert output == "False False"


def test_doctor_json_and_optional_saved_hardware_report(tmp_path, monkeypatch, capsys):
    result = {"cuda": False, "mps": True, "versions": {"torch": "recorded"}}
    monkeypatch.setattr(cli, "_workflow", lambda: SimpleNamespace(doctor=lambda: result))
    output = tmp_path / "hardware.json"
    assert cli.main(["doctor", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == result
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize("command", ["doctor", "verify-losses", "report"])
def test_readonly_stage_dry_run_does_not_load_workflow_or_write_files(
    tmp_path, monkeypatch, capsys, command
):
    def forbidden_workflow():
        raise AssertionError("dry-run must not execute the workflow")

    monkeypatch.setattr(cli, "_workflow", forbidden_workflow)
    output = tmp_path / ("REPORT.md" if command == "report" else "report.json")
    assert cli.main([command, "--dry-run", "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run"
    assert not output.exists()


@pytest.mark.parametrize("command,target", list(cli.CONFIG_STAGES.items()))
def test_stage_mapping_and_defaults_forward_dry_run_without_training(
    monkeypatch, capsys, command, target
):
    calls = []

    def stage(config, **kwargs):
        calls.append((config, kwargs))
        return {"stage": target, "status": "dry_run", "data": config["data"]["name"]}

    monkeypatch.setattr(cli, "_workflow", lambda: SimpleNamespace(**{target: stage}))
    assert cli.main([command, "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    expected = {"pilot-rays": "rotating_rays", "pilot-toy": "gmm"}.get(command, "mnist")
    assert result == {"stage": target, "status": "dry_run", "data": expected}
    assert len(calls) == 1 and calls[0][1]["dry_run"]
    assert calls[0][0]["evaluation"]["final_seed"] == FINAL_SAMPLE_SEED
    if command in cli.RESUMABLE:
        assert calls[0][1]["resume"] is False


def test_explicit_resume_and_rays_override_are_forwarded(monkeypatch, capsys):
    calls = []

    def stage(config, **kwargs):
        calls.append((config, kwargs))
        return {"status": "dry_run"}

    monkeypatch.setattr(cli, "_workflow", lambda: SimpleNamespace(train_teacher_stage=stage))
    assert (
        cli.main(
            [
                "train-teacher",
                "--config",
                "replication/configs/rays_cpu.yaml",
                "--resume",
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert calls[0][0]["data"]["name"] == "rotating_rays"
    assert calls[0][1] == {"resume": True, "dry_run": True}


def test_unknown_yaml_key_fails_before_importing_workflow(tmp_path, monkeypatch, capsys):
    config = tmp_path / "invalid.yaml"
    config.write_text("student:\n  unknown_optimizer_magic: true\n")

    def forbidden():
        raise AssertionError("invalid configuration must fail before backend loading")

    monkeypatch.setattr(cli, "_workflow", forbidden)
    assert cli.main(["train-teacher", "--config", str(config)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["error_type"] == "ValueError" and "Unknown config" in error["error"]


def test_malformed_yaml_is_actionable_configuration_error(tmp_path, monkeypatch, capsys):
    config = tmp_path / "malformed.yaml"
    config.write_text("teacher: [unterminated\n")
    monkeypatch.setattr(
        cli, "_workflow", lambda: pytest.fail("YAML parse failure must precede backend import")
    )
    assert cli.main(["train-teacher", "--config", str(config), "--dry-run"]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["status"] == "failed"


def test_pilot_command_cannot_silently_switch_to_mnist(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_workflow", lambda: pytest.fail("wrong benchmark must be rejected first")
    )
    assert (
        cli.main(["pilot-rays", "--config", "replication/configs/mnist_mvp.yaml", "--dry-run"]) == 2
    )
    assert "requires data.name=rotating_rays" in json.loads(capsys.readouterr().err)["error"]


def test_workflow_progress_is_stderr_and_stdout_is_one_json_document(monkeypatch, capsys):
    def doctor():
        print("Reading local hardware")
        return {"mps": False}

    monkeypatch.setattr(cli, "_workflow", lambda: SimpleNamespace(doctor=doctor))
    assert cli.main(["doctor"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"mps": False}
    assert captured.err.strip() == "Reading local hardware"


def test_nonfinite_backend_output_is_reported_as_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_workflow", lambda: SimpleNamespace(doctor=lambda: {"unmeasured": float("nan")})
    )
    assert cli.main(["doctor"]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["error_type"] == "ValueError"


def test_report_infers_configuration_and_accepts_custom_output_name(tmp_path, monkeypatch, capsys):
    calls = []

    def report_stage(runs, report_dir, config):
        calls.append((runs, report_dir, config))
        from pathlib import Path

        directory = Path(report_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "REPORT.md").write_text("Recorded results\n")
        return {"report": str(directory / "REPORT.md")}

    monkeypatch.setattr(cli, "_workflow", lambda: SimpleNamespace(report_stage=report_stage))
    output = tmp_path / "reports" / "pilot.md"
    assert cli.main(["report", "--runs", str(tmp_path / "runs"), "--output", str(output)]) == 0
    assert output.read_text() == "Recorded results\n"
    assert calls[0] == (str(tmp_path / "runs"), str(output.parent), None)
    assert json.loads(capsys.readouterr().out)["report"] == str(output)


def test_verify_losses_passes_explicit_output_to_numeric_workflow(tmp_path, monkeypatch, capsys):
    calls = []

    def verify_losses(*, output):
        calls.append(output)
        return {"passed": True, "dtype": "float64"}

    monkeypatch.setattr(cli, "_workflow", lambda: SimpleNamespace(verify_losses=verify_losses))
    path = tmp_path / "numeric_checks.json"
    assert cli.main(["verify-losses", "--output", str(path)]) == 0
    assert calls == [str(path)]
    assert json.loads(capsys.readouterr().out)["dtype"] == "float64"


def test_keyboard_interrupt_preserves_non_success_exit_status(monkeypatch, capsys):
    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_workflow", lambda: SimpleNamespace(doctor=interrupted))
    assert cli.main(["doctor"]) == 130
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["status"] == "interrupted"


def test_failed_numeric_verification_is_not_a_successful_command(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_workflow",
        lambda: SimpleNamespace(
            verify_losses=lambda **kwargs: {"passed": False, "dtype": "float64", "tests": []}
        ),
    )
    assert cli.main(["verify-losses"]) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_unsupported_flag_is_an_argparse_error():
    with pytest.raises(SystemExit) as error:
        cli.main(["evaluate-suite", "--resume"])
    assert error.value.code == 2


def _forbid_mnist_access(monkeypatch):
    import recheck.data

    def forbidden(*arguments, **options):
        pytest.fail("This command must not instantiate/download an MNIST dataset")

    monkeypatch.setattr(recheck.data, "_mnist_factory", forbidden)


def test_real_doctor_is_readonly_without_dataset_access(tmp_path, monkeypatch, capsys):
    _forbid_mnist_access(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["doctor"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), dict)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", list(cli.CONFIG_STAGES))
def test_actual_workflow_dry_run_does_not_download_train_or_create_runs(
    tmp_path, monkeypatch, capsys, command
):
    _forbid_mnist_access(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert cli.main([command, "--dry-run"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), dict)
    assert list(tmp_path.iterdir()) == []


def test_actual_prepare_mnist_refuses_missing_rays_gate_before_dataset_access(
    tmp_path, monkeypatch, capsys
):
    _forbid_mnist_access(monkeypatch)
    config = tmp_path / "mnist.yaml"
    config.write_text(
        "data:\n  name: mnist\nmodel:\n  kind: unet\n"
        f"output_dir: {tmp_path / 'runs'}\nresources:\n  rays_gate: {tmp_path / 'absent_gate.json'}\n"
    )
    assert cli.main(["prepare-mnist", "--config", str(config)]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    error = json.loads(captured.err)
    assert "gate" in error["error"].lower()
    assert not (tmp_path / "runs").exists()


def test_suite_resume_restores_existing_methods_and_starts_unvisited_methods(
    tmp_path, monkeypatch, capsys
):
    import yaml

    import recheck.workflow as workflow
    from recheck.config import method_name, resolve_config

    run_root = tmp_path / "runs"
    run_root.mkdir()
    (run_root / "calibration.json").write_text("{}")
    parent_teacher = object()
    parent_digest = "f" * 64
    (run_root / "teacher_gate.json").write_text(
        json.dumps({"passed": True, "teacher_sha256": parent_digest})
    )
    existing = run_root / "students" / "seed101_M0"
    existing.mkdir(parents=True)
    (existing / "training.pt").write_bytes(b"checkpoint marker")
    config = resolve_config({"output_dir": str(run_root), "distillation": {"seeds": [101]}})
    config_path = tmp_path / "resume.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(workflow, "_teacher", lambda config: (parent_teacher, {}))

    def fixed_teacher_hash(model):
        assert model is parent_teacher
        return parent_digest

    monkeypatch.setattr(workflow, "model_hash", fixed_teacher_hash)
    monkeypatch.setattr(workflow, "benchmark_data", lambda config: object())
    calls = []

    def train_stub(config, teacher, samples, calibration, *, loss, rho, seed, resume):
        assert teacher is parent_teacher
        directory = run_root / "students" / f"seed{seed}_{method_name(loss, rho)}"
        calls.append((method_name(loss, rho), resume))
        assert resume == (directory / "training.pt").is_file(), (
            "Unvisited methods must start rather than restoring a nonexistent checkpoint"
        )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "training.pt").write_bytes(b"checkpoint marker")
        (directory / "history.json").write_text(
            json.dumps([{"common_rng_trace": {"data": "fixed", "interval": 0}}])
        )
        return {"directory": str(directory), "manifest": {"status": "completed"}}

    monkeypatch.setattr(workflow, "train_student", train_stub)
    assert cli.main(["distill-suite", "--config", str(config_path), "--resume"]) == 0
    assert calls == [("M0", True), ("M1", False), ("E0", False), ("E1", False)]
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
