"""Small real CLI runs validate exported research artifacts and resume lifecycle."""

import csv
import json
import sys

import numpy as np
import pytest
import torch
import yaml

from diffusion_lab import cli
from diffusion_lab.pipeline import ResearchDiffusionPipeline


def _config():
    return {
        "seed": 9,
        "device": "cpu",
        "num_threads": 1,
        "environment": {"name": "gaussian_mixture_1d", "n_samples": 24, "n_classes": 2},
        "model": {"hidden_dim": 16, "num_layers": 1, "conditioner": "one_hot"},
        "scheduler": {"kind": "ddpm", "num_train_timesteps": 12, "prediction_type": "v_prediction"},
        "training": {"epochs": 2, "batch_size": 8, "ema_decay": 0.8},
        "sampling": {"num_samples": 16, "steps": 4, "guidance_scale": 1.0},
    }


def _invoke(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["diffusion-lab", *map(str, arguments)])
    cli.main()


def _write_config(tmp_path, config=None):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config() if config is None else config))
    return path


def test_train_export_sample_report_and_resume_cli_lifecycle(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    run = tmp_path / "run"
    _invoke(monkeypatch, "train", "--config", config_path, "--output", run)
    checkpoint = torch.load(run / "checkpoint.pt", weights_only=True)
    assert checkpoint["completed_epochs"] == 2
    assert checkpoint["global_step"] == 6
    assert len(json.loads((run / "history.json").read_text())) == 2
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["dataset_signature"]
    assert manifest["exported_weights"] == "ema"
    pipeline = ResearchDiffusionPipeline.from_pretrained(run / "pipeline", local_files_only=True)
    assert tuple(pipeline.model.sample_shape) == (1,)
    assert pipeline.scheduler.config.prediction_type == "v_prediction"
    for name, parameter in pipeline.model.state_dict().items():
        torch.testing.assert_close(parameter, checkpoint["ema"][name], atol=0, rtol=0)
    metrics = json.loads((run / "report" / "metrics.json").read_text())
    assert np.isfinite(metrics["sliced_wasserstein"])
    report_parameters = json.loads((run / "report" / "sampling_config.json").read_text())
    assert report_parameters["seed"] == 9
    assert report_parameters["steps"] == 4
    assert report_parameters["scheduler_class"] == "DDIMScheduler"
    for artifact in (
        "reference.svg",
        "generated.svg",
        "vector_field.svg",
        "potential.svg",
        "field_data.npz",
        "potential_diagnostics.json",
        "learning_curve.svg",
    ):
        assert (run / "report" / artifact).stat().st_size > 0

    for output in (tmp_path / "samples_a", tmp_path / "samples_b"):
        _invoke(
            monkeypatch,
            "sample",
            run / "pipeline",
            "--output",
            output,
            "--num-samples",
            6,
            "--steps",
            4,
            "--guidance",
            1,
            "--condition",
            "1",
            "--scheduler",
            "ddim",
            "--seed",
            15,
        )
        samples = np.load(output / "samples.npy")
        assert samples.shape == (6, 1)
        assert np.isfinite(samples).all()
        np.testing.assert_array_equal(np.load(output / "conditions.npy"), np.ones(6))
        assert (output / "samples.svg").stat().st_size > 0
        sample_parameters = json.loads((output / "sampling_config.json").read_text())
        assert sample_parameters["seed"] == 15
        assert sample_parameters["guidance_scale"] == 1
        assert sample_parameters["scheduler_class"] == "DDIMScheduler"
    np.testing.assert_array_equal(
        np.load(tmp_path / "samples_a" / "samples.npy"),
        np.load(tmp_path / "samples_b" / "samples.npy"),
    )

    report = tmp_path / "regenerated_report"
    _invoke(
        monkeypatch,
        "report",
        run,
        "--output",
        report,
        "--num-samples",
        12,
        "--steps",
        4,
        "--guidance",
        1,
        "--seed",
        9,
    )
    assert np.load(report / "samples.npy").shape == (12, 1)
    assert np.isfinite(json.loads((report / "metrics.json").read_text())["sliced_wasserstein"])
    assert json.loads((report / "sampling_config.json").read_text())["num_samples"] == 12

    _invoke(
        monkeypatch,
        "train",
        "--config",
        config_path,
        "--output",
        run,
        "--resume",
        run / "checkpoint.pt",
        "--epochs",
        3,
        "--no-report",
    )
    resumed = torch.load(run / "checkpoint.pt", weights_only=True)
    assert resumed["completed_epochs"] == 3
    assert resumed["global_step"] == 9
    assert len(json.loads((run / "history.json").read_text())) == 3
    assert yaml.safe_load((run / "config.yaml").read_text())["training"]["epochs"] == 3
    with pytest.raises(ValueError, match="already exists"):
        _invoke(monkeypatch, "train", "--config", config_path, "--output", run, "--no-report")


def test_sweep_saves_independent_conditioner_runs_and_comparison(tmp_path, monkeypatch):
    config = _config()
    config["environment"]["n_samples"] = 12
    config_path = _write_config(tmp_path, config)
    seen = []

    def light_report(pipeline, dataset, output, **kwargs):
        seen.append(pipeline.model.config.conditioner)
        assert len(kwargs["history"]) == 1
        return {"sliced_wasserstein": 0.0}

    monkeypatch.setattr(cli, "export_report", light_report)
    output = tmp_path / "sweep"
    _invoke(
        monkeypatch,
        "sweep",
        "--config",
        config_path,
        "--output",
        output,
        "--parameter",
        "model.conditioner",
        "--values",
        "one_hot",
        "embedding",
        "--epochs",
        1,
    )
    assert seen == ["one_hot", "embedding"]
    rows = json.loads((output / "comparison.json").read_text())
    assert [row["value"] for row in rows] == seen
    with (output / "comparison.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
    for index, conditioner in enumerate(seen):
        run = output / f"run_{index:02d}"
        model_config = json.loads((run / "pipeline" / "model" / "config.json").read_text())
        assert model_config["conditioner"] == conditioner
        checkpoint = torch.load(run / "checkpoint.pt", weights_only=True)
        assert checkpoint["completed_epochs"] == 1


def test_omitted_environment_default_remains_reportable(tmp_path, monkeypatch):
    config = _config()
    config.pop("environment")
    config["training"].update(epochs=1, batch_size=256)
    config_path = _write_config(tmp_path, config)
    run = tmp_path / "default_environment"
    _invoke(monkeypatch, "train", "--config", config_path, "--output", run, "--no-report")
    observed = []

    def light_report(pipeline, dataset, output, **kwargs):
        observed.append(dataset.metadata["name"])
        assert dataset.sample_shape == pipeline.model.sample_shape == (2,)
        return {"finite_fraction": 1.0}

    monkeypatch.setattr(cli, "export_report", light_report)
    _invoke(monkeypatch, "report", run, "--num-samples", 6, "--steps", 4)
    assert observed == ["rotating_rays"]


@pytest.mark.parametrize("steps,guidance", [(0, 1), (13, 1), (4, -1), (4, "nan")])
def test_cli_rejects_invalid_sampling_parameters(tmp_path, monkeypatch, steps, guidance):
    dataset, pipeline, trainer = cli.build_experiment(_config())
    model_path = tmp_path / "pipeline"
    pipeline.save_pretrained(model_path)
    with pytest.raises(ValueError):
        _invoke(
            monkeypatch,
            "sample",
            model_path,
            "--output",
            tmp_path / "bad",
            "--num-samples",
            3,
            "--steps",
            steps,
            "--guidance",
            guidance,
        )


def test_cli_condition_file_batch_and_unknown_config_errors_are_clear(tmp_path, monkeypatch):
    _, pipeline, _ = cli.build_experiment(_config())
    model_path = tmp_path / "pipeline"
    pipeline.save_pretrained(model_path)
    conditions = tmp_path / "conditions.npy"
    np.save(conditions, np.array([0, 1]))
    with pytest.raises(ValueError, match="batch_size"):
        _invoke(
            monkeypatch,
            "sample",
            model_path,
            "--output",
            tmp_path / "bad_conditions",
            "--num-samples",
            3,
            "--steps",
            4,
            "--conditions-file",
            conditions,
        )
    with pytest.raises(FileNotFoundError):
        _invoke(
            monkeypatch,
            "sample",
            model_path,
            "--output",
            tmp_path / "missing_conditions",
            "--num-samples",
            3,
            "--steps",
            4,
            "--conditions-file",
            tmp_path / "missing.npy",
        )
    with pytest.raises(SystemExit) as error:
        _invoke(
            monkeypatch,
            "sample",
            model_path,
            "--output",
            tmp_path / "conflicting_conditions",
            "--num-samples",
            3,
            "--steps",
            4,
            "--conditions-file",
            conditions,
            "--condition",
            "0",
        )
    assert error.value.code == 2
    config_path = _write_config(tmp_path, {"unknown_section": {}})
    with pytest.raises(ValueError, match="Unknown config sections"):
        _invoke(monkeypatch, "train", "--config", config_path, "--output", tmp_path / "bad_config")
