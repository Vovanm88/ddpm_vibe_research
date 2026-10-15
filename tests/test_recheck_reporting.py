"""Reports preserve pairing, honest scope, and exact saved animation states."""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from recheck.reporting import (
    _mnist_claims,
    _mnist_preview,
    _set_trajectory_frame,
    _trajectory_figure,
    actual_frame_indices,
    build_report,
    compute_contrasts,
    save_comparison_gif,
)


def _histories():
    initial = torch.tensor([[-0.75, -0.5], [0.75, 0.5]], dtype=torch.float64)
    long = torch.stack([initial + torch.tensor([index * 0.1, index * 0.04]) for index in range(5)])
    short = torch.stack(
        [initial + torch.tensor([-index * 0.12, index * 0.08]) for index in range(3)]
    )
    return {"Original N4": long, "M0": short}


def test_actual_frame_selection_uses_preceding_existing_node_no_interpolation():
    fractions, indices = actual_frame_indices(_histories(), max_frames=5)
    assert fractions == [0, 0.25, 0.5, 0.75, 1]
    assert indices["Original N4"] == [0, 1, 2, 3, 4]
    assert indices["M0"] == [0, 0, 1, 1, 2]


def test_trajectory_heads_and_segments_correspond_exactly_to_saved_nodes():
    histories = _histories()
    figure, artists = _trajectory_figure(
        histories, particles=2, real=None, bounds=([-1, -1], [2, 2])
    )
    try:
        _set_trajectory_frame(artists, {"Original N4": 2, "M0": 1})
        for _, name, path, trail, head in artists:
            index = 2 if name == "Original N4" else 1
            assert np.array_equal(head.get_offsets(), histories[name][index].numpy())
            assert len(trail.get_segments()) == index * 2
            assert np.array_equal(trail.get_segments()[0], np.stack((path[0, 0], path[1, 0])))
    finally:
        plt.close(figure)


def test_gif_has_moving_frames_saved_provenance_and_no_source_mutation(tmp_path):
    histories = _histories()
    original = {name: value.clone() for name, value in histories.items()}
    metadata = save_comparison_gif(
        histories, tmp_path / "comparison.gif", max_frames=3, particles=2, fps=4, dpi=45
    )
    assert metadata["interpolation"] is False
    assert metadata["actual_node_indices"] == {"Original N4": [0, 2, 4], "M0": [0, 1, 2]}
    assert "red start" in metadata["colors"] and "violet finish" in metadata["colors"]
    with Image.open(tmp_path / "comparison.gif") as image:
        assert image.n_frames == 3
        image.seek(0)
        first = np.asarray(image.convert("RGB")).copy()
        image.seek(2)
        last = np.asarray(image.convert("RGB")).copy()
        assert np.mean(first != last) > 0.01
    assert all(torch.equal(histories[name], value) for name, value in original.items())


def test_mismatched_noise_fails_before_comparison(tmp_path):
    histories = _histories()
    histories["M0"][0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="same initial noise"):
        save_comparison_gif(histories, tmp_path / "bad.gif")
    assert not (tmp_path / "bad.gif").exists()


def _rows():
    rows = [
        {
            "run_id": "teacher4",
            "method": "Original N4",
            "student_seed": None,
            "nfe": 4,
            "quality": {"mmd2_real": 0.001, "sliced_wasserstein": 0.04},
            "fidelity": {"paired_endpoint_mse": 0.0},
            "cost": {"training_seconds": 0.0, "median_sampling_seconds": 0.02},
        },
        {
            "run_id": "teacher2",
            "method": "Original N2",
            "student_seed": None,
            "nfe": 2,
            "quality": {"mmd2_real": 0.002, "sliced_wasserstein": 0.05},
            "fidelity": {"paired_endpoint_mse": 0.01},
            "cost": {"training_seconds": 0.0, "median_sampling_seconds": 0.01},
        },
    ]
    effects = {"M0": 0.0, "M1": -0.0001, "E0": 0.0002, "E1": -0.0002}
    for index, seed in enumerate((101, 202, 303)):
        for method, effect in effects.items():
            rows.append(
                {
                    "run_id": f"seed{seed}_{method}",
                    "method": method,
                    "student_seed": seed,
                    "nfe": 2,
                    "quality": {
                        "mmd2_real": 0.001 + index * 0.00003 + effect,
                        "sliced_wasserstein": 0.04 + effect * 10,
                    },
                    "fidelity": {"paired_endpoint_mse": 0.002},
                    "cost": {"training_seconds": 2.0, "median_sampling_seconds": 0.01},
                }
            )
    return rows


def test_contrasts_pair_training_seeds_and_do_not_replicate_frozen_baseline():
    rows = compute_contrasts(_rows())
    first = next(
        row for row in rows if row["contrast"] == "E0_minus_M0" and row["metric"] == "mmd2_real"
    )
    assert first["n_student_seeds"] == 3
    assert first["student_seeds"] == [101, 202, 303]
    assert first["values"] == pytest.approx([0.0002] * 3)
    assert all("Original" not in row.get("first", "") for row in rows)
    interaction = next(
        row
        for row in rows
        if row["contrast"] == "OT_x_tube_interaction" and row["metric"] == "mmd2_real"
    )
    assert interaction["mean"] == pytest.approx(-0.0003)


def test_duplicate_seed_results_rejected_and_missing_pairs_are_explicit():
    rows = _rows()
    with pytest.raises(ValueError, match="Duplicate"):
        compute_contrasts(rows + [rows[-1]])
    subset = [row for row in rows if row["method"] != "E0" or row["student_seed"] == 101]
    contrast = next(row for row in compute_contrasts(subset) if row["contrast"] == "E0_minus_M0")
    assert contrast["n_student_seeds"] == 1
    assert contrast["paired_t_ci"] is None


def test_report_builds_csv_figures_actual_gifs_and_keeps_mnist_not_tested(tmp_path):
    root, report_dir = tmp_path / "run", tmp_path / "report"
    evaluation = root / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "results.json").write_text(
        json.dumps(
            {
                "rows": _rows(),
                "metadata": {
                    "protocol": "synthetic_report_test",
                    "teacher_sha": "fixed",
                    "noise_sha": "same",
                },
            }
        )
    )
    histories = _histories()
    for method in ("M1", "E0", "E1"):
        histories[method] = histories["M0"].clone()
    histories["Original N2"] = histories["M0"].clone()
    torch.save(histories, evaluation / "trajectories.pt")
    torch.save({"test": histories["Original N4"][-1]}, root / "datasets.pt")
    config = {
        "protocol_id": "synthetic_report_test",
        "distillation": {"seeds": [101, 202, 303], "teacher_steps": 4, "student_steps": 2},
        "evaluation": {"gif_particles": 2, "gif_fps": 4},
    }
    summary = build_report(root, report_dir, config)
    assert summary["rows"] == 14
    assert summary["student_seed_count"] == 3
    assert summary["mnist_status"] == "not_tested"
    assert len(summary["animations"]) == 7
    assert (report_dir / "quality_vs_nfe.svg").is_file()
    assert (report_dir / "trajectories_comparison.gif").is_file()
    with (report_dir / "results_per_seed.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 14
    assert sum(row["student_seed"] == "" for row in rows) == 2
    claims = json.loads((report_dir / "claims_verdict.json").read_text())
    mnist = [row for row in claims if row["scope"] == "unconditional_pixel_MNIST"]
    assert len(mnist) == 4
    assert all(
        row["status"] == "not_tested"
        and row["effect_mean"] is None
        and row["student_seed_count"] == 0
        for row in mnist
    )
    text = (report_dir / "REPORT.md").read_text()
    assert "не результаты MNIST" in text
    assert "не доказывает эквивалентность" in text
    assert "0.0002" in text  # A worse OT contrast is retained rather than hidden.
    manifest = json.loads((report_dir / "animations_manifest.json").read_text())
    assert manifest["student_seed"] == 101
    assert all(not item["interpolation"] for item in manifest["animations"])


def test_mnist_feature_contrasts_and_claims_use_actual_rows_not_ray_placeholder(tmp_path):
    rows = _rows()
    for row in rows:
        row["quality"] = {
            "mmd2_mnist_features_real": row["quality"]["mmd2_real"],
            "digit_frequency_tv_real": 0.02,
        }
    contrasts = compute_contrasts(rows)
    assert all(
        row["metric"] in ("mmd2_mnist_features_real", "digit_frequency_tv_real")
        for row in contrasts
    )
    claims = _mnist_claims(
        rows, contrasts, "teacher4_student2_euler", tmp_path / "results.json", tmp_path
    )
    h1 = next(row for row in claims if row["claim_id"] == "H1")
    assert h1["status"] == "no_advantage_in_tested_setting"
    assert h1["effect_mean"] == pytest.approx(0.0002)
    assert h1["student_seed_count"] == 3
    h3 = next(row for row in claims if row["claim_id"] == "H3")
    assert h3["status"] == "not_tested"
    assert h3["effect_mean"] is None


def test_mnist_claims_quality_improvement_cannot_establish_tube_robustness(tmp_path):
    rows = _rows()
    for row in rows:
        row["quality"] = {
            "mmd2_mnist_features_real": row["quality"]["mmd2_real"],
        }
    contrasts = compute_contrasts(rows)
    quality = next(row for row in contrasts if row["contrast"] == "M1_minus_M0")
    assert quality["paired_t_ci"][1] < 0  # Convincing quality evidence is insufficient.
    h2 = next(
        row
        for row in _mnist_claims(
            rows, contrasts, "teacher4_student2", tmp_path / "results.json", tmp_path
        )
        if row["claim_id"] == "H2"
    )
    assert h2["status"] == "not_tested"
    assert h2["metric"] == "local_rollout_robustness"
    assert h2["effect_mean"] is None and h2["paired_t_ci"] is None
    assert h2["student_seed_count"] == 0
    assert h2["quality_proxy"]["status"] == "supported_on_small_benchmark"
    assert h2["quality_proxy"]["effect_mean"] == pytest.approx(-0.0001)
    assert any("cannot establish robustness" in text for text in h2["limitations"])
    # The measured M1-minus-M0 quality contrast is retained for the CSV producer.
    assert quality in compute_contrasts(rows)


@pytest.mark.parametrize("costs_recorded", [False, True])
def test_mnist_claims_quality_gain_requires_a_cost_qualification_for_h4(tmp_path, costs_recorded):
    rows = _rows()
    for row in rows:
        row["quality"] = {
            "mmd2_mnist_features_real": row["quality"]["mmd2_real"],
        }
        if not costs_recorded:
            row.pop("cost")
        elif row["method"] == "M0":
            row["cost"] = {"training_seconds": 1_000_000.0, "median_sampling_seconds": 500.0}
    h4 = next(
        row
        for row in _mnist_claims(
            rows, compute_contrasts(rows), "teacher4_student2", tmp_path / "results.json", tmp_path
        )
        if row["claim_id"] == "H4"
    )
    assert h4["quality_status"] == "supported_on_small_benchmark"
    assert h4["paired_t_ci"][1] < 0
    assert h4["status"] == "inconclusive"
    assert h4["cost_qualification"]["status"] == "not_assessed"
    records = h4["cost_qualification"]["recorded_costs"]
    assert len(records) == 4  # Three independent students and one frozen short-grid original.
    assert {row["run_id"] for row in records} == {
        "teacher2",
        "seed101_M0",
        "seed202_M0",
        "seed303_M0",
    }
    if costs_recorded:
        assert all(
            row["cost"]["training_seconds"] == 1_000_000.0
            for row in records
            if row["run_id"].endswith("M0")
        )
    else:
        assert all(row["cost"] == {} for row in records)
    assert any(
        "recorded cost columns alone cannot establish H4" in text for text in h4["limitations"]
    )


def test_mnist_preview_uses_actual_nodes_fixed_clamp_and_common_noise(tmp_path):
    initial = torch.full((2, 1, 28, 28), -2.0)
    histories = {
        "Original N2": torch.stack((initial, initial + 2, initial + 4)),
        "M0": torch.stack((initial, initial + 1, initial + 3)),
    }
    config = {"evaluation": {"gif_particles": 2, "gif_fps": 4}}
    animations, figures = _mnist_preview(histories, tmp_path, config)
    assert animations[0]["actual_node_indices"] == {"M0": [0, 1, 2], "Original N2": [0, 1, 2]}
    assert animations[0]["interpolation"] is False
    assert "clamp" in animations[0]["visualization_preprocessing"]
    assert len(figures) == 2
    assert torch.equal(histories["M0"][0], initial)
    with Image.open(animations[0]["file"]) as image:
        assert image.n_frames == 3
