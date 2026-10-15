"""Numerical contracts for independent quality and paired fidelity."""

import math

import numpy as np
import pytest
import torch

from recheck.evaluate import (
    evaluator_preprocess,
    exact_empirical_w2_squared,
    fit_feature_reference,
    fixed_projection_bank,
    metric_device_parity_report,
    mmd2_biased,
    mmd2_unbiased,
    mnist_quality,
    nearest_train_distances,
    paired_fidelity,
    paired_statistics,
    pixel_ssim,
    rays_quality,
    sliced_wasserstein_fixed,
    timed_generation,
    toy_quality,
)


def _random(shape, seed=12):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _dense_kernel(x, y, ell2):
    cost = (x[:, None] - y[None]).square().sum(-1)
    return sum(torch.exp(-cost / (2 * scale**2 * ell2)) for scale in (0.5, 1.0, 2.0)) / 3


def test_unbiased_mmd_blocks_match_independent_dense_estimator():
    x, y, bandwidth = _random((5, 3)), _random((7, 3), 13), 1.7
    xx, yy, xy = (
        _dense_kernel(x, x, bandwidth),
        _dense_kernel(y, y, bandwidth),
        _dense_kernel(x, y, bandwidth),
    )
    expected = (xx.sum() - xx.diag().sum()) / 20 + (yy.sum() - yy.diag().sum()) / 42 - 2 * xy.mean()
    for block_size in (1, 2, 4, 20):
        assert mmd2_unbiased(x, y, bandwidth, block_size=block_size) == pytest.approx(
            float(expected), abs=1e-14
        )
    assert mmd2_unbiased(
        x.flip(0), y[[2, 1, 6, 3, 4, 5, 0]], bandwidth, block_size=2
    ) == pytest.approx(float(expected), abs=1e-14)


def test_mmd_cross_diagonal_is_retained_and_u_statistic_can_be_negative():
    x = torch.tensor([[0.0], [10.0]], dtype=torch.float64)
    # This is a deliberately dependent identity example, not two independent samples.
    assert mmd2_unbiased(x, x, 1, block_size=1) < -0.99
    assert mmd2_biased(x, x, 1, block_size=1) == pytest.approx(0, abs=1e-14)
    with pytest.raises(ValueError, match=">=2"):
        mmd2_unbiased(x[:1], x, 1)


def test_reference_uses_fixed_train_statistics_original_ids_and_private_rng():
    train = torch.tensor([[1.0, 4.0], [3.0, 4.0], [5.0, 4.0], [7.0, 4.0]], dtype=torch.float64)
    state = torch.random.get_rng_state().clone()
    ref = fit_feature_reference(train, seed=3, bandwidth_samples=3, indices=torch.arange(4) + 50)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(ref.mean, torch.tensor([4.0, 4.0], dtype=torch.float64))
    assert ref.std[0] == pytest.approx(math.sqrt(5))
    assert ref.std[1] == pytest.approx(1e-6)
    assert set(ref.indices.tolist()) <= set(range(50, 54))
    assert torch.equal(
        ref.indices,
        fit_feature_reference(
            train, seed=3, bandwidth_samples=3, indices=torch.arange(4) + 50
        ).indices,
    )
    assert ref.transform(train + 10)[:, 0].mean() > 4  # No generated-set renormalization.
    raw = fit_feature_reference(train, standardize=False, bandwidth_samples=4)
    assert torch.equal(raw.mean, torch.zeros(2, dtype=torch.float64))
    assert torch.equal(raw.std, torch.ones(2, dtype=torch.float64))


def test_degenerate_reference_is_explicit_and_positive():
    reference = fit_feature_reference(torch.ones((5, 2)), bandwidth_samples=5)
    assert reference.bandwidth_degenerate
    assert reference.ell2 == 1e-8
    assert reference.to_dict()["fit_split"] == "train_only"


def test_fixed_swd_reuses_bank_and_preserves_existing_w1_convention():
    x = torch.tensor([[0.0, 0], [1, 0]], dtype=torch.float64)
    y = x + torch.tensor([2.0, 0])
    bank = torch.tensor([[1.0, 0], [0, 1]], dtype=torch.float64)
    assert sliced_wasserstein_fixed(x, y, bank) == pytest.approx(1.0)
    assert sliced_wasserstein_fixed(x, x.flip(0), bank) == 0
    state = torch.random.get_rng_state().clone()
    assert torch.equal(fixed_projection_bank(), fixed_projection_bank())
    assert torch.equal(state, torch.random.get_rng_state())
    reference = fit_feature_reference(_random((32, 2)), standardize=False)
    result = rays_quality(y, x, reference, bank, block_size=1)
    assert result["swd_projections"] == 2
    assert result["sliced_wasserstein"] == 1


def test_exact_w2_assignment_translation_and_permutation():
    x = _random((19, 2))
    shift = torch.tensor([2.0, -0.75], dtype=torch.float64)
    assert exact_empirical_w2_squared(x, (x + shift).flip(0)) == pytest.approx(
        float(shift.square().sum()), abs=1e-12
    )
    assert exact_empirical_w2_squared(x, x.flip(0)) == pytest.approx(0, abs=1e-14)


def test_toy_occupancy_compares_observed_assignments_and_records_five_subsamples():
    centers = torch.tensor([[-0.6, 0], [0.6, 0], [0, 0.65]], dtype=torch.float64)
    real = centers.repeat(8, 1)
    generated = real.clone()
    generated[0] = torch.tensor([10.0, 0])
    reference = fit_feature_reference(real, standardize=False)
    state = torch.random.get_rng_state().clone()
    result = toy_quality(generated, real, reference, subsample_size=16, repeats=5)
    assert torch.equal(state, torch.random.get_rng_state())
    assert len(result["exact_w2_subsamples"]) == 5
    assert result["offmode_fraction_generated"] == pytest.approx(1 / 24)
    assert result["offmode_fraction_real"] == 0
    assert result["occupancy_tv_real"] == pytest.approx(1 / 24)
    with pytest.raises(ValueError, match="registered number"):
        toy_quality(real, real, reference)  # Do not silently shrink protocol 512.


def test_nearest_train_blocking_matches_dense():
    generated, train = _random((7, 4)), _random((13, 4), 15)
    expected = torch.cdist(generated, train).min(1).values
    assert torch.allclose(
        nearest_train_distances(generated, train, block_size=3), expected, atol=1e-14
    )


def test_preprocessing_records_range_before_common_clamp():
    raw = torch.tensor([[[[-3.0, -1], [1, 3]]]])
    pixel, result = evaluator_preprocess(raw)
    assert torch.equal(pixel, torch.tensor([[[[0.0, 0], [1, 1]]]]))
    assert result == {"out_of_range_fraction": 0.5, "max_range_violation": 1.0}


class _Evaluator(torch.nn.Module):
    def forward(self, pixel01, return_features=False):
        assert pixel01.min() >= 0 and pixel01.max() <= 1
        mean = pixel01.flatten(1).mean(1)
        features = torch.stack((mean, mean.square()), 1)
        logits = torch.stack((1 - mean, mean), 1) * 10
        return (logits, features) if return_features else logits


def test_mnist_quality_runs_on_synthetic_images_without_dataset_or_rng_changes():
    evaluator = _Evaluator().train()
    raw = _random((8, 1, 8, 8))
    pixels, _ = evaluator_preprocess(raw)
    real_logits, real_features = evaluator(pixels, return_features=True)
    reference = fit_feature_reference(real_features, bandwidth_samples=8)
    state = torch.random.get_rng_state().clone()
    result = mnist_quality(
        raw,
        real_features,
        real_logits,
        real_features,
        evaluator,
        reference,
        batch_size=3,
        block_size=3,
    )
    assert evaluator.training
    assert torch.equal(state, torch.random.get_rng_state())
    assert result["digit_frequency_tv_real"] == 0
    assert result["nearest_train_feature_distance_mean"] == pytest.approx(0, abs=1e-8)
    assert result["mmd2_mnist_features_real"] < 0  # Dependent identity case is kept explicit.


def test_paired_fidelity_raw_mse_ssim_range_and_zero_psnr():
    x = _random((3, 1, 12, 12))
    identity = paired_fidelity(x, x, data_range=2)
    assert identity["paired_endpoint_mse"] == 0
    assert identity["pixel_ssim"] == pytest.approx(1, abs=1e-14)
    assert math.isinf(identity["psnr_db"])
    y = x + 0.2
    result = paired_fidelity(y, x, data_range=2)
    assert result["paired_endpoint_mse"] == pytest.approx(0.04)
    assert result["ssim_data_range"] == 2
    assert result["psnr_db"] == pytest.approx(20)


def test_ssim_matches_constant_image_luminance_term_and_batch_one():
    x, y = torch.full((1, 1, 9, 9), 0.25), torch.full((1, 1, 9, 9), 0.75)
    expected = (2 * 0.25 * 0.75 + 0.01**2) / (0.25**2 + 0.75**2 + 0.01**2)
    assert pixel_ssim(x, y, data_range=1).item() == pytest.approx(expected, abs=1e-10)


def test_paired_statistics_uses_sample_std_and_seed_level_t_interval():
    report = paired_statistics([3, 4, 5], [1, 1, 1])
    assert report["values"] == [2, 3, 4]
    assert report["mean"] == 3
    assert report["sample_std"] == 1
    assert report["paired_t_ci"] == pytest.approx(
        [3 - 4.3026527299 / 3**0.5, 3 + 4.3026527299 / 3**0.5]
    )
    assert paired_statistics([0.2])["paired_t_ci"] is None
    with pytest.raises(ValueError):
        paired_statistics([0.1, 0.2], [0.1])


def test_nonfinite_metrics_fail_instead_of_masking_bad_runs():
    with pytest.raises(ValueError, match="finite"):
        mmd2_unbiased([[0.0], [float("nan")]], [[0.0], [1.0]], 1)
    assert np.isfinite(mmd2_unbiased([[0.0], [1.0]], [[2.0], [3.0]], 1))


def test_timed_generation_synchronizes_mps_before_and_after_each_repeat(monkeypatch):
    calls, samples = [], []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("sync"))
    result = timed_generation(lambda: samples.append("sample"), repeats=3, warmup=1, device="mps")
    assert len(calls) == 6
    assert len(samples) == 4
    assert result["device"] == "mps"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Requires a real Metal device outside the sandbox"
)
def test_actual_mps_metrics_match_cpu_float64_after_safe_transport():
    report = metric_device_parity_report("mps")
    assert report["passed"], report
