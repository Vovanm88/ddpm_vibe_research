"""Independent quality estimates, paired fidelity, and small-sample statistics.

Evaluation uses local CPU generators and never calls ``manual_seed`` globally.
Kernel calibration is train-only and reused unchanged for every method. The
unbiased MMD estimator may be negative; an identity V-statistic is provided as
a separate mathematical check, not substituted for the principal estimator.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t
from scipy.stats import wasserstein_distance


def _matrix(value):
    # PyTorch 2.14.1 MPS can silently zero a fused MPS32→CPU64 conversion.
    # Separate the device transfer from the dtype cast on every metric path.
    result = torch.as_tensor(value).detach().cpu().double()
    if result.ndim < 2 or len(result) == 0:
        raise ValueError("Expected a nonempty batch with at least one feature dimension")
    result = result.flatten(1)
    if not torch.isfinite(result).all():
        raise ValueError("Metrics require finite features")
    return result


@dataclass(frozen=True)
class FeatureReference:
    """Frozen train feature statistics and the RBF squared bandwidth."""

    mean: torch.Tensor
    std: torch.Tensor
    ell2: float
    indices: torch.Tensor
    seed: int
    bandwidth_degenerate: bool = False

    def transform(self, values):
        values = _matrix(values)
        if values.shape[1] != self.mean.numel():
            raise ValueError("Feature dimension differs from the train reference")
        return (values - self.mean) / self.std

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "ell2": self.ell2,
            "ell": self.ell2**0.5,
            "indices": self.indices.tolist(),
            "seed": self.seed,
            "bandwidth_degenerate": self.bandwidth_degenerate,
            "kernel_scales": [0.5, 1.0, 2.0],
            "std_floor": 1e-6,
            "fit_split": "train_only",
        }


def fit_feature_reference(
    train_features, seed=41, bandwidth_samples=2000, *, standardize=True, indices=None
):
    """Fit on real train only; all pairwise distances include duplicate points.

    Only self diagonals are removed for the bandwidth median. An explicit floor
    and flag handle degenerate train features instead of hiding the degeneracy.
    ``standardize=False`` is useful for a fixed RBF in the physical toy plane.
    """
    features = _matrix(train_features)
    if len(features) < 2 or bandwidth_samples < 2:
        raise ValueError("At least two train features are needed for a bandwidth")
    mean = features.mean(0) if standardize else torch.zeros(features.shape[1], dtype=features.dtype)
    std = features.std(0, correction=0).clamp_min(1e-6) if standardize else torch.ones_like(mean)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    chosen = torch.randperm(len(features), generator=generator)[
        : min(bandwidth_samples, len(features))
    ]
    selected = (features[chosen] - mean) / std
    distances = torch.pdist(selected).square()
    ell2_raw = float(distances.median())
    ids = chosen if indices is None else torch.as_tensor(indices, dtype=torch.long)[chosen]
    return FeatureReference(mean, std, max(ell2_raw, 1e-8), ids, int(seed), ell2_raw < 1e-8)


def _kernel_sum(x, y, ell2, scales, block_size, exclude_diagonal=False):
    total = torch.zeros((), dtype=torch.float64)
    for i in range(0, len(x), block_size):
        for j in range(0, len(y), block_size):
            distances = torch.cdist(x[i : i + block_size], y[j : j + block_size]).square()
            kernel = sum(torch.exp(-distances / (2 * float(scale) ** 2 * ell2)) for scale in scales)
            kernel = kernel / len(scales)
            if exclude_diagonal:
                # A block's local diagonal is not a global diagonal when i != j.
                row = torch.arange(i, min(i + block_size, len(x)))[:, None]
                col = torch.arange(j, min(j + block_size, len(y)))[None, :]
                kernel = kernel.masked_fill(row == col, 0)
            total += kernel.sum()
    return total


def mmd2_unbiased(x, y, ell2, *, block_size=512, scales=(0.5, 1.0, 2.0)):
    """Two-sample U-statistic, assuming independent x/y samples.

    Cross diagonals are retained even if the two batch lengths happen to match.
    Do not interpret using the same examples twice as an unbiased identity test.
    """
    x, y = _matrix(x), _matrix(y)
    if len(x) < 2 or len(y) < 2 or x.shape[1] != y.shape[1]:
        raise ValueError("MMD needs >=2 examples in each batch and matching feature dimensions")
    if not np.isfinite(ell2) or ell2 <= 0 or block_size < 1 or not scales or min(scales) <= 0:
        raise ValueError("Bandwidth, kernel scales, and block size must be positive")
    xx = _kernel_sum(x, x, ell2, scales, block_size, True) / (len(x) * (len(x) - 1))
    yy = _kernel_sum(y, y, ell2, scales, block_size, True) / (len(y) * (len(y) - 1))
    xy = _kernel_sum(x, y, ell2, scales, block_size) / (len(x) * len(y))
    return float(xx + yy - 2 * xy)


def mmd2_biased(x, y, ell2, *, block_size=512, scales=(0.5, 1.0, 2.0)):
    """V-statistic for identity sanity checks; not the principal quality metric."""
    x, y = _matrix(x), _matrix(y)
    if x.shape[1] != y.shape[1] or ell2 <= 0 or block_size < 1:
        raise ValueError("Invalid kernel inputs")
    xx = _kernel_sum(x, x, ell2, scales, block_size) / len(x) ** 2
    yy = _kernel_sum(y, y, ell2, scales, block_size) / len(y) ** 2
    xy = _kernel_sum(x, y, ell2, scales, block_size) / (len(x) * len(y))
    return float(xx + yy - 2 * xy)


def fixed_projection_bank(dimension=2, n_projections=1024, seed=1717):
    if dimension < 1 or n_projections < 1:
        raise ValueError("Projection dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bank = torch.randn((n_projections, dimension), generator=generator, dtype=torch.float64)
    return bank / bank.norm(dim=1, keepdim=True)


def sliced_wasserstein_fixed(x, y, projection_bank):
    """Mean projected W1, matching the existing project's SWD quality metric.

    This differs from the dimension-scaled squared SW2 *training loss*.
    """
    x, y, bank = _matrix(x).numpy(), _matrix(y).numpy(), _matrix(projection_bank).numpy()
    if x.shape[1] != y.shape[1] or bank.shape[1] != x.shape[1]:
        raise ValueError("Projection bank has the wrong feature dimension")
    if not np.allclose(np.linalg.norm(bank, axis=1), 1, atol=1e-8):
        raise ValueError("Evaluation directions must be unit vectors")
    a, b = x @ bank.T, y @ bank.T
    if len(x) == len(y):
        return float(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).mean())
    return float(np.mean([wasserstein_distance(a[:, i], b[:, i]) for i in range(len(bank))]))


def rays_quality(generated, real, reference, projection_bank=None, *, block_size=512):
    if projection_bank is None:
        projection_bank = fixed_projection_bank(_matrix(real).shape[1])
    return {
        "mmd2_real": mmd2_unbiased(
            reference.transform(generated),
            reference.transform(real),
            reference.ell2,
            block_size=block_size,
        ),
        "sliced_wasserstein": sliced_wasserstein_fixed(real, generated, projection_bank),
        "n_generated": len(generated),
        "n_real": len(real),
        "swd_projections": len(projection_bank),
    }


def exact_empirical_w2_squared(x, y):
    """Squared Euclidean assignment cost, without dividing by dimension."""
    x, y = _matrix(x), _matrix(y)
    if x.shape != y.shape:
        raise ValueError("Equal size empirical measures are required for linear assignment")
    cost = torch.cdist(x, y).square().numpy()
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())


def toy_quality(
    generated,
    real,
    reference,
    *,
    seed=1818,
    subsample_size=512,
    repeats=5,
    centers=((-0.6, 0.0), (0.6, 0.0), (0.0, 0.65)),
    sigma=0.15,
    block_size=512,
):
    generated, real = _matrix(generated), _matrix(real)
    if len(generated) < subsample_size or len(real) < subsample_size or repeats < 1:
        raise ValueError("Toy W2 requires the registered number of examples in both sets")
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(repeats):
        gi = rng.choice(len(generated), subsample_size, replace=False)
        ri = rng.choice(len(real), subsample_size, replace=False)
        values.append(exact_empirical_w2_squared(generated[gi], real[ri]))
    centers = torch.as_tensor(centers, dtype=torch.float64)
    gd, rd = torch.cdist(generated, centers), torch.cdist(real, centers)
    go = torch.bincount(gd.argmin(1), minlength=len(centers)).double() / len(generated)
    ro = torch.bincount(rd.argmin(1), minlength=len(centers)).double() / len(real)
    return {
        "mmd2_real": mmd2_unbiased(
            reference.transform(generated),
            reference.transform(real),
            reference.ell2,
            block_size=block_size,
        ),
        "exact_empirical_w2_squared": float(np.mean(values)),
        "exact_w2_subsamples": values,
        "exact_w2_subsample_size": subsample_size,
        "exact_w2_repeats": repeats,
        "nearest_center_occupancy_generated": go.tolist(),
        "nearest_center_occupancy_real": ro.tolist(),
        "occupancy_tv_real": float(0.5 * (go - ro).abs().sum()),
        "offmode_fraction_generated": float((gd.min(1).values > 3 * sigma).double().mean()),
        "offmode_fraction_real": float((rd.min(1).values > 3 * sigma).double().mean()),
        "occupancy_definition": "nearest center, not latent component labels",
    }


def evaluator_preprocess(raw_images):
    """Identical FM [-1,1] -> pixel [0,1] conversion for every method."""
    raw = torch.as_tensor(raw_images)
    if raw.ndim != 4 or not torch.isfinite(raw).all():
        raise ValueError("Expected finite raw FM images [N,C,H,W]")
    pixel = (raw + 1) / 2
    violation = torch.maximum(-pixel, pixel - 1).clamp_min(0)
    return pixel.clamp(0, 1), {
        "out_of_range_fraction": float(((pixel < 0) | (pixel > 1)).float().mean()),
        "max_range_violation": float(violation.max()),
    }


@torch.no_grad()
def extract_mnist_features(evaluator, raw_images, *, batch_size=128):
    pixel, diagnostics = evaluator_preprocess(raw_images)
    pixel = pixel.detach().cpu()
    parameter = next(evaluator.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else pixel.dtype
    was_training = evaluator.training
    evaluator.eval()
    features, logits = [], []
    try:
        for chunk in pixel.split(batch_size):
            output = evaluator(chunk.to(dtype=dtype).to(device=device), return_features=True)
            if isinstance(output, dict):
                logit, feature = output["logits"], output["features"]
            else:
                logit, feature = output
            logits.append(logit.detach().cpu())
            features.append(feature.detach().cpu())
    finally:
        evaluator.train(was_training)
    return torch.cat(features), torch.cat(logits), diagnostics


def nearest_train_distances(generated_features, train_features, *, block_size=512):
    generated, train = _matrix(generated_features), _matrix(train_features)
    minimum = []
    for chunk in generated.split(block_size):
        best = torch.full((len(chunk),), float("inf"), dtype=torch.float64)
        for train_chunk in train.split(block_size):
            best = torch.minimum(
                best,
                torch.cdist(chunk, train_chunk, compute_mode="donot_use_mm_for_euclid_dist")
                .min(1)
                .values,
            )
        minimum.append(best)
    return torch.cat(minimum)


def _within_class_diversity(features, classes, num_classes):
    values = []
    for label in range(num_classes):
        part = features[classes == label]
        values.append(float(part.var(0, correction=1).mean()) if len(part) > 1 else None)
    return values


def mnist_quality(
    raw_images,
    real_features,
    real_logits,
    train_features,
    evaluator,
    reference,
    *,
    batch_size=128,
    block_size=512,
):
    features, logits, diagnostics = extract_mnist_features(
        evaluator, raw_images, batch_size=batch_size
    )
    generated = reference.transform(features)
    real, train = reference.transform(real_features), reference.transform(train_features)
    logits, real_logits = _matrix(logits), _matrix(real_logits)
    probability = logits.softmax(1)
    classes, real_classes = logits.argmax(1), real_logits.argmax(1)
    nclasses = logits.shape[1]
    occupancy = torch.bincount(classes, minlength=nclasses).double() / len(classes)
    real_occupancy = torch.bincount(real_classes, minlength=nclasses).double() / len(real_classes)
    nearest = nearest_train_distances(generated, train, block_size=block_size)
    return {
        "mmd2_mnist_features_real": mmd2_unbiased(
            generated, real, reference.ell2, block_size=block_size
        ),
        "predicted_digit_frequencies": occupancy.tolist(),
        "real_predicted_digit_frequencies": real_occupancy.tolist(),
        "digit_frequency_tv_real": float(0.5 * (occupancy - real_occupancy).abs().sum()),
        "mean_confidence": float(probability.max(1).values.mean()),
        "mean_predictive_entropy_nats": float(
            -(probability * probability.clamp_min(1e-30).log()).sum(1).mean()
        ),
        "within_class_feature_variance_generated": _within_class_diversity(
            generated, classes, nclasses
        ),
        "within_class_feature_variance_real": _within_class_diversity(real, real_classes, nclasses),
        "nearest_train_feature_distance_mean": float(nearest.mean()),
        "nearest_train_feature_distance_median": float(nearest.median()),
        "nearest_train_feature_distance_p05": float(torch.quantile(nearest, 0.05)),
        "n_generated": len(features),
        "n_real": len(real),
        **diagnostics,
    }


def pixel_ssim(x, y, *, data_range, window_size=11, sigma=1.5):
    """Per-image Gaussian-window SSIM (valid windows, population moments)."""
    x, y = torch.as_tensor(x).detach().cpu().double(), torch.as_tensor(y).detach().cpu().double()
    if x.ndim != 4 or x.shape != y.shape or data_range <= 0:
        raise ValueError("SSIM requires equal [N,C,H,W] tensors and explicit positive data_range")
    size = min(window_size, x.shape[-2], x.shape[-1])
    size -= 1 - size % 2
    axis = torch.arange(size, dtype=x.dtype, device=x.device) - size // 2
    gaussian = torch.exp(-axis.square() / (2 * sigma**2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(x.shape[1], 1, size, size)

    def conv(z):
        return F.conv2d(z, kernel, groups=x.shape[1])

    mx, my = conv(x), conv(y)
    vx, vy, cov = (
        conv(x.square()) - mx.square(),
        conv(y.square()) - my.square(),
        conv(x * y) - mx * my,
    )
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    scores = ((2 * mx * my + c1) * (2 * cov + c2)) / (
        (mx.square() + my.square() + c1) * (vx + vy + c2)
    )
    return scores.flatten(1).mean(1)


def paired_fidelity(
    student,
    teacher,
    *,
    data_range=2.0,
    student_logits=None,
    teacher_logits=None,
    student_history=None,
    teacher_history=None,
):
    student = torch.as_tensor(student).detach().cpu().double()
    teacher = torch.as_tensor(teacher).detach().cpu().double()
    if (
        student.shape != teacher.shape
        or not torch.isfinite(student).all()
        or not torch.isfinite(teacher).all()
    ):
        raise ValueError("Paired fidelity needs finite samples of the same shape and ordering")
    mse = float((student - teacher).square().mean())
    result = {"paired_endpoint_mse": mse, "paired_examples": len(student)}
    if student.ndim == 4:
        result["pixel_ssim"] = float(pixel_ssim(student, teacher, data_range=data_range).mean())
        result["ssim_data_range"] = float(data_range)
        result["psnr_db"] = float("inf") if mse == 0 else float(10 * np.log10(data_range**2 / mse))
    if student_logits is not None or teacher_logits is not None:
        if student_logits is None or teacher_logits is None:
            raise ValueError("Both paired classifier logits must be provided")
        result["class_agreement"] = float(
            (
                torch.as_tensor(student_logits).detach().cpu().argmax(1)
                == torch.as_tensor(teacher_logits).detach().cpu().argmax(1)
            )
            .float()
            .mean()
        )
    if student_history is not None or teacher_history is not None:
        if student_history is None or teacher_history is None:
            raise ValueError("Both trajectories must contain the same common grid nodes")
        sh = torch.as_tensor(student_history).detach().cpu().double()
        th = torch.as_tensor(teacher_history).detach().cpu().double()
        if sh.shape != th.shape:
            raise ValueError("Trajectory tensors must be matched on common grid nodes")
        result["rollout_mse_per_node"] = (sh - th).square().flatten(1).mean(1).tolist()
    return result


def paired_statistics(first, second=None, *, confidence=0.95):
    """Mean/sample-std and paired t CI across independent *training* seeds.

    Inputs are first-minus-second, never images as pseudo-replicate trainings.
    A CI containing zero is not an equivalence finding. No relative percentage
    is computed: unbiased MMD can be close to zero or negative.
    """
    values = np.asarray(first, dtype=np.float64)
    if second is not None:
        other = np.asarray(second, dtype=np.float64)
        if values.shape != other.shape:
            raise ValueError("Paired seed arrays must have the same shape")
        values = values - other
    if (
        values.ndim != 1
        or len(values) < 1
        or not np.isfinite(values).all()
        or not 0 < confidence < 1
    ):
        raise ValueError("Expected finite per-training values and confidence in (0,1)")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else None
    half = (
        float(student_t.ppf((1 + confidence) / 2, len(values) - 1) * std / len(values) ** 0.5)
        if std is not None
        else None
    )
    return {
        "n_student_seeds": len(values),
        "values": values.tolist(),
        "mean": mean,
        "sample_std": std,
        "confidence": confidence,
        "paired_t_ci": [mean - half, mean + half] if half is not None else None,
        "limitations": "Approximate t interval assumes independent training seeds and approximately normal paired effects; n=3 is a pilot. Conditional on the fixed teacher and shared reference/noise.",
    }


def timed_generation(sample: Callable, *, repeats=10, warmup=2, device="cpu"):
    """Synchronized median/p95; caller specifies the batch size explicitly."""
    import time

    if repeats < 1 or warmup < 0:
        raise ValueError("Timing needs at least one repeat and nonnegative warmup")
    device = torch.device(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        sample()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        sample()
        sync()
        elapsed.append(time.perf_counter() - start)
    return {
        "median_seconds": float(np.median(elapsed)),
        "p95_seconds": float(np.percentile(elapsed, 95)),
        "repeats": repeats,
        "warmup": warmup,
        "device": str(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
    }


def evaluation_numerical_report():
    """Persistable numerical evidence for ``tests_report.json``."""
    global_rng = torch.random.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(9191)
    x = torch.randn((11, 3), generator=generator, dtype=torch.float64)
    y = torch.randn((13, 3), generator=generator, dtype=torch.float64)
    blocked = mmd2_unbiased(x, y, 1.7, block_size=4)
    dense = mmd2_unbiased(x, y, 1.7, block_size=100)
    permuted = mmd2_unbiased(x.flip(0), y.flip(0), 1.7, block_size=4)
    separated = torch.tensor([[0.0], [10.0]], dtype=torch.float64)
    shift = x.new_tensor([0.25, -0.5, 0.75])
    image = torch.randn((2, 1, 12, 12), generator=generator, dtype=torch.float64)
    return {
        "dtype": "torch.float64",
        "mmd_blockwise_dense_abs_error": abs(blocked - dense),
        "mmd_permutation_abs_error": abs(blocked - permuted),
        "mmd_biased_identity_abs_error": abs(mmd2_biased(x, x, 1.7, block_size=4)),
        "mmd_dependent_identity_u_statistic": mmd2_unbiased(separated, separated, 1),
        "assignment_translation_abs_error": abs(
            exact_empirical_w2_squared(x, (x + shift).flip(0)) - float(shift.square().sum())
        ),
        "ssim_identity_max_abs_error": float(
            (pixel_ssim(image, image, data_range=2) - 1).abs().max()
        ),
        "evaluation_leaves_global_rng_unchanged": torch.equal(
            global_rng, torch.random.get_rng_state()
        ),
    }


def metric_device_parity_report(device="mps"):
    """Explicit backend smoke: transport state to CPU before float64 metrics."""
    generator = torch.Generator(device="cpu").manual_seed(9393)
    x, y = torch.randn((11, 3), generator=generator), torch.randn((13, 3), generator=generator)
    image = torch.randn((2, 1, 12, 12), generator=generator)
    xd, yd = x.to(device), y.to(device)
    host_ref, device_ref = fit_feature_reference(x), fit_feature_reference(xd)
    host_fidelity = paired_fidelity(image, image + 0.2, data_range=2)
    device_fidelity = paired_fidelity(image.to(device), (image + 0.2).to(device), data_range=2)
    result = {
        "device": str(device),
        "source_dtype": "torch.float32",
        "metric_dtype": "torch.float64_cpu",
        "transport_max_abs_error": float((_matrix(xd) - x.double()).abs().max()),
        "mmd2_abs_error": abs(
            mmd2_unbiased(xd, yd, 1.7, block_size=4) - mmd2_unbiased(x, y, 1.7, block_size=4)
        ),
        "train_feature_mean_max_abs_error": float((host_ref.mean - device_ref.mean).abs().max()),
        "train_feature_std_max_abs_error": float((host_ref.std - device_ref.std).abs().max()),
        "train_bandwidth_abs_error": abs(host_ref.ell2 - device_ref.ell2),
        "paired_mse_abs_error": abs(
            host_fidelity["paired_endpoint_mse"] - device_fidelity["paired_endpoint_mse"]
        ),
        "ssim_abs_error": abs(host_fidelity["pixel_ssim"] - device_fidelity["pixel_ssim"]),
    }
    result["passed"] = all(value == 0 for name, value in result.items() if name.endswith("error"))
    return result
