"""Fixed train-only teacher feature scales, shared by tube/no-tube methods."""

import hashlib
import math
import warnings
from typing import Any

import torch
from torch import Tensor


@torch.no_grad()
def calibrate_interval_features(
    teacher_features: Tensor,
    alpha: float = 0.05,
    median_floor: float = 1e-8,
) -> dict[str, Any]:
    """Calibrate one representation/interval before fitting any student.

    The caller supplies clean train-only anchors (256 in the registered
    protocol), then saves this immutable result in every paired run manifest.
    This helper deliberately receives no current student features or tube rho.
    """
    if teacher_features.ndim != 2 or teacher_features.shape[0] < 2:
        raise ValueError("calibration expects at least two full-state [B, P] features")
    if (
        not math.isfinite(alpha)
        or alpha <= 0
        or not math.isfinite(median_floor)
        or median_floor <= 0
    ):
        raise ValueError("alpha and median_floor must be positive and finite")
    # Separate device transfer/cast: combined MPS32 -> CPU64 can yield zeros in
    # torch 2.14.1, falsely declaring all teacher supports degenerate.
    values = teacher_features.detach().cpu().to(dtype=torch.float64).contiguous()
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite calibration features")
    costs = torch.cdist(values, values).square()
    i, j = torch.triu_indices(len(values), len(values), offset=1)
    median = float(torch.quantile(costs[i, j], 0.5))
    degenerate = median < median_floor
    if degenerate:
        warnings.warn(
            f"Degenerate teacher median cost {median:g}; applying floor {median_floor:g}. "
            "Investigate widespread degeneracy before training.",
            RuntimeWarning,
            stacklevel=2,
        )
    effective_median = max(median, median_floor)
    return {
        "median_off_diagonal_cost": median,
        "effective_median_cost": effective_median,
        "epsilon": alpha * effective_median,
        "alpha": alpha,
        "median_floor": median_floor,
        "degenerate": degenerate,
        "support_size": len(values),
        "feature_dimension": values.shape[1],
        "dtype": "float64",
        "teacher_features_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
        "anchor_source": "clean_train_only_caller_verified",
        "shared_across_tube_conditions": True,
    }
