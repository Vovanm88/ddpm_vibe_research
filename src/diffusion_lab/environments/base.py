"""Common tensor contract for point distributions and trajectory tasks."""

from __future__ import annotations

import json
from typing import Any

import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """In-memory research dataset with explicit sample and condition shapes.

    Categorical labels stay integer IDs: condition encoders decide whether to use
    one-hot vectors, learned embeddings, or another representation. Continuous
    conditions are separate floating-point feature vectors.
    """

    def __init__(
        self,
        samples: Any,
        conditions: Any | None = None,
        *,
        labels: Any | None = None,
        metadata: dict[str, Any] | None = None,
        num_classes: int | None = None,
    ) -> None:
        self.samples = torch.as_tensor(samples, dtype=torch.float32).contiguous()
        if self.samples.ndim < 2 or len(self.samples) == 0:
            raise ValueError("samples must have shape [N, *sample_shape] with N > 0")
        if not torch.isfinite(self.samples).all():
            raise ValueError("samples must be finite")
        self.sample_shape = tuple(self.samples.shape[1:])
        self.conditions = None
        self.condition_kind = "none"
        self.condition_dim = 0
        self.num_classes = 0
        if conditions is not None:
            condition = torch.as_tensor(conditions)
            if condition.ndim not in (1, 2) or len(condition) != len(self.samples):
                raise ValueError("conditions must have shape [N] or [N, K]")
            if condition.ndim == 1:
                if condition.is_floating_point() and not torch.equal(condition, condition.round()):
                    raise ValueError("categorical conditions must be integer IDs")
                condition = condition.long()
                if (condition < 0).any():
                    raise ValueError("categorical conditions must be nonnegative")
                self.condition_kind = "categorical"
                self.num_classes = (
                    int(condition.max()) + 1 if num_classes is None else int(num_classes)
                )
                if self.num_classes <= int(condition.max()):
                    raise ValueError("num_classes must include all observed labels")
            else:
                condition = condition.float()
                if condition.shape[1] == 0 or not torch.isfinite(condition).all():
                    raise ValueError("continuous conditions need finite, nonempty features")
                self.condition_kind = "continuous"
                self.condition_dim = int(condition.shape[1])
            self.conditions = condition.contiguous()
        self.labels = (
            self.conditions
            if self.condition_kind == "categorical"
            else None
            if labels is None
            else torch.as_tensor(labels, dtype=torch.long)
        )
        if self.labels is not None and self.labels.shape != (len(self.samples),):
            raise ValueError("labels must have shape [N]")
        self.metadata = dict(metadata or {})
        json.dumps(self.metadata)  # fail early for nonportable experiment metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {"samples": self.samples[index]}
        if self.conditions is not None:
            result["conditions"] = self.conditions[index]
        return result
