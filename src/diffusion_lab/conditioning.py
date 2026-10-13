"""Condition encoders. A learned null vector supports classifier-free guidance."""

import torch
from torch import nn
from torch.nn import functional as F

CONDITIONERS = {}


def register_conditioner(name):
    """Register factory(hidden_dim, num_classes, condition_dim) -> nn.Module."""

    def decorate(factory):
        if name in CONDITIONERS:
            raise ValueError(f"Conditioner {name!r} already registered")
        CONDITIONERS[name] = factory
        return factory

    return decorate


@register_conditioner("one_hot")
class OneHotConditioner(nn.Module):
    def __init__(self, hidden_dim, num_classes, condition_dim):
        super().__init__()
        if num_classes < 1:
            raise ValueError("one_hot requires num_classes >= 1")
        self.num_classes = num_classes
        self.projection = nn.Linear(num_classes, hidden_dim)

    def forward(self, conditions):
        return self.projection(
            F.one_hot(_class_ids(conditions, self.num_classes), self.num_classes).to(
                self.projection.weight.dtype
            )
        )


def _class_ids(conditions, num_classes):
    if conditions.ndim != 1 or conditions.dtype not in (torch.int32, torch.int64):
        raise ValueError("Categorical conditions must be integer IDs of shape [B]")
    if torch.any((conditions < 0) | (conditions >= num_classes)):
        raise ValueError(f"Class IDs must be in [0, {num_classes})")
    return conditions.long()


@register_conditioner("embedding")
class EmbeddingConditioner(nn.Module):
    def __init__(self, hidden_dim, num_classes, condition_dim):
        super().__init__()
        if num_classes < 1:
            raise ValueError("embedding requires num_classes >= 1")
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, hidden_dim)

    def forward(self, conditions):
        return self.embedding(_class_ids(conditions, self.num_classes))


@register_conditioner("mlp")
class ContinuousConditioner(nn.Module):
    def __init__(self, hidden_dim, num_classes, condition_dim):
        super().__init__()
        if condition_dim < 1:
            raise ValueError("mlp requires condition_dim >= 1")
        self.condition_dim = condition_dim
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, conditions):
        if conditions.ndim != 2 or conditions.shape[1] != self.condition_dim:
            raise ValueError(f"Continuous conditions must have shape [B,{self.condition_dim}]")
        if not torch.isfinite(conditions).all():
            raise ValueError("Conditions must be finite")
        return self.net(conditions.to(self.net[0].weight.dtype))
