"""Loss registry: noise/velocity regression plus geometric path regularizers."""

import torch
from torch.nn import functional as F

LOSSES = {}
REGULARIZERS = {}


def _register(registry, name):
    def decorate(function):
        if name in registry:
            raise ValueError(f"{name!r} already registered")
        registry[name] = function
        return function

    return decorate


def register_loss(name):
    """Custom loss(prediction,target)->[B] (unreduced per-example values)."""
    return _register(LOSSES, name)


def register_regularizer(name):
    """Custom regularizer(clean_prediction,conditions,metadata)->scalar."""
    return _register(REGULARIZERS, name)


@register_loss("mse")
def mse(prediction, target):
    return (prediction - target).square().flatten(1).mean(1)


@register_loss("huber")
def huber(prediction, target):
    return F.smooth_l1_loss(prediction, target, reduction="none").flatten(1).mean(1)


@register_regularizer("smoothness")
def smoothness(paths, conditions, metadata):
    if paths.ndim != 3 or paths.shape[1] < 3:
        raise ValueError("smoothness requires paths [B,L>=3,D]")
    return (paths[:, 2:] - 2 * paths[:, 1:-1] + paths[:, :-2]).square().mean()


@register_regularizer("endpoint")
def endpoint(paths, conditions, metadata):
    if paths.ndim != 3 or conditions is None:
        raise ValueError("endpoint requires paths and continuous start/goal conditions")
    dim = paths.shape[-1]
    return (
        (paths[:, 0] - conditions[:, :dim]).square().mean()
        + (paths[:, -1] - conditions[:, dim : 2 * dim]).square().mean()
    ) / 2


@register_regularizer("collision")
def collision(paths, conditions, metadata):
    from .environments import collision_loss, walls_from_conditions

    if paths.ndim != 3 or "walls" not in metadata:
        raise ValueError("collision requires navigation paths and wall metadata")
    walls = (
        walls_from_conditions(conditions, paths.shape[-1])
        if conditions is not None
        else torch.as_tensor(metadata["walls"], device=paths.device, dtype=paths.dtype)
    )
    return collision_loss(paths, walls, margin=metadata.get("margin", 0.035))


@register_regularizer("bounds")
def bounds(paths, conditions, metadata):
    if "bounds" not in metadata:
        raise ValueError("bounds regularizer requires navigation bounds")
    limits = torch.as_tensor(metadata["bounds"], device=paths.device, dtype=paths.dtype)
    return (F.relu(limits[0] - paths).square() + F.relu(paths - limits[1]).square()).mean()
