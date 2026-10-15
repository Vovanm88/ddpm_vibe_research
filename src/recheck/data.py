"""Seeded unconditional toy data and explicitly prepared, train-only MNIST splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

GMM_CENTERS = ((-0.6, 0.0), (0.6, 0.0), (0.0, 0.65))
GMM_SIGMA = 0.15


def tensor_sha256(tensor):
    tensor = torch.as_tensor(tensor).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def split_seed(seed: int, split: str) -> int:
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation or test")
    # Unlike Python hash(), this definition is stable across processes/platforms.
    return int.from_bytes(
        hashlib.sha256(f"recheck:data:{seed}:{split}".encode()).digest()[:8], "little"
    ) % (2**63 - 1)


def make_toy_dataset(name="rotating_rays", n_samples=4096, seed=41, split="train", **kwargs):
    """Return only x, with no labels/conditioning available to the generator."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    seed = split_seed(seed, split)
    if name in ("gmm", "gmm_2d", "gaussian_mixture_2d"):
        centers = torch.as_tensor(kwargs.pop("centers", GMM_CENTERS), dtype=torch.float32)
        sigma = float(kwargs.pop("sigma", GMM_SIGMA))
        if kwargs:
            raise ValueError(f"unknown GMM options: {sorted(kwargs)}")
        if centers.shape != (3, 2) or sigma <= 0:
            raise ValueError("GMM requires three 2-D centers and sigma > 0")
        generator = torch.Generator().manual_seed(seed)
        component = torch.randint(len(centers), (n_samples,), generator=generator)
        return centers[component] + sigma * torch.randn(n_samples, 2, generator=generator)
    if name not in ("rotating_rays", "rays"):
        raise ValueError("toy name must be rotating_rays or gmm")
    from diffusion_lab.environments import make_environment

    # NumPy's generator accepts the same stable positive integer seed.
    return make_environment("rotating_rays", n_samples, seed, **kwargs).samples.clone()


def sample_train(samples, batch_size, generator):
    """Independent with-replacement train sampling using only the supplied RNG."""
    if batch_size < 1 or len(samples) < 1:
        raise ValueError("positive sample count and batch size required")
    indices = torch.randint(len(samples), (batch_size,), generator=generator, device="cpu")
    return samples[indices.to(samples.device)]


def stratified_split_indices(labels, n_validation=5000, seed=41):
    """Exact stratified sizes; ties in class allocation use ascending class IDs."""
    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    if labels.ndim != 1 or not 0 < n_validation < len(labels):
        raise ValueError("labels must be [N] and validation size strictly between 0 and N")
    classes, counts = torch.unique(labels, sorted=True, return_counts=True)
    if (counts < 2).any():
        raise ValueError("each class must have at least two examples")
    fractions = counts.double() * n_validation / len(labels)
    quotas = fractions.floor().long()
    remainder = n_validation - int(quotas.sum())
    order = sorted(range(len(classes)), key=lambda i: (-float(fractions[i] - quotas[i]), i))
    for index in order[:remainder]:
        quotas[index] += 1
    if (quotas < 1).any() or (quotas >= counts).any():
        raise ValueError("split size cannot represent every class in both partitions")
    generator = torch.Generator().manual_seed(seed)
    train, validation = [], []
    for label, quota in zip(classes, quotas):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        validation.append(indices[: int(quota)])
        train.append(indices[int(quota) :])
    train, validation = torch.cat(train), torch.cat(validation)
    train = train[torch.randperm(len(train), generator=generator)]
    validation = validation[torch.randperm(len(validation), generator=generator)]
    return train, validation


def _mnist_factory(dataset_factory=None):
    if dataset_factory is not None:
        return dataset_factory
    try:
        from torchvision.datasets import MNIST
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "MNIST requires a torchvision build matching this PyTorch installation. "
            "Install the project's mnist extra in the chosen environment; "
            "toy experiments do not require torchvision."
        ) from error
    return MNIST


def _load_dataset(root, train, download, dataset_factory=None):
    return _mnist_factory(dataset_factory)(root=str(root), train=train, download=download)


def prepare_mnist(
    root,
    output_dir,
    *,
    seed=41,
    n_validation=5000,
    download=False,
    expected_train_size=60000,
    expected_test_size=10000,
    dataset_factory=None,
):
    """Explicit preparation only. The official test split is never instantiated here.

    ``download=True`` is only intended for the CLI prepare command after the
    required rotating-rays gate. Importing this module does not access a dataset.
    """
    dataset = _load_dataset(root, True, download, dataset_factory)
    images, labels = torch.as_tensor(dataset.data), torch.as_tensor(dataset.targets).long()
    if len(labels) != expected_train_size or images.shape != (expected_train_size, 28, 28):
        raise ValueError(f"official MNIST train must contain {expected_train_size} 28x28 images")
    if not torch.equal(torch.unique(labels), torch.arange(10)):
        raise ValueError("MNIST train must contain exactly the ten digit classes")
    train, validation = stratified_split_indices(labels, n_validation, seed)
    payload = {"train": train, "validation": validation}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices_path = output_dir / "mnist_split_indices.pt"
    manifest_path = output_dir / "mnist_split_manifest.json"
    manifest = {
        "dataset": "torchvision.datasets.MNIST",
        "split_seed": seed,
        "official_train_size": expected_train_size,
        "official_test_size": expected_test_size,
        "test_accessed": False,
        "train_size": len(train),
        "validation_size": len(validation),
        "indices_file": indices_path.name,
        "train_indices_sha256": tensor_sha256(train),
        "validation_indices_sha256": tensor_sha256(validation),
        "official_train_images_sha256": tensor_sha256(images),
        "official_train_labels_sha256": tensor_sha256(labels),
        "normalization": "float32(pixel_uint8)/255; FM=2*pixel-1; no augmentation",
        "class_counts": {
            split: torch.bincount(labels[index], minlength=10).tolist()
            for split, index in payload.items()
        },
    }
    # An existing split cannot be silently changed by another preparation call.
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise ValueError("existing MNIST split differs; use a new artifact directory")
    else:
        torch.save(payload, indices_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_mnist_split(root, split, manifest_path, *, allow_test=False, dataset_factory=None):
    """Load FM-normalized real images. Test needs a separate explicit authorization flag."""
    if split not in ("train", "validation", "test"):
        raise ValueError("unknown MNIST split")
    if split == "test" and not allow_test:
        raise ValueError(
            "MNIST test is closed for training/tuning; pass allow_test=True only for final evaluation"
        )
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    dataset = _load_dataset(root, split != "test", False, dataset_factory)
    images, labels = torch.as_tensor(dataset.data), torch.as_tensor(dataset.targets).long()
    if split == "test":
        if images.shape != (manifest["official_test_size"], 28, 28) or (
            labels.shape != (manifest["official_test_size"],)
        ):
            raise ValueError("official MNIST test size/shape differs from the registered dataset")
    else:
        if tensor_sha256(images) != manifest["official_train_images_sha256"] or (
            tensor_sha256(labels) != manifest["official_train_labels_sha256"]
        ):
            raise ValueError("MNIST train content differs from the registered split")
        indices = torch.load(manifest_path.parent / manifest["indices_file"], weights_only=True)
        train, validation = indices["train"], indices["validation"]
        if tensor_sha256(train) != manifest["train_indices_sha256"] or (
            tensor_sha256(validation) != manifest["validation_indices_sha256"]
        ):
            raise ValueError("saved MNIST split indices failed their hash checks")
        if len(torch.unique(torch.cat((train, validation)))) != len(labels):
            raise ValueError("saved train/validation indices overlap or omit examples")
        images, labels = images[indices[split]], labels[indices[split]]
    pixels = images.float().unsqueeze(1) / 255.0
    return pixels.mul(2).sub(1), labels.clone()


def evaluator_preprocess(raw_fm_images):
    """One conversion shared by all methods; report range errors before clamping."""
    raw = (raw_fm_images + 1) / 2
    if not torch.isfinite(raw).all():
        raise ValueError("non-finite generated pixels cannot be evaluated")
    violation = torch.maximum(-raw, raw - 1).clamp_min(0)
    diagnostics = {
        "out_of_range_fraction": float(((raw < 0) | (raw > 1)).float().mean()),
        "max_range_violation": float(violation.max()),
        "preprocessing": "(raw_fm+1)/2 then clamp(0,1); diagnostics measured before clamp",
    }
    return raw.clamp(0, 1), diagnostics
