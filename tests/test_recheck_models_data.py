"""Architecture, normalized-time, split and no-test-leakage invariants."""

import copy
import json

import pytest
import torch
from torch import nn

from recheck.data import (
    evaluator_preprocess,
    load_mnist_split,
    make_toy_dataset,
    prepare_mnist,
    sample_train,
    stratified_split_indices,
    tensor_sha256,
)
from recheck.models import (
    FMMLP,
    EvaluatorCNN,
    SmallUNetModel,
    architecture_manifest,
    state_dict_sha256,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("constructor,shape", [(FMMLP, (4, 2)), (SmallUNetModel, (2, 1, 28, 28))])
def test_unconditional_models_shape_time_and_all_parameter_gradients(constructor, shape):
    model = constructor()
    sample = torch.randn(shape)
    output = model(sample, torch.linspace(0, 1, shape[0])).sample
    assert output.shape == sample.shape and output.dtype == sample.dtype
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.square().sum()) for p in model.parameters()) > 0
    assert torch.equal(
        model(sample, 0.5).sample, model(sample, torch.full((shape[0],), 0.5)).sample
    )
    assert not torch.equal(model(sample, 0.0).sample, model(sample, 1.0).sample)
    with pytest.raises(ValueError, match="normalized"):
        model(sample, 32)
    with pytest.raises(TypeError):
        model(sample, 0.5, conditions=torch.zeros(shape[0]))


def test_architecture_is_small_and_unconditioned():
    mlp = FMMLP()
    assert len([m for m in mlp.modules() if isinstance(m, nn.Linear)]) == 4
    assert len([m for m in mlp.modules() if isinstance(m, nn.SiLU)]) == 3
    assert mlp.config.hidden_dim == 128 and mlp.config.time_embedding_dim == 32
    unet = SmallUNetModel()
    resolutions = []
    handles = [
        b[-1].register_forward_hook(lambda module, args, out: resolutions.append(out.shape[-1]))
        for b in unet.down_blocks
    ]
    unet(torch.randn(1, 1, 28, 28), 0.5)
    for handle in handles:
        handle.remove()
    assert resolutions == [28, 14, 7]
    assert [level[-1].conv2.out_channels for level in unet.down_blocks] == [32, 64, 128]
    assert all(len(level) == 2 for level in (*unet.down_blocks, *unet.up_blocks))
    assert not any(
        isinstance(m, (nn.Dropout, nn.BatchNorm2d, nn.MultiheadAttention)) for m in unet.modules()
    )
    manifest = architecture_manifest(unet)
    assert 0 < manifest["parameter_count"] < 2_000_000
    assert manifest["flops_estimate"] is None


@pytest.mark.parametrize("constructor", [FMMLP, SmallUNetModel])
def test_teacher_student_copy_hash_freeze_and_native_roundtrip(constructor, tmp_path):
    teacher = constructor().eval().requires_grad_(False)
    student = copy.deepcopy(teacher).train().requires_grad_(True)
    assert state_dict_sha256(student) == state_dict_sha256(teacher)
    initial_hash = state_dict_sha256(teacher)
    sample = torch.randn(2, *teacher.sample_shape)
    # Compare with the same no-grad context; convolution backends may use a
    # different accumulation path when one of the two passes records autograd.
    with torch.no_grad():
        assert torch.equal(teacher(sample, 0.5).sample, student(sample, 0.5).sample)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    student(sample, 0.5).sample.square().mean().backward()
    optimizer.step()
    assert state_dict_sha256(teacher) == initial_hash
    assert all(p.grad is None for p in teacher.parameters())
    assert state_dict_sha256(student) != initial_hash
    student.save_pretrained(tmp_path)
    loaded = constructor.from_pretrained(tmp_path)
    with torch.no_grad():
        assert torch.equal(loaded(sample, 0.5).sample, student(sample, 0.5).sample)


def test_evaluator_features_are_fc128_after_activation():
    model = EvaluatorCNN()
    pixels = torch.rand(3, 1, 28, 28)
    logits, features = model(pixels, return_features=True)
    assert features.shape == (3, 128) and logits.shape == (3, 10)
    assert (features >= 0).all()
    assert torch.equal(features, model.features(pixels))
    assert torch.equal(logits, model.logits(features))


@pytest.mark.parametrize("name", ["rotating_rays", "gmm"])
def test_toy_splits_are_independent_and_never_use_global_rng(name):
    state = torch.random.get_rng_state().clone()
    train = make_toy_dataset(name, 100, seed=41, split="train")
    again = make_toy_dataset(name, 100, seed=41, split="train")
    validation = make_toy_dataset(name, 100, seed=41, split="validation")
    test = make_toy_dataset(name, 100, seed=41, split="test")
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(train, again)
    assert len({tensor_sha256(x) for x in (train, validation, test)}) == 3
    assert train.dtype == torch.float32 and train.shape == (100, 2)
    generator = torch.Generator().manual_seed(123)
    batch = sample_train(train, 12, generator)
    repeated = sample_train(train, 12, torch.Generator().manual_seed(123))
    assert torch.equal(batch, repeated)


def test_full_mnist_split_is_exact_stratified_and_reproducible():
    labels = torch.arange(60000) % 10
    train, validation = stratified_split_indices(labels, 5000, seed=41)
    assert train.shape == (55000,) and validation.shape == (5000,)
    assert torch.equal(torch.bincount(labels[validation]), torch.full((10,), 500))
    assert len(torch.unique(torch.cat((train, validation)))) == 60000
    assert torch.equal(train, stratified_split_indices(labels, 5000, seed=41)[0])
    assert not torch.equal(train, stratified_split_indices(labels, 5000, seed=42)[0])


class FakeMNIST:
    calls = []

    def __init__(self, root, train=True, download=False):
        type(self).calls.append({"root": root, "train": train, "download": download})
        generator = torch.Generator().manual_seed(110 if train else 220)
        self.data = torch.randint(256, (100, 28, 28), generator=generator, dtype=torch.uint8)
        self.targets = torch.arange(100) % 10


def test_mnist_preparation_never_opens_test_and_verifies_saved_indices(tmp_path):
    FakeMNIST.calls.clear()
    root, output = tmp_path / "data", tmp_path / "split"
    manifest = prepare_mnist(
        root,
        output,
        n_validation=20,
        expected_train_size=100,
        expected_test_size=100,
        dataset_factory=FakeMNIST,
    )
    path = output / "mnist_split_manifest.json"
    assert all(call["train"] and not call["download"] for call in FakeMNIST.calls)
    assert manifest["test_accessed"] is False
    train, labels = load_mnist_split(root, "train", path, dataset_factory=FakeMNIST)
    assert train.shape == (80, 1, 28, 28) and labels.shape == (80,)
    assert train.min() == -1 and train.max() == 1
    with pytest.raises(ValueError, match="test is closed"):
        load_mnist_split(root, "test", path, dataset_factory=FakeMNIST)
    assert all(call["train"] for call in FakeMNIST.calls)
    indices_path = output / manifest["indices_file"]
    indices = torch.load(indices_path, weights_only=True)
    indices["train"][0] = indices["validation"][0]
    torch.save(indices, indices_path)
    with pytest.raises(ValueError, match="hash checks"):
        load_mnist_split(root, "train", path, dataset_factory=FakeMNIST)
    assert json.loads(path.read_text())["train_size"] == 80


def test_evaluator_range_diagnostics_are_before_shared_clamp():
    raw = torch.tensor([-1.4, -1.0, 0.0, 1.0, 1.6]).reshape(1, 1, 1, 5)
    original = raw.clone()
    pixel, diagnostic = evaluator_preprocess(raw)
    assert torch.equal(raw, original)
    assert torch.allclose(pixel.flatten(), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    assert diagnostic["out_of_range_fraction"] == pytest.approx(2 / 5)
    assert diagnostic["max_range_violation"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="non-finite"):
        evaluator_preprocess(torch.tensor([float("nan")]))
