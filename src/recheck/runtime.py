"""Independent RNG streams, portable trusted checkpoints and resource guards."""

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch


def stable_seed(seed, stream):
    return int.from_bytes(
        hashlib.sha256(f"recheck-v1:{seed}:{stream}".encode()).digest()[:8], "big"
    ) % (2**63 - 1)


class RNGStreams:
    names = ("data", "interval", "noise", "tube", "swd", "preview", "bootstrap")

    def __init__(self, seed):
        self.seeds = {name: stable_seed(seed, name) for name in self.names}
        self.generators = {
            name: torch.Generator().manual_seed(value) for name, value in self.seeds.items()
        }

    def __getitem__(self, name):
        return self.generators[name]

    def state_dict(self):
        return {name: generator.get_state() for name, generator in self.generators.items()}

    def load_state_dict(self, state):
        for name, value in state.items():
            self.generators[name].set_state(value.cpu())


def tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(t.shape), str(t.dtype))).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def environment():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    versions = {}
    for package in (
        "torch",
        "diffusers",
        "geomloss",
        "torchvision",
        "numpy",
        "scipy",
        "diffusion-lab",
    ):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "git_commit": commit,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
    }


def choose_device(name):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Run the registered CPU rays pilot; full MNIST is not completed."
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "This process cannot access MPS. On macOS run outside a GPU-restricted sandbox."
        )
    return torch.device(name)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def algorithm_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for p in sorted(root.glob("*.py")):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def require_rays_gate(config):
    if config["data"]["name"] != "mnist":
        return
    path = Path(config["resources"]["rays_gate"])
    if not path.is_file():
        raise RuntimeError(
            "MNIST is gated: run pilot-rays and inspect its animations first; rays_gate.json is missing"
        )
    gate = json.loads(path.read_text())
    if not gate.get("passed") or gate.get("algorithm_sha256") != algorithm_hash():
        raise RuntimeError(
            "Rays gate is failed/stale after algorithm changes. Repeat rays verification before MNIST."
        )
    if not gate.get("animations") or not gate.get("all_mvp_methods_completed"):
        raise RuntimeError("Rays gate lacks required trained methods or trajectory animations")


class ResourceBudget:
    def __init__(self, config, device):
        self.config, self.device, self.started = config, device, time.monotonic()
        self.last = self.started
        self.closed = False
        self.gpu = str(device).startswith(("cuda", "mps"))
        self.ledger_path = Path(config["output_dir"]) / "resource_ledger.json"
        self.lock = None
        if self.gpu:
            import fcntl
            import tempfile

            self.lock = (
                Path(tempfile.gettempdir()) / f"recheck-accelerator-{os.getuid()}.lock"
            ).open("a+")
            try:
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.lock.close()
                self.lock = None
                raise RuntimeError(
                    "Another recheck accelerator process is running; only one GPU process is allowed"
                ) from None
        try:
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        now = time.monotonic()
        ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}
        key = "gpu_training_seconds" if self.gpu else "cpu_training_wall_seconds"
        ledger[key] = ledger.get(key, 0.0) + now - self.last
        self.last = now
        atomic_json(self.ledger_path, ledger)
        if self.gpu and ledger[key] / 3600 >= self.config["resources"]["max_gpu_hours"]:
            raise RuntimeError(
                "Cumulative registered GPU-time budget reached; preserve checkpoints and register a new budget"
            )
        root = Path(self.config["output_dir"])
        size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
        if size > self.config["resources"]["max_artifact_gib"] * 1024**3:
            raise RuntimeError("Registered artifact disk budget exceeded")

    def close(self):
        if not self.closed:
            self.closed = True
            if self.lock is not None:
                self.lock.close()
                self.lock = None

    def __del__(self):
        self.close()
