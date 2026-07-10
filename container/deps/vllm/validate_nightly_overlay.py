#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_BASE_COMMIT = "2c17d33f4291a55b447317640c81eb61077b1b00"
EXPECTED_BASE_DIGEST = (
    "sha256:5bda7078b1bb17f74d369e3ded63115a77d5ea5eeb9eab6ca9a52d108f9a262d"
)
EXPECTED_VLLM_HEAD = "ec308a7178bc77dbc90c0673309dac0eb4e2959d"
EXPECTED_FLASHINFER_SHA = "f2f9646ec388d9f178b2fbda6ae0ec4246d8e7dc"
EXPECTED_FLASHINFER_VERSION = "0.6.15"
EXPECTED_AMD64_DIGEST = (
    "sha256:1ebb205a272a55abb60b09ecbf2adc63831ef2377910afd527478de720788cd8"
)
BASELINE_PATH = Path("/opt/dynamo/nightly-base-provenance.json")
OVERLAY_PROVENANCE_PATH = Path("/opt/dynamo/vllm-overlay-provenance.txt")
SOURCE_PROVENANCE_PATH = Path("/opt/dynamo/source-provenance.txt")
FLASHINFER_SHA_PATH = Path("/opt/dynamo/flashinfer-source-sha.txt")

OVERLAY_PATHS = (
    "vllm/distributed/device_communicators/all2all.py",
    "vllm/distributed/device_communicators/base_device_communicator.py",
    "vllm/distributed/device_communicators/cuda_communicator.py",
    "vllm/distributed/device_communicators/flashinfer_all_reduce.py",
    "vllm/distributed/parallel_state.py",
    "vllm/v1/worker/gpu_model_runner.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(name: str) -> metadata.Distribution:
    matches = [
        dist
        for dist in metadata.distributions()
        if (dist.metadata.get("Name") or "").lower().replace("_", "-") == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} distribution, found {len(matches)}")
    return matches[0]


def native_vllm_state(package_dir: Path) -> dict[str, str]:
    extensions = sorted(package_dir.rglob("*.so"))
    if not extensions:
        raise RuntimeError(f"No vLLM native extensions found under {package_dir}")
    return {
        str(path.relative_to(package_dir)): file_sha256(path) for path in extensions
    }


def nccl_distribution() -> metadata.Distribution:
    matches = [
        dist
        for dist in metadata.distributions()
        if (dist.metadata.get("Name") or "").lower().replace("_", "-")
        in {"nvidia-nccl-cu12", "nvidia-nccl-cu13"}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one NVIDIA NCCL distribution, found {matches}")
    return matches[0]


def capture_state() -> dict[str, Any]:
    import torch

    vllm_dist = distribution("vllm")
    vllm_package = Path(vllm_dist.locate_file("vllm")).resolve()
    torch_dist = distribution("torch")
    torch_spec = importlib.util.find_spec("torch._C")
    if torch_spec is None or torch_spec.origin is None:
        raise RuntimeError("The nightly torch._C extension is missing")
    torch_c = Path(torch_spec.origin).resolve()
    nccl_dist = nccl_distribution()
    nccl_dso = Path(nccl_dist.locate_file("nvidia/nccl/lib/libnccl.so.2")).resolve()
    if not nccl_dso.is_file():
        raise RuntimeError(f"The nightly NCCL DSO is missing: {nccl_dso}")

    return {
        "vllm": {
            "version": vllm_dist.version,
            "package": str(vllm_package),
            "native_extensions": native_vllm_state(vllm_package),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "git_version": torch.version.git_version,
            "package": str(Path(torch_dist.locate_file("torch")).resolve()),
            "extension_sha256": file_sha256(torch_c),
        },
        "nccl": {
            "name": nccl_dist.metadata["Name"],
            "version": nccl_dist.version,
            "dso_sha256": file_sha256(nccl_dso),
        },
    }


def assert_no_shim() -> None:
    forbidden_env = ("VLLM_NCCL_SO_PATH", "NCCL_CHECKPOINT_SHIM", "LD_PRELOAD")
    present = {name: os.environ[name] for name in forbidden_env if os.environ.get(name)}
    if present:
        raise RuntimeError(f"Forbidden NCCL override environment: {present}")
    preload = Path("/etc/ld.so.preload")
    if preload.exists() and preload.read_text().strip():
        raise RuntimeError(f"Unexpected system preload: {preload.read_text()!r}")
    if importlib.util.find_spec("nccl_checkpoint") is not None:
        raise RuntimeError("The nccl_checkpoint package must not be importable")


def parse_source_provenance(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", maxsplit=1)
        for line in path.read_text().splitlines()
        if "=" in line
    )


def verify_overlay_files(package_dir: Path) -> None:
    provenance = {}
    for line in OVERLAY_PROVENANCE_PATH.read_text().splitlines():
        digest, relative_path = line.split(maxsplit=1)
        provenance[relative_path] = digest
    if set(provenance) != set(OVERLAY_PATHS):
        raise RuntimeError(f"Unexpected overlay provenance: {sorted(provenance)}")
    for relative_path, expected_hash in provenance.items():
        if file_sha256(package_dir.parent / relative_path) != expected_hash:
            raise RuntimeError(f"Overlay file changed: {relative_path}")


def verify_flashinfer(source: dict[str, str]) -> None:
    if not Path("/usr/local/cuda/include/nvrtc.h").is_file():
        raise RuntimeError("FlashInfer runtime JIT header is missing")
    stale_jit_cache = [
        dist.metadata.get("Name")
        for dist in metadata.distributions()
        if (dist.metadata.get("Name") or "").lower().replace("_", "-")
        == "flashinfer-jit-cache"
    ]
    if stale_jit_cache:
        raise RuntimeError(f"Stale FlashInfer JIT cache remains: {stale_jit_cache}")
    from flashinfer import _build_meta
    from flashinfer_cubin import __git_version__ as cubin_git_version
    from flashinfer_cubin import list_cubins

    if _build_meta.__git_version__ != EXPECTED_FLASHINFER_SHA:
        raise RuntimeError(
            f"FlashInfer source is {_build_meta.__git_version__}, "
            f"expected {EXPECTED_FLASHINFER_SHA}"
        )
    if distribution("flashinfer-python").version != EXPECTED_FLASHINFER_VERSION:
        raise RuntimeError("Unexpected installed FlashInfer version")
    if distribution("flashinfer-cubin").version != EXPECTED_FLASHINFER_VERSION:
        raise RuntimeError("Unexpected installed FlashInfer cubin version")
    if cubin_git_version != EXPECTED_FLASHINFER_SHA:
        raise RuntimeError(
            f"FlashInfer cubin source is {cubin_git_version}, "
            f"expected {EXPECTED_FLASHINFER_SHA}"
        )
    if not list_cubins():
        raise RuntimeError("FlashInfer cubin package contains no cubins")
    if source.get("flashinfer_source_version") != EXPECTED_FLASHINFER_VERSION:
        raise RuntimeError(f"Unexpected FlashInfer version provenance: {source}")
    if FLASHINFER_SHA_PATH.read_text().strip() != EXPECTED_FLASHINFER_SHA:
        raise RuntimeError("FlashInfer durable source provenance is incorrect")


def verify_nvrtc(source: dict[str, str]) -> None:
    installed_version = subprocess.run(
        ["dpkg-query", "-W", "-f=${Version}", "cuda-nvrtc-dev-13-0"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if (
        not installed_version
        or source.get("nvrtc_package_version") != installed_version
    ):
        raise RuntimeError("NVRTC package provenance is missing or incorrect")
    if not Path("/usr/local/cuda/include/nvrtc.h").is_file():
        raise RuntimeError("NVRTC development headers are missing")


def verify_gms_backend_in_fresh_process() -> None:
    program = r"""
from types import SimpleNamespace

from dynamo.vllm.snapshot_backend import (
    GMS_BACKEND_NAME,
    register_dynamo_gms_snapshot_backend,
    select_dynamo_gms_snapshot_backend,
)
from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

register_dynamo_gms_snapshot_backend()
assert GMS_BACKEND_NAME in SleepModeBackendFactory._registry
backend = SleepModeBackendFactory.get_backend_class(GMS_BACKEND_NAME)
assert backend.preserves_communicators()
assert backend.preserves_graphs_with_communicators()
config = SimpleNamespace(
    load_config=SimpleNamespace(load_format="gms"),
    model_config=SimpleNamespace(sleep_mode_backend="cumem"),
)
select_dynamo_gms_snapshot_backend(config)
assert config.model_config.sleep_mode_backend == GMS_BACKEND_NAME
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def capture() -> None:
    assert_no_shim()
    BASELINE_PATH.write_text(
        json.dumps(capture_state(), indent=2, sort_keys=True) + "\n"
    )


def validate() -> None:
    baseline = json.loads(BASELINE_PATH.read_text())
    current = capture_state()
    if current != baseline:
        raise RuntimeError("Nightly native vLLM/Torch/NCCL stack changed")
    source = parse_source_provenance(SOURCE_PROVENANCE_PATH)
    expected_provenance = {
        "install_mode": "python-overlay",
        "vllm_runtime_base_image": ("vllm/vllm-openai@" + EXPECTED_BASE_DIGEST),
        "vllm_runtime_index_digest": EXPECTED_BASE_DIGEST,
        "vllm_runtime_amd64_digest": EXPECTED_AMD64_DIGEST,
        "vllm_base_commit": EXPECTED_BASE_COMMIT,
        "vllm_source_sha": EXPECTED_VLLM_HEAD,
        "vllm_overlay_commits": "2",
        "vllm_overlay_files": str(len(OVERLAY_PATHS)),
        "flashinfer_source_sha": EXPECTED_FLASHINFER_SHA,
        "flashinfer_source_version": EXPECTED_FLASHINFER_VERSION,
    }
    mismatches = {
        key: source.get(key)
        for key, expected in expected_provenance.items()
        if source.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Unexpected source provenance: {mismatches}")
    verify_overlay_files(Path(current["vllm"]["package"]))
    verify_flashinfer(source)
    verify_nvrtc(source)
    assert_no_shim()
    verify_gms_backend_in_fresh_process()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "validate"))
    args = parser.parse_args()
    capture() if args.action == "capture" else validate()


if __name__ == "__main__":
    main()
