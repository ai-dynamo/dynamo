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

EXPECTED_BASE_COMMIT = "93d8f834dd8acf33eb0e2a75b2711b628cb6e226"
EXPECTED_FLASHINFER_SHA = "330cc8e1a09f59c1241084459f3df3204b9b8327"
EXPECTED_FLASHINFER_VERSION = "0.6.14"
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
    "vllm/model_executor/warmup/kernel_warmup.py",
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


def nccl_distribution() -> metadata.Distribution:
    matches = []
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name in {"nvidia-nccl-cu12", "nvidia-nccl-cu13"}:
            matches.append(dist)
    if len(matches) != 1:
        names = [dist.metadata.get("Name") for dist in matches]
        raise RuntimeError(f"Expected one NVIDIA NCCL distribution, found {names}")
    return matches[0]


def native_vllm_state(package_dir: Path) -> dict[str, str]:
    extensions = sorted(package_dir.rglob("*.so"))
    if not extensions:
        raise RuntimeError(f"No vLLM native extensions found under {package_dir}")
    return {
        str(path.relative_to(package_dir)): file_sha256(path) for path in extensions
    }


def vllm_extension(package_dir: Path) -> Path:
    candidates = sorted(
        path.resolve()
        for path in package_dir.glob("_C*.so")
        if path.name.startswith("_C.")
    )
    if len(candidates) != 1:
        native_files = sorted(path.name for path in package_dir.glob("*.so"))
        raise RuntimeError(
            "Expected one nightly vllm._C extension, found "
            f"{[str(path) for path in candidates]}; "
            f"top-level native files are {native_files}"
        )
    return candidates[0]


def capture_state() -> dict[str, Any]:
    import torch

    vllm_dist = distribution("vllm")
    vllm_package = Path(vllm_dist.locate_file("vllm")).resolve()
    vllm_c = vllm_extension(vllm_package)

    torch_dist = distribution("torch")
    torch_spec = importlib.util.find_spec("torch._C")
    if torch_spec is None or torch_spec.origin is None:
        raise RuntimeError("The nightly torch._C extension is missing")
    torch_c = Path(torch_spec.origin).resolve()

    nccl_dist = nccl_distribution()
    nccl_name = nccl_dist.metadata["Name"]
    nccl_dso = Path(nccl_dist.locate_file("nvidia/nccl/lib/libnccl.so.2")).resolve()
    if not nccl_dso.is_file():
        raise RuntimeError(f"The nightly NCCL DSO is missing: {nccl_dso}")

    return {
        "vllm_build_commit": os.environ.get("VLLM_BUILD_COMMIT"),
        "vllm": {
            "version": vllm_dist.version,
            "package": str(vllm_package),
            "extension": str(vllm_c),
            "extension_sha256": file_sha256(vllm_c),
            "native_extensions": native_vllm_state(vllm_package),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "debug": torch.version.debug,
            "git_version": torch.version.git_version,
            "package": str(Path(torch_dist.locate_file("torch")).resolve()),
            "extension": str(torch_c),
            "extension_sha256": file_sha256(torch_c),
        },
        "nccl": {
            "name": nccl_name,
            "version": nccl_dist.version,
            "dso": str(nccl_dso),
            "dso_sha256": file_sha256(nccl_dso),
        },
    }


def assert_no_shim() -> None:
    forbidden_env = ("VLLM_NCCL_SO_PATH", "NCCL_CHECKPOINT_SHIM", "LD_PRELOAD")
    present_env = {
        name: os.environ[name] for name in forbidden_env if os.environ.get(name)
    }
    if present_env:
        raise RuntimeError(f"Forbidden NCCL override environment: {present_env}")

    preload = Path("/etc/ld.so.preload")
    if preload.exists() and preload.read_text().strip():
        raise RuntimeError(f"Unexpected system preload: {preload.read_text()!r}")

    forbidden_paths = (
        Path("/opt/nccl-checkpoint"),
        Path("/opt/dynamo/nccl"),
        Path("/usr/local/bin/build_nccl_checkpoint"),
        Path("/usr/local/lib/validate_pynccl_checkpoint_binding.py"),
    )
    present_paths = [str(path) for path in forbidden_paths if path.exists()]
    if present_paths:
        raise RuntimeError(f"Forbidden NCCL shim/custom DSO paths: {present_paths}")

    try:
        metadata.distribution("nccl-checkpoint")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("The nccl-checkpoint distribution must not be installed")
    if importlib.util.find_spec("nccl_checkpoint") is not None:
        raise RuntimeError("The nccl_checkpoint package must not be importable")


def parse_overlay_provenance() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in OVERLAY_PROVENANCE_PATH.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        hashes[path] = digest
    return hashes


def parse_source_provenance() -> dict[str, str]:
    return dict(
        line.split("=", maxsplit=1)
        for line in SOURCE_PROVENANCE_PATH.read_text().splitlines()
    )


def verify_overlay_files(package_dir: Path) -> None:
    expected = set(OVERLAY_PATHS)
    provenance = parse_overlay_provenance()
    if set(provenance) != expected:
        raise RuntimeError(
            f"Overlay provenance paths are {sorted(provenance)}, "
            f"expected {sorted(expected)}"
        )
    for relative_path, expected_hash in provenance.items():
        actual_hash = file_sha256(package_dir.parent / relative_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Overlay file changed after installation: {relative_path}"
            )


def verify_flashinfer() -> None:
    flashinfer_dist = distribution("flashinfer-python")
    if flashinfer_dist.version != EXPECTED_FLASHINFER_VERSION:
        raise RuntimeError(
            f"FlashInfer is {flashinfer_dist.version}, "
            f"expected {EXPECTED_FLASHINFER_VERSION}"
        )
    from flashinfer import _build_meta

    if _build_meta.__git_version__ != EXPECTED_FLASHINFER_SHA:
        raise RuntimeError(
            f"FlashInfer source is {_build_meta.__git_version__}, "
            f"expected {EXPECTED_FLASHINFER_SHA}"
        )
    cubin_dist = distribution("flashinfer-cubin")
    if cubin_dist.version != EXPECTED_FLASHINFER_VERSION:
        raise RuntimeError(
            f"FlashInfer cubin is {cubin_dist.version}, "
            f"expected {EXPECTED_FLASHINFER_VERSION}"
        )
    from flashinfer_cubin import _build_meta as cubin_build_meta

    if cubin_build_meta.__git_version__ != EXPECTED_FLASHINFER_SHA:
        raise RuntimeError(
            f"FlashInfer cubin source is {cubin_build_meta.__git_version__}, "
            f"expected {EXPECTED_FLASHINFER_SHA}"
        )
    try:
        metadata.distribution("flashinfer-jit-cache")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("A stale FlashInfer JIT cache must not remain installed")
    if FLASHINFER_SHA_PATH.read_text().strip() != EXPECTED_FLASHINFER_SHA:
        raise RuntimeError("FlashInfer durable source provenance is incorrect")


def verify_plugin_in_fresh_process() -> None:
    program = r"""
from importlib.metadata import entry_points

matches = [
    entry_point
    for entry_point in entry_points(group="vllm.general_plugins")
    if entry_point.name == "dynamo_snapshot"
]
assert len(matches) == 1, matches
assert (
    matches[0].value
    == "dynamo.vllm.snapshot_backend:register_dynamo_snapshot_backend"
), matches[0]

from vllm.plugins import load_general_plugins

load_general_plugins()
from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

assert "dynamo_snapshot" in SleepModeBackendFactory._registry
backend = SleepModeBackendFactory.get_backend_class("dynamo_snapshot")
assert backend.preserves_nccl()
assert backend.preserves_communicators()
assert backend.preserves_graphs_with_nccl()
assert backend.preserves_graphs_with_communicators()
print(f"fresh-process plugin registered: {backend.__module__}.{backend.__name__}")
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def capture() -> None:
    if os.environ.get("VLLM_BUILD_COMMIT") != EXPECTED_BASE_COMMIT:
        raise RuntimeError(
            f"VLLM_BUILD_COMMIT is {os.environ.get('VLLM_BUILD_COMMIT')}, "
            f"expected {EXPECTED_BASE_COMMIT}"
        )
    assert_no_shim()
    state = capture_state()
    BASELINE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(state, indent=2, sort_keys=True))


def validate() -> None:
    baseline = json.loads(BASELINE_PATH.read_text())
    current = capture_state()
    if current != baseline:
        raise RuntimeError(
            "Nightly native stack changed:\n"
            f"baseline={json.dumps(baseline, indent=2, sort_keys=True)}\n"
            f"current={json.dumps(current, indent=2, sort_keys=True)}"
        )

    package_dir = Path(current["vllm"]["package"])
    verify_overlay_files(package_dir)
    source = parse_source_provenance()
    if source.get("install_mode") != "python-overlay":
        raise RuntimeError(f"Unexpected install mode: {source}")
    if source.get("vllm_base_commit") != EXPECTED_BASE_COMMIT:
        raise RuntimeError(f"Unexpected vLLM base: {source}")

    if os.environ.get("REQUIRE_VLLM_C_IMPORT") == "1":
        module = importlib.import_module("vllm._C")
        if Path(module.__file__).resolve() != Path(current["vllm"]["extension"]):
            raise RuntimeError(
                f"vllm._C loaded from an unexpected path: {module.__file__}"
            )

    verify_flashinfer()
    assert_no_shim()
    verify_plugin_in_fresh_process()
    print(json.dumps(current, indent=2, sort_keys=True))
    print("Validated exact nightly Python overlay and stock native stack")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "validate"))
    args = parser.parse_args()
    if args.action == "capture":
        capture()
    else:
        validate()


if __name__ == "__main__":
    main()
