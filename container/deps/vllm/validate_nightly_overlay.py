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

EXPECTED_FLASHINFER_SHA = "330cc8e1a09f59c1241084459f3df3204b9b8327"
BASELINE_PATH = Path("/opt/dynamo/nightly-base-provenance.json")
OVERLAY_PROVENANCE_PATH = Path("/opt/dynamo/vllm-overlay-provenance.txt")
SOURCE_PROVENANCE_PATH = Path("/opt/dynamo/source-provenance.txt")
FLASHINFER_SHA_PATH = Path("/opt/dynamo/flashinfer-source-sha.txt")

ALLOWED_TUPLES = {
    "current-697158": {
        "vllm_runtime_base_image": (
            "vllm/vllm-openai@sha256:"
            "184914ac7c32e4aa7789bb686bfaa0817dd56dbdc8ee05fc0ec671aa0b1792f0"
        ),
        "vllm_runtime_base_index_digest": (
            "sha256:" "184914ac7c32e4aa7789bb686bfaa0817dd56dbdc8ee05fc0ec671aa0b1792f0"
        ),
        "vllm_runtime_amd64_digest": (
            "sha256:" "1fd4323d0aafe8d92b4a4b568ad33661ecaf3bfc7f40860c95d09fed4e6ccd58"
        ),
        "vllm_base_commit": "69715823df89b11ee684b84066390cbb9092d5c1",
        "vllm_git_url": "https://github.com/galletas1712/vllm.git",
        "vllm_git_ref": "schwinns/exp-cuda-zero-page-234",
        "vllm_source_sha": "17355f6f668857d9b85e0e7714529b42757e0730",
        "vllm_overlay_files": "13",
        "compliance_baseline_sbom": "vllm-openai@184914ac",
        "flashinfer_git_url": "https://github.com/galletas1712/flashinfer.git",
        "flashinfer_git_ref": "schwinns/checkpoint-collectives-integration",
        "flashinfer_source_sha": EXPECTED_FLASHINFER_SHA,
        "flashinfer_source_version": "0.6.14",
    },
    "crossover-93d8": {
        "vllm_runtime_base_image": (
            "vllm/vllm-openai@sha256:"
            "5da1eb79b49d3edb3b3601a116273f019adb7cab403e86790f61130f8596810a"
        ),
        "vllm_runtime_base_index_digest": (
            "sha256:" "7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9"
        ),
        "vllm_runtime_amd64_digest": (
            "sha256:" "5da1eb79b49d3edb3b3601a116273f019adb7cab403e86790f61130f8596810a"
        ),
        "vllm_base_commit": "93d8f834dd8acf33eb0e2a75b2711b628cb6e226",
        "vllm_git_url": "https://github.com/galletas1712/vllm.git",
        "vllm_git_ref": (
            "schwinns/exp-93d8-current-overlay-zero-regression-20260708t082747z"
        ),
        "vllm_source_sha": "7e48076f13710677c223daf6e4e1af039c0f016e",
        "vllm_overlay_files": "13",
        "compliance_baseline_sbom": "vllm-openai@7c5a10e9",
        "flashinfer_git_url": "https://github.com/galletas1712/flashinfer.git",
        "flashinfer_git_ref": "schwinns/checkpoint-collectives-integration",
        "flashinfer_source_sha": EXPECTED_FLASHINFER_SHA,
        "flashinfer_source_version": "0.6.14",
    },
}

OVERLAY_PATHS = (
    "vllm/distributed/device_communicators/all2all.py",
    "vllm/distributed/device_communicators/base_device_communicator.py",
    "vllm/distributed/device_communicators/cuda_communicator.py",
    "vllm/distributed/device_communicators/flashinfer_all_reduce.py",
    "vllm/distributed/parallel_state.py",
    "vllm/envs.py",
    "vllm/model_executor/warmup/kernel_warmup.py",
    "vllm/utils/mem_utils.py",
    "vllm/v1/attention/backends/flashinfer.py",
    "vllm/v1/engine/core.py",
    "vllm/v1/worker/gpu/attn_utils.py",
    "vllm/v1/worker/gpu_model_runner.py",
    "vllm/v1/worker/gpu_worker.py",
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


def verify_source_provenance(source: dict[str, str]) -> None:
    tuple_name = source.get("vllm_overlay_tuple", "")
    expected = ALLOWED_TUPLES.get(tuple_name)
    if expected is None:
        raise RuntimeError(f"Unknown vLLM overlay tuple: {source}")
    actual = source.copy()
    actual["vllm_runtime_base_image"] = actual.get(
        "vllm_runtime_base_image", ""
    ).removeprefix("docker.io/")
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if source.get("install_mode") != "python-overlay" or mismatches:
        raise RuntimeError(
            f"Unexpected vLLM overlay provenance for {tuple_name}: {mismatches}"
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


def verify_flashinfer() -> None:
    if not Path("/usr/local/cuda/include/nvrtc.h").is_file():
        raise RuntimeError("FlashInfer runtime JIT header is missing")
    from flashinfer import _build_meta

    if _build_meta.__git_version__ != EXPECTED_FLASHINFER_SHA:
        raise RuntimeError(
            f"FlashInfer source is {_build_meta.__git_version__}, "
            f"expected {EXPECTED_FLASHINFER_SHA}"
        )
    if FLASHINFER_SHA_PATH.read_text().strip() != EXPECTED_FLASHINFER_SHA:
        raise RuntimeError("FlashInfer durable source provenance is incorrect")


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
    expected_base_commit = os.environ.get("VLLM_EXPECTED_BASE_COMMIT")
    if (
        not expected_base_commit
        or os.environ.get("VLLM_BUILD_COMMIT") != expected_base_commit
    ):
        raise RuntimeError(
            "The upstream vLLM build commit does not match the selected tuple: "
            f"build={os.environ.get('VLLM_BUILD_COMMIT')}, "
            f"expected={expected_base_commit}"
        )
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
    verify_source_provenance(source)
    verify_overlay_files(Path(current["vllm"]["package"]))
    verify_flashinfer()
    assert_no_shim()
    verify_gms_backend_in_fresh_process()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "validate"))
    args = parser.parse_args()
    capture() if args.action == "capture" else validate()


if __name__ == "__main__":
    main()
