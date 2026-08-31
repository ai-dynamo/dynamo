# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-GPU DGDR support matrix for H100 clusters."""

from __future__ import annotations

import logging
import os

import pytest

from tests.deploy.dgdr_utils import ManagedDGDR, build_dgdr, run_lifecycle, unique_name

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.k8s,
    pytest.mark.deploy,
    pytest.mark.e2e,
    pytest.mark.integration,
    pytest.mark.nightly,
    pytest.mark.h100,
    pytest.mark.gpu_8,
    pytest.mark.slow,
]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        logger.warning("%s is invalid; using %s", name, default)
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        logger.warning("%s is invalid; using %s", name, default)
        return default


H100_TTFT_MS = _env_float("DGDR_H100_TTFT_MS", 500.0)
H100_ITL_MS = _env_float("DGDR_H100_ITL_MS", 30.0)
H100_ISL = _env_int("DGDR_H100_ISL", 3000)
H100_OSL = _env_int("DGDR_H100_OSL", 300)
H100_GPU_COUNT = 8

BACKENDS = ["vllm", "sglang", "trtllm"]


def _h100_spec(model: str, backend: str) -> dict:
    return {
        "model": model,
        "backend": backend,
        "searchStrategy": "rapid",
        "autoApply": True,
        "sla": {"ttft": H100_TTFT_MS, "itl": H100_ITL_MS},
        "workload": {"isl": H100_ISL, "osl": H100_OSL},
        "hardware": {
            "gpuSku": "h100_sxm",
            "vramMb": 81920.0,
            "numGpusPerNode": H100_GPU_COUNT,
            "totalGpus": H100_GPU_COUNT,
        },
        "features": {
            "planner": {
                "mode": "disagg",
                "enable_throughput_scaling": True,
                "enable_load_scaling": True,
                "max_gpu_budget": H100_GPU_COUNT,
            }
        },
    }


def _h100_manifest(
    manager: ManagedDGDR,
    name: str,
    model: str,
    backend: str,
    dgd_override: dict | None = None,
) -> dict:
    spec = _h100_spec(model, backend)
    if dgd_override:
        spec["overrides"] = {"dgd": dgd_override}
    return build_dgdr(
        manager.config,
        unique_name(manager.config, name),
        spec_overrides=spec,
    )


def _require_real_h100(manager: ManagedDGDR) -> None:
    if manager.config.mocker:
        pytest.skip("H100 support-matrix tests require --dgdr-no-mocker")


@pytest.mark.model("Qwen/Qwen3-32B")
@pytest.mark.timeout(4260)
@pytest.mark.parametrize("backend", BACKENDS)
async def test_dgdr_h100_qwen3_32b(dgdr_manager: ManagedDGDR, backend: str) -> None:
    _require_real_h100(dgdr_manager)
    dgdr = _h100_manifest(dgdr_manager, f"qwen32b-{backend}", "Qwen/Qwen3-32B", backend)
    await run_lifecycle(
        dgdr_manager,
        dgdr,
        expected_services={"Planner": 1},
        verify_inference=True,
    )


@pytest.mark.model("Qwen/Qwen3-235B-A22B-FP8")
@pytest.mark.timeout(4260)
@pytest.mark.parametrize("backend", BACKENDS)
async def test_dgdr_h100_qwen3_235b(dgdr_manager: ManagedDGDR, backend: str) -> None:
    _require_real_h100(dgdr_manager)
    override = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "q235"},
        "spec": {
            "services": {
                "prefill": {"sharedMemory": {"size": "256Gi"}},
                "decode": {"sharedMemory": {"size": "256Gi"}},
            }
        },
    }
    dgdr = _h100_manifest(
        dgdr_manager,
        f"qwen235b-{backend}",
        "Qwen/Qwen3-235B-A22B-FP8",
        backend,
        override,
    )
    await run_lifecycle(
        dgdr_manager,
        dgdr,
        expected_services={"Planner": 1},
        verify_inference=True,
    )


@pytest.mark.model("openai/gpt-oss-20b")
@pytest.mark.timeout(4260)
async def test_dgdr_h100_gpt_oss_20b(dgdr_manager: ManagedDGDR) -> None:
    _require_real_h100(dgdr_manager)
    override = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "gptoss"},
        "spec": {"services": {"worker": {"sharedMemory": {"size": "80Gi"}}}},
    }
    dgdr = _h100_manifest(
        dgdr_manager,
        "gptoss20b",
        "openai/gpt-oss-20b",
        "trtllm",
        override,
    )
    await run_lifecycle(dgdr_manager, dgdr, verify_inference=True)


@pytest.mark.model("meta-llama/Meta-Llama-3.1-70B")
@pytest.mark.timeout(4260)
@pytest.mark.parametrize("backend", BACKENDS)
async def test_dgdr_h100_llama_3_1_70b(dgdr_manager: ManagedDGDR, backend: str) -> None:
    _require_real_h100(dgdr_manager)
    override = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "llama31-70b"},
        "spec": {
            "services": {
                "prefill": {"sharedMemory": {"size": "256Gi"}},
                "decode": {"sharedMemory": {"size": "256Gi"}},
            }
        },
    }
    dgdr = _h100_manifest(
        dgdr_manager,
        f"llama31-70b-{backend}",
        "meta-llama/Meta-Llama-3.1-70B",
        backend,
        override,
    )
    await run_lifecycle(
        dgdr_manager,
        dgdr,
        expected_services={"Planner": 1},
        verify_inference=True,
    )
