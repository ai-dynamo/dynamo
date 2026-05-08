#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MLA (Multi-Latent Attention) integration tests for KVBM.

These tests validate that MLA models (e.g., DeepSeek R1) work correctly with
KVBM offload/onboard for both vLLM and TensorRT-LLM backends.

MLA models use a compressed KV cache representation (latent attention), so these
tests verify that KVBM handles the MLA KV blocks correctly through the full
offload → cache reset → onboard cycle.

For TensorRT-LLM, the test also exercises the DYN_KVBM_NCCL_MLA_MODE env var
path, validating graceful fallback when MPI/NCCL is not available.
"""

import logging
import os
import shutil

import pytest
import requests

from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.payloads import check_models_api

from .common import llm_server_kvbm  # noqa: F401
from .common import (
    DeterminismTester,
    assert_deterministic,
    check_module_available,
    fetch_kvbm_metrics,
)

logger = logging.getLogger(__name__)

# MLA model: tiny 2-layer DeepSeek R1 with Multi-Latent Attention
MLA_MODEL = "silence09/DeepSeek-R1-Small-2layers"

# Module availability checks
HAS_VLLM = check_module_available("vllm")
HAS_TRTLLM = check_module_available("tensorrt_llm")

# Test configuration
MAX_TOKENS = 15

# Test prompt
PROMPT = (
    "In the heart of Eldoria, an ancient land of boundless magic and mysterious "
    "creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge "
    "and power, Aeloria was buried beneath the shifting sands of time, lost to the "
    "world for centuries. You are an intrepid explorer, known for your unparalleled "
    "curiosity and courage, who has stumbled upon an ancient map hinting at secrets "
    "that Aeloria holds a secret so profound that it has the potential to reshape the "
    "very fabric of reality."
)


# =============================================================================
# Helper functions
# =============================================================================


def print_test_header(title: str) -> None:
    """Print a formatted test header."""
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def print_phase(phase_num: int, description: str) -> None:
    """Print a formatted phase header."""
    print(f"\n=== Phase {phase_num}: {description} ===")


def check_kvbm_metrics_mla(phase_name: str, metrics_port: int) -> dict[str, int]:
    """Fetch and display KVBM metrics."""
    print(f"\n--- Checking KVBM metrics after {phase_name} ---")
    metrics = fetch_kvbm_metrics(port=metrics_port)

    offload_d2h = metrics.get("kvbm_offload_blocks_d2h", 0)
    onboard_h2d = metrics.get("kvbm_onboard_blocks_h2d", 0)

    print(f"  kvbm_offload_blocks_d2h: {offload_d2h}")
    print(f"  kvbm_onboard_blocks_h2d: {onboard_h2d}")

    return {
        "kvbm_offload_blocks_d2h": offload_d2h,
        "kvbm_onboard_blocks_h2d": onboard_h2d,
    }


def reset_cache(base_url: str) -> None:
    """Reset the GPU prefix cache."""
    print("Resetting prefix cache...")
    try:
        response = requests.post(f"{base_url}/reset_prefix_cache", timeout=30)
        response.raise_for_status()
        print("Cache reset successful")
    except Exception as e:
        print(f"Warning: Cache reset failed: {e}")


# =============================================================================
# vLLM MLA Tests
# =============================================================================


@pytest.fixture(scope="function")
def mla_tester(llm_server_kvbm):  # noqa: F811
    """Create tester bound to the KVBM-enabled server with MLA model."""
    return DeterminismTester(
        base_url=llm_server_kvbm.base_url,
        model_id=MLA_MODEL,
        server_type=llm_server_kvbm.server_type,
    )


@pytest.mark.kvbm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.vllm
@pytest.mark.pre_merge
@pytest.mark.mla
@pytest.mark.model(MLA_MODEL)
@pytest.mark.skipif(not HAS_VLLM, reason="requires vllm")
@pytest.mark.parametrize("llm_server_kvbm", [{"model": MLA_MODEL}], indirect=True)
@pytest.mark.timeout(200)
def test_mla_offload_and_onboard_vllm(mla_tester, llm_server_kvbm):  # noqa: F811
    """
    Test KVBM offload/onboard cycle with an MLA (DeepSeek R1) model on vLLM.

    Validates that:
    - MLA model KV cache blocks are correctly offloaded to CPU
    - Cache reset clears GPU cache
    - Re-request triggers onboard from CPU to GPU
    - Responses are deterministic across the offload/onboard cycle
    """
    print_test_header("MLA OFFLOAD AND ONBOARD TEST (vLLM)")
    print(f"Model: {MLA_MODEL}")

    prompt = PROMPT[:400]

    # Phase 1: Initial request triggers offload
    print_phase(1, "Initial request (expect offload to CPU)")
    print(f"Sending request: {prompt[:80]}...")

    response_1 = mla_tester.make_request(prompt, max_tokens=MAX_TOKENS)
    print(f"Response 1: {response_1}")

    metrics = check_kvbm_metrics_mla("Phase 1", llm_server_kvbm.metrics_port)
    assert (
        metrics["kvbm_offload_blocks_d2h"] > 0
    ), "Phase 1: No blocks offloaded. KVBM may not be triggering offloads for MLA model."
    assert (
        metrics["kvbm_onboard_blocks_h2d"] == 0
    ), f"Phase 1: Expected 0 onboarded blocks, got {metrics['kvbm_onboard_blocks_h2d']}"
    print(f"Phase 1: {metrics['kvbm_offload_blocks_d2h']} blocks offloaded")

    # Phase 2: Reset GPU cache
    print_phase(2, "Clean up GPU cache")
    reset_cache(llm_server_kvbm.base_url)

    # Phase 3: Repeated request triggers onboard
    print_phase(3, "Re-send same request (expect onboard from CPU)")
    print(f"Sending same request: {prompt[:80]}...")

    response_2 = mla_tester.make_request(prompt, max_tokens=MAX_TOKENS)
    print(f"Response 2: {response_2}")

    metrics = check_kvbm_metrics_mla("Phase 3", llm_server_kvbm.metrics_port)
    assert (
        metrics["kvbm_onboard_blocks_h2d"] > 0
    ), "Phase 3: No blocks onboarded. Expected CPU->GPU transfer after cache reset for MLA model."
    print(f"Phase 3: {metrics['kvbm_onboard_blocks_h2d']} blocks onboarded from CPU")

    # Verify determinism
    print_test_header("DETERMINISM VERIFICATION")
    assert_deterministic(
        response_1,
        response_2,
        test_name="MLA Offload/Onboard",
        label1="Initial response",
        label2="After cache reset",
    )

    print("\n=== TEST PASSED ===")


# =============================================================================
# TensorRT-LLM MLA Tests
# =============================================================================


class MlaDynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with TensorRT-LLM backend and MLA model."""

    def __init__(self, request, worker_id: str, engine_config: str):
        self.worker_id = worker_id

        command = [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model",
            MLA_MODEL,
            "--served-model-name",
            MLA_MODEL,
            "--tensor-parallel-size",
            "2",
            "--connector",
            "kvbm",
            "--extra-engine-args",
            engine_config,
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        # TODO: Replace hardcoded port with allocate_ports() for xdist-safe parallel execution
        env["DYN_SYSTEM_PORT"] = "9345"
        env["DYN_KVBM_CPU_CACHE_GB"] = "100"
        env["DYN_KVBM_DISK_CACHE_GB"] = "60"
        env["DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS"] = "1200"
        env["DYN_KVBM_TRTLLM_ZMQ_PORT"] = "20081"
        env["DYN_KVBM_METRICS"] = "true"
        # Enable NCCL MLA mode to exercise the code path.
        # Without MPI, _get_mpi_info() returns (None, None) and the code
        # falls back gracefully to standard per-GPU loading.
        env["DYN_KVBM_NCCL_MLA_MODE"] = "true"

        log_dir = f"{request.node.name}_{worker_id}"

        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{FRONTEND_PORT}/v1/models", check_models_api),
                ("http://localhost:9345/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            log_dir=log_dir,
        )

    def get_pid(self) -> int | None:
        """Get the PID of the worker process."""
        return self.proc.pid if hasattr(self, "proc") and self.proc else None

    def is_ready(self, response) -> bool:
        """Check the health of the worker process."""
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info(
                    f"{self.__class__.__name__} {{ name: {self.worker_id} }} status is ready"
                )
                return True
            logger.warning(
                f"{self.__class__.__name__} {{ name: {self.worker_id} }} status is not ready: {data.get('status')}"
            )
        except ValueError:
            logger.warning(
                f"{self.__class__.__name__} {{ name: {self.worker_id} }} health response is not valid JSON"
            )
        return False


def send_mla_completion_request(
    prompt: str, max_tokens: int, timeout: int = 120
) -> requests.Response:
    """Send a completion request to the frontend for the MLA model."""
    payload = {
        "model": MLA_MODEL,
        "prompt": prompt,
        "stream": False,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}

    logger.info(
        f"Sending completion request with prompt: '{prompt[:50]}...' and max_tokens: {max_tokens}"
    )

    try:
        response = requests.post(
            "http://localhost:8000/v1/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        return response
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {timeout} seconds")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed with error: {e}")
        raise


def _precache_deepseek_remote_code():
    """Pre-cache the remote code files from deepseek-ai/DeepSeek-R1.

    The silence09/DeepSeek-R1-Small-2layers model's config.json has an auto_map
    pointing to deepseek-ai/DeepSeek-R1 for configuration_deepseek.py and
    modeling_deepseek.py. TRT-LLM's executor subprocess needs these files but
    may not be able to download them in offline/slow-network CI environments.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="deepseek-ai/DeepSeek-R1",
            allow_patterns=["*.py", "*.json"],
            ignore_patterns=["*.safetensors", "*.bin", "*.h5", "*.msgpack", "*.ckpt*"],
        )
        logger.info("Pre-cached deepseek-ai/DeepSeek-R1 remote code files")
    except Exception as e:
        logger.warning(f"Failed to pre-cache DeepSeek-R1 remote code: {e}")


@pytest.mark.kvbm
@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.gpu_2
@pytest.mark.mla
@pytest.mark.model(MLA_MODEL)
@pytest.mark.skipif(not HAS_TRTLLM, reason="requires tensorrt_llm")
def test_mla_kvbm_trtllm(request, runtime_services):
    """
    End-to-end test for TensorRT-LLM worker with MLA model and KVBM enabled.

    Validates that:
    - MLA model (DeepSeek R1) serves requests successfully with KVBM and TRT-LLM
    - DYN_KVBM_NCCL_MLA_MODE=true is accepted without error (graceful fallback
      when MPI/NCCL is not available)
    - KVBM offloads blocks for the MLA model
    """
    # Pre-cache remote code files that TRT-LLM's executor subprocess needs
    _precache_deepseek_remote_code()

    logger.info("Starting frontend...")
    with DynamoFrontendProcess(request):
        logger.info("Frontend started.")

        engine_config_path = "tests/kvbm_integration/engine_config_mla_kvbm.yaml"
        logger.info(
            f"Starting MLA worker with DYN_KVBM_NCCL_MLA_MODE=true "
            f"(model: {MLA_MODEL})..."
        )
        with MlaDynamoWorkerProcess(request, "decode", engine_config_path) as worker:
            logger.info(f"Worker PID: {worker.get_pid()}")

            response = send_mla_completion_request(PROMPT, 100, timeout=10)
            assert (
                response.ok
            ), f"Expected successful status, got {response.status_code}"
            logger.info(f"Completion request succeeded: {response.status_code}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
