#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinism test for KVBM in aggregated mode.

To make sure KVBM's accuracy, this test suite checks if the model produces
deterministic outputs when same requests are served 1) without KVBM onboarded KV
blocks and 2) with KVBM onboarded KV blocks, when given the same inputs with
fixed seed and temperature=0.

The expected results should be 100% match between the two cases. Compared to
disaggregated mode, aggregated mode has less randomness chances.

These tests are slow by default (~368s and ~601s). For faster runs with
fewer iterations, run the following command (expected to finish in ~58s + ~152s):

    KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2 \
        pytest tests/kvbm_integration/test_determinism_agg.py -v --tb=short
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pytest
import requests

from tests.utils.test_output import resolve_test_output_path

from .common import DeterminismTester, LLMServerManager, ServerType
from .common import TestDeterminism as BaseTestDeterminism
from .common import check_module_available

HAS_VLLM_BENCH = check_module_available("vllm")

# KVBM env vars that drive test duration (used to compute timeouts below).
_KVBM_MAX_ITERATIONS = int(os.environ.get("KVBM_MAX_ITERATIONS", "100"))
_KVBM_NUM_ITERATIONS = int(os.environ.get("KVBM_NUM_ITERATIONS", "15"))
_KVBM_REQUEST_DELAY = int(os.environ.get("KVBM_REQUEST_DELAY", "30"))

# Compute timeouts from the same env vars that control test duration.
# test_determinism_agg_with_cache_reset: runs warmup + 2 phases of KVBM_MAX_ITERATIONS,
# each iteration ~4s (request + overhead), plus ~50s setup/teardown.
_CACHE_RESET_TIMEOUT = 2 * (_KVBM_MAX_ITERATIONS * 4 + 50)
# test_concurrent_determinism_under_load: dominated by
# (KVBM_NUM_ITERATIONS - 1) * KVBM_REQUEST_DELAY seconds of sleep,
# plus ~150s overhead (server startup, benchmark ramp, teardown).
_CONCURRENT_TIMEOUT = 2 * ((_KVBM_NUM_ITERATIONS - 1) * _KVBM_REQUEST_DELAY + 150)

# Test markers to align with repository conventions
# Todo: enable the rest when kvbm is built in the ci
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.gpu_1,
    pytest.mark.nightly,
]


class AggDeterminismTester(DeterminismTester):
    """Aggregated architecture specific determinism tester."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        super().__init__(base_url, model_id, server_type)

    def reset_prefix_cache(self):
        """Reset the prefix cache."""
        print("Resetting prefix cache...")
        if self.server_type == ServerType.trtllm:
            # TRTLLM doesn't support reset_prefix_cache endpoint API
            # 300 shakespeare content could evict the 0.1 x 80G (~1700 blocks) on-device cache
            shakespeare_count = 300
            for seq_idx in range(1, shakespeare_count + 1):
                start_word = (seq_idx - 1) * self.word_count
                content = self.get_shakespeare_content(start_word)

                if content:
                    print(
                        f"Resetting Shakespeare sequence {seq_idx} (words {start_word}-{start_word + self.word_count - 1})..."
                    )
                    try:
                        self.make_request(content)
                    except Exception as e:
                        print(f"Resetting request failed: {e}")
        else:
            response = requests.post(
                f"{self.base_url}/reset_prefix_cache",
                timeout=int(os.environ.get("KVBM_HTTP_TIMEOUT", "30")),
            )
            response.raise_for_status()
        print("Cache reset done")


@pytest.fixture(scope="function")
def llm_server(request, runtime_services):
    """Start and stop a LLM server for each test with optional cache block overrides.

    To parametrize, use:
      @pytest.mark.parametrize("llm_server", [{"cpu_blocks": 10000, "gpu_blocks": 2048}], indirect=True)
    """
    logger = logging.getLogger("pytest")
    logger.setLevel(logging.INFO)

    cpu_blocks = getattr(request, "param", {}).get("cpu_blocks", None)
    gpu_blocks = getattr(request, "param", {}).get("gpu_blocks", None)
    port = getattr(request, "param", {}).get("port", None)

    # Put logs in the per-test directory set up by tests/conftest.py
    log_dir = Path(resolve_test_output_path(request.node.name))

    if check_module_available("vllm"):
        server_type = ServerType.vllm
    elif check_module_available("tensorrt_llm"):
        server_type = ServerType.trtllm
    else:
        raise Exception(
            "Neither the vllm nor the tensorrt_llm module is available in the current environment."
        )

    server_manager = LLMServerManager(
        port=port,
        cpu_cache_blocks=cpu_blocks,
        gpu_cache_blocks=gpu_blocks,
        log_dir=log_dir,
        server_type=server_type,
    )

    start_timeout = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "300"))
    if not server_manager.start_server(timeout=start_timeout):
        pytest.fail(
            f"Failed to start {server_type} server (cpu_blocks={cpu_blocks}, gpu_blocks={gpu_blocks}, port={server_manager.port})"
        )

    yield server_manager

    server_manager.stop_server()


@pytest.fixture(scope="function")
def tester(llm_server):
    """Create determinism tester bound to the running server's base URL."""
    t = AggDeterminismTester(
        base_url=llm_server.base_url,
        server_type=llm_server.server_type,
    )
    t.download_shakespeare_text()
    return t


class TestDeterminismAgg(BaseTestDeterminism):
    """Test class for determinism validation."""

    @pytest.mark.parametrize(
        "llm_server",
        [
            {
                "cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000")),
                "gpu_blocks": int(os.environ.get("KVBM_GPU_BLOCKS", "2048")),
            },
        ],
        indirect=True,
    )
    @pytest.mark.kvbm
    @pytest.mark.timeout(
        _CACHE_RESET_TIMEOUT
    )  # ~368s actual measured on 32-core machine
    def test_determinism_agg_with_cache_reset(
        self, tester, llm_server, runtime_services
    ):
        """Test determinism across cache reset: run test with warmup, reset cache, run again without warmup."""
        # Call the base class implementation
        super().base_test_determinism_with_cache_reset(
            tester, llm_server, runtime_services
        )

    @pytest.mark.parametrize(
        "llm_server",
        [
            {
                "cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "30000")),
                "gpu_blocks": int(os.environ.get("KVBM_GPU_BLOCKS", "2048")),
            },
        ],
        indirect=True,
    )
    @pytest.mark.kvbm_concurrency
    @pytest.mark.skipif(
        not HAS_VLLM_BENCH, reason="requires vllm bench (vllm module not found)"
    )
    @pytest.mark.timeout(
        _CONCURRENT_TIMEOUT
    )  # ~601s actual measured on 32-core machine
    def test_concurrent_determinism_under_load(
        self, tester, llm_server, runtime_services
    ):
        """Test Spanish prompt determinism under high concurrency load.

        Reproduces the bug where Spanish responses become English or corrupted.
        """
        # Get the Spanish prompt path relative to this test file
        spanish_prompt_path = Path(
            os.path.join(os.path.dirname(__file__), "es_prompt.txt")
        ).absolute()

        # Call the base class implementation
        super().base_test_spanish_prompt_determinism_under_load(
            tester, llm_server, runtime_services, spanish_prompt_path
        )


if __name__ == "__main__":
    # Allow running as script
    pytest.main([__file__, "-v", "-s"])
