#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinism test for KVBM in disaggregated mode.

To make sure KVBM's accuracy, this test suite checks if the model produces
deterministic outputs when same requests are served 1) without KVBM onboarded KV
blocks and 2) with KVBM onboarded KV blocks, when given the same inputs with
fixed seed and temperature=0.

The expected results should be at least 95% match between the two cases.
Compared to aggregated mode, disaggregated mode has some known randomness.
Example reference: https://github.com/vllm-project/vllm/issues/7779#issuecomment-2304967870
"""

import os

import pytest

from .common import TestDeterminism as BaseTestDeterminism

# Register fixtures from common.py
pytest_plugins = ["tests.kvbm_integration.common"]

# Test markers to align with repository conventions
pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.gpu_2,
    pytest.mark.nightly,
]


SUCCESS_RATE_THRESHOLD = 0.95


# =============================================================================
# Test Classes
# =============================================================================


class TestDeterminismDisagg(BaseTestDeterminism):
    """Test class for determinism validation in disaggregated mode."""

    @pytest.mark.parametrize(
        "disagg_llm_server",
        [
            {
                "cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000")),
                "gpu_blocks": int(os.environ.get("KVBM_GPU_BLOCKS", "1000")),
            },
        ],
        indirect=True,
    )
    def test_determinism_disagg_with_cache_reset(
        self, disagg_tester, disagg_llm_server, runtime_services
    ):
        """Test determinism across cache reset: run test with warmup, reset cache, run again without warmup."""
        super().base_test_determinism_with_cache_reset(
            disagg_tester,
            disagg_llm_server,
            runtime_services,
            success_rate_threshold=SUCCESS_RATE_THRESHOLD,
        )

    @pytest.mark.parametrize(
        "disagg_llm_server",
        [
            {
                "cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000")),
                "gpu_blocks": int(os.environ.get("KVBM_GPU_BLOCKS", "1000")),
            },
        ],
        indirect=True,
    )
    @pytest.mark.kvbm_v2
    def test_determinism_disagg_with_cache_reset_v2(
        self, disagg_tester, disagg_llm_server, runtime_services, monkeypatch
    ):
        """Test determinism across cache reset with V2 transfer."""
        monkeypatch.setenv("DYN_KVBM_USE_V2_TRANSFER_EXPERIMENTAL", "1")
        super().base_test_determinism_with_cache_reset(
            disagg_tester,
            disagg_llm_server,
            runtime_services,
            success_rate_threshold=SUCCESS_RATE_THRESHOLD,
        )


if __name__ == "__main__":
    # Allow running as script
    pytest.main([__file__, "-v", "-s"])
