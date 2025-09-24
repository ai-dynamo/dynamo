# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for profile_sla dry-run functionality.

This test ensures that the profile_sla script can successfully run in dry-run mode
for vllm, sglang, and trtllm backends with their respective disagg.yaml configurations.
"""

import sys
from pathlib import Path

import pytest

# Add the project root to sys.path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.profiler.profile_sla import run_profile  # noqa: E402


class TestProfileSLADryRun:
    """Test class for profile_sla dry-run functionality."""

    @pytest.fixture
    def vllm_args(self):
        """Create arguments for vllm backend dry-run test."""

        class Args:
            def __init__(self):
                self.backend = "vllm"
                self.config = "components/backends/vllm/deploy/disagg.yaml"
                self.output_dir = "/tmp/test_profiling_results"
                self.namespace = "test-namespace"
                self.min_num_gpus_per_engine = 1
                self.max_num_gpus_per_engine = 8
                self.skip_existing_results = False
                self.force_rerun = False
                self.isl = 3000
                self.osl = 500
                self.ttft = 50
                self.itl = 10
                self.max_context_length = 16384
                self.prefill_interpolation_granularity = 16
                self.decode_interpolation_granularity = 6
                self.service_name = ""
                self.is_moe_model = False
                self.dry_run = True
                self.use_ai_configurator = False
                self.aic_system = None
                self.aic_model_name = None
                self.backend_version = None
                self.num_gpus_per_node = 8

        return Args()

    @pytest.fixture
    def sglang_args(self):
        """Create arguments for sglang backend dry-run test."""

        class Args:
            def __init__(self):
                self.backend = "sglang"
                self.config = "components/backends/sglang/deploy/disagg.yaml"
                self.output_dir = "/tmp/test_profiling_results"
                self.namespace = "test-namespace"
                self.min_num_gpus_per_engine = 1
                self.max_num_gpus_per_engine = 8
                self.skip_existing_results = False
                self.force_rerun = False
                self.isl = 3000
                self.osl = 500
                self.ttft = 50
                self.itl = 10
                self.max_context_length = 16384
                self.prefill_interpolation_granularity = 16
                self.decode_interpolation_granularity = 6
                self.service_name = ""
                self.is_moe_model = False
                self.dry_run = True
                self.use_ai_configurator = False
                self.aic_system = None
                self.aic_model_name = None
                self.backend_version = None
                self.num_gpus_per_node = 8

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.asyncio
    async def test_vllm_dryrun(self, vllm_args):
        """Test that profile_sla dry-run works for vllm backend with disagg.yaml config."""
        # Run the profile in dry-run mode - should complete without errors
        await run_profile(vllm_args)

    @pytest.mark.pre_merge
    @pytest.mark.asyncio
    async def test_sglang_dryrun(self, sglang_args):
        """Test that profile_sla dry-run works for sglang backend with disagg.yaml config."""
        # Run the profile in dry-run mode - should complete without errors
        await run_profile(sglang_args)

    @pytest.fixture
    def trtllm_args(self):
        """Create arguments for trtllm backend dry-run test."""

        class Args:
            def __init__(self):
                self.backend = "trtllm"
                self.config = "components/backends/trtllm/deploy/disagg.yaml"
                self.output_dir = "/tmp/test_profiling_results"
                self.namespace = "test-namespace"
                self.min_num_gpus_per_engine = 1
                self.max_num_gpus_per_engine = 8
                self.skip_existing_results = False
                self.force_rerun = False
                self.isl = 3000
                self.osl = 500
                self.ttft = 50
                self.itl = 10
                self.max_context_length = 16384
                self.prefill_interpolation_granularity = 16
                self.decode_interpolation_granularity = 6
                self.service_name = ""
                self.is_moe_model = False
                self.dry_run = True
                self.use_ai_configurator = False
                self.aic_system = None
                self.aic_model_name = None
                self.backend_version = None
                self.num_gpus_per_node = 8

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.asyncio
    async def test_trtllm_dryrun(self, trtllm_args):
        """Test that profile_sla dry-run works for trtllm backend with disagg.yaml config."""
        # Run the profile in dry-run mode - should complete without errors
        await run_profile(trtllm_args)

    @pytest.fixture
    def sglang_moe_args(self):
        """Create arguments for trtllm backend dry-run test."""

        class Args:
            def __init__(self):
                self.backend = "sglang"
                self.config = (
                    "recipes/deepseek-r1/sglang-wideep/tep16p-dep16d-disagg.yaml"
                )
                self.output_dir = "/tmp/test_profiling_results"
                self.namespace = "test-namespace"
                self.min_num_gpus_per_engine = 8
                self.max_num_gpus_per_engine = 32
                self.skip_existing_results = False
                self.force_rerun = False
                self.isl = 3000
                self.osl = 500
                self.ttft = 50
                self.itl = 10
                self.max_context_length = 16384
                self.prefill_interpolation_granularity = 16
                self.decode_interpolation_granularity = 6
                self.service_name = ""
                self.is_moe_model = True
                self.dry_run = True
                self.use_ai_configurator = False
                self.aic_system = None
                self.aic_model_name = None
                self.backend_version = None
                self.num_gpus_per_node = 8

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.asyncio
    async def test_sglang_moe_dryrun(self, sglang_moe_args):
        """Test that profile_sla dry-run works for sglang backend with MoE config."""
        # Run the profile in dry-run mode - should complete without errors
        await run_profile(sglang_moe_args)
