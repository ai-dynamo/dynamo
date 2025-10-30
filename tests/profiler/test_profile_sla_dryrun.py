# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for profile_sla dry-run functionality.

This test ensures that the profile_sla script can successfully run in dry-run mode
for vllm, sglang, and trtllm backends with their respective disagg.yaml configurations.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add the project root to sys.path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.profiler.profile_sla import run_profile  # noqa: E402
from benchmarks.profiler.utils.search_space_autogen import (  # noqa: E402
    auto_generate_search_space,
)


# Override the logger fixture from conftest.py to prevent directory creation
@pytest.fixture(autouse=True)
def logger(request):
    """Override the logger fixture to prevent test directory creation.

    This replaces the logger fixture from tests/conftest.py that creates
    directories named after each test.
    """
    # Simply do nothing - no directories created, no file handlers added
    yield


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
                self.model = ""
                self.dgd_image = ""
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
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8
                self.deploy_after_profile = False

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
                self.model = ""
                self.dgd_image = ""
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
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8
                self.deploy_after_profile = False

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
                self.model = ""
                self.dgd_image = ""
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
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8
                self.deploy_after_profile = False

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
                self.config = "recipes/deepseek-r1/sglang/disagg-16gpu/deploy.yaml"
                self.output_dir = "/tmp/test_profiling_results"
                self.namespace = "test-namespace"
                self.model = ""
                self.dgd_image = ""
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
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8
                self.deploy_after_profile = False

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.asyncio
    async def test_sglang_moe_dryrun(self, sglang_moe_args):
        """Test that profile_sla dry-run works for sglang backend with MoE config."""
        # Run the profile in dry-run mode - should complete without errors
        await run_profile(sglang_moe_args)

    # Example tests with mocked GPU inventory
    @pytest.fixture
    def mock_h100_gpu_info(self):
        """Mock GPU info for H100 80GB cluster."""
        return {
            "gpus_per_node": 8,
            "model": "h100_sxm",
            "vram": 81920,  # 80GB in MiB
        }

    @pytest.fixture
    def mock_model_info(self):
        """Mock model info for DeepSeek-R1-Distill-Llama-8B."""
        return {
            "model_size": 16384,  # 16GB model in MiB
            "is_moe": False,
            "max_context_length": 16384,  # 16K tokens
        }

    @pytest.fixture
    def vllm_args_with_model_autogen(self):
        """Create arguments for vllm backend with model-based search space autogeneration."""

        class Args:
            def __init__(self):
                self.backend = "vllm"
                self.config = ""
                self.output_dir = "/tmp/test_profiling_results"
                self.namespace = "test-namespace"
                self.model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Specify model for autogen
                self.dgd_image = ""
                self.min_num_gpus_per_engine = (
                    1  # Will be overridden by auto-generation
                )
                self.max_num_gpus_per_engine = (
                    8  # Will be overridden by auto-generation
                )
                self.skip_existing_results = False
                self.force_rerun = False
                self.isl = 3000
                self.osl = 500
                self.ttft = 50
                self.itl = 10
                self.max_context_length = 0
                self.prefill_interpolation_granularity = 16
                self.decode_interpolation_granularity = 6
                self.service_name = ""
                self.is_moe_model = False
                self.dry_run = True
                self.use_ai_configurator = False
                self.aic_system = None
                self.aic_model_name = None
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8  # Will be overridden by auto-generation
                self.deploy_after_profile = False

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.asyncio
    @patch("benchmarks.profiler.utils.search_space_autogen.get_gpu_summary")
    @patch("benchmarks.profiler.utils.search_space_autogen.get_model_info")
    async def test_profile_with_autogen_search_space_h100(
        self,
        mock_get_model_info,
        mock_get_gpu_summary,
        vllm_args_with_model_autogen,
        mock_h100_gpu_info,
        mock_model_info,
    ):
        """Test profile_sla with auto-generated search space on mocked H100 cluster.

        This test demonstrates how search space is auto-generated based on model
        size and available GPU memory.
        """
        # Configure the mocks to return the appropriate info
        mock_get_model_info.return_value = mock_model_info
        mock_get_gpu_summary.return_value = mock_h100_gpu_info

        # Run the profile - the search space will be auto-generated
        # based on the model and mocked GPU info
        auto_generate_search_space(vllm_args_with_model_autogen)
        await run_profile(vllm_args_with_model_autogen)

    @pytest.fixture
    def sglang_args_with_model_autogen(self):
        """Create arguments for sglang backend with model-based search space autogeneration."""

        class Args:
            def __init__(self):
                self.backend = "sglang"
                self.config = ""
                self.output_dir = "/tmp/test_profiling_results"
                self.namespace = "test-namespace"
                self.model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Specify model for autogen
                self.dgd_image = ""
                self.min_num_gpus_per_engine = (
                    1  # Will be overridden by auto-generation
                )
                self.max_num_gpus_per_engine = (
                    8  # Will be overridden by auto-generation
                )
                self.skip_existing_results = False
                self.force_rerun = False
                self.isl = 3000
                self.osl = 500
                self.ttft = 50
                self.itl = 10
                self.max_context_length = 0
                self.prefill_interpolation_granularity = 16
                self.decode_interpolation_granularity = 6
                self.service_name = ""
                self.is_moe_model = False
                self.dry_run = True
                self.use_ai_configurator = False
                self.aic_system = None
                self.aic_model_name = None
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8  # Will be overridden by auto-generation
                self.deploy_after_profile = False

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.asyncio
    @patch("benchmarks.profiler.utils.search_space_autogen.get_gpu_summary")
    @patch("benchmarks.profiler.utils.search_space_autogen.get_model_info")
    async def test_sglang_profile_with_autogen_search_space_h100(
        self,
        mock_get_model_info,
        mock_get_gpu_summary,
        sglang_args_with_model_autogen,
        mock_h100_gpu_info,
        mock_model_info,
    ):
        """Test profile_sla with auto-generated search space for sglang on mocked H100 cluster.

        This test demonstrates how search space is auto-generated based on model
        size and available GPU memory for sglang backend.
        """
        # Configure the mocks to return the appropriate info
        mock_get_model_info.return_value = mock_model_info
        mock_get_gpu_summary.return_value = mock_h100_gpu_info

        # Run the profile - the search space will be auto-generated
        # based on the model and mocked GPU info
        auto_generate_search_space(sglang_args_with_model_autogen)
        await run_profile(sglang_args_with_model_autogen)

    @pytest.fixture
    def trtllm_args_with_model_autogen(self):
        """Create arguments for trtllm backend with model-based search space autogeneration."""

        class Args:
            def __init__(self):
                self.backend = "trtllm"
                self.config = ""
                self.output_dir = "/tmp/test_profiling_results"
                self.namespace = "test-namespace"
                self.model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Specify model for autogen
                self.dgd_image = ""
                self.min_num_gpus_per_engine = (
                    1  # Will be overridden by auto-generation
                )
                self.max_num_gpus_per_engine = (
                    8  # Will be overridden by auto-generation
                )
                self.skip_existing_results = False
                self.force_rerun = False
                self.isl = 3000
                self.osl = 500
                self.ttft = 50
                self.itl = 10
                self.max_context_length = 0
                self.prefill_interpolation_granularity = 16
                self.decode_interpolation_granularity = 6
                self.service_name = ""
                self.is_moe_model = False
                self.dry_run = True
                self.use_ai_configurator = False
                self.aic_system = None
                self.aic_model_name = None
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8  # Will be overridden by auto-generation
                self.deploy_after_profile = False

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.asyncio
    @patch("benchmarks.profiler.utils.search_space_autogen.get_gpu_summary")
    @patch("benchmarks.profiler.utils.search_space_autogen.get_model_info")
    async def test_trtllm_profile_with_autogen_search_space_h100(
        self,
        mock_get_model_info,
        mock_get_gpu_summary,
        trtllm_args_with_model_autogen,
        mock_h100_gpu_info,
        mock_model_info,
    ):
        """Test profile_sla with auto-generated search space for trtllm on mocked H100 cluster.

        This test demonstrates how search space is auto-generated based on model
        size and available GPU memory for trtllm backend.
        """
        # Configure the mocks to return the appropriate info
        mock_get_model_info.return_value = mock_model_info
        mock_get_gpu_summary.return_value = mock_h100_gpu_info

        # Run the profile - the search space will be auto-generated
        # based on the model and mocked GPU info
        auto_generate_search_space(trtllm_args_with_model_autogen)
        await run_profile(trtllm_args_with_model_autogen)
