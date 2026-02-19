# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

try:
    from dynamo.profiler.profile_sla import run_profile  # noqa: E402
    from dynamo.profiler.utils.defaults import SearchStrategy  # noqa: E402
    from dynamo.profiler.utils.model_info import ModelInfo  # noqa: E402
    from dynamo.profiler.utils.search_space_autogen import (  # noqa: E402
        auto_generate_search_space,
    )
except ImportError as _e:
    pytest.skip(f"Skip testing (refactor in progress): {_e}", allow_module_level=True)


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
    def vllm_args(self, request):
        """Create arguments for vllm backend dry-run test."""

        class Args:
            def __init__(self):
                self.backend = "vllm"
                self.config = "examples/backends/vllm/deploy/disagg.yaml"
                # Use unique output directory per test for parallel execution
                self.output_dir = f"/tmp/test_profiling_results_{request.node.name}"
                self.namespace = f"test-namespace-{request.node.name}"
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
                self.dry_run = True
                self.aic_system = None
                self.aic_hf_id = None
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8
                self.search_strategy = SearchStrategy.THOROUGH
                self.system = ""
                self.deploy_after_profile = False
                self.pick_with_webui = False
                self.model_cache_pvc_name = ""
                self.model_cache_pvc_path = ""
                self.model_cache_pvc_mount_path = "/opt/model-cache"
                # Provide minimal model_info to avoid HF queries
                self.model_info = ModelInfo(
                    model_size=16384.0,
                    architecture="TestArchitecture",
                    is_moe=False,
                    max_context_length=self.max_context_length,
                )

        return Args()

    @pytest.fixture
    def sglang_args(self, request):
        """Create arguments for sglang backend dry-run test."""

        class Args:
            def __init__(self):
                self.backend = "sglang"
                self.config = "examples/backends/sglang/deploy/disagg.yaml"
                # Use unique output directory per test for parallel execution
                self.output_dir = f"/tmp/test_profiling_results_{request.node.name}"
                self.namespace = f"test-namespace-{request.node.name}"
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
                self.dry_run = True
                self.aic_system = None
                self.aic_hf_id = None
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8
                self.search_strategy = SearchStrategy.THOROUGH
                self.system = ""
                self.deploy_after_profile = False
                self.pick_with_webui = False
                self.model_cache_pvc_name = ""
                self.model_cache_pvc_path = ""
                self.model_cache_pvc_mount_path = "/opt/model-cache"
                self.model_info = ModelInfo(
                    model_size=16384.0,
                    architecture="TestArchitecture",
                    is_moe=False,
                    max_context_length=self.max_context_length,
                )

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.parallel
    @pytest.mark.asyncio
    @pytest.mark.gpu_0
    @pytest.mark.integration
    @pytest.mark.vllm
    async def test_vllm_dryrun(self, vllm_args):
        """Test that profile_sla dry-run works for vllm backend with disagg.yaml config."""
        # Run the profile in dry-run mode - should complete without errors
        await run_profile(vllm_args)

    @pytest.mark.pre_merge
    @pytest.mark.parallel
    @pytest.mark.asyncio
    @pytest.mark.gpu_0
    @pytest.mark.integration
    @pytest.mark.sglang
    async def test_sglang_dryrun(self, sglang_args):
        """Test that profile_sla dry-run works for sglang backend with disagg.yaml config."""
        # Run the profile in dry-run mode - should complete without errors
        await run_profile(sglang_args)

    @pytest.fixture
    def trtllm_args(self, request):
        """Create arguments for trtllm backend dry-run test."""

        class Args:
            def __init__(self):
                self.backend = "trtllm"
                self.config = "examples/backends/trtllm/deploy/disagg.yaml"
                # Use unique output directory per test for parallel execution
                self.output_dir = f"/tmp/test_profiling_results_{request.node.name}"
                self.namespace = f"test-namespace-{request.node.name}"
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
                self.dry_run = True
                self.aic_system = None
                self.aic_hf_id = None
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8
                self.search_strategy = SearchStrategy.THOROUGH
                self.system = ""
                self.deploy_after_profile = False
                self.pick_with_webui = False
                self.model_cache_pvc_name = ""
                self.model_cache_pvc_path = ""
                self.model_cache_pvc_mount_path = "/opt/model-cache"
                self.model_info = ModelInfo(
                    model_size=16384.0,
                    architecture="TestArchitecture",
                    is_moe=False,
                    max_context_length=self.max_context_length,
                )

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.parallel
    @pytest.mark.asyncio
    @pytest.mark.gpu_0
    @pytest.mark.integration
    @pytest.mark.trtllm
    async def test_trtllm_dryrun(self, trtllm_args):
        """Test that profile_sla dry-run works for trtllm backend with disagg.yaml config."""
        # Run the profile in dry-run mode - should complete without errors
        await run_profile(trtllm_args)

    @pytest.fixture
    def sglang_moe_args(self, request):
        """Create arguments for trtllm backend dry-run test."""

        class Args:
            def __init__(self):
                self.backend = "sglang"
                self.config = "recipes/deepseek-r1/sglang/disagg-16gpu/deploy.yaml"
                # Use unique output directory per test for parallel execution
                self.output_dir = f"/tmp/test_profiling_results_{request.node.name}"
                self.namespace = f"test-namespace-{request.node.name}"
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
                self.dry_run = True
                self.aic_system = None
                self.aic_hf_id = None
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8
                self.search_strategy = SearchStrategy.THOROUGH
                self.system = ""
                self.deploy_after_profile = False
                self.pick_with_webui = False
                # Added in newer profiler versions; keep Args compatible with search_space_autogen
                self.model_cache_pvc_name = ""
                self.model_cache_pvc_path = ""
                self.model_cache_pvc_mount_path = "/opt/model-cache"
                self.model_info = ModelInfo(
                    model_size=65536.0,
                    architecture="TestMoEArchitecture",
                    is_moe=True,
                    max_context_length=self.max_context_length,
                    num_experts=16,
                )

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.parallel
    @pytest.mark.asyncio
    @pytest.mark.gpu_0
    @pytest.mark.integration
    @pytest.mark.sglang
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
        return ModelInfo(
            model_size=16384.0,  # 16GB model in MiB
            architecture="LlamaForCausalLM",
            is_moe=False,
            max_context_length=16384,
        )

    @pytest.fixture
    def vllm_args_with_model_autogen(self, request):
        """Create arguments for vllm backend with model-based search space autogeneration."""

        class Args:
            def __init__(self):
                self.backend = "vllm"
                self.config = ""
                # Use unique output directory per test for parallel execution
                self.output_dir = f"/tmp/test_profiling_results_{request.node.name}"
                self.namespace = f"test-namespace-{request.node.name}"
                self.model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Specify model for autogen
                self.dgd_image = ""
                # Set to 0 to trigger auto-generation path
                self.min_num_gpus_per_engine = 0
                self.max_num_gpus_per_engine = 0
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
                self.dry_run = True
                self.system = "h100_sxm"  # Renamed from aic_system, moved to hardware
                self.search_strategy = SearchStrategy.RAPID  # New top-level arg
                # GPU discovery values (auto-populated by Operator)
                self.num_gpus_per_node = 8
                self.gpu_model = "H100-SXM5-80GB"
                self.gpu_vram_mib = 81920
                self.deploy_after_profile = False
                self.pick_with_webui = False
                self.model_cache_pvc_name = ""
                self.model_cache_pvc_path = ""
                self.model_cache_pvc_mount_path = "/opt/model-cache"

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.parallel
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.gpu_0
    @pytest.mark.vllm
    @patch("dynamo.profiler.utils.model_info.get_model_info")
    async def test_profile_with_autogen_search_space_h100(
        self,
        mock_get_model_info,
        vllm_args_with_model_autogen,
        mock_model_info,
    ):
        """Test profile_sla with auto-generated search space on mocked H100 cluster.

        This test demonstrates how search space is auto-generated based on model
        size and available GPU memory. GPU info is provided via command-line
        arguments injected by the Operator into the profiling config (DYN-2135).
        """
        # Configure the mock to return the appropriate model info
        mock_get_model_info.return_value = mock_model_info

        # Run the profile - the search space will be auto-generated
        # based on the model and GPU info from args
        auto_generate_search_space(vllm_args_with_model_autogen)
        await run_profile(vllm_args_with_model_autogen)

    @pytest.fixture
    def sglang_args_with_model_autogen(self, request):
        """Create arguments for sglang backend with model-based search space autogeneration."""

        class Args:
            def __init__(self):
                self.backend = "sglang"
                self.config = ""
                # Use unique output directory per test for parallel execution
                self.output_dir = f"/tmp/test_profiling_results_{request.node.name}"
                self.namespace = f"test-namespace-{request.node.name}"
                self.model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Specify model for autogen
                self.dgd_image = ""
                self.min_num_gpus_per_engine = 0
                self.max_num_gpus_per_engine = 0
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
                self.dry_run = True
                self.system = "h100_sxm"  # Renamed from aic_system, moved to hardware
                self.search_strategy = SearchStrategy.RAPID  # New top-level arg
                # GPU discovery values (auto-populated by Operator)
                self.num_gpus_per_node = 8
                self.gpu_model = "H100-SXM5-80GB"
                self.gpu_vram_mib = 81920
                self.deploy_after_profile = False
                self.pick_with_webui = False
                self.model_cache_pvc_name = ""
                self.model_cache_pvc_path = ""
                self.model_cache_pvc_mount_path = "/opt/model-cache"

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.parallel
    @pytest.mark.asyncio
    @pytest.mark.gpu_0
    @pytest.mark.integration
    @pytest.mark.sglang
    @pytest.mark.skip(
        reason="Blocked on AI Configurator database format: sglang 0.5.6.post2 database "
        "is in legacy format missing 'gemm_dtype' field. "
        "See: KeyError in aiconfigurator/sdk/perf_database.py"
    )
    @patch("dynamo.profiler.utils.model_info.get_model_info")
    async def test_sglang_profile_with_autogen_search_space_h100(
        self,
        mock_get_model_info,
        sglang_args_with_model_autogen,
        mock_model_info,
    ):
        """Test profile_sla with auto-generated search space for sglang on mocked H100 cluster.

        This test demonstrates how search space is auto-generated based on model
        size and available GPU memory for sglang backend. GPU info is provided via
        command-line arguments injected by the Operator into the profiling config (DYN-2135).

        NOTE: Currently skipped due to AI Configurator database format issue.
        The sglang 0.5.6.post2 database for h100_sxm is in legacy format and missing
        the required 'gemm_dtype' field, causing KeyError during database loading.
        """
        # Configure the mock to return the appropriate model info
        mock_get_model_info.return_value = mock_model_info

        # Run the profile - the search space will be auto-generated
        # based on the model and GPU info from args
        auto_generate_search_space(sglang_args_with_model_autogen)
        await run_profile(sglang_args_with_model_autogen)

    @pytest.fixture
    def trtllm_args_with_model_autogen(self, request):
        """Create arguments for trtllm backend with model-based search space autogeneration."""

        class Args:
            def __init__(self):
                self.backend = "trtllm"
                self.config = ""
                # Use unique output directory per test for parallel execution
                self.output_dir = f"/tmp/test_profiling_results_{request.node.name}"
                self.namespace = f"test-namespace-{request.node.name}"
                self.model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Specify model for autogen
                self.dgd_image = ""
                self.min_num_gpus_per_engine = 0
                self.max_num_gpus_per_engine = 0
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
                self.dry_run = True
                self.system = "h100_sxm"  # Renamed from aic_system, moved to hardware
                self.search_strategy = SearchStrategy.RAPID  # New top-level arg
                # GPU discovery values (auto-populated by Operator)
                self.num_gpus_per_node = 8
                self.gpu_model = "H100-SXM5-80GB"
                self.gpu_vram_mib = 81920
                self.deploy_after_profile = False
                self.pick_with_webui = False
                self.model_cache_pvc_name = ""
                self.model_cache_pvc_path = ""
                self.model_cache_pvc_mount_path = "/opt/model-cache"

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.parallel
    @pytest.mark.asyncio
    @pytest.mark.gpu_0
    @pytest.mark.integration
    @pytest.mark.trtllm
    @patch("dynamo.profiler.utils.model_info.get_model_info")
    async def test_trtllm_profile_with_autogen_search_space_h100(
        self,
        mock_get_model_info,
        trtllm_args_with_model_autogen,
        mock_model_info,
    ):
        """Test profile_sla with auto-generated search space for trtllm on mocked H100 cluster.

        This test demonstrates how search space is auto-generated based on model
        size and available GPU memory for trtllm backend. GPU info is provided via
        command-line arguments injected by the Operator into the profiling config (DYN-2135).
        """
        # Configure the mock to return the appropriate model info
        mock_get_model_info.return_value = mock_model_info

        # Run the profile - the search space will be auto-generated
        # based on the model and GPU info from args
        auto_generate_search_space(trtllm_args_with_model_autogen)
        await run_profile(trtllm_args_with_model_autogen)

    # Unit tests for search_strategy and system attributes
    @pytest.mark.pre_merge
    @pytest.mark.unit
    @pytest.mark.gpu_0
    def test_vllm_args_has_search_strategy(self, vllm_args):
        """Test that vllm_args fixture has search_strategy attribute."""
        assert hasattr(vllm_args, "search_strategy")
        assert vllm_args.search_strategy == SearchStrategy.THOROUGH
        assert hasattr(vllm_args, "system")
        assert vllm_args.system == ""

    @pytest.mark.pre_merge
    @pytest.mark.unit
    @pytest.mark.gpu_0
    def test_sglang_args_has_search_strategy(self, sglang_args):
        """Test that sglang_args fixture has search_strategy attribute."""
        assert hasattr(sglang_args, "search_strategy")
        assert sglang_args.search_strategy == SearchStrategy.THOROUGH
        assert hasattr(sglang_args, "system")
        assert sglang_args.system == ""

    @pytest.mark.pre_merge
    @pytest.mark.unit
    @pytest.mark.gpu_0
    def test_trtllm_args_has_search_strategy(self, trtllm_args):
        """Test that trtllm_args fixture has search_strategy attribute."""
        assert hasattr(trtllm_args, "search_strategy")
        assert trtllm_args.search_strategy == SearchStrategy.THOROUGH
        assert hasattr(trtllm_args, "system")
        assert trtllm_args.system == ""

    @pytest.mark.pre_merge
    @pytest.mark.unit
    @pytest.mark.gpu_0
    def test_sglang_moe_args_has_search_strategy(self, sglang_moe_args):
        """Test that sglang_moe_args fixture has search_strategy attribute."""
        assert hasattr(sglang_moe_args, "search_strategy")
        assert sglang_moe_args.search_strategy == SearchStrategy.THOROUGH
        assert hasattr(sglang_moe_args, "system")
        assert sglang_moe_args.system == ""

    @pytest.mark.pre_merge
    @pytest.mark.unit
    @pytest.mark.gpu_0
    def test_model_autogen_args_have_rapid_strategy(
        self,
        vllm_args_with_model_autogen,
        sglang_args_with_model_autogen,
        trtllm_args_with_model_autogen,
    ):
        """Test that model autogen fixtures have RAPID search strategy and GPU info."""
        for args_fixture in [
            vllm_args_with_model_autogen,
            sglang_args_with_model_autogen,
            trtllm_args_with_model_autogen,
        ]:
            assert hasattr(args_fixture, "search_strategy")
            assert args_fixture.search_strategy == SearchStrategy.RAPID
            assert hasattr(args_fixture, "system")
            assert args_fixture.system == "h100_sxm"
            # Verify GPU discovery attributes
            assert hasattr(args_fixture, "num_gpus_per_node")
            assert args_fixture.num_gpus_per_node == 8
            assert hasattr(args_fixture, "gpu_model")
            assert args_fixture.gpu_model == "H100-SXM5-80GB"
            assert hasattr(args_fixture, "gpu_vram_mib")
            assert args_fixture.gpu_vram_mib == 81920
