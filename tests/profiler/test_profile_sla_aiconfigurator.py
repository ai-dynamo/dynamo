# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for profile_sla aiconfigurator functionality.

profile_sla should be able to use aiconfigurator functionality
even without access to any GPU system.
"""

import sys
from pathlib import Path

import pytest

# Add the project root to sys.path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.profiler.profile_sla import run_profile  # noqa: E402
from benchmarks.profiler.utils.model_info import ModelInfo  # noqa: E402


# Override the logger fixture from conftest.py to prevent directory creation
@pytest.fixture(autouse=True)
def logger(request):
    """Override the logger fixture to prevent test directory creation.

    This replaces the logger fixture from tests/conftest.py that creates
    directories named after each test.
    """
    # Simply do nothing - no directories created, no file handlers added
    yield


class TestProfileSlaAiconfigurator:
    """Test class for profile_sla aiconfigurator functionality."""

    @pytest.fixture
    def llm_args(self, request):
        class Args:
            def __init__(self):
                self.model = ""
                self.dgd_image = ""
                self.backend = "trtllm"
                self.config = "examples/backends/trtllm/deploy/disagg.yaml"
                # Use unique output directory per test for parallel execution
                self.output_dir = f"/tmp/test_profiling_results_{request.node.name}"
                self.namespace = f"test-namespace-{request.node.name}"
                self.min_num_gpus_per_engine = 1
                self.max_num_gpus_per_engine = 8
                self.skip_existing_results = False
                self.force_rerun = False
                self.isl = 3000
                self.osl = 500
                self.ttft = 50
                self.itl = 10
                self.prefill_interpolation_granularity = 16
                self.decode_interpolation_granularity = 6
                self.service_name = ""
                self.dry_run = False
                self.use_ai_configurator = True
                self.aic_system = "h200_sxm"
                self.aic_hf_id = "Qwen/Qwen3-32B"
                self.aic_backend = ""
                self.aic_backend_version = None
                self.num_gpus_per_node = 8
                self.deploy_after_profile = False
                self.pick_with_webui = False
                # Provide minimal model_info to avoid HF queries
                self.model_info = ModelInfo(
                    model_size=16384.0,
                    architecture="TestArchitecture",
                    is_moe=False,
                    max_context_length=16384,
                )

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    @pytest.mark.performance
    @pytest.mark.parallel
    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing_arg", ["aic_system", "aic_hf_id"])
    async def test_aiconfigurator_missing_args(self, llm_args, missing_arg):
        # Check that validation error happens when a required arg is missing.
        # Note: aic_backend_version is optional - when None, auto-detects latest version
        setattr(llm_args, missing_arg, None)
        with pytest.raises(ValueError):
            await run_profile(llm_args)

    @pytest.mark.pre_merge
    @pytest.mark.gpu_0
    @pytest.mark.performance
    @pytest.mark.parallel
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "arg_name, bad_value",
        [
            # these values don't exist in the aiconfigurator database.
            ("aic_system", "fake_gpu_system"),
            ("aic_backend_version", "0.1.0"),
        ],
    )
    async def test_aiconfigurator_no_data(self, llm_args, arg_name, bad_value):
        # Check that an appropriate error is raised when the system/model/backend
        # is not found in the aiconfigurator database.
        setattr(llm_args, arg_name, bad_value)
        with pytest.raises(ValueError, match="Database not found"):
            await run_profile(llm_args)

    @pytest.mark.pre_merge
    @pytest.mark.parallel
    @pytest.mark.asyncio
    @pytest.mark.gpu_1
    @pytest.mark.integration
    async def test_trtllm_aiconfigurator_single_model(self, llm_args):
        # Test that profile_sla works with the model & backend in the llm_args fixture.
        await run_profile(llm_args)

    @pytest.mark.parallel
    @pytest.mark.asyncio
    @pytest.mark.gpu_1
    @pytest.mark.integration
    @pytest.mark.nightly
    @pytest.mark.parametrize(
        "backend, aic_backend_version",
        [
            ("trtllm", None),
            ("trtllm", "0.20.0"),
            ("trtllm", "1.0.0rc3"),
            ("vllm", None),
            ("vllm", "0.11.0"),
            ("sglang", None),
            ("sglang", "0.5.1.post1"),
        ],
    )
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "Qwen/Qwen3-32B",
            "meta-llama/Llama-3.1-405B",
        ],
    )
    async def test_aiconfigurator_dense_models(
        self, llm_args, hf_model_id, backend, aic_backend_version
    ):
        # Test that profile_sla works with a variety of backend versions and model names.
        llm_args.aic_hf_id = hf_model_id
        llm_args.backend = backend
        llm_args.aic_backend_version = aic_backend_version
        await run_profile(llm_args)
