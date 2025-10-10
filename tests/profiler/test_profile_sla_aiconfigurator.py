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


class TestProfileSlaAiconfigurator:
    """Test class for profile_sla aiconfigurator functionality."""

    @pytest.fixture
    def trtllm_args(self):
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
                self.dry_run = False
                self.use_ai_configurator = True
                self.aic_system = "h200_sxm"
                self.aic_model_name = "QWEN3_32B"
                self.aic_backend = ""
                self.aic_backend_version = "0.20.0"
                self.num_gpus_per_node = 8
                self.deploy_after_profile = False

        return Args()

    @pytest.mark.pre_merge
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "missing_arg", ["aic_system", "aic_model_name", "aic_backend_version"]
    )
    async def test_aiconfigurator_missing_args(self, trtllm_args, missing_arg):
        # Check that validation error happens when a required arg is missing.
        setattr(trtllm_args, missing_arg, None)
        with pytest.raises(ValueError):
            await run_profile(trtllm_args)

    @pytest.mark.pre_merge
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "arg_name, bad_value",
        [
            # these values don't exist in the aiconfigurator database.
            ("aic_system", "fake_gpu_system"),
            ("aic_backend_version", "0.1.0"),
        ],
    )
    async def test_aiconfiguator_no_data(self, trtllm_args, arg_name, bad_value):
        # Check that an appropriate error is raised when the system/model/backend
        # is not found in the aiconfigurator database.
        setattr(trtllm_args, arg_name, bad_value)
        with pytest.raises(ValueError, match="Database not found"):
            await run_profile(trtllm_args)

    @pytest.mark.pre_merge
    @pytest.mark.asyncio
    async def test_trtllm_aiconfigurator_single_model(self, trtllm_args):
        # Test that profile_sla works with the model & backend in the trtllm_args fixture.
        await run_profile(trtllm_args)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "backend, aic_backend_version",
        [
            ("trtllm", "0.20.0"),
            ("trtllm", "1.0.0rc3"),
        ],
    )
    @pytest.mark.parametrize("model_name", ["QWEN3_32B", "GPT_7B", "LLAMA3.1_405B"])
    async def test_trtllm_aiconfigurator_many(
        self, trtllm_args, model_name, backend, aic_backend_version
    ):
        # Test that profile_sla works with a variety of backend versions and model names.
        trtllm_args.aic_model_name = model_name
        trtllm_args.backend = backend
        trtllm_args.aic_backend_version = aic_backend_version
        await run_profile(trtllm_args)
