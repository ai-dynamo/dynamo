# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test suite for profile_sla dry-run functionality.

This test ensures that the profile_sla script can successfully run in dry-run mode
for both vllm and sglang backends with their respective disagg.yaml configurations.
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
            backend = "vllm"
            config = "components/backends/vllm/deploy/disagg.yaml"
            output_dir = "/tmp/test_profiling_results"
            namespace = "test-namespace"
            min_num_gpus_per_engine = 1
            max_num_gpus_per_engine = 8
            skip_existing_results = False
            force_rerun = False
            isl = 3000
            osl = 500
            ttft = 50
            itl = 10
            max_context_length = 16384
            prefill_interpolation_granularity = 16
            decode_interpolation_granularity = 6
            service_name = ""
            dry_run = True

        return Args()

    @pytest.fixture
    def sglang_args(self):
        """Create arguments for sglang backend dry-run test."""

        class Args:
            backend = "sglang"
            config = "components/backends/sglang/deploy/disagg.yaml"
            output_dir = "/tmp/test_profiling_results"
            namespace = "test-namespace"
            min_num_gpus_per_engine = 1
            max_num_gpus_per_engine = 8
            skip_existing_results = False
            force_rerun = False
            isl = 3000
            osl = 500
            ttft = 50
            itl = 10
            max_context_length = 16384
            prefill_interpolation_granularity = 16
            decode_interpolation_granularity = 6
            service_name = ""
            dry_run = True

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
