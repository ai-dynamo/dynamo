# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sanity for the 4P:2D:2F + gpu-memory-utilization=0.5 (A100-40GB emulation)
# variant. Same pass criteria as test_sanity_vllm_qwen3_30b.py: all pods
# Ready, NIXL path up, frontend serves a small burst with zero errors.
# Purpose-built to flush out boot-time OOMs from the tighter memory cap
# before kicking off the capacity probe.

import pytest

from tests.fault_tolerance.deploy.checks import LoadCompleted, MinRequests, ZeroErrors
from tests.fault_tolerance.deploy.events import (
    StartLoad,
    WaitForLoadCompletion,
    WaitForModelReady,
)
from tests.fault_tolerance.deploy.reports import FaultToleranceReport
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_sanity_vllm_qwen3_30b_40gb(runtime_env):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_4p2d_2f_40gb.yaml"
    )
    served_model = spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=1500),
            StartLoad(
                load_config=LoadConfig(
                    model_name=served_model,
                    tokenizer=served_model,
                    request_count=10,
                    concurrency=2,
                    input_tokens_mean=128,
                    output_tokens_mean=32,
                )
            ),
            WaitForLoadCompletion(),
        ],
        checks=[
            LoadCompleted(),
            MinRequests(min_count=10),
            ZeroErrors(),
        ],
        reports=[FaultToleranceReport()],
        test_name="test_sanity_vllm_qwen3_30b_40gb",
        runtime_env=runtime_env,
    )
