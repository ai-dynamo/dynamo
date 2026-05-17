# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sanity for the prod-matching FP8 4P:2D:2F template (mirrors the
# 2026-05-03 disagg cascade deployment). 10-request burst at c=2, ISL=128,
# OSL=32 — verifies all pods Ready, FP8 GEMM kernels initialize,
# enable-expert-parallel works under TP=2, NIXL path stable.

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
async def test_sanity_vllm_qwen3_30b_prod(runtime_env):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_4p2d_2f_prod.yaml"
    )
    served_model = spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=spec,
        events=[
            # FP8 GEMM kernel JIT + expert-parallel init can take a while
            # on first boot; give the model-ready poll a wide window.
            WaitForModelReady(timeout=1800),
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
        test_name="test_sanity_vllm_qwen3_30b_prod",
        runtime_env=runtime_env,
    )
