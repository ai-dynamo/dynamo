# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke test for v1.1.0 on dynamo-nebius-2 (H200). Verifies image pulls,
# model loads from PVC, NIXL initializes, basic completions work.
#
# H200 has 141 GB HBM per GPU. To approximate prod's A100-40GB envelope
# (gpu-memory-utilization 0.45 ≈ 18 GB), we use 0.13 here (~18 GB), then
# loosen for the head-to-head once we know baseline behavior.
#
# Unlike aks-dev, nebius nodes have ~955 GB ephemeral storage and don't
# need FE node-pinning or IPC_LOCK overrides.

import pytest

from tests.fault_tolerance.deploy.checks import (
    KvCacheUsagePeak,
    LoadApplied,
    LoadCompleted,
    RestartCountIncreased,
    WorkerPanics,
)
from tests.fault_tolerance.deploy.events import (
    StartLoad,
    WaitForLoadCompletion,
    WaitForModelReady,
)
from tests.fault_tolerance.deploy.reports import (
    ErrorBreakdownReport,
    FaultToleranceReport,
    PerWorkerLatencyReport,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


_TEMPLATE = (
    "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
    "disagg_qwen3_30b_unit_prod.yaml"
)

_V110_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0"


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_nebius_smoke_v110(runtime_env, request):
    spec = DeploymentSpec(_TEMPLATE)

    for svc in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].image = _V110_IMAGE

    # H200 141 GB HBM → 0.13 ≈ 18 GB usable per GPU, matching prod A100-40GB
    # at gpu-mem-util 0.45.
    for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].set_arg("--gpu-memory-utilization", "0.13")

    served_model = spec["VllmDecodeWorker"].model

    cfg = LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        input_tokens_mean=1600,
        input_tokens_stddev=400,
        output_tokens_mean=200,
        output_tokens_stddev=50,
        concurrency=24,
        duration_minutes=3.0,
        request_timeout_seconds=60,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
        connection_reuse_strategy="never",
    )

    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=2400),
            StartLoad(load_config=cfg, name="smoke"),
            WaitForLoadCompletion(name="smoke"),
        ],
        checks=[
            LoadApplied(load_name="smoke", min_requests=50),
            KvCacheUsagePeak(
                services=["VllmDecodeWorker"],
                threshold=0.10,
                within_seconds=600,
            ),
            RestartCountIncreased(
                services=["VllmDecodeWorker", "Frontend"],
                expect_min_increment=0,
            ),
            WorkerPanics(
                services=["VllmDecodeWorker", "VllmPrefillWorker", "Frontend"],
                acceptable=False,
            ),
            LoadCompleted(name="smoke"),
        ],
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
