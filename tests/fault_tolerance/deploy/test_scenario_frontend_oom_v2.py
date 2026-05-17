# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Q3 v2 — high-RPS variant of the frontend-OOM test. The c=24 / 10-min
# stall variant in test_scenario_frontend_oom.py absorbed cleanly because
# admission rate was tiny relative to FE memory. v2 drives a much higher
# concurrency (c=128 default, override via --fault-concurrency) and
# stalls every decode pod for 25 min so the FE queued-request buffer
# can accumulate. RestartCount delta on any FE = OOMKill signature.

import pytest

from tests.fault_tolerance.deploy.backend_processes import VLLM
from tests.fault_tolerance.deploy.checks import (
    LoadCompleted,
    RestartCountIncreased,
)
from tests.fault_tolerance.deploy.events import (
    ALL,
    StallProcess,
    StartLoad,
    Wait,
    WaitForLoadCompletion,
    WaitForModelReady,
)
from tests.fault_tolerance.deploy.reports import (
    ErrorBreakdownReport,
    ErrorTrackingReport,
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


def _scale_to_units(spec, units):
    for service in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[service].replicas = spec[service].replicas * units


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_frontend_oom_v2_high_rps_long_stall(
    runtime_env, request, fault_concurrency
):
    spec = DeploymentSpec(_TEMPLATE)
    _scale_to_units(spec, units=3)
    served_model = spec["VllmDecodeWorker"].model

    # Use shorter ISL/OSL than full prod (1641/2) so admission rate is
    # high even with the in-cluster Service round-robin between FE pods.
    # Don't gate on goodput SLO — the goal is RSS growth, not SLO.
    cfg = LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        input_tokens_mean=1641, input_tokens_stddev=2800,
        output_tokens_mean=2, output_tokens_stddev=100,
        concurrency=fault_concurrency,
        # 2 min warmup + 25 min stall + 3 min observation
        duration_minutes=30.0,
        request_timeout_seconds=120,
        streaming=True, ignore_eos=True, warmup_requests=0,
        connection_reuse_strategy="never",
    )

    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=cfg, name="load"),
            Wait(duration=120),
            # Stall the EngineCore on every decode pod — incoming
            # requests pile up at the FE waiting for decode to come back.
            StallProcess(
                services=["VllmDecodeWorker"],
                process_name=VLLM.engine_core,
                pod_indices=ALL,
                duration=1500.0,  # 25 min
                name="all_decode_engine_stall_25min",
            ),
            WaitForLoadCompletion(name="load"),
        ],
        checks=[
            LoadCompleted(name="load"),
            RestartCountIncreased(
                services=["Frontend"],
                expect_min_increment=1,
                expect_zero=False,
            ),
        ],
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
            ErrorTrackingReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
