# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stalls every decode worker under sustained load to probe whether the
# frontend's queued-request buffer grows unboundedly (the "Frontend
# OOMKill" pattern documented in the 2026-05-04 disagg-cascade
# follow-on report). Liveness signature:
# `containerStatuses[0].lastState.terminated.reason == "OOMKilled"`
# on a Frontend pod within the observation window.

import pytest

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
from tests.fault_tolerance.deploy.backend_processes import VLLM
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
async def test_frontend_oom_under_stalled_decode(runtime_env, request):
    spec = DeploymentSpec(_TEMPLATE)
    _scale_to_units(spec, units=3)
    served_model = spec["VllmDecodeWorker"].model

    cfg = LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        input_tokens_mean=7000, input_tokens_stddev=0,
        output_tokens_mean=100, output_tokens_stddev=0,
        concurrency=24,
        duration_minutes=15.0,  # 2 min warmup + 10 min stall + 3 min observation
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
            StallProcess(
                services=["VllmDecodeWorker"],
                process_name=VLLM.engine_core,
                pod_indices=ALL,
                duration=600.0,
                name="all_decode_engine_stall",
            ),
            WaitForLoadCompletion(name="load"),
        ],
        checks=[
            LoadCompleted(name="load"),
            # Detect OOMKill via restart-count delta + lastReason in logs.
            # When this triggers, the fault_verification.txt entry shows
            # `lastReason=OOMKilled` on the affected FE pod.
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
