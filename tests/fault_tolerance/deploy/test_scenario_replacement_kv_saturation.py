# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reproduces the 2026-05-03 "replacement-pod immediate KV saturation"
# signature: kill a decode pod under steady load; the DGD operator
# schedules a replacement pod; the replacement reaches ≥ 90 %
# kv_cache_usage_perc within minutes because upstream prefill peers
# had a queue of pending NIXL transfers waiting for the dead decode
# to come back.

import pytest

from tests.fault_tolerance.deploy.checks import (
    KvCacheUsagePeak,
    LoadCompleted,
)
from tests.fault_tolerance.deploy.events import (
    DeletePod,
    StartLoad,
    Wait,
    WaitForLoadCompletion,
    WaitForModelReady,
)
from tests.fault_tolerance.deploy.reports import (
    ErrorBreakdownReport,
    ErrorTrackingReport,
    FaultToleranceReport,
    GpuMemoryReport,
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
async def test_replacement_kv_saturation(runtime_env, request):
    spec = DeploymentSpec(_TEMPLATE)
    _scale_to_units(spec, units=3)
    served_model = spec["VllmDecodeWorker"].model

    cfg = LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        input_tokens_mean=7000, input_tokens_stddev=0,
        output_tokens_mean=100, output_tokens_stddev=0,
        concurrency=24,  # below saturation knee, sustained
        duration_minutes=12.0,
        request_timeout_seconds=120,
        streaming=True, ignore_eos=True, warmup_requests=0,
        connection_reuse_strategy="never",
    )

    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=cfg, name="load"),
            Wait(duration=180),  # 3-min steady state
            DeletePod(
                services=["VllmDecodeWorker"],
                pod_indices=[0],
                force=True,
                name="delete_decode_0",
            ),
            WaitForLoadCompletion(name="load"),
        ],
        checks=[
            LoadCompleted(name="load"),
            KvCacheUsagePeak(
                services=["VllmDecodeWorker"],
                threshold=0.9,
                within_seconds=240.0,  # 4 minutes per prod observation
            ),
        ],
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
            GpuMemoryReport(max_gb_per_gpu=40.0),
            ErrorTrackingReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
