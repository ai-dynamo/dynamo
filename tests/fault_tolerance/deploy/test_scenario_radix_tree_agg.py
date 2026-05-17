# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reproduces the 2026-05-03 disagg-cascade pre-condition: kv-router
# radix_tree WARN storm on AGG-enum decode pods. Two parametrizations:
#   - "AGG"    : decode worker missing --disaggregation-mode flag
#                → DisaggregationMode.AGGREGATED → kv-events publisher
#                wired up → LocalKvIndexer consumes the worker's own
#                events → radix_tree.rs:341/431 fires at high rate.
#   - "DECODE" : decode worker has --disaggregation-mode decode
#                → publisher-skip gate fires at main.py:316
#                → no events flow to LocalKvIndexer → warnings silenced.
#
# 10 min sustained closed-loop load at c=24 (per-unit c=8 × N=3) so
# the worker generates enough kv-events to fan out the desync.

import pytest

from tests.fault_tolerance.deploy.checks import (
    LoadCompleted,
    ServiceLogPatternRate,
)
from tests.fault_tolerance.deploy.events import (
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


_AGG_TEMPLATE = (
    "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
    "disagg_qwen3_30b_unit_prod_AGG_decode.yaml"
)
_DECODE_TEMPLATE = (
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
@pytest.mark.parametrize("enum", ["AGG", "DECODE"])
async def test_radix_tree_warning_rate(runtime_env, request, enum):
    template = _AGG_TEMPLATE if enum == "AGG" else _DECODE_TEMPLATE
    spec = DeploymentSpec(template)
    _scale_to_units(spec, units=3)
    served_model = spec["VllmDecodeWorker"].model

    cfg = LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        input_tokens_mean=7000, input_tokens_stddev=0,
        output_tokens_mean=100, output_tokens_stddev=0,
        concurrency=24,
        duration_minutes=10.0,
        request_timeout_seconds=60,
        streaming=True, ignore_eos=True, warmup_requests=0,
        connection_reuse_strategy="never",
    )

    if enum == "AGG":
        # Production saw ~1,200/sec on a heavy-traffic prod pod. Floor here
        # is much more conservative (we have less concurrency / more pods).
        radix_check = ServiceLogPatternRate(
            services=["VllmDecodeWorker"],
            pattern=r"radix_tree\.rs:(341|431)",
            min_rate_per_sec=1.0,
        )
    else:
        radix_check = ServiceLogPatternRate(
            services=["VllmDecodeWorker"],
            pattern=r"radix_tree\.rs:(341|431)",
            expect_zero=True,
        )

    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=cfg, name="load"),
            WaitForLoadCompletion(name="load"),
        ],
        checks=[LoadCompleted(name="load"), radix_check],
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
            ErrorTrackingReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
