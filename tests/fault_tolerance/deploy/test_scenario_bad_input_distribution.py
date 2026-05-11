# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bad-input-distribution scenario.
#
# Real failure mode observed in the c16 capacity probe: vllm sampling
# rejected requests with `ValueError: min_tokens must be less than or
# equal to max_tokens=67, got 100`. Cause: ISL stddev pushed individual
# requests near `max_model_len` (8192). Per-request `max_tokens` budget
# = max_model_len − ISL = 67 for that sample, but client's `min_tokens`
# is 100 (from output_tokens_mean). vllm rejects → request fails →
# error counter climbs.
#
# This scenario deliberately reproduces that. Watch on the timeline /
# Grafana / TUI:
#   - vllm:request_failures_total (or *errors* in cascade_timeline)
#   - dynamo_frontend_request_errors_total
#   - vllm:num_requests_running stays low (errors short-circuit before
#     the engine schedules them)
#   - latency p99 may LOOK fine because errored requests don't count

import pytest

from tests.fault_tolerance.deploy.checks import LoadCompleted
from tests.fault_tolerance.deploy.events import (
    StartLoad,
    WaitForLoadCompletion,
    WaitForModelReady,
)
from tests.fault_tolerance.deploy.reports import FaultToleranceReport
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig

# ISL distribution that shoves the upper tail across the budget. With
# max_model_len=8192 and OSL min=100, requests with ISL > 8092 fail.
# A distribution centered at 8000 with stddev=300 will fail ~33%.
_ISL_MEAN = 8000
_ISL_STDDEV = 300
_OSL_MEAN = 100
_OSL_STDDEV = 0  # fixed, so min_tokens is unambiguous
_DURATION_S = 300
_CONCURRENCY = 16


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_scenario_bad_input_distribution(runtime_env):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_4p2d_2f.yaml"
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
                    input_tokens_mean=_ISL_MEAN,
                    input_tokens_stddev=_ISL_STDDEV,
                    output_tokens_mean=_OSL_MEAN,
                    output_tokens_stddev=_OSL_STDDEV,
                    concurrency=_CONCURRENCY,
                    duration_minutes=_DURATION_S / 60.0,
                    request_timeout_seconds=300,
                    streaming=True,
                    ignore_eos=True,
                    warmup_requests=0,
                    # Force rigid output length so vllm rejects requests
                    # whose ISL leaves <100 tokens of budget. Default
                    # framework min_tokens=1 would NOT trip this.
                    extra_inputs={"min_tokens": _OSL_MEAN},
                ),
                name="bad_input",
            ),
            WaitForLoadCompletion(name="bad_input"),
        ],
        # Don't gate on ZeroErrors — the *whole point* is errors. We
        # want LoadCompleted to confirm the load ran to its planned
        # duration; downstream analysis looks at the error metrics.
        checks=[LoadCompleted(name="bad_input")],
        reports=[FaultToleranceReport()],
        test_name="test_scenario_bad_input_distribution",
        runtime_env=runtime_env,
    )
