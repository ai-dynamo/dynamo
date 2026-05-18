# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Capacity probe — push the 4P:2D:2F deployment until it falls over.
#
# Goal: identify X*, the concurrency level where the system breaks (not just
# where queueing onsets). Run consecutive `StartLoad` rungs at increasing
# concurrency, sustained ISL=N(7000, σ=600) / OSL=N(100, σ=40). Each rung is
# its own clean event boundary in the metrics; spike_view buckets them.
#
# Watch (in Grafana / TUI / spike_view post-run): which signal moves FIRST
# when load crosses the breakpoint —
#   A: prefill workers' vllm:num_requests_waiting (or dynamo_frontend_queued_requests)
#   B: decode workers' vllm:num_requests_waiting
#   C: vllm:nixl_post_time_seconds p99 / upper-bucket migration
#   D: decode worker pod transitions to Failed/CrashLoopBackOff
#
# This is the natural-stress version of the cascade — no injected fault.

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

# Concurrency rungs. Aggressive escalation past presumed cascade point.
# Each rung runs `_RUNG_DURATION_S` seconds before stepping up.
# Capacity sweep: progressive degradation. c32 already over-saturates the
# 4P:2D:2F deployment at ISL=7000/OSL=100 (NIXL p99 4.8s, KV usage 0.82,
# RTT p99 62s observed in earlier run). Lower rungs find clean queueing-
# onset; c48/c64 push it firmly into the bad state to show what the
# cascade looks like before any fault is injected.
_RUNGS = [8, 16, 24, 32, 48, 64]
_RUNG_DURATION_S = 180  # 3 min/rung → ~18 min total at full sweep
_ISL_MEAN = 7000
_ISL_STDDEV = 600
_OSL_MEAN = 100
_OSL_STDDEV = 40


def _load_config(model: str, concurrency: int) -> LoadConfig:
    return LoadConfig(
        model_name=model,
        tokenizer=model,
        input_tokens_mean=_ISL_MEAN,
        input_tokens_stddev=_ISL_STDDEV,
        output_tokens_mean=_OSL_MEAN,
        output_tokens_stddev=_OSL_STDDEV,
        concurrency=concurrency,
        duration_minutes=_RUNG_DURATION_S / 60.0,
        request_timeout_seconds=300,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_capacity_probe_qwen3_30b(runtime_env):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_4p2d_2f.yaml"
    )
    served_model = spec["VllmDecodeWorker"].model

    # Build chained StartLoad / WaitForLoadCompletion events. Each rung is
    # back-to-back; the StopLoad between them lets the framework cleanly
    # finalize one ManagedLoad before starting the next.
    events = [WaitForModelReady(timeout=1500)]
    for concurrency in _RUNGS:
        events += [
            StartLoad(
                load_config=_load_config(served_model, concurrency),
                name=f"c{concurrency}",
            ),
            WaitForLoadCompletion(name=f"c{concurrency}"),
        ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        # Don't hard-fail on errors — we WANT to see the deployment fall
        # over and observe the cascade signature. Errors during higher
        # rungs are the signal, not a test failure.
        checks=[LoadCompleted(name=f"c{_RUNGS[0]}")],
        reports=[FaultToleranceReport()],
        test_name="test_capacity_probe_qwen3_30b",
        runtime_env=runtime_env,
    )
