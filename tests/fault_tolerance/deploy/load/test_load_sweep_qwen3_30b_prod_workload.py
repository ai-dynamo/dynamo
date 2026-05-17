# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Capacity sweep with the production ISL/OSL distribution (instead of
# our usual fixed 7000/100). Target: 40 RPS per unit at 10 s e2e p99 SLO.
# At N=3 that's 120 RPS total. The fixed-ISL synthetic shortcut runs
# ~56× below this number; this sweep determines whether prod-shaped
# load is in reach on H100 N=3.
#
# ISL: μ=1641, σ≈2800 (truncated; P99 ≈ 7K) — matches the upstream
#      disagg-cascade workload P50/P99 observed in field.
# OSL: μ=2, σ≈100 (truncated; P99 ≈ 207).
# Goodput SLO: e2e ≤ 10 s, TTFT ≤ 5 s.

import pytest

from tests.fault_tolerance.deploy.checks import LoadCompleted
from tests.fault_tolerance.deploy.events import (
    StartLoad,
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


# Finer-grained granularity around the expected goodput knee.
_CONCURRENCY_RUNGS_PER_UNIT = [1, 2, 3, 4, 6, 8, 10, 12, 16]
_RUNG_DURATION_S = 180

_ISL_MEAN = 1641; _ISL_STDDEV = 2800
_OSL_MEAN = 2;    _OSL_STDDEV = 100


def _scale_to_units(spec, units):
    for service in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[service].replicas = spec[service].replicas * units


def _cfg(model, concurrency, request_timeout_seconds, goodput):
    return LoadConfig(
        model_name=model, tokenizer=model,
        input_tokens_mean=_ISL_MEAN, input_tokens_stddev=_ISL_STDDEV,
        output_tokens_mean=_OSL_MEAN, output_tokens_stddev=_OSL_STDDEV,
        concurrency=concurrency,
        duration_minutes=_RUNG_DURATION_S / 60.0,
        request_timeout_seconds=request_timeout_seconds,
        streaming=True, ignore_eos=True, warmup_requests=0,
        connection_reuse_strategy="never",
        goodput=goodput,
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_load_sweep_qwen3_30b_prod_workload(
    runtime_env, units, request, request_timeout_seconds, goodput_slos
):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_unit_prod.yaml"
    )
    _scale_to_units(spec, units)
    served_model = spec["VllmDecodeWorker"].model

    events = [WaitForModelReady(timeout=1800)]
    for per_unit in _CONCURRENCY_RUNGS_PER_UNIT:
        c = per_unit * units
        events += [
            StartLoad(
                load_config=_cfg(served_model, c, request_timeout_seconds, goodput_slos),
                name=f"c{c}",
            ),
            WaitForLoadCompletion(name=f"c{c}"),
        ]

    first_rung = _CONCURRENCY_RUNGS_PER_UNIT[0] * units
    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=[LoadCompleted(name=f"c{first_rung}")],
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
