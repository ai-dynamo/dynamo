# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Same rung schedule + load shape as test_load_sweep_qwen3_30b_prod_unit.py,
# but against the unconstrained template (no --gpu-memory-utilization override
# → vLLM default ≈ 0.9 → ~72 GB KV pool on H100-80GB instead of ~36 GB).
# No 40 GB cap in GpuMemoryReport because the GPU is intentionally allowed
# to consume up to ~72 GB.

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

_CONCURRENCY_RUNGS_PER_UNIT = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
_RATE_RUNGS_PER_UNIT = [1, 2, 4, 8, 16, 32, 50, 64]

_RUNG_DURATION_S = 180
_ISL_MEAN = 7000
_ISL_STDDEV = 0
_OSL_MEAN = 100
_OSL_STDDEV = 0


def _scale_to_units(spec: DeploymentSpec, units: int) -> None:
    assert units >= 1, f"--units must be >= 1, got {units}"
    for service_name in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[service_name].replicas = spec[service_name].replicas * units


def _concurrency_load_config(model: str, concurrency: int) -> LoadConfig:
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
        connection_reuse_strategy="never",
    )


def _rate_load_config(model: str, total_rate: int) -> LoadConfig:
    return LoadConfig(
        model_name=model,
        tokenizer=model,
        input_tokens_mean=_ISL_MEAN,
        input_tokens_stddev=_ISL_STDDEV,
        output_tokens_mean=_OSL_MEAN,
        output_tokens_stddev=_OSL_STDDEV,
        concurrency=16384,
        request_rate=float(total_rate),
        duration_minutes=_RUNG_DURATION_S / 60.0,
        request_timeout_seconds=600,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
        connection_reuse_strategy="never",
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_load_sweep_qwen3_30b_unconstrained_unit(runtime_env, units, request):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_unit_unconstrained.yaml"
    )
    _scale_to_units(spec, units)
    served_model = spec["VllmDecodeWorker"].model

    events = [WaitForModelReady(timeout=1800)]

    for per_unit in _CONCURRENCY_RUNGS_PER_UNIT:
        c = per_unit * units
        events += [
            StartLoad(
                load_config=_concurrency_load_config(served_model, c),
                name=f"c{c}",
            ),
            WaitForLoadCompletion(name=f"c{c}"),
        ]

    for per_unit in _RATE_RUNGS_PER_UNIT:
        r = per_unit * units
        events += [
            StartLoad(
                load_config=_rate_load_config(served_model, r),
                name=f"r{r}",
            ),
            WaitForLoadCompletion(name=f"r{r}"),
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
            # Unconstrained: vLLM default ~0.9 → up to ~72 GB; bump soft
            # threshold to 80 GB (H100 physical max) so the report only
            # flags actual OOM-territory usage.
            GpuMemoryReport(max_gb_per_gpu=80.0),
            ErrorTrackingReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
