# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Load sweep on the prod-FP8 unit deployment. One "unit" is
# 1 FE + 2 PF (TP=2) + 1 DE (TP=2). Scale with --units N.
#
# Two phases in one DGD lifecycle:
#   1. Concurrency sweep (closed-loop, aiperf --concurrency):
#        per-unit:  [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
#        actual at N=k: each value × k
#   2. Rate sweep (open-loop, aiperf --request-rate, req/s):
#        per-unit:  [1, 2, 4, 8, 16, 32, 50, 64]
#        actual at N=k: each value × k
#
# Per-rung artifacts land in:
#   test_outputs/<test_name>/load/load-<c|r><N>-<hash>/
#       profile_export_aiperf.{json,jsonl,csv}
#       server_metrics_export.{json,jsonl,csv}
#       aiperf.log
# After successful local extract the framework clears the same sub-path
# on the shared `framework-logs` PVC; the test output dir on disk is
# preserved indefinitely.
#
# Reports emitted at end of run:
#   FaultToleranceReport      — aiperf request_latency pre/post per fault
#   ErrorBreakdownReport      — aiperf error types per load
#   PerWorkerLatencyReport    — per-worker TTFT/queue/decode/NIXL p99 per rung
#   GpuMemoryReport           — per-GPU max DCGM_FI_DEV_FB_USED per rung
#                                (enforces 40 GB envelope vs --gpu-memory-utilization=0.45)
#   ErrorTrackingReport       — aiperf + vLLM error counters + pod restart count

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

# Phase 1: closed-loop concurrency sweep, per-unit (scaled × units at runtime).
_CONCURRENCY_RUNGS_PER_UNIT = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
# Phase 2: open-loop request-rate sweep, per-unit (scaled × units at runtime).
_RATE_RUNGS_PER_UNIT = [1, 2, 4, 8, 16, 32, 50, 64]

_RUNG_DURATION_S = 180
# Fixed-size synthetic inputs/outputs (stddev=0) for deterministic per-rung
# load. With variance, low-concurrency rungs see only a handful of requests
# whose token-count noise dominates the latency measurement; fixing the
# sizes makes the only varying signal across concurrency / rate sweeps the
# scheduling side of the engine.
_ISL_MEAN = 7000
_ISL_STDDEV = 0
_OSL_MEAN = 100
_OSL_STDDEV = 0


def _scale_to_units(spec: DeploymentSpec, units: int) -> None:
    """Multiply each service's replica count by `units`."""
    assert units >= 1, f"--units must be >= 1, got {units}"
    for service_name in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[service_name].replicas = spec[service_name].replicas * units


def _concurrency_load_config(
    model: str,
    concurrency: int,
    request_timeout_seconds: float,
    goodput: list[str] | None,
) -> LoadConfig:
    return LoadConfig(
        model_name=model,
        tokenizer=model,
        input_tokens_mean=_ISL_MEAN,
        input_tokens_stddev=_ISL_STDDEV,
        output_tokens_mean=_OSL_MEAN,
        output_tokens_stddev=_OSL_STDDEV,
        concurrency=concurrency,
        duration_minutes=_RUNG_DURATION_S / 60.0,
        request_timeout_seconds=request_timeout_seconds,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
        # Force per-request fresh connection so the k8s Service ClusterIP
        # round-robins across all FE pods (per-connection iptables/IPVS).
        # Verified after the run via dynamo_frontend_requests_total
        # per-pod distribution.
        connection_reuse_strategy="never",
        goodput=goodput,
    )


def _rate_load_config(
    model: str,
    total_rate: int,
    request_timeout_seconds: float,
    goodput: list[str] | None,
) -> LoadConfig:
    return LoadConfig(
        model_name=model,
        tokenizer=model,
        input_tokens_mean=_ISL_MEAN,
        input_tokens_stddev=_ISL_STDDEV,
        output_tokens_mean=_OSL_MEAN,
        output_tokens_stddev=_OSL_STDDEV,
        # Open-loop — aiperf still needs a concurrency cap to bound
        # in-flight if the server collapses. Set to a value well above
        # what 64 req/s × 30 s e2e target would imply, so the cap never
        # binds during healthy operation.
        concurrency=16384,
        request_rate=float(total_rate),
        duration_minutes=_RUNG_DURATION_S / 60.0,
        request_timeout_seconds=request_timeout_seconds,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
        connection_reuse_strategy="never",
        goodput=goodput,
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_load_sweep_qwen3_30b_prod_unit(
    runtime_env, units, request, request_timeout_seconds, goodput_slos
):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_unit_prod.yaml"
    )
    _scale_to_units(spec, units)
    served_model = spec["VllmDecodeWorker"].model

    events = [WaitForModelReady(timeout=1800)]

    # Phase 1: closed-loop concurrency rungs.
    for per_unit in _CONCURRENCY_RUNGS_PER_UNIT:
        c = per_unit * units
        events += [
            StartLoad(
                load_config=_concurrency_load_config(
                    served_model, c, request_timeout_seconds, goodput_slos
                ),
                name=f"c{c}",
            ),
            WaitForLoadCompletion(name=f"c{c}"),
        ]

    # Phase 2: open-loop request-rate rungs.
    for per_unit in _RATE_RUNGS_PER_UNIT:
        r = per_unit * units
        events += [
            StartLoad(
                load_config=_rate_load_config(
                    served_model, r, request_timeout_seconds, goodput_slos
                ),
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
            GpuMemoryReport(max_gb_per_gpu=40.0),
            ErrorTrackingReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
