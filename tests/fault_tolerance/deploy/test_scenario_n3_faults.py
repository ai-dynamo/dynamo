# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Failure-scenario sweep against the N=3 prod-mirror DGD held at near-saturation
# load (per-unit c6 = total c18, just under the c8 per-unit knee from the
# constrained sweep where N=3 hit 2.44 RPS at 32s p99).
#
# Each scenario:
#   1. Wait for model ready.
#   2. Start sustained load (concurrency=18) for ~10 min.
#   3. Wait 180s for steady-state.
#   4. Inject fault (targets pod-0 of the service; rank_index=0 for rank ops).
#   5. Wait 300s to observe cascade + any recovery.
#   6. StopLoad → WaitForLoadCompletion → reports.
#
# Process targeting (vLLM):
#   - rank_*    targets ``VLLM::Worker``     (one rank in one pod)
#   - engine_*  targets ``VLLM::EngineCore`` (the coordinator in one pod)
#   - pod_delete deletes the whole pod (k8s reschedules)
#
# Scenarios (one parametrize-id each → its own test_outputs dir):
#   decode_rank_stall / decode_rank_kill / decode_engine_stall /
#   decode_engine_kill / prefill_rank_stall / prefill_rank_kill /
#   prefill_engine_stall / prefill_engine_kill / decode_pod_delete

import pytest

from tests.fault_tolerance.deploy.backend_processes import VLLM
from tests.fault_tolerance.deploy.checks import LoadCompleted, WorkerPanics
from tests.fault_tolerance.deploy.events import (
    ALL,
    RANDOM,
    DeletePod,
    RstInjection,
    StallProcess,
    StartLoad,
    StopLoad,
    TerminateProcess,
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


_RUN_DURATION_MINUTES = 10.0  # 600s: 180 warmup + 30 inject + 300 observe + margin
_ISL_MEAN = 7000
_OSL_MEAN = 100
_STALL_DURATION_S = 30.0


def _parse_selection(value, default=0):
    """Parse a pod- or rank-selection token (from --fault-pod / --fault-rank
    or DYN_TEST_FAULT_POD / DYN_TEST_FAULT_RANK):

        "0"       → 0          (pin to a specific index)
        "0,1"     → [0, 1]     (list of explicit indices)
        "random"  → RANDOM     (one random choice per run)
        "all"     → ALL        (hit every matching item)
    """
    if value is None:
        return default
    v = str(value).strip().lower()
    if v == "random":
        return RANDOM
    if v == "all":
        return ALL
    if "," in v:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        try:
            return [int(p) for p in parts]
        except ValueError:
            return default
    try:
        return int(v)
    except ValueError:
        return default


def _pod_arg(value):
    """Convert a parsed pod selection into the ``pod_indices`` argument
    shape: sentinel string for RANDOM / ALL, list[int] for explicit pods.
    """
    if value in (RANDOM, ALL):
        return value
    if isinstance(value, list):
        return value
    return [value]


def _load_config(
    model: str,
    concurrency: int,
    request_timeout_seconds: float,
    goodput: list[str] | None,
) -> LoadConfig:
    return LoadConfig(
        model_name=model,
        tokenizer=model,
        input_tokens_mean=_ISL_MEAN,
        input_tokens_stddev=0,
        output_tokens_mean=_OSL_MEAN,
        output_tokens_stddev=0,
        concurrency=concurrency,
        duration_minutes=_RUN_DURATION_MINUTES,
        request_timeout_seconds=request_timeout_seconds,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
        connection_reuse_strategy="never",
        goodput=goodput,
    )


def _stall(service, process_name, pod_arg, rank_index, name):
    return StallProcess(
        services=[service],
        process_name=process_name,
        pod_indices=pod_arg,
        rank_index=rank_index,
        duration=_STALL_DURATION_S,
        name=name,
    )


def _kill(service, process_name, pod_arg, rank_index, name):
    return TerminateProcess(
        services=[service],
        process_name=process_name,
        pod_indices=pod_arg,
        rank_index=rank_index,
        signal="SIGKILL",
        name=name,
    )


# Each scenario factory takes the resolved pod_arg (from --fault-pod) and
# rank_arg (from --fault-rank) and returns the list of events to inject
# between warmup and observation. Engine-level scenarios ignore rank_arg
# (there is exactly one EngineCore per pod) — they pass rank_index=None
# so the events pick the single match.
_SCENARIOS = {
    "decode_rank_stall":   lambda pod, rank: [_stall("VllmDecodeWorker",  VLLM.worker, pod, rank, "decode_rank_stall")],
    "decode_rank_kill":    lambda pod, rank: [_kill ("VllmDecodeWorker",  VLLM.worker, pod, rank, "decode_rank_kill")],
    "prefill_rank_stall":  lambda pod, rank: [_stall("VllmPrefillWorker", VLLM.worker, pod, rank, "prefill_rank_stall")],
    "prefill_rank_kill":   lambda pod, rank: [_kill ("VllmPrefillWorker", VLLM.worker, pod, rank, "prefill_rank_kill")],
    "decode_engine_stall":  lambda pod, rank: [_stall("VllmDecodeWorker",  VLLM.engine_core, pod, None, "decode_engine_stall")],
    "decode_engine_kill":   lambda pod, rank: [_kill ("VllmDecodeWorker",  VLLM.engine_core, pod, None, "decode_engine_kill")],
    "prefill_engine_stall": lambda pod, rank: [_stall("VllmPrefillWorker", VLLM.engine_core, pod, None, "prefill_engine_stall")],
    "prefill_engine_kill":  lambda pod, rank: [_kill ("VllmPrefillWorker", VLLM.engine_core, pod, None, "prefill_engine_kill")],
    "decode_pod_delete":   lambda pod, rank: [DeletePod(services=["VllmDecodeWorker"], pod_indices=pod, force=True, name="decode_pod_delete")],
    # Force TCP RSTs into the target pod for 30s — repro for PR #8254.
    "decode_rst_inject":   lambda pod, rank: [RstInjection(service="VllmDecodeWorker",  pod_indices=pod, duration=_STALL_DURATION_S, name="decode_rst_inject")],
    "prefill_rst_inject":  lambda pod, rank: [RstInjection(service="VllmPrefillWorker", pod_indices=pod, duration=_STALL_DURATION_S, name="prefill_rst_inject")],
}


def _scale_to_units(spec: DeploymentSpec, units: int) -> None:
    assert units >= 1
    for service_name in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[service_name].replicas = spec[service_name].replicas * units


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("scenario_id", list(_SCENARIOS.keys()))
async def test_n3_fault_scenario(
    runtime_env,
    scenario_id,
    request,
    fault_pod,
    fault_rank,
    fault_concurrency,
    request_timeout_seconds,
    goodput_slos,
):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_unit_prod.yaml"
    )
    _scale_to_units(spec, units=3)
    served_model = spec["VllmDecodeWorker"].model

    pod_arg = _pod_arg(_parse_selection(fault_pod))
    rank_arg = _parse_selection(fault_rank)
    fault_events = _SCENARIOS[scenario_id](pod_arg, rank_arg)

    events = [
        WaitForModelReady(timeout=1800),
        StartLoad(
            load_config=_load_config(
                served_model,
                fault_concurrency,
                request_timeout_seconds,
                goodput_slos,
            ),
            name="load",
        ),
        Wait(duration=180),  # warmup / steady-state
        *fault_events,
        Wait(duration=300),  # observe cascade
        WaitForLoadCompletion(name="load"),
    ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=[
            LoadCompleted(name="load"),
            # No Rust panics anywhere — Frontend or workers. The first
            # signal that the PR #8254 panic path got hit again is a
            # ``thread '...' panicked at tcp/...`` line in the logs.
            WorkerPanics(
                services=["Frontend", "VllmPrefillWorker", "VllmDecodeWorker"],
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
