# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# C1 — Cascade repro: panic flood -> degraded tokio runtime ->
# /live handler latency grows past liveness probe timeout ->
# kubelet SIGTERM -> pod restart -> repeat.
#
# Uses the mocker engine so we can scale FE/decode replicas without
# GPUs and iterate fast. Same dynamo TCP request plane as vLLM workers
# (panic site lives in dynamo_runtime, not engine code).
#
# Knobs (env vars, all optional):
#   CASCADE_CONCURRENCY            (default 100)
#   CASCADE_FRONTEND_REPLICAS      (default 1)
#   CASCADE_DECODE_REPLICAS        (default 1)
#   CASCADE_THREADS_FRONTEND       (default unset -> num_cores)
#   CASCADE_THREADS_DECODE         (default unset -> num_cores)
#   CASCADE_FRONTEND_CPU_LIMIT     (default unset; e.g. "500m")
#   CASCADE_DECODE_CPU_LIMIT       (default unset; e.g. "500m")
#   CASCADE_LIVE_PERIOD_S          (default 10  — operator default for FE; 5 for worker)
#   CASCADE_LIVE_TIMEOUT_S         (default 1   — operator default for FE)
#   CASCADE_LIVE_FAILURE_THRESHOLD (default 3   — operator default for FE)
#   CASCADE_RST_BURSTS             (default 3)
#   CASCADE_RST_DURATION_S         (default 60)
#   CASCADE_RST_GAP_S              (default 90)
#   CASCADE_RST_TARGET             (default "decode"; "Frontend" to flip)
#   CASCADE_LOAD_MINUTES           (default 12)
#
# Image: vllm-runtime:1.0.1 (bug-present). For the regression gate,
# pass a post-#8254 main-built image via --image.

import os

import pytest

from tests.fault_tolerance.deploy.checks import RestartCountIncreased
# Note: WorkerPanics and ServiceLogPatternRate currently break in post-teardown
# phase because they call ctx.deployment.collect_service_logs() after
# scenario.py sets ctx.deployment = None. Until that's fixed, we verify
# panics by scanning extracted PVC logs at test_outputs/<test>/<svc>/*.log.
# RestartCountIncreased reads ctx.pod_restart_state snapshot and works.
from tests.fault_tolerance.deploy.events import (
    RstInjection,
    StartLoad,
    Wait,
    WaitForLoadCompletion,
    WaitForModelReady,
)
from tests.fault_tolerance.deploy.reports import (
    ErrorBreakdownReport,
    ErrorTrackingReport,
    FaultToleranceReport,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_str(name: str, default):
    raw = os.environ.get(name)
    return raw if raw else default


def _apply_liveness_override(service, period_s: int, timeout_s: int, failure_threshold: int, port_name: str) -> None:
    """Inject a livenessProbe override into the DGD service spec.

    The operator merges DGD-provided probes entirely (no partial merge),
    so we provide the full Probe shape with our knob values. Path "/live"
    matches the operator defaults for both Frontend and worker components.
    """
    service._spec["livenessProbe"] = {
        "httpGet": {"path": "/live", "port": port_name},
        "periodSeconds": period_s,
        "timeoutSeconds": timeout_s,
        "failureThreshold": failure_threshold,
    }


def _apply_cpu_limit(service, cpu_limit: str) -> None:
    """Set a CPU resources.limits on this service (e.g. '500m')."""
    service._spec.setdefault("resources", {}).setdefault("limits", {})["cpu"] = cpu_limit
    # Mirror to requests so the pod schedules and CFS quota actually applies.
    service._spec["resources"].setdefault("requests", {})["cpu"] = cpu_limit


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_cascade_repro_mocker(runtime_env, request):
    """Cascade repro via mocker: panic flood -> /live degradation -> SIGTERM."""

    spec = DeploymentSpec.from_backend("mocker", "agg")

    # Drop --planner-profile-data: the template hard-codes a newer source
    # layout path not present in vllm-runtime:1.0.1. The arg is optional in
    # the 1.0.1 mocker (returns no profile, serves with built-in defaults).
    _decode_args = spec["decode"]._get_args()
    if "--planner-profile-data" in _decode_args:
        idx = _decode_args.index("--planner-profile-data")
        del _decode_args[idx:idx + 2]
    spec["decode"].set_arg("--startup-time", "5")

    # --- replicas ---
    spec["Frontend"].replicas = _env_int("CASCADE_FRONTEND_REPLICAS", 1)
    spec["decode"].replicas = _env_int("CASCADE_DECODE_REPLICAS", 1)

    # --- tokio thread caps ---
    threads_fe = os.environ.get("CASCADE_THREADS_FRONTEND")
    if threads_fe:
        spec["Frontend"].set_env_var("DYN_RUNTIME_NUM_WORKER_THREADS", threads_fe)
    threads_decode = os.environ.get("CASCADE_THREADS_DECODE")
    if threads_decode:
        spec["decode"].set_env_var("DYN_RUNTIME_NUM_WORKER_THREADS", threads_decode)

    # --- cpu limits ---
    cpu_fe = os.environ.get("CASCADE_FRONTEND_CPU_LIMIT")
    if cpu_fe:
        _apply_cpu_limit(spec["Frontend"], cpu_fe)
    cpu_decode = os.environ.get("CASCADE_DECODE_CPU_LIMIT")
    if cpu_decode:
        _apply_cpu_limit(spec["decode"], cpu_decode)

    # --- liveness probe overrides ---
    live_period = _env_int("CASCADE_LIVE_PERIOD_S", 10)
    live_timeout = _env_int("CASCADE_LIVE_TIMEOUT_S", 1)
    live_failures = _env_int("CASCADE_LIVE_FAILURE_THRESHOLD", 3)
    _apply_liveness_override(
        spec["Frontend"], live_period, live_timeout, live_failures, port_name="http"
    )
    # Worker liveness uses the "system" port (operator default), not "http".
    _apply_liveness_override(
        spec["decode"], live_period, live_timeout, live_failures, port_name="system"
    )

    # --- load config ---
    served_model = spec["decode"].model
    concurrency = _env_int("CASCADE_CONCURRENCY", 100)
    load_minutes = _env_float("CASCADE_LOAD_MINUTES", 12.0)

    cfg = LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        input_tokens_mean=512, input_tokens_stddev=0,
        output_tokens_mean=2000, output_tokens_stddev=0,
        concurrency=concurrency,
        duration_minutes=load_minutes,
        request_timeout_seconds=120,
        streaming=True, ignore_eos=True, warmup_requests=0,
        connection_reuse_strategy="never",
    )

    # --- RST burst sequence ---
    rst_target = _env_str("CASCADE_RST_TARGET", "decode")
    n_bursts = _env_int("CASCADE_RST_BURSTS", 3)
    burst_dur = _env_float("CASCADE_RST_DURATION_S", 60.0)
    burst_gap = _env_float("CASCADE_RST_GAP_S", 90.0)

    events = [
        WaitForModelReady(timeout=600),
        StartLoad(load_config=cfg, name="load"),
        Wait(duration=120),  # warmup so streams are flowing
    ]
    for i in range(n_bursts):
        events.append(
            RstInjection(
                service=rst_target,
                pod_indices=[0],
                duration=burst_dur,
                name=f"rst_burst_{i}",
            )
        )
        if i < n_bursts - 1:
            events.append(Wait(duration=burst_gap))
    events.append(WaitForLoadCompletion(name="load"))

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=[
            # The cascade signal: did the panic flood drive a kubelet
            # liveness-probe-induced restart? Reads from ctx.pod_restart_state
            # snapshot taken before deployment teardown; works post-teardown.
            # Panic count itself is verified by scanning extracted PVC logs.
            RestartCountIncreased(
                services=["Frontend", "decode"],
                expect_min_increment=1,
                expect_zero=False,
            ),
        ],
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            # ErrorTrackingReport records per-pod last-terminated reason —
            # look there for "Killing" / probe-failure provenance vs panic
            # process-death.
            ErrorTrackingReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
