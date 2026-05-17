# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase A — fault-primitive verification (V1..V8).
#
# Each V-test stands up a SHARED N=1 prod-mirror DGD (cluster wide:
# 1 FE / 2 PF (TP=2) / 1 DE (TP=2) = 6 GPUs, fits one 8×H100 node),
# drives a small open-loop background load so fault signatures show
# fast, then injects ONE fault and asserts:
#   - local effect (PID gone, container restartCount delta, pod
#     replaced, helper-pod stdout, etc.)
#   - downstream observable signature (request flow stops, partner
#     queue grows, panic line in worker log, etc.)
#
# Tests are parametrized so a single pytest invocation runs all V*'s
# on ONE deployment via --skip-service-restart between tests.
#
# Output: each V test produces fault_verification.txt and per-pod
# server_metrics_export.jsonl. After the run, a small offline driver
# builds fault_primitive_verification.md summarising pass/fail.

import pytest

from tests.fault_tolerance.deploy.backend_processes import VLLM
from tests.fault_tolerance.deploy.checks import (
    LoadCompleted,
    RestartCountIncreased,
    ServiceLogPatternRate,
)
from tests.fault_tolerance.deploy.events import (
    ALL,
    DeletePod,
    NetworkPartition,
    RstFromInsidePod,
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
    PerWorkerLatencyReport,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


_TEMPLATE = (
    "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
    "disagg_qwen3_30b_unit_prod.yaml"
)


def _load_config(model: str, *, request_rate: float = 20.0) -> LoadConfig:
    """Tiny open-loop load so fault effect visible within ~30 s.

    ISL=500 / OSL=20: each request completes in ~1 s, so per-second
    state churn is high and any throughput drop manifests immediately.
    """
    return LoadConfig(
        model_name=model,
        tokenizer=model,
        input_tokens_mean=500, input_tokens_stddev=0,
        output_tokens_mean=20, output_tokens_stddev=0,
        concurrency=4096,
        request_rate=request_rate,
        duration_minutes=4.0,   # 30 s warm + ~60 s fault + 90 s observe
        request_timeout_seconds=60,
        streaming=True, ignore_eos=True, warmup_requests=0,
        connection_reuse_strategy="never",
    )


def _common_reports():
    return [
        FaultToleranceReport(),
        ErrorBreakdownReport(),
        PerWorkerLatencyReport(),
        ErrorTrackingReport(),
    ]


# Each V-test is one async function so we can document the asserts inline
# and keep the failure mode explicit per primitive. They share the same
# DGD via --skip-service-restart.

@pytest.fixture
def n1_spec():
    spec = DeploymentSpec(_TEMPLATE)
    # N=1 — no scaling needed, template defaults are 1 FE / 2 PF / 1 DE.
    return spec


# ---------------------------------------------------------------------------
# V1 — Worker rank stall (SIGSTOP), parametrized by rpc_timeout_seconds.
#
# `rpc_timeout_seconds=None` (the "stall" arm): leave VLLM_EXECUTE_MODEL_TIMEOUT
# at its 300s default. The 60s SIGSTOP completes before the timeout fires, so
# the engine simply pauses + resumes. Verifies: SIGSTOP delivered (state=T),
# no restart, no `sample_tokens timed out` line.
#
# `rpc_timeout_seconds=20` (the "force engine death" arm): patch the decode
# worker env to drop the RPC timeout to 20s, then SIGSTOP for 30s — past the
# timeout. EngineCore raises `TimeoutError("RPC call to sample_tokens timed
# out.")`, `_send_engine_dead()` fires, container exits, kubelet restarts.
# Same path as the 2026-05-01 prefill rank wedge in prod.
#
# Code refs (vllm 0.16.0 in vllm-runtime:1.0.1):
#   - executor/multiproc_executor.py:282  sample_tokens(...)
#       → collective_rpc(timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
#       → mq.dequeue(timeout=...) raises TimeoutError(...)
#   - engine/core.py:1009  except Exception: _send_engine_dead(); raise
# ---------------------------------------------------------------------------
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "rpc_timeout_seconds,stall_duration",
    [
        pytest.param(None, 60.0, id="stall_no_timeout"),
        pytest.param(20, 30.0, id="stall_force_engine_death"),
    ],
)
async def test_V1_worker_rank_stall(
    runtime_env, request, n1_spec, rpc_timeout_seconds, stall_duration
):
    decode = n1_spec["VllmDecodeWorker"]
    if rpc_timeout_seconds is not None:
        # Shorten so the SIGSTOP outlives the RPC timeout → engine death.
        decode.set_env_var(
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", str(rpc_timeout_seconds)
        )
    served_model = decode.model
    cfg = _load_config(served_model)
    if rpc_timeout_seconds is not None:
        # Longer load duration so we observe the kubelet restart in-window.
        cfg.duration_minutes = 6.0

    expect_engine_death = (
        rpc_timeout_seconds is not None and stall_duration > rpc_timeout_seconds
    )

    checks = [LoadCompleted(name="load")]
    if expect_engine_death:
        checks += [
            RestartCountIncreased(
                services=["VllmDecodeWorker"], expect_min_increment=1,
            ),
            # Production log line — proves it was the vLLM RPC timeout path
            # that killed the engine, not the liveness probe.
            ServiceLogPatternRate(
                services=["VllmDecodeWorker"],
                pattern=r"RPC call to sample_tokens timed out",
                min_rate_per_sec=0.001,
            ),
        ]

    await run_scenario(
        deployment_spec=n1_spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=cfg, name="load"),
            Wait(duration=30),
            StallProcess(
                services=["VllmDecodeWorker"], process_name=VLLM.worker,
                pod_indices=[0], rank_index=0, duration=stall_duration,
                name=request.node.name,
            ),
            Wait(duration=180 if expect_engine_death else 60),
            WaitForLoadCompletion(name="load"),
        ],
        checks=checks,
        reports=_common_reports(),
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ---------------------------------------------------------------------------
# V2 — EngineCore stall
# ---------------------------------------------------------------------------
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_V2_engine_core_stall(runtime_env, request, n1_spec):
    served_model = n1_spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=n1_spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=_load_config(served_model), name="load"),
            Wait(duration=30),
            StallProcess(
                services=["VllmDecodeWorker"], process_name=VLLM.engine_core,
                pod_indices=[0], duration=60.0, name="V2_engine_stall",
            ),
            Wait(duration=60),
            WaitForLoadCompletion(name="load"),
        ],
        checks=[LoadCompleted(name="load")],
        reports=_common_reports(),
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ---------------------------------------------------------------------------
# V3 — Worker rank kill (SIGKILL; framework auto-demotes to SIGINT only
# when pid==1, which Worker is not, so SIGKILL is delivered for real)
# ---------------------------------------------------------------------------
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_V3_worker_rank_kill(runtime_env, request, n1_spec):
    served_model = n1_spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=n1_spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=_load_config(served_model), name="load"),
            Wait(duration=30),
            TerminateProcess(
                services=["VllmDecodeWorker"], process_name=VLLM.worker,
                pod_indices=[0], rank_index=0, signal="SIGKILL",
                name="V3_worker_kill",
            ),
            Wait(duration=60),
            WaitForLoadCompletion(name="load"),
        ],
        checks=[
            LoadCompleted(name="load"),
            RestartCountIncreased(
                services=["VllmDecodeWorker"], expect_min_increment=1,
            ),
        ],
        reports=_common_reports(),
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ---------------------------------------------------------------------------
# V4 — EngineCore kill
# ---------------------------------------------------------------------------
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_V4_engine_core_kill(runtime_env, request, n1_spec):
    served_model = n1_spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=n1_spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=_load_config(served_model), name="load"),
            Wait(duration=30),
            TerminateProcess(
                services=["VllmDecodeWorker"], process_name=VLLM.engine_core,
                pod_indices=[0], signal="SIGKILL",
                name="V4_engine_kill",
            ),
            Wait(duration=60),
            WaitForLoadCompletion(name="load"),
        ],
        checks=[
            LoadCompleted(name="load"),
            RestartCountIncreased(
                services=["VllmDecodeWorker"], expect_min_increment=1,
            ),
        ],
        reports=_common_reports(),
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ---------------------------------------------------------------------------
# V5 — DeletePod
# ---------------------------------------------------------------------------
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_V5_decode_pod_delete(runtime_env, request, n1_spec):
    served_model = n1_spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=n1_spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=_load_config(served_model), name="load"),
            Wait(duration=30),
            DeletePod(
                services=["VllmDecodeWorker"], pod_indices=[0],
                force=True, name="V5_pod_delete",
            ),
            Wait(duration=120),  # replacement appearance can take 60s
            WaitForLoadCompletion(name="load"),
        ],
        checks=[LoadCompleted(name="load")],
        reports=_common_reports(),
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ---------------------------------------------------------------------------
# V6 — RstInjection (iptables FORWARD REJECT)
# Documented as likely-failing on AWS VPC CNI; we run it to confirm the
# negative and avoid relying on it for stress tests.
# ---------------------------------------------------------------------------
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_V6_rst_injection_iptables(runtime_env, request, n1_spec):
    served_model = n1_spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=n1_spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=_load_config(served_model), name="load"),
            Wait(duration=30),
            RstInjection(
                service="VllmDecodeWorker", pod_indices=[0], duration=30.0,
                name="V6_rst_iptables",
            ),
            Wait(duration=60),
            WaitForLoadCompletion(name="load"),
        ],
        checks=[
            LoadCompleted(name="load"),
            # If iptables REJECT is actually on the packet path, we'd see
            # ECONNRESET / connection error lines in the worker log. We
            # don't assert that yet — V6 is a probe to verify whether
            # iptables-FORWARD-REJECT works on this CNI.
        ],
        reports=_common_reports(),
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ---------------------------------------------------------------------------
# V7 — RstFromInsidePod (SO_LINGER=0 partial-frame client; upstream-blessed
# shape from arm_b_rst.py).
#
# Target the FRONTEND, not the worker. The Frontend hosts the
# Dynamo TcpStreamServer (lib/runtime/src/pipeline/network/tcp/server.rs) —
# workers connect OUT to it to push response streams. The panic at
# server.rs:584 `Some(Err(_)) => panic!("invalid message issued over
# socket; this should never happen")` fires when the framed reader on an
# accepted connection hits an I/O error (ECONNRESET from SO_LINGER=0).
# ---------------------------------------------------------------------------
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_V7_rst_from_inside_pod(runtime_env, request, n1_spec):
    served_model = n1_spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=n1_spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=_load_config(served_model), name="load"),
            Wait(duration=30),
            RstFromInsidePod(
                service="Frontend", pod_indices=[0],
                count=50, inter_delay=0.01,
                # target_port=None → auto-discover Dynamo TcpStreamServer
                # ports from /proc/net/tcp on the FE pod (excludes 9090
                # axum status, 8000 OpenAI HTTP, 5600 NIXL UCX).
                target_port=None,
                name="V7_rst_so_linger",
            ),
            Wait(duration=60),
            WaitForLoadCompletion(name="load"),
        ],
        checks=[
            LoadCompleted(name="load"),
            # PR #8254 is in vllm-runtime:1.0.1 — the panic site at
            # server.rs:584 is now a WARN at server.rs:374. Verify the
            # WARN fires (proves RST reached the framed reader) AND
            # that no panic happened (proves the regression is fixed).
            ServiceLogPatternRate(
                services=["Frontend"],
                pattern=(
                    r"failed to handle tcp connection: I/O error: "
                    r"Connection reset by peer \(os error 104\)"
                ),
                min_rate_per_sec=1.0,
            ),
            ServiceLogPatternRate(
                services=["Frontend"],
                pattern=r"panicked at .*tcp/(client|server)\.rs",
                min_rate_per_sec=0.0,
                expect_zero=True,
            ),
        ],
        reports=_common_reports(),
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ---------------------------------------------------------------------------
# V8 — NetworkPartition (existing event).
# ---------------------------------------------------------------------------
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_V8_network_partition(runtime_env, request, n1_spec):
    served_model = n1_spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=n1_spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=_load_config(served_model), name="load"),
            Wait(duration=30),
            NetworkPartition(
                source="Frontend", target="VllmDecodeWorker",
                duration=30.0, name="V8_netpart",
            ),
            Wait(duration=60),
            WaitForLoadCompletion(name="load"),
        ],
        checks=[LoadCompleted(name="load")],
        reports=_common_reports(),
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
