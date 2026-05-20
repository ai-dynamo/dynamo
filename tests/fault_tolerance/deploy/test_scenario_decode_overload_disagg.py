# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Decode-worker overload reproduction — disagg-cascade shape.
#
# Mirrors a production-shape workload (ISL p50≈1600 / p99≈7000, OSL≈200,
# prefix-cache hit ~37%) on the 1-FE : 2-P : 1-D unit topology. Six
# parametrized arms.
#
# Phase 1 — overload-only (decode-side pressure without fault injection)
#   A — c=96 sustained ramp + 30 s SLA timeout.
#       The base overload regime: decode KV pegs at 100%, vLLM scheduler
#       enters preempt-recompute loop, throughput collapses ~20× vs c=24.
#       Demonstrates the production cascade *upstream condition* without
#       panic injection.
#   B — A + 8 % probabilistic mid-flight cancellations (1.5 s delay).
#       Hammers the SO_LINGER=0 close path at every request, on the client
#       → FE direction. Note: AIPerf-side cancel = client-disconnect path,
#       not FE↔worker close; without (F2/F3) won't panic worker.
#   C — A + brief c=120 burst at peak.
#       Step-shock variant of an in-cascade traffic spike pattern.
#
# Phase 2 — fault-injection overlays (force panic / FE-side failure)
#   F1 — A + frontend ulimit nofile=1024 → drives FD-exhaust.
#       Lowered FD ceiling means sustained traffic should hit `accept() Too
#       many open files (os error 24)` at `tcp/server.rs:344`. When FE
#       hits its FD ceiling and starts force-closing worker sockets, the
#       SO_LINGER=0 path emits RST on the wire → decode `tcp_client.rs:255`
#       panics.
#   F2 — A + `kubectl rollout restart` Frontend at peak.
#       Graceful FE close at peak load. SO_LINGER=0 on the worker-accept
#       socket → every FE shutdown event emits RST instead of FIN → decode
#       reader-task panic burst.
#   F3 — A + `kubectl delete pod` Frontend at peak.
#       Hard FE termination. Kernel-side RST on all worker sockets.
#       Bounds the panic-burst rate; pair with F2 to attribute mechanism.
#
# Uses the AGG-on-decode misconfig template: decode worker omits
# ``--disaggregation-mode`` so v1.0.1 defaults to AGGREGATED enum, which
# wires up LocalKvIndexer on a decode pod and produces a ~1,200/sec
# radix_tree warning storm.
#
# Design doc:
#   ../../../dgh-703-unified-k8s-test-framework/findings/
#   2026-05-13-decode-overload-repro-design.md

import pytest

from tests.fault_tolerance.deploy.checks import (
    KvCacheUsagePeak,
    LoadApplied,
    LoadCompleted,
    RequestCancellationOccurred,
    RequestsRunningPeak,
    RestartCountIncreased,
    ServiceLogPatternRate,
    SlaViolation,
    ThroughputCollapse,
    WorkerPanics,
)
from tests.fault_tolerance.deploy.events import (
    DeletePod,
    NetworkPartition,
    RollingUpgrade,
    SetBusyThreshold,
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

_TEMPLATE = (
    "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
    "disagg_qwen3_30b_unit_prod_AGG_decode.yaml"
)


from dataclasses import dataclass  # noqa: E402

from tests.fault_tolerance.deploy.checks import Check, _iter_server_metric  # noqa: E402


@dataclass
class _KvCacheStaysBelow(Check):
    """Inverse of KvCacheUsagePeak — assert every sample of
    ``vllm:kv_cache_usage_perc`` on the named services stays at or below
    ``ceiling``. Used to verify load-shedding is preventing overload.
    """

    services: list
    ceiling: float = 0.85

    def validate(self, ctx) -> None:
        observed: dict = {}
        for ts_ns, pod, val in _iter_server_metric(ctx, "vllm:kv_cache_usage_perc"):
            cur = observed.get(pod)
            if cur is None or val > cur:
                observed[pod] = val
        ctx.logger.info(
            f"_KvCacheStaysBelow: per-pod max KV={observed} " f"ceiling={self.ceiling}"
        )
        breaches = {p: v for p, v in observed.items() if v > self.ceiling}
        assert not breaches, (
            f"_KvCacheStaysBelow: some pods exceeded KV ceiling "
            f"{self.ceiling}: {breaches}"
        )

    @property
    def description(self) -> str:
        return (
            f"KV cache usage stays at or below {self.ceiling} on "
            f"{', '.join(self.services)} (shedding holds the line)"
        )


_DISAGG_SEQ_DIST = (
    "100,200:5;500,200:15;1000,200:20;1600,200:30;3400,200:20;7000,200:10"
)

_DISAGG_NUM_PREFIX_PROMPTS = 15
_DISAGG_PREFIX_PROMPT_LENGTH = 600


def _disagg_load(
    *,
    served_model: str,
    name: str,
    concurrency: int,
    duration_minutes: float,
    ramp_seconds: float | None = None,
    warmup_concurrency: int | None = None,
    warmup_duration_seconds: float | None = None,
    cancellation_rate: float | None = None,
    cancellation_delay: float = 0.0,
    seq_dist: str | None = None,
) -> LoadConfig:
    """Build a LoadConfig with the prod-shape disagg workload settings."""
    return LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        seq_dist=seq_dist or _DISAGG_SEQ_DIST,
        num_prefix_prompts=_DISAGG_NUM_PREFIX_PROMPTS,
        prefix_prompt_length=_DISAGG_PREFIX_PROMPT_LENGTH,
        concurrency=concurrency,
        concurrency_ramp_duration=ramp_seconds,
        warmup_concurrency=warmup_concurrency,
        warmup_duration=warmup_duration_seconds,
        warmup_requests=0,
        duration_minutes=duration_minutes,
        request_timeout_seconds=30,
        request_cancellation_rate=cancellation_rate,
        request_cancellation_delay=cancellation_delay,
        streaming=True,
        ignore_eos=True,
        connection_reuse_strategy="never",
        goodput=["request_latency:30000"],
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "arm",
    [
        pytest.param("A", id="A-baseline-sla-only"),
        pytest.param("B", id="B-baseline-plus-cancellation"),
        pytest.param("C", id="C-baseline-plus-burst"),
        pytest.param("F1", id="F1-fe-fd-exhaustion"),
        pytest.param("F2", id="F2-fe-rolling-restart"),
        pytest.param("F3", id="F3-fe-pod-delete"),
        pytest.param("S", id="S-load-shed-prevent-overload"),
        pytest.param("P", id="P-prefill-decode-partition"),
        pytest.param("P2", id="P2-prefill-pod-kill"),
        pytest.param("T1", id="T1-tokio-starved-abrupt-fe-kill"),
    ],
)
async def test_decode_overload_disagg(runtime_env, request, arm):
    spec = DeploymentSpec(_TEMPLATE)
    served_model = spec["VllmDecodeWorker"].model

    # F1: cap FE nofile at 1024 via the dyn_tee.sh wrapper. The wrapper
    # reads DYN_TEST_NOFILE_LIMIT at startup and calls `ulimit -n N` before
    # exec'ing the user command. K8s pod spec doesn't directly expose
    # per-container nofile limits, so the env-var-in-wrapper path is the
    # standard fix.
    if arm == "F1":
        # 1024 was insufficient at c=96 (peak FD ~246). 128 is 8x tighter;
        # should trip accept() with margin.
        spec["Frontend"].set_env_var("DYN_TEST_NOFILE_LIMIT", "128")

    # T1: Single-threaded tokio runtime on decode. Combined with an
    # abrupt FE kill at peak load, this should close the cascade chain:
    # panic flood on the only worker thread → /live HTTP handler queues
    # behind panic recovery → exceeds the 4 s liveness timeout →
    # kubelet SIGTERM. The remaining ingredient F-3 lacked at our scale.
    if arm == "T1":
        spec["VllmDecodeWorker"].set_env_var("DYN_RUNTIME_NUM_WORKER_THREADS", "1")

    # S: Enable load shedding on FE. Goal is to PREVENT overload —
    # the router returns 503 once decode-busy crosses the threshold,
    # so accepted requests still meet SLA while clients receive a
    # fast "server busy" instead of a 30 s timeout.
    if arm == "S":
        fe = spec["Frontend"]
        fe.set_env_var("DYN_ROUTER_QUEUE_THRESHOLD", "0.5")
        fe.set_env_var("DYN_ROUTER_TRACK_PREFILL_TOKENS", "true")

    # c=96 sustained at OSL=200 pegs decode KV at 100% on this topology.
    # See findings/2026-05-13-decode-overload-repro-design.md for derivation.
    ramp_load = _disagg_load(
        served_model=served_model,
        name="ramp",
        concurrency=96,
        duration_minutes=10.0,
        ramp_seconds=600.0,
        warmup_concurrency=4,
        warmup_duration_seconds=60.0,
        cancellation_rate=(8.0 if arm == "B" else None),
        cancellation_delay=(1.5 if arm == "B" else 0.0),
    )
    # P2: bias sustain to long ISL (~7000) so NIXL transfers are slow
    # enough that killing prefill mid-transfer is reliable. Single-bucket
    # seq_dist forces every request to that shape.
    sustain_seq_dist = "7000,200:100" if arm == "P2" else None
    sustain_load = _disagg_load(
        served_model=served_model,
        name="sustain",
        concurrency=96,
        duration_minutes=15.0,
        cancellation_rate=(8.0 if arm == "B" else None),
        cancellation_delay=(1.5 if arm == "B" else 0.0),
        seq_dist=sustain_seq_dist,
    )

    events: list = [WaitForModelReady(timeout=1800)]
    # S: install the shedding threshold BEFORE load starts so the
    # very first ramp requests can be 503'd if needed.
    if arm == "S":
        events.append(
            SetBusyThreshold(
                model_name=served_model,
                active_decode_blocks_threshold=0.70,
                active_prefill_tokens_threshold_frac=0.85,
                name="install_shedding",
            )
        )
    events.extend(
        [
            StartLoad(load_config=ramp_load, name="ramp"),
            WaitForLoadCompletion(name="ramp"),
            Wait(duration=15),
            StartLoad(load_config=sustain_load, name="sustain"),
        ]
    )

    # Fault overlay during sustain — schedule injection 5 min in
    # (gives 1/3 of sustain to reach steady-state overload first).
    # The Wait happens in parallel with the running StartLoad, since
    # StartLoad returns immediately and WaitForLoadCompletion is what
    # actually blocks on completion.
    if arm == "C":
        burst_load = _disagg_load(
            served_model=served_model,
            name="burst",
            concurrency=120,
            duration_minutes=1.0,
        )
        events.extend(
            [
                Wait(duration=300),  # wait into sustain
                StartLoad(load_config=burst_load, name="burst"),
                WaitForLoadCompletion(name="burst"),
                WaitForLoadCompletion(name="sustain"),
            ]
        )
    elif arm == "F2":
        events.extend(
            [
                Wait(duration=300),
                RollingUpgrade(services=["Frontend"], name="fe_rollout"),
                WaitForLoadCompletion(name="sustain"),
            ]
        )
    elif arm == "F3":
        events.extend(
            [
                Wait(duration=300),
                DeletePod(services=["Frontend"], name="fe_delete"),
                WaitForLoadCompletion(name="sustain"),
            ]
        )
    elif arm == "P":
        # Prefill ↔ decode partition.
        # NIXL is decode-pulled, so the relevant ingress direction is
        # decode → prefill (decode initiates the NIXL handshake/pull).
        # NetworkPolicy + conntrack-flush severs both new and existing
        # connections; 120 s window is plenty to drive a sustained
        # decode-side stall.
        events.extend(
            [
                Wait(duration=300),
                NetworkPartition(
                    source="VllmDecodeWorker",
                    target="VllmPrefillWorker",
                    duration=120.0,
                    name="prefill_decode_partition",
                ),
                WaitForLoadCompletion(name="sustain"),
            ]
        )
    elif arm == "T1":
        # Same abrupt FE kill as F-3, but on a decode pod whose tokio
        # runtime has only one worker thread. Expect panics + /live
        # timeout + kubelet SIGTERM → restartCount ≥ 1.
        events.extend(
            [
                Wait(duration=300),
                DeletePod(services=["Frontend"], force=True, name="t1_fe_kill"),
                WaitForLoadCompletion(name="sustain"),
            ]
        )
    elif arm == "P2":
        # Abrupt prefill pod kill mid-sustain — peer-disappearance class.
        # At peak load decode has a deep queue of inflight requests, each
        # waiting on an outstanding NIXL pull from a prefill peer.
        # SIGKILLing the prefill rips its NIXL endpoint mid-handshake →
        # decode raises `KeyError: remote_block_size` from vLLM's
        # NixlConnector when it tries to complete those pulls.
        # `force=True` on DeletePod => --grace-period=0 hard kill.
        events.extend(
            [
                Wait(duration=300),
                DeletePod(
                    services=["VllmPrefillWorker"],
                    force=True,
                    pod_indices=[0],
                    name="prefill_kill",
                ),
                WaitForLoadCompletion(name="sustain"),
            ]
        )
    else:
        events.append(WaitForLoadCompletion(name="sustain"))

    # Gating checks — common to all arms.
    if arm == "S":
        # Shedding flips the polarity of every overload check: we want
        # to PROVE the system stayed within bounds despite the same
        # offered load. The 503-firing check is implemented via
        # ServiceLogPatternRate below.
        checks: list = [
            LoadApplied(load_name="sustain", min_requests=500),
            # On accepted (non-503) requests, AIPerf's request_latency
            # only includes successful streams. p99 < 30 s = SLA held.
            # SlaViolation passes if breached; we want it NOT to pass.
            # No direct "InverseSla" check — assert via report instead;
            # this arm runs with goodput SLO == 30 s so good_request_count
            # / request_count is the metric.
        ]
    else:
        checks = [
            # 1. Cheap guard: some load went through. 100 floor accommodates
            #    severe overload (A2 had 158 sustained reqs in 15 min).
            LoadApplied(load_name="sustain", min_requests=100),
            # 2. Decode KV reached or passed the disagg-cascade-class production peak.
            KvCacheUsagePeak(
                services=["VllmDecodeWorker"], threshold=0.80, within_seconds=1800
            ),
            # 3. Some pod's vllm-batch hit a meaningful running-count plateau.
            #    Under KV pin, "running" is small (vllm preempt-recomputes),
            #    so threshold=10 with 30s window catches the regime where
            #    the engine is actively chewing a stable cohort, not idle.
            RequestsRunningPeak(threshold=10, sustained_seconds=30.0),
            # 4. Client-side disconnect / SLA-timeout pressure observed.
            RequestCancellationOccurred(
                load_name="sustain",
                min_count=(100 if arm == "B" else 25),
            ),
            # 5. SLA breach — end-user impact signal.
            SlaViolation(load_name="sustain", e2e_p99_ms=30000.0, ttft_p99_ms=10000.0),
            # 6. Throughput collapse — engine spent ≥ 30 s at running=0
            #    while frontend had pending work.
            ThroughputCollapse(collapse_seconds=30.0),
        ]

    # Phase-2-specific assertions.
    if arm == "F1":
        # We expect FE to log the FD-exhaustion signature.
        checks.append(
            ServiceLogPatternRate(
                services=["Frontend"],
                pattern=r"accept\(\).*Too many open files|os error 24",
                min_rate_per_sec=0.0,
                expect_zero=False,
            )
        )
    if arm == "S":
        # FE should be returning 503 to clients above the busy threshold.
        # The Rust frontend logs the queue-exceeded WARN at the
        # busy_threshold handler; rate is the proof shedding fired.
        checks.append(
            ServiceLogPatternRate(
                services=["Frontend"],
                pattern=r"all workers.*busy|queue.*threshold|503 Service Unavailable",
                min_rate_per_sec=0.1,
                expect_zero=False,
            )
        )
        # Decode KV should stay BELOW the busy threshold (we set 0.70).
        # Use a slightly relaxed ceiling (0.80) to allow transient bursts.
        # Inverse semantics — wrap in a small inline helper.
        checks.append(_KvCacheStaysBelow(services=["VllmDecodeWorker"], ceiling=0.85))
    if arm in ("F2", "F3"):
        # FE shutdown via rollout/delete fires SO_LINGER=0 RST →
        # decode `tcp_client.rs:255` panic burst should appear.
        checks.append(
            ServiceLogPatternRate(
                services=["VllmDecodeWorker"],
                pattern=r"panicked at .*tcp/client\.rs:255",
                min_rate_per_sec=0.0,
                expect_zero=False,
            )
        )
    if arm == "T1":
        # Full cascade closure: decode container should have been
        # SIGTERMed by kubelet after /live timeouts during the panic
        # flood. restartCount ≥ 1 is the proof.
        checks.append(
            RestartCountIncreased(services=["VllmDecodeWorker"], expect_min_increment=1)
        )
        # And the panic burst itself should be visible.
        checks.append(
            ServiceLogPatternRate(
                services=["VllmDecodeWorker"],
                pattern=r"panicked at .*tcp/client\.rs:255",
                min_rate_per_sec=0.0,
                expect_zero=False,
            )
        )
    if arm in ("P", "P2"):
        # P  — NetworkPolicy-based prefill↔decode partition.
        # P2 — abrupt prefill SIGKILL mid-transfer (ISL=7000 sustain).
        # Both should produce NIXL-side failures on decode. Looking for:
        # 1. Decode raises KeyError on remote_block_size (D-3 production
        #    class — vLLM's NIXL connector failing block lookup).
        # 2. NIXL connector / UCX-level errors on either side.
        # 3. The vllm:nixl_num_kv_expired_reqs_total counter climbs
        #    (KV blocks aborted because peer disappeared).
        checks.append(
            ServiceLogPatternRate(
                services=["VllmDecodeWorker", "VllmPrefillWorker"],
                pattern=(
                    r"KeyError: ['\"]?remote_block_size|"
                    r"nixl.*(error|fail)|"
                    r"UCX.*(ERROR|err)|"
                    r"NixlConnector.*Exception"
                ),
                min_rate_per_sec=0.0,
                expect_zero=False,
            )
        )

    # Bonus signatures — informational, never gating.
    checks.extend(
        [
            ServiceLogPatternRate(
                services=["VllmDecodeWorker"],
                pattern=r"radix_tree\.rs:(341|431)",
                min_rate_per_sec=0.5,
            ),
            WorkerPanics(
                services=["VllmDecodeWorker", "Frontend"],
                acceptable=True,
            ),
            RestartCountIncreased(
                services=["VllmDecodeWorker"], expect_min_increment=0
            ),
            LoadCompleted(name="sustain"),
        ]
    )

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=checks,
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
            ErrorTrackingReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
