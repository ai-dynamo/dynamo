# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# v1.1.0 head-to-head — three arms covering routing + threshold space.
#
# Per the dynamo-observe KV-routing test plan
# (deployments/amazon-ads/2026-05-16-kv-routing-test-plan.md), the
# rejection-thresholds env vars alone CAN'T prevent per-worker KV
# imbalance: under the default `round-robin` (or `least-loaded`) router,
# requests are dispatched by stream count, not actual KV pressure. One
# decode pod can sit at 100% KV while peers stay cool — the all-busy
# 503 floor only fires when *every* worker is above threshold.
#
# Switching to `--router-mode kv` with all prefix-match weights zeroed
# turns selection into pure argmin on `active_decode_blocks`. The router
# routes around hot workers until peers catch up, then thresholds keep
# overall fleet within bounds with 503s only at true saturation.
#
# Three arms (all on N=3: 3 FE : 6 PF TP=2 : 3 DE TP=2 = 18 GPUs).
# `--max-num-seqs=256` is held constant across all three arms so the only
# moving variable is routing+threshold:
#
#   arm                       | router-mode    | thresholds | KV-zeroed
#   --------------------------+----------------+------------+---------
#   A. baseline_no_threshold   | default (RR)   | unset      | n/a
#   B. threshold_only          | default (RR)   | 0.85/0.85  | n/a
#   C. kv_route_threshold      | kv             | 0.85/0.85  | yes (overlap=0, T=0)
#
# All three: vllm-runtime:1.1.0, kv-events OFF on all workers
# (matches the production AZ-c configuration).
#
# Ladder per arm:
#   1. Pre-warm phase (2 min, c=30, ISL=8K low-OSL) to bias KV pressure
#      unevenly across decode pods under RR. Skipped on arm A (cascade
#      is the story, not imbalance).
#   2. Steady rungs: c=72, 144, 216, 288, 432 — 10 min each.
#
# Pass criteria summary:
#   Arm A (baseline_no_threshold): cascade at upper rungs (KV ≥ 0.95, SLA breach).
#   Arm B (threshold_only):
#       - decode pods do NOT all stay below 0.90 → imbalance gap ≥ 30pp
#         (max-min per-pod KV across decode pods on rung 4-5)
#       - 503 rate may be near zero even with thresholds set
#       - SLA breach observed on the hot pod
#   Arm C (kv_route_threshold):
#       - per-pod KV imbalance gap ≤ 15pp on every rung
#       - 503 fires only when ALL pods are over threshold (rung 5)
#       - SLA held until full-fleet saturation

import pytest

from tests.fault_tolerance.deploy.checks import (
    Check,
    KvCacheUsagePeak,
    LoadApplied,
    LoadCompleted,
    RestartCountIncreased,
    ServiceLogPatternRate,
    SlaViolation,
    WorkerPanics,
    _iter_server_metric,
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


_TEMPLATE = (
    "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
    "disagg_qwen3_30b_unit_prod.yaml"
)

_V110_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0"

_PROD_SEQ_DIST = (
    "100,200:5;500,200:15;1000,200:20;1600,200:30;3400,200:20;7000,200:10"
)
_NUM_PREFIX_PROMPTS = 15
_PREFIX_PROMPT_LENGTH = 600

# Single-bucket warmup shape — long ISL, short OSL — creates concentrated
# KV pressure that decays slowly. Drives initial per-worker asymmetry.
_PREWARM_SEQ_DIST = "8000,50:100"

# Cluster-wide closed-loop concurrency rungs (N=3 = 3 FE pods).
# Per-set: 24, 48, 72, 96, 144. Design point ~96/set; rung 5 = 1.5× design.
_RUNGS = [
    ("c72",  72,  10.0),
    ("c144", 144, 10.0),
    ("c216", 216, 10.0),
    ("c288", 288, 10.0),
    ("c432", 432, 10.0),
]


from dataclasses import dataclass


@dataclass
class _KvCacheStaysBelow(Check):
    """Assert every sample of ``vllm:kv_cache_usage_perc`` on the named
    services stays at or below ``ceiling``."""

    services: list
    ceiling: float = 0.90

    def validate(self, ctx) -> None:
        observed: dict = {}
        for ts_ns, pod, val in _iter_server_metric(
            ctx, "vllm:kv_cache_usage_perc"
        ):
            cur = observed.get(pod)
            if cur is None or val > cur:
                observed[pod] = val
        ctx.logger.info(
            f"_KvCacheStaysBelow: per-pod max KV={observed} "
            f"ceiling={self.ceiling}"
        )
        breaches = {p: v for p, v in observed.items() if v > self.ceiling}
        assert not breaches, (
            f"_KvCacheStaysBelow: pods exceeded KV ceiling "
            f"{self.ceiling}: {breaches}"
        )

    @property
    def description(self) -> str:
        return (
            f"KV cache usage stays ≤ {self.ceiling} on "
            f"{', '.join(self.services)} (routing/shedding holds the line)"
        )


@dataclass
class _KvImbalanceGap(Check):
    """Compute per-pod max(``vllm:kv_cache_usage_perc``) across the named
    services. Assert that ``max(per-pod peaks) - min(per-pod peaks)`` is
    either below ``max_gap`` (proves rebalancing) or above ``min_gap``
    (proves imbalance reproduced)."""

    services: list
    max_gap: float | None = None
    min_gap: float | None = None

    def validate(self, ctx) -> None:
        observed: dict = {}
        for ts_ns, pod, val in _iter_server_metric(
            ctx, "vllm:kv_cache_usage_perc"
        ):
            cur = observed.get(pod)
            if cur is None or val > cur:
                observed[pod] = val
        if len(observed) < 2:
            ctx.logger.warning(
                f"_KvImbalanceGap: only {len(observed)} pods observed; "
                f"cannot compute imbalance"
            )
            return
        peaks = list(observed.values())
        gap = max(peaks) - min(peaks)
        ctx.logger.info(
            f"_KvImbalanceGap: per-pod peaks={observed} "
            f"max={max(peaks):.3f} min={min(peaks):.3f} gap={gap:.3f}"
        )
        if self.max_gap is not None:
            assert gap <= self.max_gap, (
                f"_KvImbalanceGap: gap {gap:.3f} > max_gap {self.max_gap} "
                f"(per-pod peaks={observed})"
            )
        if self.min_gap is not None:
            assert gap >= self.min_gap, (
                f"_KvImbalanceGap: gap {gap:.3f} < min_gap {self.min_gap} "
                f"(per-pod peaks={observed}) — imbalance not reproduced"
            )

    @property
    def description(self) -> str:
        if self.max_gap is not None and self.min_gap is None:
            return (
                f"Per-pod KV peak gap across {', '.join(self.services)} "
                f"≤ {self.max_gap} (routing rebalanced)"
            )
        if self.min_gap is not None and self.max_gap is None:
            return (
                f"Per-pod KV peak gap across {', '.join(self.services)} "
                f"≥ {self.min_gap} (imbalance reproduced)"
            )
        return f"KV-pod imbalance gap on {', '.join(self.services)}"


def _scale_to_units(spec, units):
    for service in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[service].replicas = spec[service].replicas * units


def _prod_load(*, served_model, concurrency, duration_minutes, name,
               seq_dist=None):
    return LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        seq_dist=seq_dist or _PROD_SEQ_DIST,
        num_prefix_prompts=_NUM_PREFIX_PROMPTS,
        prefix_prompt_length=_PREFIX_PROMPT_LENGTH,
        concurrency=concurrency,
        duration_minutes=duration_minutes,
        request_timeout_seconds=20,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
        connection_reuse_strategy="never",
        goodput=["request_latency:20000"],
    )


_ARMS = [
    pytest.param("baseline_no_threshold", id="baseline_no_threshold"),
    pytest.param("threshold_only", id="threshold_only"),
    pytest.param("kv_route_threshold", id="kv_route_threshold"),
]


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("arm", _ARMS)
async def test_n3_v110_routing_threshold_compare(runtime_env, request, arm):
    spec = DeploymentSpec(_TEMPLATE)
    _scale_to_units(spec, units=3)

    # Pin all arms to v1.1.0 (templates default to v1.0.1).
    for svc in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].image = _V110_IMAGE

    # aks-dev: pin Frontend to the A100 node pool (1.87 TB ephemeral).
    fe_pod = spec["Frontend"]._spec.setdefault("extraPodSpec", {})
    fe_pod["nodeSelector"] = {
        "nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB"
    }
    fe_pod["tolerations"] = [
        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
    ]

    # aks-dev: full unprivileged-bypass recipe for UCX/NIXL on
    # cri-containerd AppArmor + 64KB memlock default (OPS-4332).
    for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
        main = spec[svc]._spec.setdefault(
            "extraPodSpec", {}
        ).setdefault("mainContainer", {})
        secctx = main.setdefault("securityContext", {})
        secctx["privileged"] = True
        secctx["runAsUser"] = 0
        secctx["appArmorProfile"] = {"type": "Unconfined"}
        caps = secctx.setdefault("capabilities", {})
        caps.setdefault("add", []).append("IPC_LOCK")
        spec[svc].set_env_var("DYN_TEST_MEMLOCK_UNLIMITED", "1")

    # aks-dev runs A100-80GB; production runs A100-40GB. Halve the
    # gpu-memory-utilization so per-GPU effective KV envelope matches prod.
    for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].set_arg("--gpu-memory-utilization", "0.5")

    served_model = spec["VllmDecodeWorker"].model

    # ----- arm-specific config -----
    # --max-num-seqs held constant at 256 across all arms — the variable
    # is purely routing + threshold settings.
    spec["VllmPrefillWorker"].set_arg("--max-num-seqs", "256")
    spec["VllmDecodeWorker"].set_arg("--max-num-seqs", "256")

    fe = spec["Frontend"]
    # Threshold envs — both threshold arms.
    if arm in ("threshold_only", "kv_route_threshold"):
        fe.set_env_var("DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD", "0.85")
        fe.set_env_var("DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC", "0.85")

    # KV-mode routing config — arm C only.
    # Use env vars (NOT set_arg on the FE) — `set_arg` initialises an
    # empty args list that clobbers the framework's default
    # `python3 -m dynamo.frontend` injection. Env vars achieve the same
    # configuration without touching the CLI argv plane.
    if arm == "kv_route_threshold":
        fe.set_env_var("DYN_ROUTER_MODE", "kv")
        fe.set_env_var("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "0.0")
        fe.set_env_var("DYN_ROUTER_TEMPERATURE", "0.0")
        # DYN_ROUTER_USE_KV_EVENTS=false matches --no-router-kv-events.
        fe.set_env_var("DYN_ROUTER_USE_KV_EVENTS", "false")

    # ----- Events: optional pre-warm, then rung ladder -----
    events: list = [WaitForModelReady(timeout=2400)]

    # Pre-warm only for the two threshold arms — they're the ones where
    # imbalance is the question. Skip for baseline (cascade dominates).
    if arm in ("threshold_only", "kv_route_threshold"):
        prewarm = _prod_load(
            served_model=served_model,
            concurrency=30,
            duration_minutes=2.0,
            name="prewarm",
            seq_dist=_PREWARM_SEQ_DIST,
        )
        events.append(StartLoad(load_config=prewarm, name="prewarm"))
        events.append(WaitForLoadCompletion(name="prewarm"))
        events.append(Wait(duration=30))  # let KV pressure stabilize

    for name, conc, dur in _RUNGS:
        cfg = _prod_load(
            served_model=served_model,
            concurrency=conc,
            duration_minutes=dur,
            name=name,
        )
        events.append(StartLoad(load_config=cfg, name=name))
        events.append(WaitForLoadCompletion(name=name))
        events.append(Wait(duration=30))

    # ----- Checks per arm -----
    final = _RUNGS[-1][0]
    common_panics = WorkerPanics(
        services=["VllmDecodeWorker", "VllmPrefillWorker", "Frontend"],
        acceptable=True,
    )

    if arm == "baseline_no_threshold":
        # Baseline: establish the max / steady-state RPS the cluster can
        # sustain at a 20 s SLA. No thresholds, default routing. Climbing
        # the rungs measures where goodput plateaus and where SLA starts
        # to breach — that knee is the ceiling all other arms get
        # compared against.
        checks = [
            LoadApplied(load_name=final, min_requests=100),
            # Restarts informational only — baseline may or may not crash.
            RestartCountIncreased(
                services=["VllmDecodeWorker"], expect_min_increment=0
            ),
            RestartCountIncreased(
                services=["Frontend"], expect_min_increment=0
            ),
            LoadCompleted(name=final),
            common_panics,
        ]
    elif arm == "threshold_only":
        # Imbalance reproduction: thresholds are set but routing is
        # round-robin so one pod gets hot while peers stay cool.
        # Expectations:
        #  - At least one decode pod over the threshold ceiling.
        #  - Per-pod gap ≥ 30 pp (imbalance reproduced).
        #  - SLA still breached on the hot pod.
        #  - 503 rate may be near zero — all-busy gate doesn't fire
        #    because only one pod is over.
        checks = [
            LoadApplied(load_name=final, min_requests=500),
            KvCacheUsagePeak(
                services=["VllmDecodeWorker"],
                threshold=0.90,  # at least one pod hits this
                within_seconds=3600,
            ),
            _KvImbalanceGap(
                services=["VllmDecodeWorker"], min_gap=0.30
            ),
            SlaViolation(
                load_name=final, e2e_p99_ms=20000.0, ttft_p99_ms=10000.0
            ),
            RestartCountIncreased(
                services=["VllmDecodeWorker"], expect_min_increment=0
            ),
            RestartCountIncreased(
                services=["Frontend"], expect_min_increment=0
            ),
            LoadCompleted(name=final),
            common_panics,
        ]
    else:  # kv_route_threshold
        # Load-aware routing with thresholds. Pass criteria:
        #  - Every decode pod stays ≤ 0.90 (routing rebalances).
        #  - Per-pod gap ≤ 0.15 (within 15 pp of each other).
        #  - 503 fires only when ALL pods are simultaneously over the
        #    threshold (rung 5 saturates everyone).
        #  - Zero restarts.
        checks = [
            LoadApplied(load_name=final, min_requests=500),
            _KvCacheStaysBelow(
                services=["VllmDecodeWorker"], ceiling=0.95
            ),
            _KvImbalanceGap(
                services=["VllmDecodeWorker"], max_gap=0.15
            ),
            RestartCountIncreased(
                services=["VllmDecodeWorker"], expect_min_increment=0
            ),
            RestartCountIncreased(
                services=["Frontend"], expect_min_increment=0
            ),
            RestartCountIncreased(
                services=["VllmPrefillWorker"], expect_min_increment=0
            ),
            ServiceLogPatternRate(
                services=["Frontend"],
                pattern=r"all workers.*busy|busy_threshold|503 Service Unavailable",
                min_rate_per_sec=0.05,
                expect_zero=False,
            ),
            LoadCompleted(name=final),
            common_panics,
        ]

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
