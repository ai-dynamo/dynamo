# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Overload and cascade scenarios. Each test reproduces one well-defined
# failure shape under load. Reports per-pod KV imbalance, goodput, and
# error breakdown every run.
#
# Run a single scenario:
#     uv run pytest test_overload.py::test_decode_overload \
#       --namespace neelays-test --storage-class azurefile-csi-premium \
#       --image nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1 -s -v

import os
from dataclasses import dataclass

import pytest

from tests.fault_tolerance.deploy.checks import (
    Check,
    CliffContained,
    CliffPropagated,
    GenerationThroughputDropped,
    KvCacheUsagePeak,
    LoadApplied,
    LoadCompleted,
    NixlXferTimeMultiplied,
    PinningContained,
    PodMemoryGrowth,
    RejectionFired,
    RestartCountIncreased,
    ServiceLogPatternRate,
    SlaViolation,
    WaitingForKVTransferExceeds,
    WorkerPanics,
    _iter_server_metric,
)
from tests.fault_tolerance.deploy.events import (
    PodMemoryPoller,
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
from tests.utils.managed_load import LoadConfig, WorkerPin

# ─── DGD ──────────────────────────────────────────────────────────────
# The seed DGD all scenarios use. To run against a different DGD, either
# edit this line or override via `--dgd <name>` on the CLI.
dgd = "disagg_qwen3_30b_unit_prod"

_DEFAULT_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1"

# Production-shaped traffic distribution (ISL/OSL pairs with realistic
# long-tail mix). Same distribution as the n3 routing-threshold test.
_PROD_SEQ_DIST = "100,200:5;500,200:15;1000,200:20;1600,200:30;3400,200:20;7000,200:10"
_NUM_PREFIX_PROMPTS = 15
_PREFIX_PROMPT_LENGTH = 600

# Default rung ladder. Overridable via --rungs.
_DEFAULT_RUNGS = "72,216,432"
_DEFAULT_RUNG_MINUTES = 5.0


# ─── CLI ──────────────────────────────────────────────────────────────
# Image override comes from the framework-shared --image flag (already
# registered by tests/conftest.py and the deploy conftest). Test reads
# it via request.config.getoption("--image") with a 1.1.1 fallback.
def add_cli_options(parser):
    """Called from the deploy conftest's pytest_addoption."""
    g = parser.getgroup("overload", "Overload scenarios (test_overload.py)")
    g.addoption(
        "--dgd",
        default=dgd,
        help=f"DGD template basename under templates/vllm/. Default: {dgd}",
    )
    g.addoption(
        "--units",
        type=int,
        default=3,
        help="Convenience: scale frontend/prefill/decode together (prod-unit "
        "replicas). Default 3 (3F : 6P : 3D = 18 GPUs).",
    )
    g.addoption(
        "--frontend-replicas",
        type=int,
        default=None,
        help="Override frontend replicas (takes precedence over --units).",
    )
    g.addoption(
        "--prefill-replicas",
        type=int,
        default=None,
        help="Override prefill replicas (takes precedence over --units).",
    )
    g.addoption(
        "--decode-replicas",
        type=int,
        default=None,
        help="Override decode replicas (takes precedence over --units).",
    )
    g.addoption(
        "--rungs",
        default=_DEFAULT_RUNGS,
        help=f"Comma-separated concurrency rungs. Default: {_DEFAULT_RUNGS}",
    )
    g.addoption(
        "--rung-minutes",
        type=float,
        default=_DEFAULT_RUNG_MINUTES,
        help=f"Minutes per rung. Default: {_DEFAULT_RUNG_MINUTES}",
    )
    g.addoption(
        "--router-mode",
        default="kv",
        choices=[
            "kv",
            "least-loaded",
            "round-robin",
            "random",
            "power-of-two",
            "direct",
        ],
        help="Router mode applied to Frontend (DYN_ROUTER_MODE). Default: kv",
    )
    g.addoption(
        "--model-cache-pvc",
        default="",
        help="Mount this RWX PVC on workers as the HuggingFace model cache "
        "(skips re-downloading 30+GB model weights on every pod restart). "
        "On aws-dev-02 the conventional name is 'shared-model-cache' (already "
        "provisioned in neelays-test and neelays-test-b). Empty string disables.",
    )


# ─── Helpers ──────────────────────────────────────────────────────────
def _load_dgd(name: str) -> DeploymentSpec:
    """Resolve a DGD basename to a DeploymentSpec."""
    path = os.path.join(os.path.dirname(__file__), "templates", "vllm", f"{name}.yaml")
    return DeploymentSpec(path)


def _apply_topology(spec, units, fe, pf, dec):
    """Set per-service replicas. Per-component overrides win over `units`."""
    base = {
        "Frontend": spec["Frontend"].replicas,
        "VllmPrefillWorker": spec["VllmPrefillWorker"].replicas,
        "VllmDecodeWorker": spec["VllmDecodeWorker"].replicas,
    }
    spec["Frontend"].replicas = fe if fe is not None else base["Frontend"] * units
    spec["VllmPrefillWorker"].replicas = (
        pf if pf is not None else base["VllmPrefillWorker"] * units
    )
    spec["VllmDecodeWorker"].replicas = (
        dec if dec is not None else base["VllmDecodeWorker"] * units
    )


def _apply_recommended_kv_router_config(spec) -> None:
    """Apply the canonical A4 "recommended KV-aware" Frontend config.

    This is the **prescribed production config** from the prior
    KV-pressure cascade analysis (§A4 — `Recommendations —
    Without code change`). Pure-load steering with cross-FE visibility:
    KV-aware routing mode + zeroed prefix-cache scoring + cross-FE
    replica sync. The combination closes BOTH LeastLoaded blind spots
    (per-worker `_avail` vs `_free`, AND cross-FE-pod) at the cost of
    no code change.

    All env-var names and defaults were verified against the **v1.1.1
    release tag** of ai-dynamo/dynamo at
    ``components/src/dynamo/common/configuration/groups/kv_router_args.py``
    and ``components/src/dynamo/frontend/frontend_args.py``. Every
    value below is explicitly set even where it matches the v1.1.1
    default so the test asserts a known-good config rather than
    relying on shifting defaults across releases.

    Cost-function logit walked through in the cascade analysis
    (selector.rs:166-167)::

        logit = prefill_load_scale × adjusted_prefill_blocks + decode_blocks
              = 1.0 × 0 + active_decode_blocks      (overlap_weight=0)
              = active_decode_blocks                 (argmin = least-loaded)

    The two cost-function weights the cascade analysis mentions but
    that aren't env-var-exposed in v1.1.1 — ``host_cache_hit_weight``
    (default 0.75) and ``disk_cache_hit_weight`` (default 0.25) — are
    multiplied through the overlap term, so with
    ``DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT=0.0`` they contribute zero to
    the logit regardless of their values. Leaving them at defaults is
    correct for this config.
    """
    fe = spec["Frontend"]
    # Routing mode (frontend_args.py) — KV-aware is the only mode where
    # the load signal flows across all FE pods via the event plane.
    fe.set_env_var("DYN_ROUTER_MODE", "kv")
    # Pure-load steering: zero out prefix-cache scoring contribution.
    fe.set_env_var("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "0.0")
    # No KV-cache event consumption from workers — router predicts
    # cache state from its own routing decisions. The active_decode_blocks
    # load signal flows regardless (via ActiveSequenceEvent on the event
    # plane, not via vLLM KV-cache events).
    fe.set_env_var("DYN_ROUTER_USE_KV_EVENTS", "false")
    # **Critical**: cross-FE replica sync. v1.1.1 default is False, which
    # leaves each FE blind to its peers' routing decisions — the exact
    # mechanism that caused the the cross-FE cascade propagation.
    fe.set_env_var("DYN_ROUTER_REPLICA_SYNC", "true")
    # Deterministic argmin (no softmax-sampling exploration). Default 0,
    # set explicitly for clarity.
    fe.set_env_var("DYN_ROUTER_TEMPERATURE", "0.0")
    # Active-block tracking MUST be on for the load signal. Default true,
    # set explicitly.
    fe.set_env_var("DYN_ROUTER_TRACK_ACTIVE_BLOCKS", "true")
    # With overlap_weight=0 the router doesn't do prefix matching, so
    # disable KV-reuse hash tracking to suppress the radix-tree warning
    # storm observed at 1,200/s on a heavily-loaded pod. v1.1.1 default is
    # True — recommendation flips it.
    fe.set_env_var("DYN_ROUTER_ASSUME_KV_REUSE", "false")
    # Admission shedding thresholds — the all-busy 503 floor. Activates
    # the shedding code already in v1.1.0 (PR #7617/#7912/#8413, whose
    # defaults are permissive sentinels). v1.1.1 defaults: 1.0 and 10.0
    # respectively — both effectively off without these explicit sets.
    fe.set_env_var("DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD", "0.85")
    fe.set_env_var("DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC", "0.85")


def _apply_router_config(
    spec,
    *,
    decode_blocks_threshold: str = "0.85",
    prefill_tokens_threshold_frac: str = "0.85",
    router_mode: str = "kv",
    overlap_score_weight: str = "0.0",
    temperature: str = "0.0",
    use_kv_events: str = "false",
) -> None:
    """Apply the final-recommended router config to the Frontend spec.

    Defaults match the recipe from the prior cascade-reproducer doc
    (and the joint thread close-out): per-worker admission thresholds
    enabled at 0.85, KV-aware routing with prefix-cache weight zeroed,
    deterministic argmin selection, no KV events. The same config is
    applied to every comparison arm — the diff between arms is purely
    the image (1.1.1 baseline vs 1.1.2-test with the routing fix), not
    the router knobs.
    """
    fe = spec["Frontend"]
    fe.set_env_var("DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD", decode_blocks_threshold)
    fe.set_env_var(
        "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC", prefill_tokens_threshold_frac
    )
    fe.set_env_var("DYN_ROUTER_MODE", router_mode)
    fe.set_env_var("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", overlap_score_weight)
    fe.set_env_var("DYN_ROUTER_TEMPERATURE", temperature)
    fe.set_env_var("DYN_ROUTER_USE_KV_EVENTS", use_kv_events)


def _apply_cluster_portability(spec):
    """Pod-security + memory-budget tweaks that make this DGD run cleanly on
    aks-dev (A100-80GB) and aws-dev-02 (H100-80GB) shared clusters.

    Includes image-pull-secret wiring: aws-dev-02's `default` SA only carries
    `acr-token-secret` (a controller resets the list on any add). We tack on
    `ngc-pull-secret` per-service so private nvcr.io/nvidian/* images pull
    cleanly. The secret is harmless on clusters that don't have it (k8s
    silently ignores unknown imagePullSecrets in pod specs)."""
    # Pin Frontend to A100/H100 node pool with the GPU toleration so the
    # operator's default CPU scheduling doesn't land it on a small node.
    fe = spec["Frontend"]._spec.setdefault("extraPodSpec", {})
    fe.setdefault(
        "tolerations",
        [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
        ],
    )

    # Add NGC private-org pull secret on every service. Default SA is reset
    # by a controller on shared clusters; per-pod secrets bypass that.
    for svc in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        eps = spec[svc]._spec.setdefault("extraPodSpec", {})
        ips = eps.setdefault("imagePullSecrets", [])
        if not any(s.get("name") == "ngc-pull-secret" for s in ips):
            ips.append({"name": "ngc-pull-secret"})

    # Full unprivileged-bypass for UCX/NIXL on cri-containerd AppArmor +
    # 64KB memlock default (OPS-4332).
    for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
        main = (
            spec[svc]
            ._spec.setdefault("extraPodSpec", {})
            .setdefault("mainContainer", {})
        )
        secctx = main.setdefault("securityContext", {})
        secctx["privileged"] = True
        secctx["runAsUser"] = 0
        secctx["appArmorProfile"] = {"type": "Unconfined"}
        caps = secctx.setdefault("capabilities", {})
        caps.setdefault("add", []).append("IPC_LOCK")
        spec[svc].set_env_var("DYN_TEST_MEMLOCK_UNLIMITED", "1")

    # Pod-level fsGroup=1000 on every service + Frontend runAsUser=0.
    # The prior working DGD (dynamo-observe/.../2026-05-03-cascade-repro-
    # results/) set BOTH. Reasoning:
    #   - fsGroup=1000 makes k8s recursively chown the mounted volume to
    #     gid 1000 (dynamo user's group) and SGID-bit the dirs — but
    #     that's only group membership; write is still owner-only under
    #     default umask 022 (mode 755).
    #   - To actually let Frontend write under root-owned parent dirs
    #     created by workers (which need runAsUser=0 for UCX/IB), the
    #     Frontend must also be root. fsGroup is then defense-in-depth.
    for svc in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        pod_sec = (
            spec[svc]
            ._spec.setdefault("extraPodSpec", {})
            .setdefault("securityContext", {})
        )
        pod_sec.setdefault("fsGroup", 1000)
    fe_main = (
        spec["Frontend"]
        ._spec.setdefault("extraPodSpec", {})
        .setdefault("mainContainer", {})
    )
    fe_main.setdefault("securityContext", {})["runAsUser"] = 0

    # Prod runs A100-40GB; both shared clusters above run 80GB cards.
    # Halve gpu-memory-utilization to match prod KV envelope.
    for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].set_arg("--gpu-memory-utilization", "0.5")


def _parse_rungs(raw):
    """`"72,216,432"` → `[(name, conc), ...]`. Names are `c<N>`."""
    out = []
    for piece in raw.split(","):
        c = int(piece.strip())
        out.append((f"c{c}", c))
    return out


def _prod_load(*, served_model, concurrency, duration_minutes, name):
    return LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        seq_dist=_PROD_SEQ_DIST,
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


@dataclass
class _PerPodKvImbalanceGap(Check):
    """Report-only check that computes the per-pod max(KV-cache-usage) gap
    across the named services. Logged for evidence; never raises."""

    services: list

    def validate(self, ctx) -> None:
        peaks: dict = {}
        for _ts, pod, val in _iter_server_metric(ctx, "vllm:kv_cache_usage_perc"):
            cur = peaks.get(pod)
            if cur is None or val > cur:
                peaks[pod] = val
        if len(peaks) < 2:
            ctx.logger.warning(
                f"_PerPodKvImbalanceGap: only {len(peaks)} pods observed; "
                f"cannot compute imbalance"
            )
            return
        vals = list(peaks.values())
        gap = max(vals) - min(vals)
        ctx.logger.info(
            f"_PerPodKvImbalanceGap: per-pod KV peaks={peaks} "
            f"max={max(vals):.3f} min={min(vals):.3f} gap={gap:.3f}"
        )

    @property
    def description(self) -> str:
        return f"Per-pod KV peak gap across {', '.join(self.services)} (report-only)"


# ─── Scenario ─────────────────────────────────────────────────────────
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_decode_overload(runtime_env, request):
    """Drive prod-shaped load past sustainable concurrency; observe
    per-pod KV imbalance and goodput collapse on the decode pool.

    No fault injection — this is the natural-cascade scenario. Same
    function exercises both 'bug' and 'fix' images: against an image
    without the per-worker admission-threshold patch (PR #9692) the
    per-pod KV peaks spread out and goodput drops; against an image with
    the patch, the spread tightens and goodput holds longer.

    CLI flags: --overload-image, --overload-units (or per-component
    replica overrides), --overload-rungs, --overload-rung-minutes,
    --overload-dgd.
    """
    cfg = request.config
    image = cfg.getoption("--image") or _DEFAULT_IMAGE
    dgd_name = cfg.getoption("--dgd")
    units = cfg.getoption("--units")
    fe = cfg.getoption("--frontend-replicas")
    pf = cfg.getoption("--prefill-replicas")
    dec = cfg.getoption("--decode-replicas")
    rungs = _parse_rungs(cfg.getoption("--rungs"))
    rung_minutes = cfg.getoption("--rung-minutes")

    spec = _load_dgd(dgd_name)
    _apply_topology(spec, units=units, fe=fe, pf=pf, dec=dec)

    for svc in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].image = image

    _apply_cluster_portability(spec)
    _apply_router_config(spec, router_mode=cfg.getoption("--router-mode"))

    model_cache_pvc = cfg.getoption("--model-cache-pvc")
    if model_cache_pvc:
        spec.enable_model_cache(model_cache_pvc)

    served_model = spec["VllmDecodeWorker"].model

    events: list = [WaitForModelReady(timeout=2400)]
    for name, conc in rungs:
        events.append(
            StartLoad(
                load_config=_prod_load(
                    served_model=served_model,
                    concurrency=conc,
                    duration_minutes=rung_minutes,
                    name=name,
                ),
                name=name,
            )
        )
        events.append(WaitForLoadCompletion(name=name))
        events.append(Wait(duration=30))

    final = rungs[-1][0]
    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=[
            LoadApplied(load_name=final, min_requests=100),
            LoadCompleted(name=final),
            _PerPodKvImbalanceGap(services=["VllmDecodeWorker"]),
            # Report-only goodput / SLA observations. SlaViolation reads
            # aiperf summary p99s on the named load; ServiceLogPatternRate
            # scans frontend logs for the 504 / inter-token-latency signal.
            SlaViolation(load_name=final, e2e_p99_ms=30000, ttft_p99_ms=10000),
            ServiceLogPatternRate(
                services=["Frontend"],
                pattern=r"504|GatewayTimeout|inter.token.latency",
                min_rate_per_sec=0.0,
            ),
            WorkerPanics(
                services=["VllmDecodeWorker", "VllmPrefillWorker", "Frontend"],
                acceptable=True,
            ),
            RestartCountIncreased(
                services=["VllmDecodeWorker", "Frontend"],
                expect_min_increment=0,
            ),
        ],
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
            ErrorTrackingReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ─── Cascade-repro scenarios (cascade reproduction) ─────────────────────────
# These tests reproduce the cascade-failure regime on a
# small dev pool. Source spec lives in GitLab `neelays/dynamo-observe`
# under the cascade-small-pool reproducer scenarios/
# (FRAMEWORK_TASK.md + PLAN.md).
#
# Reuses the existing disagg_qwen3_30b_unit_prod template. The only
# cascade-specific deltas are applied via _apply_cascade_dgd below
# (k8s-discovery envs needed for WorkerPin, NIXL abort timer, image).

_CASCADE_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1"


def _apply_cascade_dgd(
    spec,
    *,
    image: str,
    units: int,
    abort_timeout_s: int = 480,
    vllm_extra_args: list[str] | None = None,
    override_worker_knobs: str | None = None,
):
    """Cascade-specific overrides on top of disagg_qwen3_30b_unit_prod.

    Adds the three Dynamo runtime envs required for the k8s-native
    discovery path (so each worker pod publishes a
    ``DynamoWorkerMetadata`` CR that ``WorkerPin`` can resolve), pins
    the NIXL abort timer, and forces the image. ``_apply_cluster_portability``
    and ``_apply_router_config`` are intentionally NOT called here —
    callers compose them at the test site, same as ``test_decode_overload``.

    ``vllm_extra_args``: optional list of vLLM CLI args (e.g.
    ``["--max-num-seqs", "128"]``) appended to BOTH ``VllmPrefillWorker``
    and ``VllmDecodeWorker``'s ``extraPodSpec.mainContainer.args``. Used
    by the cascade fail-fast S2 / S3 experiments (2026-05-26 handoff) to
    cap engine running-batch size and starve the cascade at the source.
    Existing args are left untouched; if a flag is already present, the
    helper overwrites its value (via ``ServiceSpec.set_arg``).

    ``override_worker_knobs``: optional inline knob string with the same
    semicolon-separated KEY=VALUE format as the ``DYN_TEST_WORKER_KNOBS``
    env. When set, it WINS over the env var so per-row knob variations
    (S3 combo: 64/16 vs S1 spike: 128/32) can be expressed in the
    parametrize tuple directly instead of forking the launcher.
    """
    _apply_topology(spec, units=units, fe=None, pf=None, dec=None)
    for svc in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].image = image
        spec[svc].set_env_var("DYN_EVENT_PLANE", "zmq")
        spec[svc].set_env_var("DYN_DISCOVERY_BACKEND", "kubernetes")
        spec[svc].set_env_var("DYN_STORE_KV", "mem")
    for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].set_env_var("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", str(abort_timeout_s))

    # Per-test admission-control knobs. Set via env at launch time so a
    # single test function can drive the cascade matrix from observe's
    # 5-hr-burst handoff (DIS-2105 needs DYN_TCP_WORK_QUEUE_SIZE, PR #9858
    # needs DYN_VLLM_REJECT_QUEUE_THRESHOLD; baseline gets neither).
    # Format: semicolon-separated KEY=VALUE list. Empty → no overrides.
    # Example: DYN_TEST_WORKER_KNOBS="DYN_TCP_WORK_QUEUE_SIZE=256;DYN_VLLM_REJECT_QUEUE_THRESHOLD=80"
    knob_source = (
        override_worker_knobs
        if override_worker_knobs is not None
        else os.environ.get("DYN_TEST_WORKER_KNOBS", "")
    )
    extra = (knob_source or "").strip()
    if extra:
        for kv in extra.split(";"):
            kv = kv.strip()
            if not kv or "=" not in kv:
                continue
            key, val = kv.split("=", 1)
            for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
                spec[svc].set_env_var(key.strip(), val.strip())

    # Worker-side vLLM CLI extras (e.g. --max-num-seqs cap). Parsed in
    # flag/value pairs and pushed through ServiceSpec.set_arg so an
    # already-present flag is updated in place rather than appended
    # twice. Lone flags (no value) get an empty string value, matching
    # set_arg's behaviour for repeated flags.
    if vllm_extra_args:
        i = 0
        flat_pairs: list[tuple[str, str]] = []
        while i < len(vllm_extra_args):
            arg = vllm_extra_args[i]
            if i + 1 < len(vllm_extra_args) and not vllm_extra_args[i + 1].startswith(
                "-"
            ):
                flat_pairs.append((arg, vllm_extra_args[i + 1]))
                i += 2
            else:
                flat_pairs.append((arg, ""))
                i += 1
        for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
            for flag, value in flat_pairs:
                spec[svc].set_arg(flag, value)


def _cascade_load(
    *,
    served_model: str,
    concurrency: int,
    duration_minutes: float,
    name: str,
    isl: int = 2000,
    osl: int = 200,
    worker_pin: WorkerPin | None = None,
    request_cancellation_rate: float | None = None,
) -> LoadConfig:
    """LoadConfig shared by the cascade scenarios.

    Defaults match the FRAMEWORK_TASK §3.1 spec: no warmup-requests, fresh
    sockets per request (so any disconnect actually closes the underlying
    TCP), goodput SLO at 20s e2e latency. ISL/OSL default to a
    steady-state production-shape pair; override per-scenario.

    ``request_cancellation_rate`` (0.0-1.0, optional): pass to AIPerf so a
    fraction of requests get mid-flight client disconnects. Used by S1's
    cliff rung to exercise the ``_DeferredAbort`` path + 480-s NIXL abort
    timer — without cancellations to feed the path, the timer never fires
    and we can't claim production-parity recovery behaviour. Per observe-
    agent NEXT_STEPS_FOR_TEST_AGENT P0.2 (sanity-suite work).
    """
    return LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        input_tokens_mean=isl,
        input_tokens_stddev=max(1, isl // 10),
        output_tokens_mean=osl,
        output_tokens_stddev=max(1, osl // 10),
        num_prefix_prompts=8,
        prefix_prompt_length=512,
        concurrency=concurrency,
        duration_minutes=duration_minutes,
        # 40s per observe NEXT_STEPS — last cycle's cliff p99 was ~21s and
        # the old 20s timeout truncated slow-but-real 503 responses, making
        # it impossible to tell "rejection fired slowly" from "didn't fire".
        request_timeout_seconds=40,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
        connection_reuse_strategy="never",
        goodput=["request_latency:20000"],
        worker_pin=worker_pin,
        request_cancellation_rate=request_cancellation_rate,
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.sanity
@pytest.mark.weekly  # Lifecycle category required by report_pytest_markers
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "concurrency,isl,duration_min",
    [
        # Default: cliff in ~90s at c=128 ISL=6000, hold 5 min.
        (128, 6000, 5),
    ],
)
async def test_overload_cascade_sanity_pinned(
    runtime_env, request, concurrency, isl, duration_min
):
    """S0a — pinned single-pod cascade sanity (~7 min after model load).

    Smallest, fastest reproduction of the per-pod cliff. Pins every
    request to ``VllmDecodeWorker#0`` + ``VllmPrefillWorker#0`` via
    ``nvext.worker_id`` (resolved at execute-time from the live
    ``DynamoWorkerMetadata`` CRs). No warmup — straight to overload so
    KV pegs within 2 min. The 5-min hold is enough to observe the
    cascade signature without burning GPU time.

    Pass = the per-pod cliff mechanism is reachable on this dev cluster.
    Failure = stop and debug before spending budget on the full S1 demo.
    """
    cfg = request.config
    image = cfg.getoption("--image") or _CASCADE_IMAGE

    spec = _load_dgd("disagg_qwen3_30b_unit_prod")
    _apply_cascade_dgd(spec, image=image, units=1, abort_timeout_s=480)
    _apply_cluster_portability(spec)
    # Don't call _apply_router_config — its admission-threshold defaults
    # (0.85) would shed load before the cliff fires. The cascade plan
    # leaves them unset so the FE inherits v1.1.1's permissive-sentinel
    # defaults. Set only the router-mode + KV-events knobs directly.
    # Under pinning the routing mode is moot; least-loaded matches S1.
    spec["Frontend"].set_env_var("DYN_ROUTER_MODE", "least-loaded")
    spec["Frontend"].set_env_var("DYN_ROUTER_USE_KV_EVENTS", "false")

    model_cache_pvc = cfg.getoption("--model-cache-pvc")
    if model_cache_pvc:
        spec.enable_model_cache(model_cache_pvc)

    served_model = spec["VllmDecodeWorker"].model

    events = [
        WaitForModelReady(timeout=2400),
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=concurrency,
                duration_minutes=duration_min,
                name="cliff",
                isl=isl,
                worker_pin=WorkerPin(
                    decode_service="VllmDecodeWorker",
                    decode_replica_index=0,
                    prefill_service="VllmPrefillWorker",
                    prefill_replica_index=0,
                ),
            ),
            name="cliff",
        ),
        WaitForLoadCompletion(name="cliff"),
    ]

    checks = [
        LoadApplied(load_name="cliff", min_requests=100),
        LoadCompleted(name="cliff"),
        KvCacheUsagePeak(
            services=["VllmDecodeWorker"],
            threshold=0.95,
            within_seconds=180,
            load_name="cliff",
        ),
        WaitingForKVTransferExceeds(
            services=["VllmDecodeWorker"],
            threshold=50,
            load_name="cliff",
        ),
        NixlXferTimeMultiplied(
            services=["VllmDecodeWorker"],
            min_multiplier=3.0,
            warmup_seconds=30,
            cliff_load_name="cliff",
        ),
        GenerationThroughputDropped(
            services=["VllmDecodeWorker"],
            min_drop_frac=0.30,
            warmup_seconds=60,
            cliff_window_seconds=120,
            cliff_load_name="cliff",
        ),
        CliffContained(
            services=["VllmDecodeWorker"],
            pinned_service="VllmDecodeWorker",
            pinned_replica_index=0,
            kv_threshold=0.85,
            peer_max_kv=0.50,
            load_name="cliff",
        ),
        # Prefill containment — the real pinning test at units=1. There
        # are 2 prefill replicas; pinning to #0 should leave #1 with
        # zero requests-running peak. If peer prefill ever saw any
        # in-flight requests, the nvext.worker_id pin leaked.
        PinningContained(
            services=["VllmPrefillWorker"],
            pinned_service="VllmPrefillWorker",
            pinned_replica_index=0,
            metric="vllm:num_requests_running",
            active_threshold=1.0,
            peer_max=0.0,
            load_name="cliff",
        ),
        # Decode containment via the same primitive — redundant with the
        # CliffContained KV-pressure assertion at units=1 (single
        # replica), but the assertion shape future-proofs the test if a
        # follow-up scenario raises decode replicas while keeping the pin.
        PinningContained(
            services=["VllmDecodeWorker"],
            pinned_service="VllmDecodeWorker",
            pinned_replica_index=0,
            metric="vllm:num_requests_running",
            active_threshold=1.0,
            peer_max=0.0,
            load_name="cliff",
        ),
        WorkerPanics(
            services=["VllmDecodeWorker", "VllmPrefillWorker", "Frontend"],
            acceptable=True,
        ),
        RestartCountIncreased(
            services=["VllmDecodeWorker", "Frontend"],
            expect_min_increment=0,
        ),
    ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=checks,
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ─── S0b — natural-routing minimum sanity (~10 min after model load) ───
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.sanity
@pytest.mark.weekly  # Lifecycle category required by report_pytest_markers
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "units,warmup_c,cliff_c,duration_min",
    [
        # Default: 1-min warmup @ c=64 + 8-min hold @ c=160 on N=2 units
        # (2F+4P+2D = 12 GPUs). c=160 across 2 decodes ≈ 80 concurrent per
        # decode, slightly above S1's per-decode ~72 to compress wall time.
        (2, 64, 160, 8),
    ],
)
async def test_overload_cascade_sanity_natural(
    runtime_env, request, units, warmup_c, cliff_c, duration_min
):
    """S0b — natural-routing minimum-topology cascade sanity (~10 min).

    Smallest topology with multiple decodes + multiple FEs (N=2) that
    proves natural LeastLoaded routing reaches the cliff regime on this
    dev cluster. Compressed: 1-min warmup, 8-min cliff hold.

    Pass = at least one decode pegs KV ≥ 0.95 under natural routing.
    Full peer-decode infection is NOT asserted at this duration — the
    LeastLoaded cross-FE blind spot may not propagate in 8 min, and
    that's fine for sanity. CliffPropagated is S1's job.
    """
    cfg = request.config
    image = cfg.getoption("--image") or _CASCADE_IMAGE

    spec = _load_dgd("disagg_qwen3_30b_unit_prod")
    _apply_cascade_dgd(spec, image=image, units=units, abort_timeout_s=480)
    _apply_cluster_portability(spec)
    # Same router knob set as S0a — natural LeastLoaded routing, no
    # admission thresholds, no KV events. Distinguishes S0a (pinned) from
    # S0b: only the absence of WorkerPin on the load is different.
    spec["Frontend"].set_env_var("DYN_ROUTER_MODE", "least-loaded")
    spec["Frontend"].set_env_var("DYN_ROUTER_USE_KV_EVENTS", "false")

    model_cache_pvc = cfg.getoption("--model-cache-pvc")
    if model_cache_pvc:
        spec.enable_model_cache(model_cache_pvc)

    served_model = spec["VllmDecodeWorker"].model

    events = [
        WaitForModelReady(timeout=2400),
        # Sub-cliff warmup rung — proves the cluster is healthy under
        # bounded load and establishes a baseline window for
        # NixlXferTimeMultiplied.
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=warmup_c,
                duration_minutes=1,
                name="warmup",
            ),
            name="warmup",
        ),
        WaitForLoadCompletion(name="warmup"),
        Wait(duration=15),
        # Cliff rung — natural routing decides which decode tips first.
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=cliff_c,
                duration_minutes=duration_min,
                name="cliff",
            ),
            name="cliff",
        ),
        WaitForLoadCompletion(name="cliff"),
    ]

    checks = [
        LoadApplied(load_name="cliff", min_requests=200),
        LoadCompleted(name="cliff"),
        # At least one decode must peg KV — but we don't assert WHICH one
        # (natural routing decides). within_seconds=240 measured from
        # cliff-rung start (KvCacheUsagePeak uses per-pod first-sample-of-load).
        KvCacheUsagePeak(
            services=["VllmDecodeWorker"],
            threshold=0.95,
            within_seconds=240,
            load_name="cliff",
        ),
        WaitingForKVTransferExceeds(
            services=["VllmDecodeWorker"],
            threshold=100,
            load_name="cliff",
        ),
        NixlXferTimeMultiplied(
            services=["VllmDecodeWorker"],
            min_multiplier=3.0,
            warmup_seconds=60,  # 1-min warmup rung window
            cliff_load_name="cliff",
        ),
        # NOTE: deliberately omitting CliffPropagated — at 8 min the
        # cross-FE blind spot may not have propagated yet. Sanity passes
        # if ONE decode cliffs; full propagation is S1's bar.
        WorkerPanics(
            services=["VllmDecodeWorker", "VllmPrefillWorker", "Frontend"],
            acceptable=True,
        ),
        RestartCountIncreased(
            services=["VllmDecodeWorker", "Frontend"],
            expect_min_increment=0,
        ),
    ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=checks,
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ─── Cascade fail-fast spike (S1 of 2026-05-26 fail-fast handoff) ──────
#
# Goal: prove the DIS-2105 TCP `work_tx.try_send()` rejection path
# actually surfaces an HTTP 503 (or `TrySendError::Full` worker log
# line) under a c=300 instant burst, no warmup. The prior cycle ran
# POOL=128 / QUEUE=32 and saw zero 503s in any log — inflight pegged at
# 128, suggesting the cap engaged silently. This test asserts the
# observable side of the cap.
#
# Launcher MUST pass the DIS-2105 patched image:
#   --image nvcr.io/nvidian/dynamo-dev/<...>/vllm-runtime:dis-2105-v1.1.1-<sha>-...
# (Hardcoding the image would couple the test to a specific build; the
# fallback ``_DEFAULT_IMAGE`` is stock 1.1.1 and will pass-fail this
# test as a control arm.)
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.sanity
@pytest.mark.weekly  # Lifecycle category required by report_pytest_markers
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "concurrency,duration_min",
    [
        # Default: c=300 instant burst, ~60s hold. Single rung, no
        # warmup ramp — this is a spike-test, not a steady-state
        # cascade.
        (300, 1.0),
    ],
)
async def test_overload_cascade_spike(runtime_env, request, concurrency, duration_min):
    """Cascade fail-fast S1 spike — DIS-2105 rejection-path surfacing.

    N=2 disagg topology, c=300 instant burst with no warmup. The TCP
    worker pool defaults (overrideable via ``DYN_TEST_WORKER_KNOBS``)
    are ``DYN_TCP_WORKER_POOL_SIZE=128, DYN_TCP_WORK_QUEUE_SIZE=32`` —
    matching the handoff's S1 row. Pass if ANY 503-rejection signal
    fires during the burst (AIPerf 503 OR worker
    ``TrySendError::Full`` log line OR Frontend ``Status code: 503``
    filtered to the load window).

    The launcher is responsible for passing the DIS-2105 patched image
    via ``--image``. Without it, ``RejectionFired`` will fail — by
    design, since that's the whole point of the experiment.
    """
    cfg = request.config
    image = cfg.getoption("--image") or _CASCADE_IMAGE

    spec = _load_dgd("disagg_qwen3_30b_unit_prod")
    # Default knobs for the S1 spike. The env var still wins so an
    # operator can sweep POOL/QUEUE values without forking the test.
    if not os.environ.get("DYN_TEST_WORKER_KNOBS", "").strip():
        os.environ[
            "DYN_TEST_WORKER_KNOBS"
        ] = "DYN_TCP_WORKER_POOL_SIZE=128;DYN_TCP_WORK_QUEUE_SIZE=32"
    # Optional vLLM extra args (e.g. --max-num-seqs=128 to bound running
    # batch). Semicolon-separated FLAG=VALUE pairs in DYN_TEST_VLLM_EXTRA_ARGS.
    # Lone flags (no value) are also supported as a bare flag with no =.
    extra_raw = os.environ.get("DYN_TEST_VLLM_EXTRA_ARGS", "").strip()
    vllm_extra_args: list[str] | None = None
    if extra_raw:
        vllm_extra_args = []
        for tok in extra_raw.split(";"):
            tok = tok.strip()
            if not tok:
                continue
            if "=" in tok:
                k, v = tok.split("=", 1)
                vllm_extra_args.extend([k.strip(), v.strip()])
            else:
                vllm_extra_args.append(tok)
    _apply_cascade_dgd(
        spec,
        image=image,
        units=2,
        abort_timeout_s=480,
        vllm_extra_args=vllm_extra_args,
    )
    _apply_cluster_portability(spec)
    spec["Frontend"].set_env_var("DYN_ROUTER_MODE", "least-loaded")
    spec["Frontend"].set_env_var("DYN_ROUTER_USE_KV_EVENTS", "false")

    model_cache_pvc = cfg.getoption("--model-cache-pvc")
    if model_cache_pvc:
        spec.enable_model_cache(model_cache_pvc)

    served_model = spec["VllmDecodeWorker"].model

    events = [
        WaitForModelReady(timeout=2400),
        # Single rung — instant burst, no warmup, no ramp.
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=concurrency,
                duration_minutes=duration_min,
                name="cliff",
            ),
            name="cliff",
        ),
        WaitForLoadCompletion(name="cliff"),
    ]

    checks = [
        # Sanity — confirm the burst actually got past the proxy and
        # SOME load made it to the cluster. min_requests is small
        # because rejection is the expected path; we still want to know
        # the load harness didn't silently no-op.
        LoadApplied(load_name="cliff", min_requests=10),
        # The point of the test — at least one of three rejection
        # signals must fire during the cliff window.
        RejectionFired(load_name="cliff", min_503_count=1),
    ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=checks,
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ─── S1 — full natural-routing cascade demonstration (~75 min) ─────────
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly  # not sanity — long-running
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "units,concurrency,duration_min,abort_timeout,vllm_max_num_seqs,override_worker_knobs",
    [
        # Default: production-mirror — N=3, c=216, 45-min hold, 480s NIXL timer.
        (3, 216, 45, 480, None, None),
        # S2 (2026-05-26 cascade fail-fast handoff): cap engine running-batch
        # at 128 via vLLM CLI. No DIS-2105 image required — stock 1.1.1
        # plus --max-num-seqs=128 alone is the lever. Pass if goodput
        # holds ≥ 50% of pre-cliff OR KV peak < 0.95 (assertions live on
        # the existing cascade-signature checks; the row exercises the
        # cap).
        (3, 216, 45, 480, 128, None),
        # S3 combo: DIS-2105 image + --max-num-seqs=128 + smaller
        # POOL/QUEUE knobs (64/16, half the S1-spike values) so the TCP
        # rejection path engages alongside the engine-side cap. Per the
        # handoff: launcher MUST pass --image
        #   nvcr.io/.../vllm-runtime:dis-2105-v1.1.1-...
        # for this row to do anything meaningful — the parametrize
        # itself only carries the knob deltas, not the image.
        (
            3,
            216,
            45,
            480,
            128,
            "DYN_TCP_WORKER_POOL_SIZE=64;DYN_TCP_WORK_QUEUE_SIZE=16",
        ),
        # Subsequent rows are entire S2-S6 scenarios (same test function);
        # add them as the campaign expands. Each row keeps the framework
        # primitives identical and only varies the cascade lever:
        # (3, 216, 45, 30, None, None),   # S4: short abort-timer
    ],
)
async def test_overload_cascade(
    runtime_env,
    request,
    units,
    concurrency,
    duration_min,
    abort_timeout,
    vllm_max_num_seqs,
    override_worker_knobs,
):
    """S1 — full cascade demonstration under natural LeastLoaded routing.

    Drives sustained load past the per-pod cliff on a production-shape N=3
    topology and asserts the full cascade signature: KV peg →
    WaitingForKVTransfer pool explosion → NIXL xfer_time blow-up →
    generation throughput collapse → cliff propagates to a second decode
    pod within the run window.

    Admission thresholds are deliberately UNSET so the FE inherits
    v1.1.1's permissive-sentinel defaults (matches production). Setting
    them is the mitigation tested in S3 (future row); this scenario is
    the unmitigated baseline.

    ~75 min wall (setup 15 + load 50 + teardown 10) on 18 GPUs.
    Marked @weekly — not part of the sanity-CI lane.
    """
    cfg = request.config
    image = cfg.getoption("--image") or _CASCADE_IMAGE

    spec = _load_dgd("disagg_qwen3_30b_unit_prod")
    vllm_extra_args = (
        ["--max-num-seqs", str(vllm_max_num_seqs)]
        if vllm_max_num_seqs is not None
        else None
    )
    _apply_cascade_dgd(
        spec,
        image=image,
        units=units,
        abort_timeout_s=abort_timeout,
        vllm_extra_args=vllm_extra_args,
        override_worker_knobs=override_worker_knobs,
    )
    _apply_cluster_portability(spec)
    spec["Frontend"].set_env_var("DYN_ROUTER_MODE", "least-loaded")
    spec["Frontend"].set_env_var("DYN_ROUTER_USE_KV_EVENTS", "false")

    model_cache_pvc = cfg.getoption("--model-cache-pvc")
    if model_cache_pvc:
        spec.enable_model_cache(model_cache_pvc)

    served_model = spec["VllmDecodeWorker"].model

    events = [
        WaitForModelReady(timeout=2400),
        # 5-min sub-cliff warmup at c=72 — sets the steady-state baseline
        # for NixlXferTimeMultiplied and proves the cluster is healthy
        # before the cliff rung. c=72 = 24 concurrent per decode at N=3,
        # comfortably below the per-pod cliff.
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=72,
                duration_minutes=5,
                name="warmup",
            ),
            name="warmup",
        ),
        WaitForLoadCompletion(name="warmup"),
        Wait(duration=30),
        # Cliff rung — the parameterized cascade target. 5% mid-flight
        # cancellations feed the `_DeferredAbort` path so the 480-s NIXL
        # abort timer actually fires during the 45-min hold. Without
        # these, no requests cancel → no stranded prefill blocks → the
        # primary production-recovery mechanism never exercises in dev.
        # Per observe-agent NEXT_STEPS P0.2 (sanity-suite work).
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=concurrency,
                duration_minutes=duration_min,
                name="cliff",
                request_cancellation_rate=0.05,
            ),
            name="cliff",
        ),
        WaitForLoadCompletion(name="cliff"),
    ]

    checks = [
        LoadApplied(load_name="cliff", min_requests=100),
        LoadCompleted(name="cliff"),
        # Cascade-signature sub-checks at S1 thresholds. Inlined rather
        # than wrapped in a CascadeReached composite so a single failure
        # gives a clean stack trace for the specific signal that missed.
        KvCacheUsagePeak(
            services=["VllmDecodeWorker"],
            threshold=0.95,
            within_seconds=600,
            load_name="cliff",
        ),
        WaitingForKVTransferExceeds(
            services=["VllmDecodeWorker"],
            threshold=100,
            load_name="cliff",
        ),
        NixlXferTimeMultiplied(
            services=["VllmDecodeWorker"],
            min_multiplier=5.0,
            warmup_seconds=300,  # use the full 5-min warmup window
            cliff_load_name="cliff",
        ),
        GenerationThroughputDropped(
            services=["VllmDecodeWorker"],
            min_drop_frac=0.30,
            warmup_seconds=300,
            cliff_window_seconds=300,
            cliff_load_name="cliff",
        ),
        # Peer-decode infection — the LeastLoaded cross-FE blind spot
        # signature. min_pods=2 means at least 2 of 3 decodes peg KV.
        CliffPropagated(
            services=["VllmDecodeWorker"],
            min_pods=2,
            kv_threshold=0.85,
            load_name="cliff",
        ),
        WorkerPanics(
            services=["VllmDecodeWorker", "VllmPrefillWorker", "Frontend"],
            acceptable=True,
        ),
        RestartCountIncreased(
            services=["VllmDecodeWorker", "Frontend"],
            expect_min_increment=0,
        ),
    ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=checks,
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ─── KV-router memory stability — recommended-A4 config ────────────────
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly  # ~40 min wall — not in the sanity-CI lane
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "router_config,units,rung_c12_min,rung_c24_ramp_min,rung_c48_min,rung_c24_cool_min,rung_c12_cool_min",
    [
        # Default: N=2, U-shape concurrency ramp.
        #   c=12 (warm)  2 min
        #   c=24 (ramp)  3 min
        #   c=48 (peak) 18 min  ← most of the run, main slope signal
        #   c=24 (cool)  3 min  ← prove memory comes down with load
        #   c=12 (cool)  2 min
        # = 28 min total load + ~5×30s gaps + ~7 min model load + ~2 min
        # teardown ≈ 40 min wall.
        #
        # Three router configs run head-to-head — same workload, same
        # topology, same image, only routing differs. Lets the observe
        # agent compare per-pod memory slopes side-by-side.
        ("a4-recommended", 2, 2.0, 3.0, 18.0, 3.0, 2.0),
        ("least-loaded-baseline", 2, 2.0, 3.0, 18.0, 3.0, 2.0),
        # Round-2 P0.1: A4 minus replica-sync. Single-variable
        # isolation of today's leak hypothesis. If FE slope drops to
        # LL territory (~2.6 MB/min) under this arm, replica-sync's
        # request-lifecycle accumulation (ActiveSequencesMultiWorker)
        # is confirmed as the dominant retention path.
        ("a4-no-replica-sync", 2, 2.0, 3.0, 18.0, 3.0, 2.0),
    ],
)
async def test_kv_router_memory_stability(
    runtime_env,
    request,
    router_config,
    units,
    rung_c12_min,
    rung_c24_ramp_min,
    rung_c48_min,
    rung_c24_cool_min,
    rung_c12_cool_min,
):
    """Memory-stability test for the recommended A4 KV-router config.

    Drives a U-shape concurrency ramp (12 → 24 → 48 → 24 → 12) under the
    canonical "recommended A4" Frontend config from the the cascade reproduction
    cascade analysis. Asserts that per-pod working-set memory
    grows at most ``ceiling`` MB/min over the run — i.e. that the
    recommended config does NOT leak.

    Why the U-shape: a leak shows as monotonic climb regardless of load
    direction. A healthy config climbs with concurrency on the way up
    AND comes back down on the cool-down. The single per-pod slope
    check (over the whole run window) catches the leak case; the
    aggregate-throughput report makes the steady-state shape visible
    for review.

    Workload: production-shaped ``_PROD_SEQ_DIST`` (long-tail ISL/OSL
    with prefix overlap) so the KV-router actually exercises its
    sequence-state plumbing under realistic prefill / decode mix.
    """
    cfg = request.config
    image = cfg.getoption("--image") or _DEFAULT_IMAGE

    spec = _load_dgd(dgd)
    _apply_topology(spec, units=units, fe=None, pf=None, dec=None)
    for svc in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].image = image
    _apply_cluster_portability(spec)

    # Branch on the parametrize axis: A4 recommended (KV-aware + cross-FE
    # sync + admission shedding at 0.85) vs LeastLoaded baseline (the
    # unmitigated production arm — same admission thresholds so the only
    # variable is the routing-mode-derived load model).
    if router_config == "a4-recommended":
        _apply_recommended_kv_router_config(spec)
    elif router_config == "least-loaded-baseline":
        fe = spec["Frontend"]
        fe.set_env_var("DYN_ROUTER_MODE", "least-loaded")
        fe.set_env_var("DYN_ROUTER_USE_KV_EVENTS", "false")
        # Keep admission shedding identical to A4 so the only behavioral
        # diff between arms is the routing-mode-derived load model.
        fe.set_env_var("DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD", "0.85")
        fe.set_env_var("DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC", "0.85")
    elif router_config == "a4-no-replica-sync":
        # Single-variable isolation: A4 recommended config with
        # replica-sync flipped off. Everything else identical to the
        # "a4-recommended" arm so the slope delta cleanly attributes
        # to ActiveSequencesMultiWorker's request-lifecycle accounting.
        # Per observe-agent NEXT_STEPS round-2 P0.1.
        _apply_recommended_kv_router_config(spec)
        spec["Frontend"].set_env_var("DYN_ROUTER_REPLICA_SYNC", "false")
    else:
        raise ValueError(f"unknown router_config: {router_config!r}")

    # Per-launch FE env-var override. Mirrors the cascade-side
    # DYN_TEST_WORKER_KNOBS pattern but for Frontend so observe's R2
    # toggle-sweep can drive single-knob isolations from the launch
    # env without forking the test function per toggle. Format:
    # semicolon-separated KEY=VALUE list, empty → no overrides.
    # Example: DYN_TEST_FE_KNOBS="DYN_ROUTER_USE_KV_EVENTS=true"
    fe_extra = os.environ.get("DYN_TEST_FE_KNOBS", "").strip()
    if fe_extra:
        for kv in fe_extra.split(";"):
            kv = kv.strip()
            if not kv or "=" not in kv:
                continue
            key, val = kv.split("=", 1)
            spec["Frontend"].set_env_var(key.strip(), val.strip())

    model_cache_pvc = cfg.getoption("--model-cache-pvc")
    if model_cache_pvc:
        spec.enable_model_cache(model_cache_pvc)

    served_model = spec["VllmDecodeWorker"].model

    def _steady_load(concurrency: float, duration_minutes: float, name: str):
        """Prod-mix LoadConfig for a single rung of the U-shape."""
        return LoadConfig(
            model_name=served_model,
            tokenizer=served_model,
            seq_dist=_PROD_SEQ_DIST,
            num_prefix_prompts=_NUM_PREFIX_PROMPTS,
            prefix_prompt_length=_PREFIX_PROMPT_LENGTH,
            concurrency=int(concurrency),
            duration_minutes=duration_minutes,
            request_timeout_seconds=30,
            streaming=True,
            ignore_eos=True,
            warmup_requests=0,
            connection_reuse_strategy="never",
            goodput=["request_latency:30000"],
        )

    events = [
        WaitForModelReady(timeout=2400),
        # Memory poller installed right after model ready so we don't
        # pick up loader/init memory in the slope. interval_s=10 gives
        # ~180 samples over a 30-min steady window — smooth slope, low
        # kubectl-exec overhead.
        PodMemoryPoller(
            services=["Frontend", "VllmPrefillWorker", "VllmDecodeWorker"],
            interval_s=10,
        ),
        StartLoad(load_config=_steady_load(12, rung_c12_min, "warmup"), name="warmup"),
        WaitForLoadCompletion(name="warmup"),
        StartLoad(load_config=_steady_load(24, rung_c24_ramp_min, "ramp"), name="ramp"),
        WaitForLoadCompletion(name="ramp"),
        StartLoad(load_config=_steady_load(48, rung_c48_min, "steady"), name="steady"),
        WaitForLoadCompletion(name="steady"),
        StartLoad(load_config=_steady_load(24, rung_c24_cool_min, "cool"), name="cool"),
        WaitForLoadCompletion(name="cool"),
        StartLoad(
            load_config=_steady_load(12, rung_c12_cool_min, "final"), name="final"
        ),
        WaitForLoadCompletion(name="final"),
    ]

    # Pass criteria:
    #   - Each rung issued enough requests to be meaningful (no AIPerf
    #     silent failures)
    #   - FE working-set growth ≤ 30 MB/min over the run window
    #   - Workers ≤ 100 MB/min (vLLM internal caches are noisier)
    #   - Zero restarts (a restart invalidates the slope)
    #   - No worker panics
    checks = [
        LoadApplied(load_name="warmup", min_requests=50),
        LoadApplied(load_name="ramp", min_requests=100),
        LoadApplied(load_name="steady", min_requests=500),
        LoadApplied(load_name="cool", min_requests=100),
        LoadApplied(load_name="final", min_requests=50),
        # window_seconds=2400 covers the entire ~30-min load span; the
        # check's sliding-window algorithm picks the longest valid pair
        # from each start point, so a 2-min startup ramp gets diluted
        # rather than tripping the ceiling. A real leak shows as
        # monotonic climb across the whole run → high overall slope.
        PodMemoryGrowth(
            services=["Frontend"],
            assert_mode="max",
            growth_bytes_per_min=30 * 1024 * 1024,  # 30 MiB/min
            window_seconds=2400,
            source="working_set",
        ),
        PodMemoryGrowth(
            services=["VllmDecodeWorker", "VllmPrefillWorker"],
            assert_mode="max",
            growth_bytes_per_min=100 * 1024 * 1024,  # 100 MiB/min
            window_seconds=2400,
            source="working_set",
        ),
        RestartCountIncreased(
            services=["Frontend", "VllmPrefillWorker", "VllmDecodeWorker"],
            expect_min_increment=0,
        ),
        WorkerPanics(
            services=["VllmDecodeWorker", "VllmPrefillWorker", "Frontend"],
            acceptable=True,
        ),
    ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=checks,
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ─── S2 / S0c — pinned single-pod overload with natural-bg traffic ─────
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "units,pin_c,pin_isl,pin_min,bg_c,bg_min",
    [
        # Default: N=2 (need ≥1 peer decode to observe propagation).
        # Pin Load A onto (decode#0, prefill#0) at c=64 ISL=6000 for 10
        # min — that pair will cliff. Concurrent Load B is natural-
        # routing c=32 ISL=2000 for 10 min — exercises peers, MAY split
        # some traffic onto the saturated decode#0 if the routing layer
        # is blind to its overload (the the cross-FE blind spot).
        (2, 64, 6000, 10.0, 32, 10.0),
    ],
)
async def test_overload_cascade_pinned_with_bg(
    runtime_env,
    request,
    units,
    pin_c,
    pin_isl,
    pin_min,
    bg_c,
    bg_min,
):
    """Pinned cliff + natural-bg traffic — cross-FE blind-spot diagnostic.

    Drives ONE decode pod into the cliff via nvext.worker_id pinning
    while simultaneously running a moderate, natural-routing background
    load against the rest of the pool. The diagnostic question is:
    does the cliff stay contained on the pinned target, or does it
    propagate to the peer decode because the FE's load model can't
    see the saturation?

    Two concurrent ``StartLoad`` events. Both AIPerf Jobs share the
    log PVC but write to subpath-isolated dirs (each ``ManagedLoad``
    generates a unique 8-hex job-name suffix). Each load is extracted
    independently at scenario teardown.

    Asserts only on framework health (loads ran, cliff fired on the
    pinned target, no panics, no restarts). The propagation-vs-
    containment classification is left to post-hoc analysis of the
    per-pod ``server_metrics_export.jsonl`` for both loads — both
    arms (LeastLoaded blind-spot vs A4 cross-FE visibility) yield
    different per-pod KV-cliff signatures that an analyst can read
    out of the JSONLs without re-running.
    """
    cfg = request.config
    image = cfg.getoption("--image") or _CASCADE_IMAGE

    spec = _load_dgd(dgd)
    _apply_topology(spec, units=units, fe=None, pf=None, dec=None)
    for svc in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].image = image
    _apply_cluster_portability(spec)

    # LeastLoaded routing with NO admission shedding — the unmitigated
    # baseline that matches the production arm. The cross-FE blind-spot
    # only manifests in this config. To probe the recommended-A4
    # remediation, swap to `_apply_recommended_kv_router_config(spec)`
    # in a future parametrize row.
    spec["Frontend"].set_env_var("DYN_ROUTER_MODE", "least-loaded")
    spec["Frontend"].set_env_var("DYN_ROUTER_USE_KV_EVENTS", "false")

    # NIXL abort timer: 30s mitigation (default 480 implicated in prior outage).
    for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].set_env_var("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", "30")

    model_cache_pvc = cfg.getoption("--model-cache-pvc")
    if model_cache_pvc:
        spec.enable_model_cache(model_cache_pvc)

    served_model = spec["VllmDecodeWorker"].model

    events = [
        WaitForModelReady(timeout=2400),
        # Load A: pinned cliff. Concentrated on (decode#0, prefill#0).
        # Starts first; AIPerf is up + scraping before bg load joins.
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=pin_c,
                duration_minutes=pin_min,
                name="cliff",
                isl=pin_isl,
                worker_pin=WorkerPin(
                    decode_service="VllmDecodeWorker",
                    decode_replica_index=0,
                    prefill_service="VllmPrefillWorker",
                    prefill_replica_index=0,
                ),
            ),
            name="cliff",
        ),
        # Load B: natural background. Starts right after A (no Wait
        # in between — both AIPerf Jobs run concurrently from this
        # point until each completes).
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=bg_c,
                duration_minutes=bg_min,
                name="bg",
                isl=2000,
            ),
            name="bg",
        ),
        # Wait for both loads to finish independently.
        WaitForLoadCompletion(name="cliff"),
        WaitForLoadCompletion(name="bg"),
    ]

    checks = [
        # Framework-health asserts only.
        LoadApplied(load_name="cliff", min_requests=50),
        LoadApplied(load_name="bg", min_requests=100),
        LoadCompleted(name="cliff"),
        LoadCompleted(name="bg"),
        # Sanity that the pin actually concentrated load on decode#0
        # (KV peg = pin worked + cluster is in the cliff regime). The
        # observe agent reads per-pod KV / W4RK / NIXL from the JSONL
        # for both loads to classify whether the cliff stayed contained
        # or propagated to the peer decode.
        KvCacheUsagePeak(
            services=["VllmDecodeWorker"],
            threshold=0.95,
            within_seconds=600,
            load_name="cliff",
        ),
        WorkerPanics(
            services=["VllmDecodeWorker", "VllmPrefillWorker", "Frontend"],
            acceptable=True,
        ),
        RestartCountIncreased(
            services=["VllmDecodeWorker", "Frontend"],
            expect_min_increment=0,
        ),
    ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=checks,
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ─── N=3 cascade-prevention chain-phased 4-arm demo ────────────────────
#
# 4 arms × 57-min cliff (cold-spike + imbalance + extended steady for memory).
# Per the 2026-05-26-n3-cascade-prevention-demo-handoff.md (revised). All
# arms use the prod-shape N=3 disagg topology (3 FE + 6 prefill + 3 decode).
#
#   A — LL no admission           v1.1.1 stock                 LeastLoaded, no admission knobs
#   B — LL + DIS-2105 rejection   v1.1.1 + 06a9efb537           LeastLoaded + pool=32, queue=8
#   C — KV no replica-sync        release/1.2.0                 KV zero-weight + queue=4.0, no replica sync
#   D — KV + replica-sync         release/1.2.0                 same as C + DYN_ROUTER_REPLICA_SYNC=true
#
# Launch via scripts/launch-n3-cascade-prevention-demo.sh which fans out
# to 4 namespaces and supplies per-arm --image.


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "arm_label",
    ["A-LL", "B-LL-DIS2105", "C-KV-no-sync", "D-KV-sync"],
)
async def test_overload_cascade_chain_phased(runtime_env, request, arm_label):
    """N=3 cluster-resilience demo across four admission strategies.

    Workload (57 min total):
      P-spike   1 min  idle → c=150 sudden  cold-onset (production-shape)
      P-recov   2 min  idle                  reset
      P0        2 min  c=24 unpinned         baseline + memory anchor
      P1        4 min  c=80 pinned to dec#0  imbalance under cap
      P2        3 min  P1 + c=24 unpinned    mixed-load below cliff
      P2.5      1 min  P1 + c=80 burst       spike-on-imbalance (total c=184)
      P3       40 min  P1 + c=24             extended steady — PRIMARY memory window
      P4        4 min  drop pin + c=24       recovery + final memory

    The pinned long load spans P1..P3 (48 min). Unpinned loads start/stop
    per phase. PodMemoryPoller captures pod_memory_growth.tsv at 10s.

    Per-arm knobs differ only in routing strategy + admission. Image
    selected via CLI ``--image`` at launch time (no per-arm hard-coded
    image — the launcher script supplies the right image per arm).
    """
    cfg = request.config
    image = cfg.getoption("--image") or _CASCADE_IMAGE

    spec = _load_dgd("disagg_qwen3_30b_unit_prod")

    # Per-arm routing + admission knob set
    vllm_extra_args = None
    fe_knobs: dict[str, str] = {}

    if arm_label == "A-LL":
        # Stock LL, no admission. The cascade-prone baseline.
        os.environ["DYN_TEST_WORKER_KNOBS"] = ""
        fe_knobs = {
            "DYN_ROUTER_MODE": "least-loaded",
            "DYN_ROUTER_USE_KV_EVENTS": "false",
        }
    elif arm_label == "B-LL-DIS2105":
        # LL + worker-side rejection (DIS-2105 fix). Image must be the
        # 06a9efb537-built v1.1.1 base.
        os.environ[
            "DYN_TEST_WORKER_KNOBS"
        ] = "DYN_TCP_WORKER_POOL_SIZE=32;DYN_TCP_WORK_QUEUE_SIZE=8"
        fe_knobs = {
            "DYN_ROUTER_MODE": "least-loaded",
            "DYN_ROUTER_USE_KV_EVENTS": "false",
        }
    elif arm_label == "C-KV-no-sync":
        # release/1.2.0 KV with zero scoring weights + FE queue threshold,
        # NO cross-FE replica sync.
        os.environ["DYN_TEST_WORKER_KNOBS"] = ""
        fe_knobs = {
            "DYN_ROUTER_MODE": "kv",
            "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT": "0.0",
            "DYN_KV_HOST_CACHE_HIT_WEIGHT": "0.0",
            "DYN_KV_DISK_CACHE_HIT_WEIGHT": "0.0",
            "DYN_ROUTER_USE_KV_EVENTS": "false",
            "DYN_ROUTER_REPLICA_SYNC": "false",
            "DYN_ROUTER_TEMPERATURE": "0.0",
            "DYN_ROUTER_ASSUME_KV_REUSE": "false",
            "DYN_ROUTER_TRACK_ACTIVE_BLOCKS": "true",
            "DYN_ROUTER_QUEUE_THRESHOLD": "4.0",
        }
    elif arm_label == "D-KV-sync":
        # Same as C but WITH replica-sync — the production-proposed config.
        os.environ["DYN_TEST_WORKER_KNOBS"] = ""
        fe_knobs = {
            "DYN_ROUTER_MODE": "kv",
            "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT": "0.0",
            "DYN_KV_HOST_CACHE_HIT_WEIGHT": "0.0",
            "DYN_KV_DISK_CACHE_HIT_WEIGHT": "0.0",
            "DYN_ROUTER_USE_KV_EVENTS": "false",
            "DYN_ROUTER_REPLICA_SYNC": "true",
            "DYN_ROUTER_TEMPERATURE": "0.0",
            "DYN_ROUTER_ASSUME_KV_REUSE": "false",
            "DYN_ROUTER_TRACK_ACTIVE_BLOCKS": "true",
            "DYN_ROUTER_QUEUE_THRESHOLD": "4.0",
        }
    else:
        raise ValueError(f"unknown arm_label: {arm_label!r}")

    _apply_cascade_dgd(
        spec,
        image=image,
        units=3,  # N=3 disagg
        abort_timeout_s=30,  # mitigation (default 480 implicated in prior outage)
        vllm_extra_args=vllm_extra_args,
    )
    _apply_cluster_portability(spec)

    # Apply per-arm FE knobs (overrides any default set by _apply_cascade_dgd)
    for k, v in fe_knobs.items():
        spec["Frontend"].set_env_var(k, v)

    model_cache_pvc = cfg.getoption("--model-cache-pvc")
    if model_cache_pvc:
        spec.enable_model_cache(model_cache_pvc)

    served_model = spec["VllmDecodeWorker"].model

    # Phase durations (minutes) per the 2026-05-26 handoff (revised)
    PSPIKE_MIN = 1.0
    PRECOV_MIN = 2.0
    P0_MIN = 2.0
    P1_MIN = 4.0
    P2_MIN = 3.0
    P2_5_MIN = 1.0
    P3_MIN = 40.0
    P4_MIN = 4.0

    # Pinned load duration: spans P1+P2+P2.5+P3 = 48 min
    PIN_DURATION_MIN = P1_MIN + P2_MIN + P2_5_MIN + P3_MIN

    pinned_workerpin = WorkerPin(
        decode_service="VllmDecodeWorker",
        decode_replica_index=0,
        prefill_service="VllmPrefillWorker",
        prefill_replica_index=0,
    )

    events = [
        WaitForModelReady(timeout=2400),
        # Continuous FE memory polling for the full 57-min run
        PodMemoryPoller(
            services=["Frontend", "VllmPrefillWorker", "VllmDecodeWorker"],
            interval_s=10,
        ),
        # P-spike — cold onset (instant idle → c=150)
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=150,
                duration_minutes=PSPIKE_MIN,
                name="p-spike",
            ),
            name="p-spike",
        ),
        WaitForLoadCompletion(name="p-spike"),
        # P-recover — idle reset
        Wait(duration=int(PRECOV_MIN * 60)),
        # P0 — warmup
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=24,
                duration_minutes=P0_MIN,
                name="p0-warmup",
            ),
            name="p0-warmup",
        ),
        WaitForLoadCompletion(name="p0-warmup"),
        # P1 starts: long pinned load (48 min — ends after P3)
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=80,
                duration_minutes=PIN_DURATION_MIN,
                name="pinned-imbalance",
                worker_pin=pinned_workerpin,
            ),
            name="pinned-imbalance",
        ),
        Wait(duration=int(P1_MIN * 60)),
        # P2 — add cross-traffic (synchronous with pinned in background)
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=24,
                duration_minutes=P2_MIN,
                name="p2-cross",
            ),
            name="p2-cross",
        ),
        WaitForLoadCompletion(name="p2-cross"),
        # P2.5 — spike burst on top of pinned
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=80,
                duration_minutes=P2_5_MIN,
                name="p2-5-spike",
            ),
            name="p2-5-spike",
        ),
        WaitForLoadCompletion(name="p2-5-spike"),
        # P3 — extended sustained mixed (40 min — memory window)
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=24,
                duration_minutes=P3_MIN,
                name="p3-sustained",
            ),
            name="p3-sustained",
        ),
        WaitForLoadCompletion(name="p3-sustained"),
        # Drain the pinned load (should be done; explicit for clarity)
        WaitForLoadCompletion(name="pinned-imbalance"),
        # P4 — recovery
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=24,
                duration_minutes=P4_MIN,
                name="p4-recovery",
            ),
            name="p4-recovery",
        ),
        WaitForLoadCompletion(name="p4-recovery"),
    ]

    # Framework-health asserts only. The per-arm A/B/C/D comparison per
    # the handoff is post-hoc — per-phase goodput, per-pod KV containment,
    # 503/Timeout/504 accounting, FE memory slope.
    checks = [
        LoadApplied(load_name="p0-warmup", min_requests=50),
        LoadApplied(load_name="pinned-imbalance", min_requests=10),
        LoadApplied(load_name="p3-sustained", min_requests=500),
        WorkerPanics(
            services=["VllmDecodeWorker", "VllmPrefillWorker", "Frontend"],
            acceptable=True,
        ),
        RestartCountIncreased(
            services=["VllmDecodeWorker", "Frontend"],
            expect_min_increment=0,
        ),
    ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=checks,
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )


# ─── N=3 NATURAL OVERLOAD — companion no-pin variant of chain-phased ───
#
# IDENTICAL to test_overload_cascade_chain_phased except: the long
# background load (P1..P3) has NO `WorkerPin`. Same 4 arms, same
# 57-min phase schedule, same images, same FE knobs. The only
# variable is whether nvext-pinning is applied to the long load.
#
# Direct A/B against the pin-based test: same workload shape and
# same routing config → any difference in cluster behavior between
# the two tests isolates the pin's effect.


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "arm_label",
    ["A-LL", "B-LL-DIS2105", "C-KV-no-sync", "D-KV-sync"],
)
async def test_overload_natural_overload(runtime_env, request, arm_label):
    """N=3 routing-balance demo across 4 arms — NO pinning.

    Clone of test_overload_cascade_chain_phased with `worker_pin=None`
    on the long background load. Everything else (phases, durations,
    concurrencies, knobs, image selection) is identical.

    Headline metric is the coefficient-of-variation of per-decode
    inflight across the 57-min run — computed post-hoc.
    """
    cfg = request.config
    image = cfg.getoption("--image") or _CASCADE_IMAGE

    # Per-arm knobs — IDENTICAL to chain_phased
    vllm_extra_args = None
    fe_knobs: dict[str, str] = {}

    if arm_label == "A-LL":
        os.environ["DYN_TEST_WORKER_KNOBS"] = ""
        fe_knobs = {
            "DYN_ROUTER_MODE": "least-loaded",
            "DYN_ROUTER_USE_KV_EVENTS": "false",
        }
    elif arm_label == "B-LL-DIS2105":
        os.environ[
            "DYN_TEST_WORKER_KNOBS"
        ] = "DYN_TCP_WORKER_POOL_SIZE=32;DYN_TCP_WORK_QUEUE_SIZE=8"
        fe_knobs = {
            "DYN_ROUTER_MODE": "least-loaded",
            "DYN_ROUTER_USE_KV_EVENTS": "false",
        }
    elif arm_label == "C-KV-no-sync":
        os.environ["DYN_TEST_WORKER_KNOBS"] = ""
        fe_knobs = {
            "DYN_ROUTER_MODE": "kv",
            "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT": "0.0",
            "DYN_KV_HOST_CACHE_HIT_WEIGHT": "0.0",
            "DYN_KV_DISK_CACHE_HIT_WEIGHT": "0.0",
            "DYN_ROUTER_USE_KV_EVENTS": "false",
            "DYN_ROUTER_REPLICA_SYNC": "false",
            "DYN_ROUTER_TEMPERATURE": "0.0",
            "DYN_ROUTER_ASSUME_KV_REUSE": "false",
            "DYN_ROUTER_TRACK_ACTIVE_BLOCKS": "true",
            "DYN_ROUTER_QUEUE_THRESHOLD": "4.0",
        }
    elif arm_label == "D-KV-sync":
        os.environ["DYN_TEST_WORKER_KNOBS"] = ""
        fe_knobs = {
            "DYN_ROUTER_MODE": "kv",
            "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT": "0.0",
            "DYN_KV_HOST_CACHE_HIT_WEIGHT": "0.0",
            "DYN_KV_DISK_CACHE_HIT_WEIGHT": "0.0",
            "DYN_ROUTER_USE_KV_EVENTS": "false",
            "DYN_ROUTER_REPLICA_SYNC": "true",
            "DYN_ROUTER_TEMPERATURE": "0.0",
            "DYN_ROUTER_ASSUME_KV_REUSE": "false",
            "DYN_ROUTER_TRACK_ACTIVE_BLOCKS": "true",
            "DYN_ROUTER_QUEUE_THRESHOLD": "4.0",
        }
    else:
        raise ValueError(f"unknown arm_label: {arm_label!r}")

    spec = _load_dgd("disagg_qwen3_30b_unit_prod")
    _apply_cascade_dgd(
        spec,
        image=image,
        units=3,
        abort_timeout_s=30,  # mitigation (default 480 implicated in prior outage)
        vllm_extra_args=vllm_extra_args,
    )
    _apply_cluster_portability(spec)
    for k, v in fe_knobs.items():
        spec["Frontend"].set_env_var(k, v)

    model_cache_pvc = cfg.getoption("--model-cache-pvc")
    if model_cache_pvc:
        spec.enable_model_cache(model_cache_pvc)

    served_model = spec["VllmDecodeWorker"].model

    # Same phase schedule as chain_phased
    PSPIKE_MIN, PRECOV_MIN = 1.0, 2.0
    P0_MIN, P1_MIN, P2_MIN, P2_5_MIN, P3_MIN, P4_MIN = 2.0, 4.0, 3.0, 1.0, 40.0, 4.0
    PIN_DURATION_MIN = P1_MIN + P2_MIN + P2_5_MIN + P3_MIN  # 48 min

    # NOTE: identical to chain_phased except worker_pin=None below.
    events = [
        WaitForModelReady(timeout=2400),
        PodMemoryPoller(
            services=["Frontend", "VllmPrefillWorker", "VllmDecodeWorker"],
            interval_s=10,
        ),
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=150,
                duration_minutes=PSPIKE_MIN,
                name="p-spike",
            ),
            name="p-spike",
        ),
        WaitForLoadCompletion(name="p-spike"),
        Wait(duration=int(PRECOV_MIN * 60)),
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=24,
                duration_minutes=P0_MIN,
                name="p0-warmup",
            ),
            name="p0-warmup",
        ),
        WaitForLoadCompletion(name="p0-warmup"),
        # The long background load — ** unpinned ** (vs chain_phased which pins this)
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=80,
                duration_minutes=PIN_DURATION_MIN,
                name="long-background",
                worker_pin=None,
            ),
            name="long-background",
        ),
        Wait(duration=int(P1_MIN * 60)),
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=24,
                duration_minutes=P2_MIN,
                name="p2-cross",
            ),
            name="p2-cross",
        ),
        WaitForLoadCompletion(name="p2-cross"),
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=80,
                duration_minutes=P2_5_MIN,
                name="p2-5-spike",
            ),
            name="p2-5-spike",
        ),
        WaitForLoadCompletion(name="p2-5-spike"),
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=24,
                duration_minutes=P3_MIN,
                name="p3-sustained",
            ),
            name="p3-sustained",
        ),
        WaitForLoadCompletion(name="p3-sustained"),
        WaitForLoadCompletion(name="long-background"),
        StartLoad(
            load_config=_cascade_load(
                served_model=served_model,
                concurrency=24,
                duration_minutes=P4_MIN,
                name="p4-recovery",
            ),
            name="p4-recovery",
        ),
        WaitForLoadCompletion(name="p4-recovery"),
    ]

    checks = [
        LoadApplied(load_name="p0-warmup", min_requests=50),
        LoadApplied(load_name="long-background", min_requests=10),
        LoadApplied(load_name="p3-sustained", min_requests=500),
        WorkerPanics(
            services=["VllmDecodeWorker", "VllmPrefillWorker", "Frontend"],
            acceptable=True,
        ),
        RestartCountIncreased(
            services=["VllmDecodeWorker", "Frontend"],
            expect_min_increment=0,
        ),
    ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=checks,
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
