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

# ─── DGD ──────────────────────────────────────────────────────────────
# The seed DGD all scenarios use. To run against a different DGD, either
# edit this line or override via `--dgd <name>` on the CLI.
dgd = "disagg_qwen3_30b_unit_prod"

_DEFAULT_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1"

# Prod-shaped traffic distribution (ISL/OSL pairs, from the 2026-05-16
# reproducer). Same as the n3 routing-threshold test.
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

    Defaults match the recipe from the 2026-05-16 cascade-reproducer doc
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
    aks-dev (A100-80GB) and aws-dev-02 (H100-80GB) shared clusters."""
    # Pin Frontend to A100/H100 node pool with the GPU toleration so the
    # operator's default CPU scheduling doesn't land it on a small node.
    fe = spec["Frontend"]._spec.setdefault("extraPodSpec", {})
    fe.setdefault(
        "tolerations",
        [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
        ],
    )

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
    _apply_router_config(spec)

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
            LoadCompleted(load_name=final),
            _PerPodKvImbalanceGap(services=["VllmDecodeWorker"]),
            # Report-only goodput / SLA observations.
            SlaViolation(
                services=["VllmDecodeWorker"], load_name=final, acceptable=True
            ),
            ServiceLogPatternRate(
                services=["Frontend"],
                pattern=r"504|GatewayTimeout|inter.token.latency",
                acceptable=True,
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
