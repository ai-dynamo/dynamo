# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Worker death mid-KV-transfer — reliable cascade reproduction.

Forces a prefill peer to disappear while it is in the middle of NIXL
KV transfers to the decode worker, then measures the cascade. A
`DeletePod(VllmPrefillWorker, force=True)` mid-transfer with sustained
heavy-prefill load drives the paired decode worker into engine death
— UCX `ucp_requests` pool warnings followed by many `raise output`
exceptions in vLLM's `generate_tokens` / `_abort_monitor` path. Same
NIXL-peer-disappearance class as a prefill-process exit under load.

The forcing function is **ISL=7000 tokens**: heavy prefill drives big
prefill→decode KV transfers continuously, so killing a prefill mid-NIXL
is the common case rather than a race.

Scoped to the smallest meaningful disagg topology (units=1 → 1 FE + 2
PF + 1 DE) so the cascade signal is observable on a 5-GPU footprint.
"""

import logging

import pytest

from tests.fault_tolerance.deploy.checks import MaxErrors, MinRequests, WorkerPanics
from tests.fault_tolerance.deploy.events import (
    DeletePod,
    StartLoad,
    StopLoad,
    Wait,
    WaitForModelReady,
    WaitForRecovery,
)
from tests.fault_tolerance.deploy.reports import (
    ErrorBreakdownReport,
    FaultToleranceReport,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.fault_tolerance.deploy.test_overload import (
    _apply_cluster_portability,
    _apply_router_config,
    _apply_topology,
    _load_dgd,
)
from tests.utils.managed_load import LoadConfig

logger = logging.getLogger(__name__)


# DGD template shared with test_overload — production-shape disagg unit.
_DGD = "disagg_qwen3_30b_unit_prod"

# Forcing-function workload: heavy prefill drives continuous prefill→decode
# KV transfers. ISL=7000 / OSL=100 is the heavy-prefill / light-decode
# regime that keeps NIXL transfer pipelines saturated end-to-end.
_ISL_TOKENS = 7000
_OSL_TOKENS = 100

# Smallest meaningful disagg footprint for parametrized service-kill.
# 1 FE + 2 PF (TP=2) + 1 DE (TP=2) = 4 pods, 5 GPUs.
# Two prefills lets us kill one and keep one peer serving so the
# kill-versus-recover dynamic is observable.
_DEFAULT_UNITS = 1
_DEFAULT_FE = 1
_DEFAULT_PF = 2
_DEFAULT_DE = 1

# Saturation point chosen so the decode hits KV=1.00 with ~c in-flight
# requests at the moment of the kill — every one holds live NIXL block
# references back to the prefills, so killing one prefill maximises the
# lost-reference set.
_DEFAULT_CONCURRENCY = 96

# Closed-loop ramp 4→c over 600s, then sustain 5 min, then kill
# mid-sustain. Default total load duration = ramp + pre-kill sustain
# + post-kill recovery soak.
_DEFAULT_RAMP_S = 600
_DEFAULT_WARMUP_C = 4
_DEFAULT_PRE_KILL_SUSTAIN_S = 300  # 5-min sustain before the kill
_DEFAULT_POST_KILL_RECOVERY_S = 300  # observation + recovery soak
_LOAD_DURATION_MIN = (
    _DEFAULT_RAMP_S + _DEFAULT_PRE_KILL_SUSTAIN_S + _DEFAULT_POST_KILL_RECOVERY_S
) / 60


def _kv_transfer_load(
    *,
    served_model: str,
    concurrency: int,
    duration_minutes: float,
    ramp_duration_s: float = 0,
    warmup_concurrency: int = 0,
) -> LoadConfig:
    """Fixed-ISL load that maximises prefill→decode KV transfer volume.

    Distinct from `_prod_load` (test_overload): no prefix-cache shaping
    (uniform prompts → bigger NIXL transfers), no `seq_dist` (fixed ISL
    means every request hits the worst case).

    When `ramp_duration_s > 0`, AIPerf ramps closed-loop concurrency
    from `warmup_concurrency` (or 1) up to `concurrency` over
    `ramp_duration_s` seconds — drives KV-saturation buildup before
    the kill so the lost-reference set is large.
    """
    return LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        input_tokens_mean=_ISL_TOKENS,
        input_tokens_stddev=200,
        output_tokens_mean=_OSL_TOKENS,
        concurrency=concurrency,
        duration_minutes=duration_minutes,
        request_timeout_seconds=120,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
        connection_reuse_strategy="never",
        concurrency_ramp_duration=ramp_duration_s if ramp_duration_s > 0 else None,
        warmup_concurrency=warmup_concurrency if warmup_concurrency > 0 else None,
        warmup_duration=60 if warmup_concurrency > 0 else None,
    )


def add_cli_options(parser):
    g = parser.getgroup(
        "worker-death-kv-transfer", "Worker death mid-KV-transfer scenario"
    )
    # Note: --units is registered by test_overload (shared topology flag).
    g.addoption(
        "--wd-concurrency",
        type=int,
        default=_DEFAULT_CONCURRENCY,
        help=f"Concurrency during fault window. Default: {_DEFAULT_CONCURRENCY}",
    )
    g.addoption(
        "--wd-isl-tokens",
        type=int,
        default=_ISL_TOKENS,
        help=f"Mean input tokens. Default: {_ISL_TOKENS} (forces heavy KV transfer)",
    )
    g.addoption(
        "--wd-target-service",
        default="VllmPrefillWorker",
        choices=["VllmPrefillWorker", "VllmDecodeWorker", "Frontend"],
        help="Which service to kill mid-transfer. Default: VllmPrefillWorker",
    )
    g.addoption(
        "--wd-ramp-s",
        type=int,
        default=_DEFAULT_RAMP_S,
        help=f"Closed-loop ramp duration before sustain. Default: {_DEFAULT_RAMP_S}s. Set 0 to disable.",
    )
    g.addoption(
        "--wd-warmup-c",
        type=int,
        default=_DEFAULT_WARMUP_C,
        help=f"Starting concurrency for the ramp. Default: {_DEFAULT_WARMUP_C}",
    )
    g.addoption(
        "--wd-pre-kill-sustain-s",
        type=int,
        default=_DEFAULT_PRE_KILL_SUSTAIN_S,
        help=f"Time at full concurrency before kill. Default: {_DEFAULT_PRE_KILL_SUSTAIN_S}s",
    )
    g.addoption(
        "--wd-post-kill-recovery-s",
        type=int,
        default=_DEFAULT_POST_KILL_RECOVERY_S,
        help=f"Observation soak after kill. Default: {_DEFAULT_POST_KILL_RECOVERY_S}s",
    )


@pytest.mark.asyncio
@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_worker_death_kv_transfer(runtime_env, request):
    """Drive heavy KV-transfer load, kill one target-service pod mid-flight,
    measure cascade signature + recovery.

    Expected signature when saturated: killing one prefill with ISL=7000
    sustained drives the paired decode to engine death within seconds
    (UCX pool warnings + many `raise output` exceptions). Whether a
    given image survives that cascade (via migration support / better
    NIXL handling) is what this test answers.
    """
    cfg = request.config
    image = cfg.getoption("--image") or "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1"
    units = cfg.getoption("--units")
    concurrency = cfg.getoption("--wd-concurrency")
    target_service = cfg.getoption("--wd-target-service")
    isl = cfg.getoption("--wd-isl-tokens")
    ramp_s = cfg.getoption("--wd-ramp-s")
    warmup_c = cfg.getoption("--wd-warmup-c")
    pre_kill_s = cfg.getoption("--wd-pre-kill-sustain-s")
    post_kill_s = cfg.getoption("--wd-post-kill-recovery-s")

    # Total AIPerf load duration spans ramp + pre-kill sustain + post-kill soak.
    load_duration_min = (ramp_s + pre_kill_s + post_kill_s) / 60

    spec = _load_dgd(_DGD)
    _apply_topology(spec, units=units, fe=_DEFAULT_FE, pf=_DEFAULT_PF, dec=_DEFAULT_DE)
    for svc in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].image = image
    _apply_cluster_portability(spec)
    _apply_router_config(spec, router_mode=cfg.getoption("--router-mode"))

    model_cache_pvc = cfg.getoption("--model-cache-pvc")
    if model_cache_pvc:
        spec.enable_model_cache(model_cache_pvc)

    served_model = spec["VllmDecodeWorker"].model

    logger.info(
        "test_worker_death_kv_transfer: units=%d target=%s c=%d isl=%d",
        units,
        target_service,
        concurrency,
        isl,
    )

    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=2400),
            StartLoad(
                load_config=_kv_transfer_load(
                    served_model=served_model,
                    concurrency=concurrency,
                    duration_minutes=load_duration_min,
                    ramp_duration_s=ramp_s,
                    warmup_concurrency=warmup_c,
                ),
                name="heavy-kv",
            ),
            # Ramp + pre-kill sustain — get to KV saturation BEFORE killing.
            # The cascade only reproduces when decode is at KV=1.00 with
            # a full in-flight queue at the moment of the kill.
            Wait(duration=ramp_s + pre_kill_s),
            # The fault: kill one pod of the target service mid-NIXL.
            # `force=True` skips graceful shutdown so in-flight transfers
            # are abruptly broken (the reliable repro condition).
            DeletePod(
                services=[target_service],
                force=True,
                pod_indices=[0],
                name="kill-mid-transfer",
            ),
            # Post-kill recovery soak — replacement-pod cold start
            # (vLLM WorkerProc init can take ~60s + model load).
            Wait(duration=post_kill_s),
            WaitForRecovery(timeout=600),
            StopLoad(),
        ],
        checks=[
            # Minimum signal that load actually ran
            MinRequests(min_count=50),
            # Allow significant errors during fault window — what matters
            # is recovery, not absence of failures. Tighten later once we
            # have baseline data.
            MaxErrors(max_errors=1_000_000),
            # Pods that survive should NOT panic on the peer's death.
            # Decode is expected to self-terminate cleanly via Runtime
            # shutdown rather than panic. `acceptable=True` for now;
            # tighten once the test image is past PR #8254 (tcp/client
            # ConnectionReset panic fix).
            WorkerPanics(
                services=["VllmPrefillWorker", "VllmDecodeWorker", "Frontend"],
                acceptable=True,
            ),
        ],
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
        ],
        test_name=f"test_worker_death_kv_transfer[{target_service}-u{units}-c{concurrency}]",
        runtime_env=runtime_env,
    )
