# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""5-hour diurnal endurance — kv vs least-loaded router comparison.

Continuous up-and-down load over 5 hours to simulate a real production
day (morning ramp, midday peak, lunch dip, evening peak, cool-down).
Run twice with different DYN_ROUTER_MODE in parallel namespaces to A/B
compare kv-aware routing vs least-loaded on identical workload.

Each diurnal segment is a separate AIPerf rung (closed-loop with optional
concurrency-ramp), so the existing per-rung report panels light up
naturally — one column per segment for every metric (KV usage, running,
waiting, inflight, derived wait-for-remote).
"""

import logging

import pytest

from tests.fault_tolerance.deploy.checks import MinRequests
from tests.fault_tolerance.deploy.events import (
    StartLoad,
    Wait,
    WaitForLoadCompletion,
    WaitForModelReady,
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


_DGD = "disagg_qwen3_30b_unit_prod"

# Production-shape ISL/OSL (matches test_overload _prod_load).
_ISL_MEAN = 1600
_OSL_MEAN = 200

# Default diurnal curve — total 5 h. Each tuple is one rung:
#   (segment_name, start_concurrency, end_concurrency, duration_minutes)
# When start == end the rung is a flat hold; otherwise AIPerf ramps
# closed-loop concurrency over `duration_minutes`.
_DEFAULT_CURVE = [
    ("morning-ramp", 4, 24, 30),
    ("morning-climb", 24, 96, 30),
    ("morning-sustain", 96, 96, 60),
    ("lunch-dip", 96, 24, 30),
    ("offpeak-hold", 24, 24, 60),
    ("evening-climb", 24, 96, 30),
    ("evening-sustain", 96, 96, 30),
    ("cool-down", 96, 4, 30),
]

_DEFAULT_UNITS = 1
_DEFAULT_FE = 1
_DEFAULT_PF = 2
_DEFAULT_DE = 1


def _segment_load(
    *, served_model: str, c_start: int, c_end: int, minutes: float
) -> LoadConfig:
    """One diurnal segment as an AIPerf LoadConfig."""
    is_ramp = c_start != c_end
    return LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        input_tokens_mean=_ISL_MEAN,
        input_tokens_stddev=200,
        output_tokens_mean=_OSL_MEAN,
        concurrency=c_end if is_ramp else c_start,
        duration_minutes=minutes,
        request_timeout_seconds=120,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
        connection_reuse_strategy="never",
        concurrency_ramp_duration=(minutes * 60) if is_ramp else None,
        warmup_concurrency=c_start if is_ramp else None,
        warmup_duration=0 if is_ramp else None,
    )


def _parse_curve(raw: str):
    """`"name:c0:c1:min,name:c0:c1:min,..."` → list of tuples."""
    out = []
    for piece in raw.split(","):
        parts = [p.strip() for p in piece.split(":")]
        if len(parts) != 4:
            raise ValueError(f"Bad curve segment {piece!r}; expect name:c0:c1:min")
        out.append((parts[0], int(parts[1]), int(parts[2]), float(parts[3])))
    return out


def add_cli_options(parser):
    g = parser.getgroup("endurance", "5h diurnal A/B endurance scenario")
    # Note: --units is registered by test_overload (shared topology flag).
    g.addoption(
        "--endurance-curve",
        type=str,
        default=None,
        help="Override curve as `name:c0:c1:min,...`. Default: built-in 5h diurnal.",
    )


@pytest.mark.asyncio
@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_endurance(runtime_env, request):
    """Drive an 8-segment diurnal curve over 5h, capture per-rung metrics.

    Run twice (different `--router-mode`, different `--namespace`) in
    parallel to A/B compare kv-aware routing vs least-loaded under
    realistic up-and-down workload.
    """
    cfg = request.config
    image = cfg.getoption("--image") or "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1"
    units = cfg.getoption("--units")
    raw_curve = cfg.getoption("--endurance-curve")
    curve = _parse_curve(raw_curve) if raw_curve else _DEFAULT_CURVE

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

    total_min = sum(seg[3] for seg in curve)
    logger.info(
        "test_endurance: units=%d segments=%d total=%.1f min router-mode=%s",
        units,
        len(curve),
        total_min,
        cfg.getoption("--router-mode"),
    )

    events: list = [WaitForModelReady(timeout=2400)]
    for seg_name, c0, c1, minutes in curve:
        events.append(
            StartLoad(
                load_config=_segment_load(
                    served_model=served_model,
                    c_start=c0,
                    c_end=c1,
                    minutes=minutes,
                ),
                name=seg_name,
            )
        )
        events.append(WaitForLoadCompletion(name=seg_name))
        events.append(Wait(duration=20))  # short gap for AIPerf cleanup

    router = cfg.getoption("--router-mode") or "default"
    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=[
            # Endurance soak — just confirm the load actually ran.
            MinRequests(min_count=1000),
        ],
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
        ],
        test_name=f"test_endurance[u{units}-{router}]",
        runtime_env=runtime_env,
    )
