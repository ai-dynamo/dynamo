# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the retune imports.

"""Configuration premises of live retune (``retune.executor: dgdr``).

P1 (Kubernetes, decision latency, explicit timeout within the interval), P2
(a router domain; scaling and a planner domain are Scenario 2), P3 (the DGDR router knobs), P5 (a pinned engine
section) and P13 (the largest grid shard fits the sweep deadline), plus the
shard-split and need-seconds helpers the executor shares with validation.
"""

from __future__ import annotations

import copy
import math

import pytest

pytest.importorskip(
    "aisim.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisim.config.engine import ParallelismRecommendationConfig
from pydantic import ValidationError

from dynamo.planner.config.planner_config import (
    DGDR_MAX_SHARD_POINTS,
    LIVE_RETUNE_K8S_TIMEOUT_SECONDS,
    DgdrExecutorConfig,
    PlannerConfig,
    balanced_shard_parts,
    dgdr_need_seconds,
    dgdr_rounds,
    largest_shard_points,
    live_retune_min_lease_seconds,
)
from dynamo.planner.retune.domains import expand_router_domain
from dynamo.planner.retune.live import dgdr

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

PRESET = {
    "tensor": 2,
    "pipeline": 1,
    "attention_data": 1,
    "moe_tensor": 1,
    "moe_expert": 1,
}
SCHEDULER = {"max_batched_tokens": 8192, "max_sequences": 256}


def _worker(**overrides):
    return {
        "parallelism": {"preset": [dict(PRESET)]},
        "scheduler": dict(SCHEDULER),
        "kv_cache": {"block_size": 64, "capacity": {"memory_fraction": 0.9}},
        **overrides,
    }


ENGINE_AGG = {
    "mode": "aggregated",
    "backend": "vllm",
    "backend_version": "0.11.0",
    "context_length": 32768,
    "workers": {"aggregated": _worker(), "prefill": None, "decode": None},
}
ENGINE_DISAGG = {
    "mode": "disaggregated",
    "backend": "vllm",
    "backend_version": {"vllm": "0.11.0"},
    "context_length": "max",
    "workers": {"prefill": _worker(), "decode": _worker()},
}
# The live retune defaults: T = 6 h, timeout = T - 20 min.
INTERVAL = 21600.0
TIMEOUT = 20400.0
# The default router domain is 3 x 9 x 4 = 108 points; with max_shards 3 the
# largest shard has 36: 900 + 300 + (1 + ceil(35 / 8)) x 1800 = 12000 s.
DEFAULT_NEED = 12000.0


def _dgdr(**overrides):
    return {
        "claim_name": "request-traces",
        "model": "Qwen/Qwen3-32B",
        "sku": "h200_sxm",
        "engine": copy.deepcopy(ENGINE_AGG),
        **overrides,
    }


def _retune(**overrides):
    return {
        "interval_seconds": INTERVAL,
        "timeout_seconds": TIMEOUT,
        "router": "default",
        "executor": "dgdr",
        "dgdr": _dgdr(),
        **overrides,
    }


def _config(retune=None, **overrides) -> PlannerConfig:
    values = {
        "namespace": "test-ns",
        "environment": "kubernetes",
        "mode": "agg",
        "enable_load_scaling": False,
        "enable_throughput_scaling": False,
        "retune": _retune() if retune is None else retune,
        **overrides,
    }
    return PlannerConfig(**values)


def _rejects(match: str, retune=None, **overrides) -> None:
    with pytest.raises(ValidationError, match=match):
        _config(retune, **overrides)


def _engine(mode: str = "agg") -> dict:
    return copy.deepcopy(ENGINE_AGG if mode == "agg" else ENGINE_DISAGG)


# ------------------------------------------------------------------ valid


@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_live_retune_config_accepts_the_design_example(mode):
    retune = _retune(dgdr=_dgdr(engine=_engine(mode)))
    config = _config(retune, mode=mode)

    assert config.retune is not None
    assert config.retune.executor == "dgdr"
    assert config.retune.dgdr is not None
    assert config.retune.dgdr.engine == _engine(mode)
    # Validation never rewrites the sections it checks.
    assert config.retune.router == "default"


def test_live_retune_accepts_a_measured_decision_latency_at_scale_one():
    config = _config(_retune(decision_latency_seconds="measured"))
    assert config.retune is not None
    assert config.retune.decision_latency_seconds == "measured"


def test_live_retune_accepts_the_largest_timeout_the_interval_allows():
    config = _config(_retune(timeout_seconds=INTERVAL - 600))
    assert config.retune is not None


def test_live_retune_accepts_the_three_dgdr_router_knobs_and_none_load_model():
    router = {
        "overlap_score_credit": {"choices": [0.0, 1.0]},
        "prefill_load_scale": {"range": {"min": 0.5, "max": 2.0, "step": 0.5}},
        "temperature": 0.0,
        "policy": "kv_router",
        "prefill_load_model": {"type": "none"},
    }
    assert _config(_retune(router=router)).retune is not None
    router["prefill_load_model"] = {"type": {"choices": ["none"]}}
    assert _config(_retune(router=router)).retune is not None


def test_replay_retune_keeps_its_rules():
    # executor local: none of the live premises apply.
    config = _config(
        {"interval_seconds": 600, "router": "default"},
        environment="virtual",
    )
    assert config.retune is not None
    assert config.retune.executor == "local"


# --------------------------------------------------------------------- P1


def test_p1_requires_kubernetes():
    _rejects(r"environment='kubernetes' \(P1\)", environment="virtual")


@pytest.mark.parametrize("latency", [30.0, 0.5])
def test_p1_rejects_a_fixed_nonzero_decision_latency(latency):
    _rejects(
        r"decision_latency_seconds must be 0 or 'measured' \(P1\)",
        _retune(decision_latency_seconds=latency),
    )


def test_p1_rejects_a_scaled_measured_decision_latency():
    # Valid for RetuneConfig (2 x 10000 <= 21600), rejected by P1 before the
    # time budget is checked.
    _rejects(
        r"decision_latency_scale 1 \(P1\)",
        _retune(
            decision_latency_seconds="measured",
            decision_latency_scale=2.0,
            timeout_seconds=10000.0,
        ),
    )


def test_p1_requires_an_explicit_timeout():
    retune = _retune()
    del retune["timeout_seconds"]
    _rejects(r"explicit retune.timeout_seconds \(P1\)", retune)


def test_p1_rejects_a_timeout_within_ten_minutes_of_the_interval():
    _rejects(
        r"timeout_seconds must be <= interval_seconds - 600 = 21000 .*\(P1\)",
        _retune(timeout_seconds=INTERVAL - 599),
    )


# --------------------------------------------------------------------- P2


# Two credits, the rest pinned: a 2-point router grid.
PINNED_ROUTER = {
    "overlap_score_credit": {"choices": [0.5, 1.0]},
    "prefill_load_scale": 1.0,
    "temperature": 0.0,
}
LOAD_ONLY = {"enable_load_scaling": True, "enable_throughput_scaling": False}
SLA_THROUGHPUT = {
    "enable_load_scaling": False,
    "enable_throughput_scaling": True,
    "optimization_target": "sla",
}


@pytest.mark.parametrize(
    ("scaling", "planner"),
    [
        (LOAD_ONLY, {"load_sensitivity": {"preset": ["default", "conservative"]}}),
        (SLA_THROUGHPUT, "default"),
        # Easy mode keeps load scaling on without the flags written.
        ({"enable_throughput_scaling": False}, None),
        (
            {**SLA_THROUGHPUT, "enable_load_scaling": True},
            {"scaling_policy": {"preset": ["hybrid_180_5", "hybrid_600_5"]}},
        ),
    ],
    ids=["load", "sla-throughput", "easy-mode", "scaling-policy"],
)
def test_p2_accepts_scaling_and_a_planner_domain(scaling, planner):
    # Scenario 2: candidates replay an inner planner from the window start.
    retune = _retune(router=PINNED_ROUTER)
    if planner is not None:
        retune["planner"] = planner
    config = PlannerConfig(
        namespace="test-ns",
        environment="kubernetes",
        mode="agg",
        retune=retune,
        **scaling,
    )
    assert config.scaling_enabled()


def test_p2_requires_a_router_domain():
    # Every DGDR Request sets the three router knobs; pin them to retune the
    # Planner alone.
    _rejects(
        r"requires retune.router \(P2\)",
        _retune(router=None, planner="default"),
        **LOAD_ONLY,
    )


# --------------------------------------------------------------------- P3


@pytest.mark.parametrize(
    "knob", ["overlap_score_credit_decay", "decode_active_request_weight"]
)
def test_p3_rejects_retune_extension_knobs(knob):
    _rejects(
        rf"retune.router \['{knob}'\] cannot be searched with executor 'dgdr' \(P3\)",
        _retune(router={knob: {"choices": [0.0, 1.0]}}),
    )


@pytest.mark.parametrize(
    "load_model",
    [{"type": "ais"}, {"type": {"choices": ["none", "ais"]}}],
)
def test_p3_requires_the_none_prefill_load_model(load_model):
    _rejects(
        r"prefill_load_model.type must be 'none' with executor 'dgdr' \(P3\)",
        _retune(router={"temperature": 0.0, "prefill_load_model": load_model}),
    )


def test_p3_keeps_the_schema_errors_for_other_policies():
    # RouterRetuneConfig already pins the policy; its error stays first.
    _rejects(
        r"retune.router.policy is fixed at startup",
        _retune(router={"policy": "round_robin"}),
    )


# --------------------------------------------------------------------- P5


def _rejects_engine(match: str, engine: dict, mode: str = "agg") -> None:
    _rejects(match, _retune(dgdr=_dgdr(engine=engine)), mode=mode)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (
            ("workers", "aggregated", "scheduler", "max_sequences"),
            {"choices": [128, 256]},
            r"engine.workers.aggregated.scheduler.max_sequences is a search "
            r"domain \(choices\); .* \(P5\)",
        ),
        (
            ("workers", "aggregated", "kv_cache", "capacity", "memory_fraction"),
            {"range": {"min": 0.8, "max": 0.9}},
            r"memory_fraction is a search domain \(range\)",
        ),
        (
            ("backend",),
            {"choices": ["vllm", "sglang"]},
            r"engine.backend is a search domain",
        ),
        (
            ("workers", "aggregated", "timing", "systems_paths"),
            ["/opt/systems"],
            r"engine.workers.aggregated.timing.systems_paths is a list; .* \(P5\)",
        ),
        (
            ("workers", "aggregated", "parallelism", "preset", 0, "replicas"),
            2,
            r"engine.workers.aggregated.parallelism.preset\[0\].replicas must "
            r"not be set: the DGD spec provides replicas at each launch \(P5\)",
        ),
        (
            ("workers", "aggregated", "parallelism", "replicas"),
            2,
            r"parallelism.replicas must not be set: the DGD spec",
        ),
        (
            ("workers", "aggregated", "parallelism", "preset"),
            [dict(PRESET), dict(PRESET, tensor=4)],
            r"parallelism.preset must be a list of exactly one mapping .* \(P5\)",
        ),
        (
            ("workers", "aggregated", "parallelism", "preset"),
            "default",
            r"parallelism.preset must be a list of exactly one mapping",
        ),
        (
            ("workers", "aggregated", "parallelism", "tensor"),
            2,
            r"parallelism sets \['tensor'\] next to preset",
        ),
        (
            ("workers", "aggregated", "parallelism", "preset", 0, "tensor"),
            True,
            r"preset\[0\].tensor must be a positive integer \(P5\)",
        ),
        (
            ("workers", "aggregated", "parallelism", "preset", 0, "expert"),
            1,
            r"preset\[0\] must set exactly .*moe_expert and optionally "
            r"prefill_context: 1 \(P5\); missing=\[\], unknown=\['expert'\]",
        ),
        (
            ("workers", "aggregated", "parallelism", "preset", 0, "decode_context"),
            1,
            r"preset\[0\].decode_context is not accepted \(P5\); the pinned "
            r"AISimulate preset schema has no decode_context",
        ),
        (
            ("workers", "aggregated", "parallelism", "preset", 0, "prefill_context"),
            2,
            r"preset\[0\].prefill_context must be 1 or omitted \(P5\); AISimulate "
            r"recommend rejects context parallelism above 1, got 2",
        ),
        (
            ("workers", "aggregated", "parallelism", "preset", 0, "prefill_context"),
            0,
            r"preset\[0\].prefill_context must be a positive integer \(P5\), got 0",
        ),
        (
            ("workers", "aggregated", "scheduler", "max_batched_tokens"),
            None,
            r"scheduler.max_batched_tokens must be a positive integer \(P5\); "
            r"omitted, AISimulate searches its default domain",
        ),
        (
            ("mode",),
            "disaggregated",
            r"engine.mode must be 'aggregated' for planner mode 'agg' \(P5\)",
        ),
        (
            ("model",),
            "Qwen/Qwen3-32B",
            r"engine.model would override the Request's typed field \(P5\); "
            r"set retune.dgdr.model",
        ),
        (
            ("hardware",),
            "h200_sxm",
            r"set retune.dgdr.sku instead",
        ),
        (
            ("workers", "decode"),
            _worker(),
            r"engine.workers.decode is not a worker role of planner mode 'agg'",
        ),
        (
            ("workers", "encoder"),
            {"tensor": 1},
            r"engine.workers.encoder is not a worker role",
        ),
    ],
)
def test_p5_rejects_an_unpinned_engine(path, value, match):
    engine = _engine()
    target = engine
    for key in path[:-1]:
        target = target.setdefault(key, {}) if isinstance(key, str) else target[key]
    target[path[-1]] = value
    _rejects_engine(match, engine)


def test_p5_rejects_a_missing_preset_knob():
    engine = _engine()
    del engine["workers"]["aggregated"]["parallelism"]["preset"][0]["moe_expert"]
    _rejects_engine(r"missing=\['moe_expert'\]", engine)


def test_p5_requires_mode():
    engine = _engine()
    del engine["mode"]
    _rejects_engine(r"an omitted mode searches both", engine)


def test_p5_requires_every_role_of_the_planner_mode():
    engine = _engine("disagg")
    del engine["workers"]["decode"]
    _rejects_engine(
        r"engine.workers.decode is required for planner mode 'disagg' \(P5\)",
        engine,
        mode="disagg",
    )


def test_p5_requires_the_disaggregated_engine_mode_in_disagg():
    engine = _engine("disagg")
    engine["mode"] = "aggregated"
    _rejects_engine(
        r"engine.mode must be 'disaggregated' for planner mode 'disagg'",
        engine,
        mode="disagg",
    )


def test_p5_engine_backend_must_be_the_deployed_backend():
    engine = _engine()
    engine["backend"] = "sglang"
    # PlannerConfig.backend defaults to vllm; every Run would fail its build.
    _rejects_engine(
        r"retune.dgdr.engine.backend 'sglang' differs from the deployed "
        r"backend 'vllm' \(P5\)",
        engine,
    )
    config = _config(_retune(dgdr=_dgdr(engine=engine)), backend="sglang")
    assert config.retune is not None and config.retune.dgdr is not None
    assert config.retune.dgdr.engine["backend"] == "sglang"
    _rejects(
        r"retune.dgdr.engine.backend 'vllm' differs from the deployed backend "
        r"'trtllm' \(P5\)",
        _retune(dgdr=_dgdr(engine=_engine())),
        backend="trtllm",
    )


def test_p5_accepts_an_omitted_engine_backend():
    # The Request fills it with the deployed backend.
    engine = _engine()
    del engine["backend"]
    for backend in ("vllm", "sglang"):
        config = _config(
            _retune(dgdr=_dgdr(engine=copy.deepcopy(engine))), backend=backend
        )
        assert config.retune is not None


def test_live_retune_accepts_nested_claim_directories():
    dgdr = _dgdr(segments_dir="retune/segments", windows_dir="retune/windows")
    assert _config(_retune(dgdr=dgdr)).retune is not None


def test_p5_accepts_unused_roles_absent_or_null():
    engine = _engine()
    del engine["workers"]["prefill"]
    engine["workers"]["encoder"] = None
    assert _config(_retune(dgdr=_dgdr(engine=engine))).retune is not None


def test_p5_accepts_prefill_context_one():
    engine = _engine()
    engine["workers"]["aggregated"]["parallelism"]["preset"][0]["prefill_context"] = 1
    assert _config(_retune(dgdr=_dgdr(engine=engine))).retune is not None


def test_p5_rejects_decode_context_on_a_disaggregated_role():
    engine = _engine("disagg")
    engine["workers"]["decode"]["parallelism"]["preset"][0]["decode_context"] = 1
    _rejects_engine(
        r"engine.workers.decode.parallelism.preset\[0\].decode_context is not "
        r"accepted \(P5\)",
        engine,
        mode="disagg",
    )


def _spec_replicas(mode: str) -> dict[str, int]:
    return {"aggregated": 2} if mode == "agg" else {"prefill": 1, "decode": 3}


def _validate_with_pinned_aisimulate(engine: dict, mode: str) -> None:
    spec_replicas = _spec_replicas(mode)
    pinned = dgdr._pinned_engine(engine, spec_replicas, "vllm")
    for role in spec_replicas:
        ParallelismRecommendationConfig.model_validate(
            pinned["workers"][role]["parallelism"]
        )


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("prefill_context", [None, 1])
def test_p5_presets_validate_against_the_pinned_aisimulate(mode, prefill_context):
    engine = _engine(mode)
    if prefill_context is not None:
        for role in _spec_replicas(mode):
            preset = engine["workers"][role]["parallelism"]["preset"]
            preset[0]["prefill_context"] = prefill_context
    config = _config(_retune(dgdr=_dgdr(engine=engine)), mode=mode)
    assert config.retune is not None and config.retune.dgdr is not None

    _validate_with_pinned_aisimulate(config.retune.dgdr.engine, mode)


def test_pinned_aisimulate_rejects_decode_context():
    # Why P5 rejects decode_context; once a pin bump makes this fail, P5 may
    # admit it.
    engine = _engine()
    engine["workers"]["aggregated"]["parallelism"]["preset"][0]["decode_context"] = 1
    with pytest.raises(ValidationError, match=r"decode_context\s+Extra inputs"):
        _validate_with_pinned_aisimulate(engine, "agg")


# -------------------------------------------------------------- claim paths


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"segments_dir": "/trace/segments"}, r"segments_dir must be a relative"),
        ({"windows_dir": "../windows"}, r"windows_dir must be a relative"),
        ({"windows_dir": "."}, r"windows_dir must be a relative"),
        ({"windows_dir": "a/./windows"}, r"windows_dir must be a relative"),
        ({"windows_dir": "windows/"}, r"windows_dir must be a relative"),
        ({"segments_dir": "a//segments"}, r"segments_dir must be a relative"),
        ({"mount_path": "trace"}, r"mount_path must be absolute"),
        ({"windows_dir": "segments"}, r"must not contain one another"),
        ({"windows_dir": "segments/windows"}, r"must not contain one another"),
    ],
)
def test_live_retune_rejects_unsafe_claim_paths(overrides, match):
    _rejects(match, _retune(dgdr=_dgdr(**overrides)))


# --------------------------------------------------------- leadership Lease


def test_live_retune_lease_fits_two_renewals_before_its_margin():
    minimum = live_retune_min_lease_seconds(LIVE_RETUNE_K8S_TIMEOUT_SECONDS)
    # 3 x 5 s margin + 2 x (2 x 5 s + 1 s) I/O budget.
    assert minimum == 37.0
    for lease in (15.0, minimum - 0.001):
        _rejects(
            r"retune.dgdr.lease_duration_seconds must be at least the lease "
            r"margin \+ 2 x the I/O budget of one renewal \(15s \+ 2 x 11s = 37s "
            r"with the 5s Kubernetes request timeout\)",
            _retune(dgdr=_dgdr(lease_duration_seconds=lease)),
        )
    config = _config(_retune(dgdr=_dgdr(lease_duration_seconds=minimum)))
    assert config.retune is not None and config.retune.dgdr is not None
    assert config.retune.dgdr.lease_duration_seconds == minimum
    schema = DgdrExecutorConfig.model_json_schema()["properties"]
    assert schema["lease_duration_seconds"]["minimum"] == minimum
    assert "At least 37 s" in schema["lease_duration_seconds"]["description"]


# -------------------------------------------------------------------- P13


def test_p13_accepts_a_shard_that_exactly_fits():
    assert _config(_retune(timeout_seconds=DEFAULT_NEED)).retune is not None


def test_p13_rejects_a_largest_shard_that_cannot_finish():
    _rejects(
        r"largest retune grid shard \(36 points\) needs 12000s: "
        r"cut_budget_seconds 900 \+ job_startup_seconds 300 \+ 6 rounds x "
        r"candidate_timeout_seconds 1800, more than "
        r"retune.timeout_seconds=11999 \(P13\)",
        _retune(timeout_seconds=DEFAULT_NEED - 1),
    )


def test_p13_counts_the_largest_shard_of_fewer_max_shards():
    # max_shards 2: 3 x 9 x 2 = 54 points -> 1 + ceil(53 / 8) = 8 rounds.
    dgdr = _dgdr(run={"max_shards": 2})
    need = 900 + 300 + 8 * 1800
    assert _config(_retune(dgdr=dgdr, timeout_seconds=need)).retune is not None
    _rejects(r"\(54 points\)", _retune(dgdr=dgdr, timeout_seconds=need - 1))


def test_p13_rejects_a_grid_that_needs_more_shards():
    _rejects(
        r"retune grid of 108 points cannot be split into at most "
        r"retune.dgdr.run.max_shards=1 Cartesian shards of at most 100 points "
        r"\(P13\); the largest would have 108",
        _retune(dgdr=_dgdr(run={"max_shards": 1})),
    )


def test_p13_shards_the_planner_points_too():
    # The 108 router points x the 3 default sensitivities: 324 points need a
    # fourth shard, which cuts the temperature axis.
    retune = _retune(planner="default")
    _rejects(
        r"retune grid of 324 points cannot be split into at most "
        r"retune.dgdr.run.max_shards=3 Cartesian shards",
        retune,
        **LOAD_ONLY,
    )
    dgdr = _dgdr(run={"max_shards": 4, "parallelism": 16})
    config = _config(_retune(planner="default", dgdr=dgdr), **LOAD_ONLY)
    assert config.retune is not None


def test_p13_budgets_the_history_read_of_a_scaling_planner():
    dgdr = _dgdr(
        cut_budget_seconds=60,
        job_startup_seconds=0,
        run={"parallelism": 36, "candidate_timeout_seconds": 100},
    )
    retune = _retune(router=PINNED_ROUTER, planner=None, dgdr=dgdr)
    # Two one-point shards: cut 60 + history 60 + 0 + 1 round x 100 = 220.
    assert _config({**retune, "timeout_seconds": 220}, **LOAD_ONLY).retune is not None
    _rejects(
        r"needs 220s: cut_budget_seconds 60 for the history \+ cut_budget_seconds 60",
        {**retune, "timeout_seconds": 219},
        **LOAD_ONLY,
    )
    # Fixed replicas read no history.
    assert _config({**retune, "timeout_seconds": 160}).retune is not None


def test_p13_bounds_the_warmup_a_request_carries():
    # 180-s buckets: 1024 fit, 1025 do not.
    retune = _retune(router=PINNED_ROUTER, planner=None)
    accepted = {**retune, "history_seconds": 180.0 * 1024}
    assert _config(accepted, **SLA_THROUGHPUT).retune is not None
    _rejects(
        r"is 1025 warmup buckets, more than the 1024 a DGDR Request carries \(P13\)",
        {**retune, "history_seconds": 180.0 * 1025},
        **SLA_THROUGHPUT,
    )


def test_p13_uses_the_configured_run_parallelism_and_timeouts():
    dgdr = _dgdr(
        cut_budget_seconds=60,
        job_startup_seconds=0,
        run={"parallelism": 36, "candidate_timeout_seconds": 100},
    )
    # 60 + 0 + (1 + ceil(35 / 36)) x 100 = 260.
    assert _config(_retune(dgdr=dgdr, timeout_seconds=260)).retune is not None
    _rejects(r"needs 260s", _retune(dgdr=dgdr, timeout_seconds=259))


# --------------------------------------------------------------- helpers


def test_balanced_shard_parts_minimizes_the_rounds_exhaustively():
    for sizes in ((3, 9, 4), (5, 7), (2, 3, 5), (12,), (1, 1, 30), (250,)):
        for max_shards in range(1, 17):
            best = min(
                largest_shard_points(sizes, (a, b, c)[: len(sizes)])
                for a in range(1, max_shards + 1)
                for b in range(1, max_shards + 1)
                for c in range(1, max_shards + 1)
                if math.prod((a, b, c)[: len(sizes)]) <= max_shards
                and all(p <= n for p, n in zip((a, b, c)[: len(sizes)], sizes))
            )
            # With the Run parallelism, the split takes the fewest rounds, so
            # its largest shard fits and finishes exactly when the smallest
            # largest shard does: P13 and the executor agree.
            for parallelism in (1, 8, 36):
                parts = balanced_shard_parts(sizes, max_shards, parallelism=parallelism)
                largest = largest_shard_points(sizes, parts)
                if best > DGDR_MAX_SHARD_POINTS:
                    assert largest == best
                else:
                    assert largest <= DGDR_MAX_SHARD_POINTS
                    assert dgdr_rounds(largest, parallelism) == dgdr_rounds(
                        best, parallelism
                    )


@pytest.mark.parametrize(
    ("sizes", "max_shards", "parallelism", "parts"),
    [
        # A shard is added only when it saves an evaluation round; the first
        # candidate runs alone, so single points save one.
        ((2, 2, 2), 3, 8, (1, 1, 1)),
        ((2, 2, 2), 16, 8, (2, 2, 2)),
        ((2, 2, 2), 3, 1, (1, 1, 2)),
        ((3, 9, 4), 3, 8, (1, 3, 1)),
    ],
)
def test_balanced_shard_parts_with_parallelism(sizes, max_shards, parallelism, parts):
    assert balanced_shard_parts(sizes, max_shards, parallelism=parallelism) == parts


@pytest.mark.parametrize(
    ("sizes", "max_shards", "parallelism"),
    [((3, 0), 3, 8), ((3,), 0, 8), ((3,), 3, 0)],
)
def test_balanced_shard_parts_rejects_bad_input(sizes, max_shards, parallelism):
    with pytest.raises(ValueError):
        balanced_shard_parts(sizes, max_shards, parallelism=parallelism)


def test_default_router_grid_splits_into_three_shards_of_36():
    retune = _config().retune
    assert retune is not None
    sizes = [len(values) for values in expand_router_domain(retune).values()]
    assert sizes == [3, 9, 4]
    assert retune.dgdr is not None
    parts = balanced_shard_parts(sizes, 3, parallelism=retune.dgdr.run.parallelism)
    assert largest_shard_points(sizes, parts) == 36 <= DGDR_MAX_SHARD_POINTS


@pytest.mark.parametrize(
    ("points", "needs_cut", "expected"),
    [
        (1, True, 900 + 300 + 1 * 1800),
        (1, False, 300 + 1 * 1800),
        (9, True, 900 + 300 + 2 * 1800),
        (10, True, 900 + 300 + 3 * 1800),
        (36, False, 300 + 6 * 1800),
    ],
)
def test_dgdr_need_seconds(points, needs_cut, expected):
    dgdr = DgdrExecutorConfig.model_validate(_dgdr())
    assert dgdr_need_seconds(points, dgdr, needs_cut=needs_cut) == expected


def test_dgdr_need_seconds_rejects_an_empty_shard():
    dgdr = DgdrExecutorConfig.model_validate(_dgdr())
    with pytest.raises(ValueError):
        dgdr_need_seconds(0, dgdr, needs_cut=True)
