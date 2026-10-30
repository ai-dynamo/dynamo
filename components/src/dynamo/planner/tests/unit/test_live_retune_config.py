# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the retune imports.

"""Configuration premises of live retune (``retune.executor: dgdr``).

P1 (Kubernetes, decision latency, explicit timeout within the interval), P2
(retune-only, router only), P3 (the DGDR router knobs), P5 (a pinned engine
section) and P13 (the largest grid shard fits the sweep deadline), plus the
shard-split and need-seconds helpers the executor shares with validation.
"""

from __future__ import annotations

import copy
import math

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from pydantic import ValidationError

from dynamo.planner.config.planner_config import (
    DGDR_MAX_SHARD_POINTS,
    DgdrExecutorConfig,
    PlannerConfig,
    balanced_shard_parts,
    dgdr_need_seconds,
    largest_shard_points,
)
from dynamo.planner.retune.domains import expand_router_domain

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
# Design §4.1 defaults: T = 6 h, timeout = T - 20 min.
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


@pytest.mark.parametrize(
    "scaling",
    [
        {"enable_load_scaling": True, "enable_throughput_scaling": False},
        {
            "enable_load_scaling": False,
            "enable_throughput_scaling": True,
            "optimization_target": "sla",
        },
    ],
)
def test_p2_requires_retune_only(scaling):
    _rejects(r"retune-only mode \(P2\)", **scaling)


def test_p2_requires_both_scaling_flags_explicitly():
    # Without both flags written, easy mode keeps load scaling on.
    with pytest.raises(ValidationError, match=r"retune-only mode \(P2\)"):
        PlannerConfig(
            namespace="test-ns",
            environment="kubernetes",
            mode="agg",
            enable_throughput_scaling=False,
            retune=_retune(),
        )


def test_p2_rejects_a_planner_domain():
    _rejects(
        r"retunes the router only \(P2\)",
        _retune(planner={"load_sensitivity": {"preset": ["default"]}}),
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
            r"preset\[0\] must set exactly .*unknown=\['expert'\]",
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


def test_p5_accepts_the_context_parallel_preset_knobs():
    engine = _engine()
    engine["workers"]["aggregated"]["parallelism"]["preset"][0]["decode_context"] = 1
    assert _config(_retune(dgdr=_dgdr(engine=engine))).retune is not None


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
        r"router grid of 108 points cannot be split into at most "
        r"retune.dgdr.run.max_shards=1 Cartesian shards of at most 100 points "
        r"\(P13\); the largest would have 108",
        _retune(dgdr=_dgdr(run={"max_shards": 1})),
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


@pytest.mark.parametrize(
    ("sizes", "max_shards", "parts", "largest"),
    [
        ((3, 9, 4), 3, (1, 3, 1), 36),
        ((3, 9, 4), 2, (1, 1, 2), 54),
        ((3, 9, 4), 1, (1, 1, 1), 108),
        # 15 shards of 1 x 2 x 4 beat 12 of 1 x 9 x 1.
        ((3, 9, 4), 16, (3, 5, 1), 8),
        ((10,), 3, (3,), 4),
        ((2, 2), 16, (2, 2), 1),
        ((), 3, (), 1),
    ],
)
def test_balanced_shard_parts(sizes, max_shards, parts, largest):
    assert balanced_shard_parts(sizes, max_shards) == parts
    assert largest_shard_points(sizes, parts) == largest
    assert math.prod(parts) <= max_shards


def test_balanced_shard_parts_minimizes_the_largest_shard_exhaustively():
    for sizes in ((3, 9, 4), (5, 7), (2, 3, 5), (12,), (1, 1, 30)):
        for max_shards in range(1, 17):
            parts = balanced_shard_parts(sizes, max_shards)
            best = min(
                largest_shard_points(sizes, (a, b, c)[: len(sizes)])
                for a in range(1, max_shards + 1)
                for b in range(1, max_shards + 1)
                for c in range(1, max_shards + 1)
                if math.prod((a, b, c)[: len(sizes)]) <= max_shards
                and all(p <= n for p, n in zip((a, b, c)[: len(sizes)], sizes))
            )
            assert largest_shard_points(sizes, parts) == best


@pytest.mark.parametrize(
    ("sizes", "max_shards"),
    [((3, 0), 3), ((3,), 0)],
)
def test_balanced_shard_parts_rejects_bad_input(sizes, max_shards):
    with pytest.raises(ValueError):
        balanced_shard_parts(sizes, max_shards)


def test_default_router_grid_splits_into_three_shards_of_36():
    retune = _config().retune
    assert retune is not None
    sizes = [len(values) for values in expand_router_domain(retune).values()]
    assert sizes == [3, 9, 4]
    parts = balanced_shard_parts(sizes, 3)
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
