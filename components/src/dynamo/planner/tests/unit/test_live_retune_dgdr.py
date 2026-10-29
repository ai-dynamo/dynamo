# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the executor imports the sweep.

"""Unit tests for the live retune DGDR executor (``retune.live.dgdr``).

Each sweep runs against ``FakeRetuneKube`` with ``FakeDgdrOperator``
advancing every Request one phase per ``sleep`` of a fake clock, so the
Request -> Run -> Candidate lifecycle, adoption, failures and timeouts are
all deterministic.
"""

from __future__ import annotations

import itertools
import math
import threading
from collections.abc import Callable, Mapping
from typing import Any, Optional

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.planner.config.planner_config import (
    DgdrExecutorConfig,
    RetuneConfig,
    RetuneEngineSpec,
)
from dynamo.planner.core.types import RetuneParams
from dynamo.planner.retune.live import dgdr
from dynamo.planner.retune.live.contracts import (
    LABEL_ATTEMPT,
    LABEL_DGD_UID,
    LABEL_SHARD,
    LABEL_SPEC,
    LABEL_WINDOW,
    NON_DECISION_FAILURES,
    DgdRef,
    KubeAlreadyExists,
    KubeError,
    KubeInvalid,
    KubeNotFound,
    WindowCutResult,
)
from dynamo.planner.retune.live.dgdr import (
    DgdrContext,
    Shard,
    build_request,
    crn_unsupported,
    input_mismatches,
    missing_fields,
    need_seconds,
    plan_shards,
    request_name,
    run_dgdr_sweep,
    spec_hash,
)
from dynamo.planner.retune.sweep import SweepRequest, score_summary
from dynamo.planner.tests.live_retune_fakes import (
    FakeDgdrOperator,
    FakeRetuneKube,
    default_metrics,
    lower_request_input,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

CREDIT = "overlap_score_credit"
SCALE = "prefill_load_scale"
TEMP = "router_temperature"
LIVE_TO_PUBLIC = {CREDIT: CREDIT, SCALE: SCALE, TEMP: "temperature"}
DEFAULT_DOMAIN = {
    CREDIT: (0.0, 0.5, 1.0),
    SCALE: (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    TEMP: (0.0, 0.2, 0.5, 1.0),
}
SMALL_DOMAIN = {CREDIT: (0.5, 1.0), SCALE: (1.0, 2.0), TEMP: (0.0, 0.5)}
WINDOW_END_MS = 1_793_000_000_000
INTERVAL_S = 3600.0
GRACE_S = 120.0
TIMEOUT_S = 3000.0
LAUNCH_S = WINDOW_END_MS / 1000.0 + GRACE_S
POLL_S = 5.0
DGD = DgdRef(name="llm", namespace="test-ns", uid="dgd-uid-1", dynamo_namespace="dyn")
AGG_ENGINE = {
    "mode": "aggregated",
    "backend": "vllm",
    "context_length": 32768,
    "workers": {
        "aggregated": {
            "parallelism": {
                "preset": [
                    {
                        "tensor": 2,
                        "pipeline": 1,
                        "attention_data": 1,
                        "moe_tensor": 1,
                        "moe_expert": 1,
                    }
                ]
            }
        },
        "prefill": None,
        "decode": None,
    },
}
DISAGG_ENGINE = {
    "mode": "disaggregated",
    "backend": "vllm",
    "workers": {
        "aggregated": None,
        "prefill": {
            "parallelism": {
                "preset": [
                    {
                        "tensor": 1,
                        "pipeline": 1,
                        "attention_data": 1,
                        "moe_tensor": 1,
                        "moe_expert": 1,
                    }
                ]
            }
        },
        "decode": {
            "parallelism": {
                "preset": [
                    {
                        "tensor": 2,
                        "pipeline": 1,
                        "attention_data": 2,
                        "moe_tensor": 1,
                        "moe_expert": 1,
                    }
                ]
            }
        },
    },
}


# ------------------------------------------------------------------ helpers


def make_cfg(**overrides: Any) -> DgdrExecutorConfig:
    values: dict[str, Any] = {
        "claim_name": "request-traces",
        "model": "Qwen/Qwen3-32B",
        "sku": "h200_sxm",
        "engine": AGG_ENGINE,
        "cut_budget_seconds": 100.0,
        "job_startup_seconds": 60.0,
        "run": {"parallelism": 8, "candidate_timeout_seconds": 300.0, "max_shards": 3},
    }
    values.update(overrides)
    return DgdrExecutorConfig.model_validate(values)


def make_retune(cfg: DgdrExecutorConfig, **overrides: Any) -> RetuneConfig:
    values: dict[str, Any] = {
        "interval_seconds": INTERVAL_S,
        "grace_seconds": GRACE_S,
        "timeout_seconds": TIMEOUT_S,
        "router": "default",
        "executor": "dgdr",
        "dgdr": cfg,
        "objective": "GoodputPerGPU",
    }
    values.update(overrides)
    return RetuneConfig.model_validate(values)


def grid(domain: Mapping[str, tuple[float, ...]]) -> list[dict[str, float]]:
    keys = sorted(domain)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(domain[key] for key in keys))
    ]


def make_request(
    domain: Mapping[str, tuple[float, ...]] = SMALL_DOMAIN,
    *,
    baseline: Optional[Mapping[str, float]] = None,
    startup: Optional[Mapping[str, float]] = None,
    deadline_s: Optional[float] = LAUNCH_S + TIMEOUT_S,
    spec_replicas: Optional[dict[str, int]] = None,
    objective: str = "GoodputPerGPU",
    router_domain: Any = "same",
    planner: Optional[dict[str, Any]] = None,
    trace_paths: tuple[str, ...] = ("/trace/segments/llm",),
) -> SweepRequest:
    points = grid(domain)
    baseline = dict(baseline or points[-1])
    startup = dict(startup or baseline)
    candidates = {RetuneParams(router=point) for point in points}
    candidates.add(RetuneParams(router=startup))
    base = RetuneParams(router=baseline, planner=planner or {})
    candidates.add(base)
    replicas = spec_replicas or {"aggregated": 3}
    disagg = "prefill" in replicas
    return SweepRequest(
        window_index=7,
        window_start_ms=WINDOW_END_MS - INTERVAL_S * 1000.0,
        window_end_ms=float(WINDOW_END_MS),
        history_start_ms=WINDOW_END_MS - INTERVAL_S * 1000.0,
        as_of_ms=WINDOW_END_MS + GRACE_S * 1000.0,
        trace_paths=trace_paths,
        arrival_speedup_ratio=1.0,
        engine=RetuneEngineSpec(
            router_mode="kv_router",
            router_config={},
            prefill_engine_args={} if disagg else None,
            decode_engine_args={} if disagg else None,
        ),
        candidates=tuple(sorted(candidates, key=RetuneParams.key)),
        baseline=base,
        replicas=dict(replicas),
        base_planner=None,
        predictor_candidates=None,
        objective=objective,  # type: ignore[arg-type]
        sla_ttft_ms=500.0,
        sla_itl_ms=30.0,
        max_parallel=1,
        timeout_s=TIMEOUT_S,
        router_domain=(
            {key: tuple(values) for key, values in domain.items()}
            if router_domain == "same"
            else router_domain
        ),
        deadline_s=deadline_s,
        spec_replicas=dict(replicas),
        generation=3,
    )


def public(point: Mapping[str, float]) -> dict[str, float]:
    return {LIVE_TO_PUBLIC[key]: value for key, value in point.items()}


def expected_score(point: Mapping[str, float], request: SweepRequest) -> float:
    score, _ = score_summary(
        default_metrics(public(point)), request.window_s, request.objective
    )
    return score


class FakeClock:
    """Wall clock for the executor; ``sleep`` advances it and runs hooks."""

    def __init__(self, start: float = LAUNCH_S) -> None:
        self.now = start
        self.sleeps: list[float] = []
        self.hooks: list[Callable[[], None]] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(seconds, 0.001)
        for hook in list(self.hooks):
            hook()


class FakeRegistry:
    def __init__(self) -> None:
        self.protected: dict[int, set[str]] = {}
        self.released: list[int] = []
        self.references: dict[int, int] = {}
        self.deleted: list[str] = []
        self.lock = threading.Lock()

    def protect(self, window_end_ms: int, request_names) -> None:
        with self.lock:
            self.protected.setdefault(window_end_ms, set()).update(request_names)
            self.references[window_end_ms] = self.references.get(window_end_ms, 0) + 1

    def release(self, window_end_ms: int) -> None:
        with self.lock:
            self.released.append(window_end_ms)
            self.references[window_end_ms] = self.references.get(window_end_ms, 0) - 1

    def balanced(self, window_end_ms: int) -> bool:
        """Every reference the sweep took on the window was released."""
        return (
            window_end_ms in self.released
            and self.references.get(window_end_ms, 0) == 0
        )

    def enqueue_delete(self, request_names) -> None:
        with self.lock:
            self.deleted.extend(request_names)


class FakeCutter:
    """Records cuts; returns ``num_requests`` rows or raises ``error``."""

    def __init__(
        self,
        *,
        num_requests: int = 1234,
        error: Optional[Exception] = None,
        on_cut: Optional[Callable[[], None]] = None,
    ) -> None:
        self.num_requests = num_requests
        self.error = error
        self.on_cut = on_cut
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        segments_root,
        start_ms,
        end_ms,
        as_of_ms,
        out_path,
        *,
        max_rows,
        deadline_s,
        cancel=None,
    ) -> WindowCutResult:
        self.calls.append(
            {
                "segments_root": segments_root,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "as_of_ms": as_of_ms,
                "out_path": out_path,
                "max_rows": max_rows,
                "deadline_s": deadline_s,
                "cancel": cancel,
            }
        )
        if self.on_cut is not None:
            self.on_cut()
        if self.error is not None:
            raise self.error
        if self.num_requests == 0:
            return WindowCutResult(
                path=None, num_requests=0, trace_block_size=None, row_counts={}
            )
        return WindowCutResult(
            path=out_path,
            num_requests=self.num_requests,
            trace_block_size=64,
            row_counts={"selected": self.num_requests},
            dropped_members=1,
            dropped_bytes=10,
        )


class CutError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Harness:
    """A fake cluster, operator, clock, registry and cutter for one window."""

    def __init__(
        self,
        tmp_path,
        *,
        cfg: Optional[DgdrExecutorConfig] = None,
        operator: bool = True,
        **retune_overrides: Any,
    ) -> None:
        self.clock = FakeClock()
        self.kube = FakeRetuneKube("test-ns", now=self.clock.time)
        self.cfg = cfg or make_cfg(mount_path=str(tmp_path / "trace"))
        self.retune = make_retune(self.cfg, **retune_overrides)
        self.registry = FakeRegistry()
        self.cutter = FakeCutter()
        self.leader = True
        self.cancel = threading.Event()
        self.abort = threading.Event()
        self.operator = FakeDgdrOperator(self.kube)
        if operator:
            self.clock.hooks.append(self.operator.step)

    def context(self, **overrides: Any) -> DgdrContext:
        values: dict[str, Any] = {
            "dgd": DGD,
            "cfg": self.cfg,
            "retune": self.retune,
            "backend": "vllm",
            "kube": self.kube,
            "registry": self.registry,
            "cut_window": self.cutter,
            "is_leader": lambda: self.leader,
            "now": self.clock.time,
            "sleep": self.clock.sleep,
            "poll_interval_s": POLL_S,
        }
        values.update(overrides)
        return DgdrContext(**values)

    def sweep(self, request: SweepRequest, **overrides: Any):
        return run_dgdr_sweep(
            request,
            cancel=self.cancel,
            abort=self.abort,
            context=self.context(**overrides),
        )

    def created(self) -> list[dict]:
        return [
            call.target
            for call in self.kube.calls_of("create_request")
            if not call.kwargs.get("dry_run")
        ]

    def request_of_shard(self, shard_id: str) -> dict:
        found = [
            obj
            for obj in self.kube.requests.values()
            if obj["metadata"]["labels"][LABEL_SHARD] == shard_id
        ]
        assert len(found) == 1, (shard_id, [o["metadata"]["name"] for o in found])
        return found[0]


def assert_scores(result, request: SweepRequest, missing=()) -> None:
    missing_keys = {RetuneParams(router=point).key() for point in missing}
    assert set(result.scores) == {candidate.key() for candidate in request.candidates}
    for candidate in request.candidates:
        key = candidate.key()
        if key in missing_keys:
            assert result.scores[key] is None, key
            assert "error" in result.metrics[key]
        else:
            assert result.scores[key] == pytest.approx(
                expected_score(candidate.router, request), rel=1e-12
            ), key
            assert result.metrics[key]["params"] == candidate.to_dict()


def kube_error(cls: type[KubeError], status: Optional[int], message: str) -> KubeError:
    """A ``Kube*`` error as ``kube.py`` raises it, with its HTTP ``status``."""

    error = cls(f"HTTP {status}: {message}")
    error.status = status  # type: ignore[attr-defined]
    return error


def lose_leadership_after_start(h: Harness, request: SweepRequest) -> None:
    """A first leader creates the window's Requests, the operator starts their
    Runs, and the leader loses leadership; the next sweep is another leader."""

    h.operator.max_phase = "started"
    h.clock.hooks.append(lambda: h.abort.set() if len(h.clock.sleeps) >= 2 else None)
    assert h.sweep(request).failure_reason == "lost_leadership"
    h.abort = threading.Event()
    h.operator.max_phase = None
    h.clock.hooks = [h.operator.step]


# ------------------------------------------------------------------ shards


def test_plan_shards_default_grid_is_three_balanced_shards_and_a_probe():
    points = grid(DEFAULT_DOMAIN)
    baseline = points[0]
    shards = plan_shards(
        DEFAULT_DOMAIN, baseline=baseline, startup=baseline, max_shards=3
    )
    grid_shards = [shard for shard in shards if shard.kind == "grid"]
    assert [shard.size for shard in grid_shards] == [36, 36, 36]
    assert [shard.id for shard in shards] == ["grid-0", "grid-1", "grid-2", "probe"]
    covered = [point for shard in grid_shards for point in shard.points]
    assert sorted(map(str, covered)) == sorted(map(str, points))
    for shard in grid_shards:
        # Each shard is the Cartesian product of its choices.
        assert shard.size == math.prod(len(v) for v in shard.choices.values())
    probe = shards[-1]
    # The middle value of each knob, a positive temperature.
    assert probe.points == ({CREDIT: 0.5, SCALE: 2.0, TEMP: 0.5},)
    assert probe.choices == {CREDIT: (0.5,), SCALE: (2.0,), TEMP: (0.5,)}


def test_plan_shards_adds_off_grid_baseline_and_startup_points():
    off_baseline = {CREDIT: 0.75, SCALE: 1.0, TEMP: 0.0}
    off_startup = {CREDIT: 1.0, SCALE: 3.0, TEMP: 0.0}
    shards = plan_shards(
        SMALL_DOMAIN,
        baseline=off_baseline,
        startup=off_startup,
        max_shards=3,
        parallelism=8,
    )
    by_id = {shard.id: shard for shard in shards}
    assert set(by_id) == {"grid-0", "point-baseline", "point-startup", "probe"}
    assert by_id["point-baseline"].points == (off_baseline,)
    assert by_id["point-baseline"].kind == "point"
    assert by_id["point-startup"].points == (off_startup,)
    # The same off-grid point once.
    shards = plan_shards(
        SMALL_DOMAIN,
        baseline=off_baseline,
        startup=off_baseline,
        max_shards=3,
        parallelism=8,
    )
    assert [shard.id for shard in shards] == ["grid-0", "point-baseline", "probe"]
    # An on-grid startup needs no shard; no probe when disabled.
    shards = plan_shards(
        SMALL_DOMAIN,
        baseline=grid(SMALL_DOMAIN)[0],
        startup=grid(SMALL_DOMAIN)[1],
        max_shards=3,
        probe=False,
        parallelism=8,
    )
    assert [shard.id for shard in shards] == ["grid-0"]


def test_plan_shards_splits_uneven_dimension_and_rejects_impossible_grids():
    domain = {CREDIT: tuple(float(v) for v in range(250))}
    zero = {CREDIT: 0.0}
    shards = plan_shards(domain, baseline=zero, startup=zero, max_shards=3, probe=False)
    assert [shard.size for shard in shards] == [84, 83, 83]
    assert shards[0].choices[CREDIT][0] == 0.0
    assert shards[-1].choices[CREDIT][-1] == 249.0
    with pytest.raises(ValueError, match="cannot be split"):
        plan_shards(domain, baseline=zero, startup=zero, max_shards=2)
    with pytest.raises(ValueError, match="cannot be split"):
        plan_shards(
            DEFAULT_DOMAIN,
            baseline=grid(DEFAULT_DOMAIN)[0],
            startup=grid(DEFAULT_DOMAIN)[0],
            max_shards=1,
        )
    point = grid(SMALL_DOMAIN)[0]

    def sizes(**kwargs):
        shards = plan_shards(
            SMALL_DOMAIN, baseline=point, startup=point, probe=False, **kwargs
        )
        return [shard.size for shard in shards]

    # Without the Run parallelism the largest shard is minimized.
    assert sizes(max_shards=16) == [1] * 8
    assert sizes(max_shards=3) == [4, 4]
    # With it, a shard is added only when it saves an evaluation round
    # (AISimulate runs the first candidate alone, so single points save one).
    assert sizes(max_shards=3, parallelism=8) == [8]
    assert sizes(max_shards=16, parallelism=8) == [1] * 8
    assert sizes(max_shards=3, parallelism=1) == [4, 4]
    # The default grid is three shards of 36 either way.
    default = plan_shards(
        DEFAULT_DOMAIN,
        baseline=grid(DEFAULT_DOMAIN)[0],
        startup=grid(DEFAULT_DOMAIN)[0],
        max_shards=3,
        parallelism=8,
    )
    assert [shard.size for shard in default] == [36, 36, 36, 1]


def test_plan_shards_validates_inputs_and_probe_falls_back_to_zero_temperature():
    point = {CREDIT: 0.5, SCALE: 1.0}
    with pytest.raises(ValueError, match="do not match"):
        plan_shards(SMALL_DOMAIN, baseline=point, startup=point, max_shards=3)
    with pytest.raises(ValueError, match="empty"):
        plan_shards({}, baseline={}, startup={}, max_shards=3)
    domain = {CREDIT: (0.0, 1.0), SCALE: (1.0,), TEMP: (0.0,)}
    shards = plan_shards(
        domain, baseline=grid(domain)[0], startup=grid(domain)[0], max_shards=3
    )
    assert shards[-1].points == ({CREDIT: 1.0, SCALE: 1.0, TEMP: 0.0},)


def test_need_seconds_follows_the_design_formula():
    cfg = make_cfg(
        cut_budget_seconds=900.0,
        job_startup_seconds=300.0,
        run={"parallelism": 8, "candidate_timeout_seconds": 1800.0},
    )
    shard = plan_shards(
        DEFAULT_DOMAIN,
        baseline=grid(DEFAULT_DOMAIN)[0],
        startup=grid(DEFAULT_DOMAIN)[0],
        max_shards=3,
    )[0]
    assert shard.size == 36
    # 1 + ceil(35 / 8) = 6 rounds.
    assert need_seconds(shard, cfg, needs_cut=True) == 900 + 300 + 6 * 1800
    assert need_seconds(shard, cfg, needs_cut=False) == 300 + 6 * 1800
    single = Shard(
        id="probe", kind="probe", choices={CREDIT: (1.0,)}, points=({CREDIT: 1.0},)
    )
    assert need_seconds(single, cfg, needs_cut=False) == 300 + 1800


# ---------------------------------------------------------------- Request


def test_spec_hash_is_canonical():
    assert spec_hash({"a": 1, "b": [1, 2]}) == spec_hash({"b": [1, 2], "a": 1})
    assert spec_hash({"a": 1}) != spec_hash({"a": 2})
    assert len(spec_hash({})) == 16
    int(spec_hash({}), 16)


def _shards(domain=SMALL_DOMAIN, **kwargs):
    point = grid(domain)[0]
    return plan_shards(
        domain, baseline=point, startup=point, max_shards=3, parallelism=8, **kwargs
    )


def test_build_request_aggregated_grid_shard():
    cfg = make_cfg()
    retune = make_retune(cfg)
    shard = _shards()[0]
    body = build_request(
        dgd=DGD,
        cfg=cfg,
        retune=retune,
        sla_ttft_ms=500,
        sla_itl_ms=30,
        window_end_ms=WINDOW_END_MS,
        shard=shard,
        spec_replicas={"aggregated": 3},
        backend="vllm",
        attempt=1,
    )
    spec = body["spec"]
    assert body["apiVersion"] == "nvidia.com/v1beta2"
    assert body["kind"] == "DynamoGraphDeploymentRequest"
    assert spec["model"] == {"name": "Qwen/Qwen3-32B"}
    assert spec["backends"] == ["vllm"]
    # 3 replicas x (tensor 2 x pipeline 1 x attention_data 1).
    assert spec["hardware"] == {"gpu": {"sku": "h200_sxm", "budget": 6}}
    assert spec["workload"] == {
        "trace": {
            "format": "dynamo",
            "source": {
                "pvc": {
                    "claimName": "request-traces",
                    "path": f"windows/llm/{WINDOW_END_MS}.jsonl.gz",
                }
            },
        }
    }
    assert spec["objective"] == {
        "mode": "Optimize",
        "metric": "GoodputPerGPU",
        "sla": {"ttftMs": 500.0, "itlMs": 30.0},
    }
    assert spec["recommendation"] == {"maxCandidates": 8}
    overrides = spec["aisimulate"]["inputOverrides"]
    workers = overrides["engine"]["workers"]
    assert workers["aggregated"]["parallelism"]["preset"][0]["replicas"] == 3
    assert workers["prefill"] is None and workers["decode"] is None
    assert overrides["engine"]["backend"] == "vllm"
    assert overrides["router"] == {
        "policy": "kv_router",
        "prefill_load_model": {"type": "none"},
        "overlap_score_credit": {"choices": [0.5, 1.0]},
        "prefill_load_scale": {"choices": [1.0, 2.0]},
        "temperature": {"choices": [0.0, 0.5]},
    }
    assert overrides["optimizer"] == {
        "algorithm": "random",
        "max_trials": 16,
        "parallelism": 8,
        "candidate_timeout_seconds": 300.0,
    }
    assert overrides["optimization"] == {"strict_sla": False}
    assert overrides["dgdr_run"] == {"generator": "direct"}
    # The config's engine is not mutated.
    assert (
        "replicas"
        not in AGG_ENGINE["workers"]["aggregated"]["parallelism"]["preset"][0]
    )

    metadata = body["metadata"]
    digest = spec_hash(spec)
    assert metadata["labels"] == {
        "nvidia.com/planner-retune-dgd": "llm",
        LABEL_DGD_UID: "dgd-uid-1",
        LABEL_WINDOW: str(WINDOW_END_MS),
        LABEL_SHARD: "grid-0",
        LABEL_SPEC: digest,
        LABEL_ATTEMPT: "1",
    }
    assert metadata["ownerReferences"] == [
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "name": "llm",
            "uid": "dgd-uid-1",
            "blockOwnerDeletion": False,
        }
    ]
    assert metadata["name"] == request_name(DGD, WINDOW_END_MS, "grid-0", digest, 1)
    assert metadata["name"].startswith("llm-rt-") and len(metadata["name"]) == 17
    assert request_name(DGD, WINDOW_END_MS, "grid-0", digest, 2) != metadata["name"]


def test_build_request_point_shard_uses_scalars_and_disagg_budget():
    cfg = make_cfg(engine=DISAGG_ENGINE)
    retune = make_retune(cfg, objective="TTFT")
    probe = _shards()[-1]
    body = build_request(
        dgd=DGD,
        cfg=cfg,
        retune=retune,
        sla_ttft_ms=200,
        sla_itl_ms=20,
        window_end_ms=WINDOW_END_MS,
        shard=probe,
        spec_replicas={"prefill": 2, "decode": 4},
        backend="vllm",
        attempt=3,
    )
    spec = body["spec"]
    # 2 x 1 + 4 x (2 x 1 x 2).
    assert spec["hardware"]["gpu"]["budget"] == 18
    assert spec["objective"]["metric"] == "TTFT"
    overrides = spec["aisimulate"]["inputOverrides"]
    workers = overrides["engine"]["workers"]
    assert workers["aggregated"] is None
    assert workers["prefill"]["parallelism"]["preset"][0]["replicas"] == 2
    assert workers["decode"]["parallelism"]["preset"][0]["replicas"] == 4
    assert overrides["router"]["temperature"] == 0.5
    assert overrides["router"]["overlap_score_credit"] == 1.0
    assert overrides["router"]["prefill_load_scale"] == 2.0
    assert overrides["optimizer"]["max_trials"] == 2
    assert spec["recommendation"]["maxCandidates"] == 1
    assert body["metadata"]["labels"][LABEL_ATTEMPT] == "3"


@pytest.mark.parametrize(
    "engine, replicas, backend, match",
    [
        ({**AGG_ENGINE, "backend": "sglang"}, {"aggregated": 1}, "vllm", "differs"),
        (
            {**AGG_ENGINE, "workers": {"aggregated": {}}},
            {"aggregated": 1},
            "vllm",
            "preset",
        ),
        (AGG_ENGINE, {"prefill": 1}, "vllm", "spec_replicas"),
        (AGG_ENGINE, {"aggregated": 0}, "vllm", "at least 1"),
    ],
)
def test_build_request_rejects_engines_it_cannot_pin(engine, replicas, backend, match):
    cfg = make_cfg(engine=engine)
    with pytest.raises(ValueError, match=match):
        build_request(
            dgd=DGD,
            cfg=cfg,
            retune=make_retune(cfg),
            sla_ttft_ms=1,
            sla_itl_ms=1,
            window_end_ms=WINDOW_END_MS,
            shard=_shards()[0],
            spec_replicas=replicas,
            backend=backend,
            attempt=1,
        )


def test_build_request_requires_the_three_dgdr_knobs():
    cfg = make_cfg()
    shard = Shard(
        id="grid-0", kind="grid", choices={CREDIT: (1.0,)}, points=({CREDIT: 1.0},)
    )
    with pytest.raises(ValueError, match="exactly"):
        build_request(
            dgd=DGD,
            cfg=cfg,
            retune=make_retune(cfg),
            sla_ttft_ms=1,
            sla_itl_ms=1,
            window_end_ms=WINDOW_END_MS,
            shard=shard,
            spec_replicas={"aggregated": 1},
            backend="vllm",
            attempt=1,
        )


def _body(shard_index=0, **kwargs):
    cfg = make_cfg(**kwargs)
    return build_request(
        dgd=DGD,
        cfg=cfg,
        retune=make_retune(cfg),
        sla_ttft_ms=500,
        sla_itl_ms=30,
        window_end_ms=WINDOW_END_MS,
        shard=_shards()[shard_index],
        spec_replicas={"aggregated": 3},
        backend="vllm",
        attempt=1,
    )


def test_input_read_back_accepts_the_operator_lowering_and_finds_drift():
    body = _body()
    body["metadata"]["namespace"] = "test-ns"
    effective = lower_request_input(body, run_name="r-1")
    assert input_mismatches(body["spec"], effective) == []

    drifted = lower_request_input(body, run_name="r-1")
    drifted["engine"]["workers"]["aggregated"]["parallelism"]["preset"][0][
        "replicas"
    ] = 1
    drifted["engine"]["workers"]["prefill"] = {}
    drifted["router"]["temperature"] = {"choices": [0.0]}
    drifted["evaluation"]["sla"]["e2e_ms"] = 10.0
    drifted["traffic"]["load"]["speedup"] = 2.0
    drifted["optimization"]["strict_sla"] = True
    problems = input_mismatches(body["spec"], drifted)
    joined = "\n".join(problems)
    for path in (
        "engine:",
        "router:",
        "evaluation.sla.e2e_ms",
        "traffic.load",
        "optimization.strict_sla",
    ):
        assert path in joined, (path, problems)

    stripped = lower_request_input(body, run_name="r-1")
    del stripped["dgdr_run"]["generator"]
    del stripped["optimization"]["constraints"]
    problems = input_mismatches(body["spec"], stripped)
    assert any(p.startswith("dgdr_run.generator") for p in problems)
    assert any("max_candidate_gpus" in p for p in problems)


def test_missing_fields_reports_pruned_leaves_and_allows_defaults():
    submitted = {"a": {"b": 1, "c": [1, 2], "gone": None}, "d": True}
    stored = {"a": {"b": 1.0, "c": [1, 2], "extra": 5}, "d": True, "default": 1}
    assert missing_fields(stored, submitted) == []
    assert missing_fields({"a": {"b": 1}}, submitted, "spec") == ["spec.a.c", "spec.d"]
    assert missing_fields(
        {"a": {"b": 1, "c": [1, 2], "gone": 3}, "d": 1}, submitted
    ) == [
        "a.gone",
        "d",
    ]


def test_crn_unsupported_compares_parseable_versions_only():
    assert not crn_unsupported("1.7.0", None, None)
    assert crn_unsupported("1.7.0", None, "1.8.0")
    assert not crn_unsupported("1.8.0", None, "1.8.0")
    assert not crn_unsupported("1.9.1rc0", None, "1.8.0")
    assert crn_unsupported(None, "nvcr.io/nvidia/planner:1.7.2@sha256:ab", "1.8")
    assert not crn_unsupported("main", "image-without-tag", "1.8.0")


# ------------------------------------------------------------ the sweep


def test_fresh_window_creates_cuts_harvests_and_scores(tmp_path):
    h = Harness(tmp_path)
    request = make_request()
    result = h.sweep(request)

    assert result.status == "ok", (result.error, result.failure_reason)
    assert result.failure_reason is None
    assert result.num_requests == 1234
    assert result.row_counts == {"selected": 1234}
    assert_scores(result, request)
    # One grid shard (8 points) and the probe.
    created = h.created()
    assert [obj["metadata"]["labels"][LABEL_SHARD] for obj in created] == [
        "grid-0",
        "probe",
    ]
    assert all(
        call.kwargs["timeout_s"] <= 10.0 for call in h.kube.calls_of("create_request")
    )
    # The window was cut once, into the claim path the Requests name.
    (cut,) = h.cutter.calls
    assert cut["out_path"] == str(
        tmp_path / "trace" / f"windows/llm/{WINDOW_END_MS}.jsonl.gz"
    )
    assert cut["segments_root"] == "/trace/segments/llm"
    assert cut["start_ms"] == WINDOW_END_MS - 3_600_000
    assert cut["end_ms"] == WINDOW_END_MS
    assert cut["as_of_ms"] == WINDOW_END_MS + 120_000
    # Absolute wall-clock seconds: now plus the cut budget.
    assert cut["deadline_s"] == LAUNCH_S + 100.0
    assert cut["max_rows"] == h.cfg.max_window_rows
    assert (tmp_path / "trace" / "windows" / "llm").is_dir()
    # Requests were protected, nothing deleted, the window released.
    assert h.registry.protected[WINDOW_END_MS] == {
        o["metadata"]["name"] for o in created
    }
    assert h.registry.deleted == []
    assert h.registry.balanced(WINDOW_END_MS)
    details = result.details
    assert details["crn_probe"]["relative_error"] == 0.0
    assert details["runtime_image"].endswith(":1.8.0")
    assert details["runtime_version"] == "1.8.0"
    assert details["missing_candidates"] == 0
    assert details["shards"]["grid-0"]["critical"] is True
    assert details["window"]["dropped_members"] == 1
    assert details["generation"] == 3
    assert result.wall_s > 0


def test_default_grid_with_off_grid_startup_runs_five_requests(tmp_path):
    h = Harness(tmp_path)
    startup = {CREDIT: 1.0, SCALE: 1.5, TEMP: 0.0}
    request = make_request(DEFAULT_DOMAIN, baseline=startup, startup=startup)
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert_scores(result, request)
    shard_ids = sorted(obj["metadata"]["labels"][LABEL_SHARD] for obj in h.created())
    assert shard_ids == ["grid-0", "grid-1", "grid-2", "point-baseline", "probe"]
    # The probe point's grid twin is critical; the baseline shard is too.
    critical = {k for k, v in result.details["shards"].items() if v["critical"]}
    assert critical == {"grid-1", "point-baseline", "probe"}


def test_disaggregated_window(tmp_path):
    h = Harness(tmp_path, cfg=make_cfg(engine=DISAGG_ENGINE, mount_path=str(tmp_path)))
    request = make_request(spec_replicas={"prefill": 1, "decode": 2})
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert_scores(result, request)
    spec = h.created()[0]["spec"]
    assert spec["hardware"]["gpu"]["budget"] == 1 + 2 * 4


def test_window_adopted_as_succeeded_is_not_cut_again(tmp_path):
    h = Harness(tmp_path)
    request = make_request()
    first = h.sweep(request)
    assert first.status == "ok"
    creates = len(h.kube.calls_of("create_request"))

    # A new leader (fresh cutter) sweeps the same window.
    h.cutter = FakeCutter(num_requests=1)
    second = h.sweep(request)
    assert second.status == "ok", second.error
    assert h.cutter.calls == []
    assert len(h.kube.calls_of("create_request")) == creates
    assert second.scores == first.scores
    # Row counts come from the Request annotation of the cut.
    assert second.num_requests == 1234
    assert second.row_counts == {"selected": 1234}
    assert all(shard["adopted"] for shard in second.details["shards"].values())


def test_lost_leadership_leaves_requests_for_the_next_leader(tmp_path):
    h = Harness(tmp_path)
    h.operator.max_phase = "started"
    h.clock.hooks.append(lambda: h.abort.set() if len(h.clock.sleeps) >= 2 else None)
    request = make_request()
    result = h.sweep(request)
    assert result.status == "failed"
    assert result.failure_reason == "lost_leadership"
    assert result.failure_reason in NON_DECISION_FAILURES
    assert h.registry.deleted == []
    assert h.registry.balanced(WINDOW_END_MS)
    names = set(h.kube.requests)
    assert len(names) == 2

    # The next leader adopts the in-flight Requests without cutting.
    h.abort = threading.Event()
    h.operator.max_phase = None
    h.clock.hooks = [h.operator.step]
    h.cutter = FakeCutter()
    second = h.sweep(request)
    assert second.status == "ok", second.error
    assert set(h.kube.requests) == names
    assert h.cutter.calls == []
    assert_scores(second, request)


def test_shutdown_returns_at_once_without_deleting(tmp_path):
    h = Harness(tmp_path)
    h.operator.max_phase = "accepted"
    h.clock.hooks.append(h.cancel.set)
    result = h.sweep(make_request())
    assert result.failure_reason == "shutdown"
    assert result.status == "failed"
    assert h.registry.deleted == []
    assert len(h.kube.requests) == 2


def test_not_leader_creates_nothing(tmp_path):
    h = Harness(tmp_path)
    h.leader = False
    result = h.sweep(make_request())
    assert result.failure_reason == "lost_leadership"
    assert h.created() == []
    assert h.cutter.calls == []


def test_abort_during_cut_reports_lost_leadership(tmp_path):
    h = Harness(tmp_path)
    h.cutter = FakeCutter(on_cut=h.abort.set, error=CutError("cancelled"))
    result = h.sweep(make_request())
    assert result.failure_reason == "lost_leadership"
    assert h.created() == []
    # The cutter's cancel event reflects abort.
    assert h.cutter.calls[0]["cancel"].is_set()


def test_stale_spec_and_older_attempts_are_queued_for_deletion(tmp_path):
    h = Harness(tmp_path)
    request = make_request()
    h.sweep(request)
    grid_request = h.request_of_shard("grid-0")
    name = grid_request["metadata"]["name"]

    # A stale shard spec and an older attempt of the same spec.
    stale = {
        "metadata": {
            "name": "llm-rt-stale",
            "labels": {
                **grid_request["metadata"]["labels"],
                LABEL_SPEC: "0" * 16,
            },
        },
        "spec": {},
    }
    older = {
        "metadata": {
            "name": "llm-rt-older",
            "labels": {**grid_request["metadata"]["labels"], LABEL_ATTEMPT: "0"},
        },
        "spec": {},
    }
    unknown_shard = {
        "metadata": {
            "name": "llm-rt-unknown",
            "labels": {**grid_request["metadata"]["labels"], LABEL_SHARD: "grid-9"},
        },
        "spec": {},
    }
    for obj in (stale, older, unknown_shard):
        h.kube.create_request(obj)
    other_window = {
        "metadata": {
            "name": "llm-rt-other",
            "labels": {**grid_request["metadata"]["labels"], LABEL_WINDOW: "1"},
        },
        "spec": {},
    }
    h.kube.create_request(other_window)
    h.registry.deleted.clear()
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert sorted(h.registry.deleted) == [
        "llm-rt-older",
        "llm-rt-stale",
        "llm-rt-unknown",
    ]
    assert result.details["shards"]["grid-0"]["request"] == name


def test_create_that_lands_but_times_out_is_adopted(tmp_path):
    h = Harness(tmp_path)
    h.kube.fail("create_request", KubeError("timeout"), landed=True)
    request = make_request()
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert len(h.kube.requests) == 2
    assert result.details["k8s_errors"] == 1
    assert_scores(result, request)


def test_create_already_exists_with_same_spec_is_adopted(tmp_path):
    h = Harness(tmp_path)
    h.kube.fail("create_request", KubeAlreadyExists("exists"), landed=True)
    result = h.sweep(make_request())
    assert result.status == "ok", result.error
    assert len(h.kube.requests) == 2


def test_create_that_fails_is_retried(tmp_path):
    h = Harness(tmp_path)
    h.kube.fail("create_request", KubeError("503"), times=2)
    result = h.sweep(make_request())
    assert result.status == "ok", result.error
    assert len(h.kube.requests) == 2
    assert result.details["k8s_errors"] == 2


@pytest.mark.parametrize(
    "error",
    [
        kube_error(KubeError, 403, "requests is forbidden: cannot create"),
        kube_error(KubeNotFound, 404, "the server could not find the resource"),
        kube_error(KubeError, 405, "method not allowed"),
        KubeNotFound("no such API"),
    ],
    ids=["403", "404", "405", "not-found"],
)
def test_permanent_create_error_fails_fast(tmp_path, error):
    h = Harness(tmp_path)
    h.kube.fail("create_request", error, times=1000)
    result = h.sweep(make_request(deadline_s=LAUNCH_S + 900.0))
    assert result.status == "failed"
    assert result.failure_reason == "k8s_error"
    assert str(error) in result.error
    # No create -> unknown -> GET -> create loop until the launch gate trips.
    assert len(h.created()) == 1
    assert h.clock.sleeps == []
    assert result.details["k8s_errors"] == 1
    assert h.kube.requests == {}


@pytest.mark.parametrize(
    "cls, status", [(KubeInvalid, 422), (KubeInvalid, 400), (KubeError, 403)]
)
def test_admission_webhook_rejection_is_request_rejected(tmp_path, cls, status):
    h = Harness(tmp_path)
    message = (
        'admission webhook "vdynamographdeploymentrequest-v1beta2.kb.io" denied '
        'the request: DynamoGraphDeploymentRequest.nvidia.com "x" is invalid: '
        'spec.hardware.gpu.sku: Invalid value: "h200_sxm": no node offers it'
    )
    h.kube.fail("create_request", kube_error(cls, status, message))
    result = h.sweep(make_request())
    assert result.status == "failed"
    assert result.failure_reason == "request_rejected"
    assert "no node offers it" in result.error
    assert len(h.created()) == 1


def test_webhook_unreachable_is_retried(tmp_path):
    h = Harness(tmp_path)
    error = kube_error(
        KubeError,
        500,
        'Internal error occurred: failed calling webhook "x.kb.io": connection refused',
    )
    h.kube.fail("create_request", error, times=2)
    request = make_request()
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert result.details["k8s_errors"] == 2
    assert_scores(result, request)


@pytest.mark.parametrize(
    "error",
    [kube_error(KubeError, 403, "requests is forbidden: cannot get"), KubeNotFound()],
    ids=["403", "not-found"],
)
def test_permanent_get_error_fails_fast(tmp_path, error):
    h = Harness(tmp_path)
    # The create's outcome is unknown, so the next poll looks the Request up.
    h.kube.fail("create_request", KubeError("timeout"))
    h.kube.fail("get_request", error, times=1000)
    result = h.sweep(make_request(deadline_s=LAUNCH_S + 900.0))
    assert result.status == "failed"
    assert result.failure_reason == "k8s_error"
    assert "get Request" in result.error
    assert h.clock.now <= LAUNCH_S + POLL_S


def test_transient_get_errors_are_retried(tmp_path):
    h = Harness(tmp_path)
    h.kube.fail("get_request", KubeError("timeout"), times=3)
    h.kube.fail("get_run", kube_error(KubeError, 503, "unavailable"), times=2)
    request = make_request()
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert result.details["k8s_errors"] == 5
    assert_scores(result, request)


def test_name_taken_by_a_foreign_request_fails_a_critical_shard(tmp_path):
    h = Harness(tmp_path)
    request = make_request()
    # Pre-create a foreign object under the grid shard's name.
    body = build_request(
        dgd=DGD,
        cfg=h.cfg,
        retune=h.retune,
        sla_ttft_ms=500,
        sla_itl_ms=30,
        window_end_ms=WINDOW_END_MS,
        shard=_shards()[0],
        spec_replicas={"aggregated": 3},
        backend="vllm",
        attempt=1,
    )
    h.kube.requests[body["metadata"]["name"]] = {
        "metadata": {"name": body["metadata"]["name"], "labels": {"foreign": "yes"}},
        "spec": {},
    }
    result = h.sweep(request)
    assert result.status == "failed"
    assert result.failure_reason == "request_conflict"
    assert body["metadata"]["name"] not in h.registry.deleted


def test_strict_rejection_is_schema_unsupported(tmp_path):
    h = Harness(tmp_path)
    h.kube.unknown_fields = {"spec.workload.trace"}
    result = h.sweep(make_request())
    assert result.status == "failed"
    assert result.failure_reason == "schema_unsupported"
    assert h.kube.requests == {}


def test_silently_pruned_field_is_schema_unsupported(tmp_path):
    h = Harness(tmp_path)
    h.kube.pruned_fields = {"spec.workload.trace.source"}
    result = h.sweep(make_request())
    assert result.failure_reason == "schema_unsupported"
    assert "spec.workload.trace.source" in result.error
    # The pruned Request is queued for deletion.
    assert h.registry.deleted == list(h.kube.requests)


def test_invalid_spec_fails_the_window_and_abandons_in_flight(tmp_path):
    h = Harness(tmp_path, operator=False)

    def reject(kube, created):
        if created["metadata"]["labels"][LABEL_SHARD] == "probe":
            kube.set_request_condition(
                created["metadata"]["name"],
                status="False",
                reason="InvalidSpec",
                message="bad engine",
            )

    h.kube.on_request_created.append(reject)
    result = h.sweep(make_request())
    assert result.failure_reason == "invalid_spec"
    assert "bad engine" in result.error
    assert sorted(h.registry.deleted) == sorted(h.kube.requests)


def test_operator_not_reconciling_after_sixty_seconds(tmp_path):
    h = Harness(tmp_path, operator=False)
    result = h.sweep(make_request())
    assert result.failure_reason == "operator_not_reconciling"
    assert h.clock.now - LAUNCH_S >= dgdr.ACCEPT_TIMEOUT_S
    assert h.clock.now - LAUNCH_S < dgdr.ACCEPT_TIMEOUT_S + 2 * POLL_S


def test_run_name_conflict_keeps_waiting(tmp_path):
    h = Harness(tmp_path, operator=False)
    phases = {"steps": 0}

    def operator():
        phases["steps"] += 1
        for name in list(h.kube.requests):
            if phases["steps"] < 30:
                h.kube.set_request_condition(
                    name, status="False", reason="RunNameConflict"
                )
        if phases["steps"] >= 30:
            h.operator.step()

    h.clock.hooks.append(operator)
    request = make_request()
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert h.clock.now - LAUNCH_S > dgdr.ACCEPT_TIMEOUT_S
    assert_scores(result, request)


def test_run_pending_until_the_deadline_times_out(tmp_path):
    h = Harness(tmp_path)
    # The operator never creates the Jobs. Like the real operator, it then
    # writes no Run status at all: the input arrives together with jobRef.
    h.operator.max_phase = "accepted"
    request = make_request(deadline_s=LAUNCH_S + 900.0)
    result = h.sweep(request)
    assert [h.kube.run_of(name)["status"] for name in h.kube.requests] == [{}, {}]
    assert result.status == "timeout"
    assert result.failure_reason == "run_pending"
    assert "Runs pending" in result.error
    assert h.clock.now >= LAUNCH_S + 900.0
    assert sorted(h.registry.deleted) == sorted(h.kube.requests)
    assert all(shard["run_pending"] for shard in result.details["shards"].values())


def test_run_pending_clears_once_the_job_exists(tmp_path, caplog):
    h = Harness(tmp_path)
    h.operator.max_phase = "accepted"

    def create_jobs_late():
        if h.clock.now - LAUNCH_S >= 3 * h.cfg.job_startup_seconds:
            h.operator.max_phase = None

    h.clock.hooks.insert(0, create_jobs_late)
    request = make_request()
    with caplog.at_level("WARNING", logger=dgdr.__name__):
        result = h.sweep(request)
    assert result.status == "ok", result.error
    assert_scores(result, request)
    assert "(run_pending)" in caplog.text
    assert not any(shard["run_pending"] for shard in result.details["shards"].values())


def test_job_created_but_not_started_is_not_run_pending(tmp_path):
    h = Harness(tmp_path, operator=False)

    def create_jobs():
        for name in list(h.kube.requests):
            if h.kube.run_of(name) is None:
                h.kube.create_job(h.kube.accept_request(name)["metadata"]["name"])

    h.clock.hooks.append(create_jobs)
    result = h.sweep(make_request(deadline_s=LAUNCH_S + 900.0))
    for name in h.kube.requests:
        status = h.kube.run_of(name)["status"]
        assert status["aisimulate"]["input"] and status["jobRef"]
        assert "startTime" not in status
    assert result.status == "timeout"
    assert result.failure_reason == "timeout"
    assert not any(shard["run_pending"] for shard in result.details["shards"].values())


def test_running_job_past_the_deadline_times_out(tmp_path):
    h = Harness(tmp_path)
    h.operator.max_phase = "started"
    result = h.sweep(make_request(deadline_s=LAUNCH_S + 900.0))
    assert result.status == "timeout"
    assert result.failure_reason == "timeout"
    assert sorted(h.registry.deleted) == sorted(h.kube.requests)


def test_launch_too_late_for_a_critical_shard(tmp_path):
    h = Harness(tmp_path)
    # The 8-point grid shard needs 100 + 60 + 2 * 300 = 760 s.
    result = h.sweep(make_request(deadline_s=LAUNCH_S + 700.0))
    assert result.status == "timeout"
    assert result.failure_reason == "launch_too_late"
    assert h.cutter.calls == []
    assert h.created() == []


def test_launch_too_late_for_an_ordinary_shard_leaves_its_points_missing(tmp_path):
    cfg = make_cfg(
        mount_path=str(tmp_path),
        run={"parallelism": 1, "candidate_timeout_seconds": 300.0, "max_shards": 3},
    )
    h = Harness(tmp_path, cfg=cfg)
    domain = {
        CREDIT: tuple(float(v) for v in range(10)),
        SCALE: (1.0,),
        TEMP: (0.0, 0.5),
    }
    baseline = {CREDIT: 2.5, SCALE: 1.0, TEMP: 0.0}  # off the grid
    shards = plan_shards(
        domain, baseline=baseline, startup=baseline, max_shards=3, parallelism=1
    )
    sizes = {shard.id: shard.size for shard in shards}
    # Ten credits in chunks of 4, 3 and 3; the probe (credit 5) is in grid-1.
    assert sizes == {
        "grid-0": 8,
        "grid-1": 6,
        "grid-2": 6,
        "point-baseline": 1,
        "probe": 1,
    }
    assert shards[-1].points[0][CREDIT] == 5.0
    need6 = need_seconds(shards[1], cfg, needs_cut=True)
    need8 = need_seconds(shards[0], cfg, needs_cut=True)
    assert need6 < need8
    request = make_request(domain, baseline=baseline, deadline_s=LAUNCH_S + need6 + 1)
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert result.details["shards"]["grid-0"]["reason"] == "launch_too_late"
    assert result.details["shards"]["grid-0"]["request"] is None
    assert result.details["shards"]["grid-2"]["state"] == "done"
    assert_scores(result, request, missing=shards[0].points)
    assert (
        "launch_too_late"
        in result.metrics[RetuneParams(router=shards[0].points[0]).key()]["error"]
    )


def test_failed_run_of_an_ordinary_shard_leaves_its_points_missing(tmp_path):
    h = Harness(tmp_path)
    startup = {CREDIT: 1.0, SCALE: 1.5, TEMP: 0.0}
    request = make_request(baseline=grid(SMALL_DOMAIN)[0], startup=startup)

    def fail_startup(kube, created):
        if created["metadata"]["labels"][LABEL_SHARD] == "point-startup":
            h.operator.fail_runs[created["metadata"]["name"]] = "SearchFailed"

    h.kube.on_request_created.append(fail_startup)
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert_scores(result, request, missing=[startup])
    shard = result.details["shards"]["point-startup"]
    assert shard["reason"] == "run_failed"
    assert "SearchFailed" in shard["message"]
    # A terminal Request is kept for audit.
    assert h.registry.deleted == []


def test_failed_run_of_the_baseline_shard_fails_the_window(tmp_path):
    h = Harness(tmp_path)

    def fail_grid(kube, created):
        if created["metadata"]["labels"][LABEL_SHARD] == "grid-0":
            h.operator.fail_runs[created["metadata"]["name"]] = "JobFailed"

    h.kube.on_request_created.append(fail_grid)
    result = h.sweep(make_request())
    assert result.status == "failed"
    assert result.failure_reason == "run_failed"
    assert "JobFailed" in result.error
    probe = h.request_of_shard("probe")["metadata"]["name"]
    grid0 = h.request_of_shard("grid-0")["metadata"]["name"]
    # The probe may already be done (kept) or in flight (deleted); the failed
    # baseline shard is kept for audit.
    assert grid0 not in h.registry.deleted
    assert set(h.registry.deleted) <= {probe}


def test_input_mismatch_of_a_created_critical_shard_fails(tmp_path):
    h = Harness(tmp_path)

    def skew(name, effective):
        effective["engine"]["context_length"] = 1024

    h.operator.mutate_input = skew
    result = h.sweep(make_request())
    assert result.status == "failed"
    assert result.failure_reason == "input_mismatch"
    assert "engine" in result.error
    assert set(h.registry.deleted) == set(h.kube.requests)


def test_adopted_attempt_with_input_mismatch_is_replaced(tmp_path):
    h = Harness(tmp_path)
    request = make_request()
    skewed: set[str] = set()

    def skew(name, effective):
        if not skewed or name in skewed:
            skewed.add(name)
            effective["router"]["prefill_load_model"] = {"type": "ais"}

    h.operator.mutate_input = skew
    # The first leader creates the Requests and loses leadership before the
    # operator accepts them.
    h.clock.hooks = [lambda: h.abort.set() if len(h.clock.sleeps) >= 2 else None]
    assert h.sweep(request).failure_reason == "lost_leadership"

    h.abort = threading.Event()
    h.clock.hooks = [h.operator.step]
    result = h.sweep(request)
    (bad,) = skewed
    assert result.status == "ok", result.error
    assert_scores(result, request)
    assert bad in h.registry.deleted
    replaced = [s for s in result.details["shards"].values() if s["attempt"] == 2]
    assert len(replaced) == 1
    # The replacement needed the window file, so the window was cut.
    assert len(h.cutter.calls) == 2


def test_adopted_attempt_whose_run_failed_is_replaced(tmp_path):
    h = Harness(tmp_path)
    request = make_request()
    lose_leadership_after_start(h, request)
    old = h.request_of_shard("grid-0")["metadata"]["name"]
    probe = h.request_of_shard("probe")["metadata"]["name"]
    # While no leader runs, the baseline shard's Job fails.
    h.kube.complete_run(
        h.kube.run_of(old)["metadata"]["name"], reason="JobFailed", message="evicted"
    )
    creates = len(h.created())

    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert_scores(result, request)
    shard = result.details["shards"]["grid-0"]
    assert shard["attempt"] == 2
    assert not shard["adopted"]
    assert shard["request"] not in (old, None)
    assert "JobFailed" in shard["message"]
    assert result.details["shards"]["probe"]["request"] == probe
    assert h.registry.deleted == [old]
    assert len(h.created()) == creates + 1
    # The new attempt needed the window file, so the window was cut again.
    assert len(h.cutter.calls) == 2


def test_adopted_failed_run_without_time_for_a_new_attempt_fails(tmp_path):
    h = Harness(tmp_path)
    # The grid shard needs 100 + 60 + 2 * 300 = 760 s with a cut.
    request = make_request(deadline_s=LAUNCH_S + 800.0)
    lose_leadership_after_start(h, request)
    old = h.request_of_shard("grid-0")["metadata"]["name"]
    h.kube.complete_run(
        h.kube.run_of(old)["metadata"]["name"], reason="JobFailed", message="evicted"
    )
    h.clock.now = LAUNCH_S + 100.0  # 700 s are left
    creates = len(h.created())

    result = h.sweep(request)
    assert result.status == "failed"
    assert result.failure_reason == "run_failed"
    assert "JobFailed" in result.error
    assert len(h.created()) == creates
    assert result.details["shards"]["grid-0"]["attempt"] == 1
    # The failed attempt is kept for audit.
    assert old not in h.registry.deleted


def test_adopted_failed_run_of_an_ordinary_shard_is_replaced(tmp_path):
    h = Harness(tmp_path)
    startup = {CREDIT: 1.0, SCALE: 1.5, TEMP: 0.0}
    request = make_request(baseline=grid(SMALL_DOMAIN)[0], startup=startup)
    lose_leadership_after_start(h, request)
    old = h.request_of_shard("point-startup")["metadata"]["name"]
    h.kube.complete_run(
        h.kube.run_of(old)["metadata"]["name"],
        reason="TraceSourceUnavailable",
        message="claim is gone",
    )

    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert_scores(result, request)
    assert result.details["shards"]["point-startup"]["attempt"] == 2
    assert h.registry.deleted == [old]


def test_failed_run_of_a_request_this_sweep_created_is_not_replaced(tmp_path):
    h = Harness(tmp_path)

    def grid0(body) -> bool:
        return body["metadata"]["labels"][LABEL_SHARD] == "grid-0"

    # The create lands but reports an error, so the sweep adopts its own
    # Request by name; its failed Run is not retried.
    h.kube.fail("create_request", KubeError("timeout"), landed=True, match=grid0)

    def fail_grid(kube, created):
        if grid0(created):
            h.operator.fail_runs[created["metadata"]["name"]] = "JobFailed"

    h.kube.on_request_created.append(fail_grid)
    result = h.sweep(make_request())
    assert result.status == "failed"
    assert result.failure_reason == "run_failed"
    shard = result.details["shards"]["grid-0"]
    assert shard["adopted"] and shard["attempt"] == 1
    assert len([b for b in h.created() if grid0(b)]) == 1


def test_input_mismatch_of_an_ordinary_shard_leaves_points_missing(tmp_path):
    h = Harness(tmp_path)
    startup = {CREDIT: 1.0, SCALE: 1.5, TEMP: 0.0}
    request = make_request(baseline=grid(SMALL_DOMAIN)[0], startup=startup)

    def skew(name, effective):
        if h.kube.requests[name]["metadata"]["labels"][LABEL_SHARD] == "point-startup":
            effective["optimizer"]["parallelism"] = 1

    h.operator.mutate_input = skew
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert_scores(result, request, missing=[startup])
    assert result.details["shards"]["point-startup"]["reason"] == "input_mismatch"


def test_missing_run_input_on_a_succeeded_run_is_a_mismatch(tmp_path):
    h = Harness(tmp_path, operator=False)

    def operator():
        for name in list(h.kube.requests):
            if h.kube.run_of(name) is None:
                h.kube.accept_request(name)
                run = h.kube.run_of(name)["metadata"]["name"]
                h.kube.start_run(run, record_input=False)
                h.kube.complete_run(run)

    h.clock.hooks.append(operator)
    result = h.sweep(make_request())
    assert result.failure_reason == "input_mismatch"
    assert "status.aisimulate.input is missing" in result.error


def test_crn_unsupported_runtime_fails_the_window(tmp_path):
    h = Harness(tmp_path)
    h.operator.dynamo_version = "1.7.0"
    result = h.sweep(make_request(), crn_min_version="1.8.0")
    assert result.failure_reason == "crn_unsupported"
    assert result.details["runtime_version"] == "1.7.0"


def test_crn_probe_mismatch_fails_the_window(tmp_path):
    def evaluate(request, point):
        metrics = default_metrics(point)
        if request["metadata"]["labels"][LABEL_SHARD] == "probe":
            metrics["goodput_output_throughput_tok_s"] *= 1.0 + 1e-6
        return metrics

    h = Harness(tmp_path)
    h.operator.evaluate = evaluate
    result = h.sweep(make_request())
    assert result.failure_reason == "crn_mismatch"
    assert result.details["crn_probe"]["relative_error"] == pytest.approx(
        1e-6, rel=1e-3
    )


def test_crn_probe_within_tolerance_passes(tmp_path):
    def evaluate(request, point):
        metrics = default_metrics(point)
        if request["metadata"]["labels"][LABEL_SHARD] == "probe":
            metrics["goodput_output_throughput_tok_s"] *= 1.0 + 1e-13
        return metrics

    h = Harness(tmp_path)
    h.operator.evaluate = evaluate
    result = h.sweep(make_request())
    assert result.status == "ok", result.error


def test_unscored_probe_is_a_crn_mismatch(tmp_path):
    probe_point = {CREDIT: 1.0, SCALE: 2.0, TEMP: 0.5}

    def evaluate(request, point):
        if request["metadata"]["labels"][LABEL_SHARD] == "probe":
            return None
        return default_metrics(point)

    h = Harness(tmp_path)
    h.operator.evaluate = evaluate
    result = h.sweep(make_request())
    assert result.failure_reason == "crn_mismatch"
    assert result.details["crn_probe"]["point"] == probe_point


def test_unpublished_and_unscorable_points_are_none(tmp_path):
    request = make_request()
    dropped = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.0}
    broken = {CREDIT: 0.5, SCALE: 2.0, TEMP: 0.0}

    def evaluate(request_obj, point):
        if point == public(dropped):
            return None
        metrics = default_metrics(point)
        if point == public(broken):
            del metrics["gpu_hours"]
        return metrics

    h = Harness(tmp_path)
    h.operator.evaluate = evaluate
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert_scores(result, request, missing=[dropped, broken])
    assert "gpu_hours" in result.metrics[RetuneParams(router=broken).key()]["error"]
    assert (
        "not published" in result.metrics[RetuneParams(router=dropped).key()]["error"]
    )
    assert result.details["missing_candidates"] == 2


def test_scores_follow_the_retune_objective(tmp_path):
    h = Harness(tmp_path, objective="TTFT")
    request = make_request(objective="TTFT")
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert_scores(result, request)
    assert h.created()[0]["spec"]["objective"]["metric"] == "TTFT"


def test_engine_drift_between_candidates_is_engine_not_pinned(tmp_path):
    h = Harness(tmp_path)

    def drift(name, parameters):
        if h.kube.requests[name]["metadata"]["labels"][LABEL_SHARD] == "probe":
            parameters["aggregated_parallelism"]["tensor"] = 4

    h.operator.mutate_parameters = drift
    result = h.sweep(make_request())
    assert result.failure_reason == "engine_not_pinned"


def test_router_policy_drift_is_engine_not_pinned(tmp_path):
    h = Harness(tmp_path)

    def drift(name, parameters):
        parameters["adapters"]["dynamo.router"]["prefill_load_model"] = {"type": "ais"}

    h.operator.mutate_parameters = drift
    result = h.sweep(make_request())
    assert result.failure_reason == "engine_not_pinned"


def test_candidate_with_an_unknown_point_is_ignored(tmp_path):
    h = Harness(tmp_path)
    request = make_request()
    odd = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.0}

    def drift(name, parameters):
        router = parameters["adapters"]["dynamo.router"]
        if (
            router["overlap_score_credit"],
            router["prefill_load_scale"],
            router["temperature"],
        ) == (0.5, 1.0, 0.0):
            router["prefill_load_scale"] = 7.0

    h.operator.mutate_parameters = drift
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert_scores(result, request, missing=[odd])
    assert "not one of the shard's values" in result.details["unmapped_candidates"][0]


def test_unscored_baseline_is_reported_for_the_controller(tmp_path):
    request = make_request(baseline=grid(SMALL_DOMAIN)[0])
    baseline = request.baseline.router

    def evaluate(request_obj, point):
        return None if point == public(baseline) else default_metrics(point)

    h = Harness(tmp_path)
    h.operator.evaluate = evaluate
    result = h.sweep(request)
    # Status ok: _decide records the window failed for its missing baseline.
    assert result.status == "ok"
    assert result.scores[request.baseline.key()] is None


def test_empty_window_creates_nothing(tmp_path):
    h = Harness(tmp_path)
    h.cutter = FakeCutter(num_requests=0)
    result = h.sweep(make_request())
    assert result.status == "empty"
    assert result.failure_reason is None
    assert result.error is None
    assert h.created() == []


@pytest.mark.parametrize(
    "error, reason",
    [
        (CutError("window_too_large"), "window_too_large"),
        (CutError("trace_corrupt"), "trace_corrupt"),
        (CutError("window_cut_timeout"), "window_cut_timeout"),
        (CutError("something_else"), "window_cut_failed"),
        (OSError("ENOENT"), "window_cut_failed"),
    ],
)
def test_cut_failures_fail_the_window(tmp_path, error, reason):
    h = Harness(tmp_path)
    h.cutter = FakeCutter(error=error)
    result = h.sweep(make_request())
    assert result.status == "failed"
    assert result.failure_reason == reason
    assert h.created() == []


def test_transient_list_failures_are_retried(tmp_path):
    h = Harness(tmp_path)
    h.kube.fail("list_requests", KubeError("timeout"), times=2)
    h.kube.fail("list_candidates", KubeError("timeout"))
    request = make_request()
    result = h.sweep(request)
    assert result.status == "ok", result.error
    assert result.details["k8s_errors"] == 3
    assert_scores(result, request)


def test_permanent_list_failure_fails_the_window(tmp_path):
    h = Harness(tmp_path)
    error = KubeError("forbidden")
    error.status = 403  # type: ignore[attr-defined]
    h.kube.fail("list_requests", error)
    result = h.sweep(make_request())
    assert result.failure_reason == "k8s_error"


def test_unexpected_exception_never_escapes(tmp_path):
    h = Harness(tmp_path)
    h.kube.fail("list_requests", RuntimeError("boom"))
    result = h.sweep(make_request())
    assert result.status == "failed"
    assert result.failure_reason == "internal_error"
    assert "boom" in result.error


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"router_domain": None}, "invalid_request"),
        ({"planner": {"load_scaling_down_sensitivity": 50}}, "invalid_request"),
        (
            {"router_domain": {CREDIT: tuple(float(v) for v in range(500))}},
            "invalid_domain",
        ),
    ],
)
def test_invalid_requests_fail_cleanly(tmp_path, kwargs, reason):
    h = Harness(tmp_path)
    if "router_domain" in kwargs and kwargs["router_domain"] is not None:
        # A domain whose keys do not match the candidates' router keys.
        request = make_request(router_domain=kwargs["router_domain"])
    else:
        request = make_request(**kwargs)
    result = h.sweep(request)
    assert result.status == "failed"
    assert result.failure_reason == reason
    assert h.created() == []


def test_invalid_engine_fails_cleanly(tmp_path):
    h = Harness(
        tmp_path,
        cfg=make_cfg(
            engine={**AGG_ENGINE, "backend": "sglang"}, mount_path=str(tmp_path)
        ),
    )
    result = h.sweep(make_request())
    assert result.failure_reason == "invalid_engine"


def test_deadline_defaults_to_the_request_timeout(tmp_path):
    h = Harness(tmp_path)
    h.operator.max_phase = "started"
    result = h.sweep(make_request(deadline_s=None))
    assert result.status == "timeout"
    assert result.details["deadline_s"] == LAUNCH_S + TIMEOUT_S
    assert h.clock.now >= LAUNCH_S + TIMEOUT_S


def test_sweep_with_the_real_window_cutter(tmp_path):
    """The executor and ``window_cut.cut_window`` agree on paths and deadline."""

    window_cut = pytest.importorskip("dynamo.planner.retune.live.window_cut")
    import gzip
    import json
    import time

    from dynamo.planner.tests.retune_sweep_helpers import trace_rows

    pod_dir = tmp_path / "trace" / "segments" / "llm" / "frontend-0"
    pod_dir.mkdir(parents=True)
    arrivals = [WINDOW_END_MS - 1000 * i for i in range(1, 6)]
    rows = trace_rows(arrivals) + trace_rows([WINDOW_END_MS + 5], prefix="late")
    with gzip.open(pod_dir / "req.000001.jsonl.gz", "wt", encoding="utf-8") as out:
        out.writelines(json.dumps(row) + "\n" for row in rows)

    h = Harness(tmp_path)
    clock = h.clock

    def real_cut(*args, deadline_s, **kwargs):
        # The fake clock runs in the past; keep the same budget in real time.
        return window_cut.cut_window(
            *args, deadline_s=time.time() + (deadline_s - clock.now), **kwargs
        )

    request = make_request(trace_paths=(str(tmp_path / "trace" / "segments" / "llm"),))
    result = h.sweep(request, cut_window=real_cut)
    assert result.status == "ok", result.error
    assert result.num_requests == 5
    window_file = tmp_path / "trace" / "windows" / "llm" / f"{WINDOW_END_MS}.jsonl.gz"
    with gzip.open(window_file, "rt", encoding="utf-8") as lines:
        assert len(lines.readlines()) == 5
    claim_path = h.created()[0]["spec"]["workload"]["trace"]["source"]["pvc"]["path"]
    assert tmp_path / "trace" / claim_path == window_file
    assert_scores(result, request)
