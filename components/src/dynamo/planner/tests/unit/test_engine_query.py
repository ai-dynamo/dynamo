# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Planner-owned engine queries over AISimulate forward-pass estimates."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional

import pytest

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.perf_model import engine_query
from dynamo.planner.core.perf_model.ais_adapter import PlannerEnginePerfModel
from dynamo.planner.core.perf_model.engine_query import (
    AISCoreEnginePerfModel,
    EngineCapacityRequest,
    EngineLoadStatus,
    EnginePerfLimits,
)
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    WorkerCapabilities,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _FakeForwardPassModel:
    def __init__(
        self,
        estimate: Callable[[list[dict[str, Any]]], Optional[float]],
    ) -> None:
        self._estimate = estimate
        self.estimate_calls: list[list[dict[str, Any]]] = []
        self.tuned_iterations: list[list[dict[str, Any]]] = []

    def estimate_forward_pass_time_ms(
        self, metrics_by_rank: list[dict[str, Any]]
    ) -> Optional[float]:
        self.estimate_calls.append(metrics_by_rank)
        return self._estimate(metrics_by_rank)

    def tune_with_fpms(self, iterations: list[list[dict[str, Any]]]) -> None:
        self.tuned_iterations.extend(iterations)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": "fallback_regression",
            "readiness": "ready",
            "retained_observations": len(self.tuned_iterations),
        }


def _model(
    *,
    worker_type: str,
    estimate: Callable[[list[dict[str, Any]]], Optional[float]],
    max_num_batched_tokens: int = 100,
    max_num_seqs: int = 16,
    max_kv_tokens: int = 100_000,
    attention_dp_size: int = 1,
    prefill_shares_decode_budget: Optional[bool] = True,
) -> tuple[AISCoreEnginePerfModel, _FakeForwardPassModel]:
    forward_pass_model = _FakeForwardPassModel(estimate)
    model = AISCoreEnginePerfModel(
        model=forward_pass_model,
        worker_type=worker_type,  # type: ignore[arg-type]
        limits=EnginePerfLimits(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_kv_tokens=max_kv_tokens,
            prefill_shares_decode_budget=prefill_shares_decode_budget,
        ),
        attention_dp_size=attention_dp_size,
        max_observations=16,
    )
    return model, forward_pass_model


def _sum_prefill_ms(metrics_by_rank: list[dict[str, Any]]) -> float:
    return float(
        sum(
            metrics["scheduled_requests"]["sum_prefill_tokens"]
            for metrics in metrics_by_rank
        )
    )


def test_best_available_uses_ais_core_wheel_facade(monkeypatch):
    sentinel = _FakeForwardPassModel(lambda _metrics: 1.0)

    class _FakeAicFacade:
        last_config = None
        last_options = None

        @classmethod
        def best_available(cls, config):
            cls.last_config = config
            return sentinel

    monkeypatch.setattr(engine_query, "AISForwardPassPerfModel", _FakeAicFacade)
    limits = EnginePerfLimits(128, 16, 10_000)
    options = {
        "max_observations": 8,
        "min_observations": 2,
        "bucket_count": 4,
        "max_num_tokens": 128,
        "max_batch_size": 16,
        "max_kv_tokens": 10_000,
    }
    config = {"model": "Qwen/Qwen3-0.6B", "worker_type": "prefill"}

    model = AISCoreEnginePerfModel.best_available(
        ais_config=config,
        worker_type="prefill",
        limits=limits,
        max_observations=options["max_observations"],
        attention_dp_size=1,
    )

    assert _FakeAicFacade.last_config is config
    assert model.diagnostics()["readiness"] == "ready"


def test_queued_prefill_uses_full_chunks_plus_tail():
    model, forward_pass_model = _model(
        worker_type="prefill",
        estimate=_sum_prefill_ms,
        max_num_batched_tokens=100,
    )
    fpm = ForwardPassMetrics(
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=3,
            sum_prefill_tokens=250,
        )
    )

    estimate_s = model.get_queued_prefill_time([fpm])

    assert estimate_s == pytest.approx(0.25)
    assert [
        call[0]["scheduled_requests"]["sum_prefill_tokens"]
        for call in forward_pass_model.estimate_calls
    ] == [100, 50]


@pytest.mark.parametrize(
    "worker_type,decode_requests,expected_chunks,expected_s",
    [
        ("aggregated", 6, (2, 1), 0.05),
        ("aggregated", 0, (8, 1), 0.02),
        ("prefill", 6, (8, 1), 0.02),
    ],
)
def test_queue_drain_reserves_decode_tokens(
    worker_type, decode_requests, expected_chunks, expected_s
):
    model, forward_pass_model = _model(
        worker_type=worker_type,
        estimate=lambda _metrics: 10.0,
        max_num_batched_tokens=8,
        max_num_seqs=8,
    )
    snapshot = ForwardPassMetrics(
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=decode_requests,
            sum_decode_kv_tokens=decode_requests * 10,
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=1, sum_prefill_tokens=9
        ),
    )

    estimate_s = model.get_queued_prefill_time([snapshot])

    assert estimate_s == pytest.approx(expected_s)
    assert (
        tuple(
            call[0]["scheduled_requests"]["sum_prefill_tokens"]
            for call in forward_pass_model.estimate_calls
        )
        == expected_chunks
    )
    for call in forward_pass_model.estimate_calls:
        scheduled = call[0]["scheduled_requests"]
        assert scheduled["sum_prefill_tokens"] + scheduled["num_decode_requests"] <= 8


@pytest.mark.parametrize(
    "queued_tokens,decode_requests,expected_s,expected_chunks",
    [
        ((9, 9), (6, 2), 0.05, ((2, 6), (2, 3), (2, 0), (1, 0))),
        ((0, 9), (8, 2), 0.02, ((0, 6), (0, 3))),
        ((1, 9), (8, 2), EngineLoadStatus.SATURATED, ()),
        ((0, 9), (9, 2), None, ()),
        ((0, 1), (9, 8), EngineLoadStatus.SATURATED, ()),
        ((0, 0), (8, 8), 0.0, ()),
    ],
)
def test_aggregate_queue_drain_uses_each_ranks_remaining_budget(
    queued_tokens, decode_requests, expected_s, expected_chunks
):
    model, forward_pass_model = _model(
        worker_type="aggregated",
        estimate=lambda _metrics: 10.0,
        max_num_batched_tokens=8,
        max_num_seqs=8,
        attention_dp_size=2,
    )
    snapshots = [
        ForwardPassMetrics(
            dp_rank=rank,
            scheduled_requests=ScheduledRequestMetrics(
                num_decode_requests=decode_requests[rank],
                sum_decode_kv_tokens=decode_requests[rank] * 10,
            ),
            queued_requests=QueuedRequestMetrics(
                num_prefill_requests=int(tokens > 0), sum_prefill_tokens=tokens
            ),
        )
        for rank, tokens in enumerate(queued_tokens)
    ]

    estimate_s = model.get_queued_prefill_time(snapshots)

    if expected_s is None or expected_s is EngineLoadStatus.SATURATED:
        assert estimate_s is expected_s
    else:
        assert estimate_s == pytest.approx(expected_s)
    assert (
        tuple(
            tuple(rank["scheduled_requests"]["sum_prefill_tokens"] for rank in call)
            for call in forward_pass_model.estimate_calls
        )
        == expected_chunks
    )
    for call in forward_pass_model.estimate_calls:
        for rank in call:
            scheduled = rank["scheduled_requests"]
            assert (
                scheduled["sum_prefill_tokens"] + scheduled["num_decode_requests"] <= 8
            )


@pytest.mark.parametrize(
    ("mixed_chunk", "expected"),
    [(False, [(224, 0)]), (True, [(192, 64), (32, 64)]), (None, [(224, 64)])],
)
def test_aggregate_prefill_pricing_uses_the_scheduler_mode(mixed_chunk, expected):
    model, forward = _model(
        worker_type="aggregated",
        estimate=lambda _: 10.0,
        max_num_batched_tokens=256,
        prefill_shares_decode_budget=mixed_chunk,
    )
    fpm = ForwardPassMetrics(
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=64, sum_decode_kv_tokens=640
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=1, sum_prefill_tokens=224
        ),
    )
    for estimate in (
        lambda: model.get_queued_prefill_time([fpm]),
        lambda: model._aggregated_ttft(224, 64, 640),
    ):
        forward.estimate_calls.clear()
        assert estimate() == pytest.approx(len(expected) * 0.01)
        rows = [call[0]["scheduled_requests"] for call in forward.estimate_calls]
        assert [
            (row["sum_prefill_tokens"], row["num_decode_requests"]) for row in rows
        ] == expected


def test_sglang_aggregate_query_rebuilds_on_mixed_chunk_change(monkeypatch):
    monkeypatch.setattr(
        engine_query.AISForwardPassPerfModel,
        "best_available",
        lambda _: _FakeForwardPassModel(lambda _metrics: 10.0),
    )
    config = PlannerConfig.model_construct(mode="agg", backend="sglang")
    limits = EngineCapabilities(
        max_num_batched_tokens=256,
        max_num_seqs=128,
        max_kv_tokens=100_000,
        sglang_mixed_chunk=False,
    )
    model = PlannerEnginePerfModel(
        worker_type="aggregated", config=config, capabilities=limits
    )
    fpm = ForwardPassMetrics(
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=64, sum_decode_kv_tokens=640
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=1, sum_prefill_tokens=224
        ),
    )

    def estimate():
        return model.estimate_queued_prefill_time(
            [fpm], max_num_batched_tokens=256, add_next_request=False
        )

    assert estimate() == pytest.approx(0.01)
    model.update_capabilities(replace(limits, sglang_mixed_chunk=True))
    assert estimate() == pytest.approx(0.02)


def test_separate_prefill_mode_prices_aggregate_capacity_as_separate_steps():
    model, forward = _model(
        worker_type="aggregated",
        estimate=lambda _: 10.0,
        max_num_batched_tokens=8,
        max_num_seqs=16,
        prefill_shares_decode_budget=False,
    )
    capacity = model.find_engine_capacity_rps(EngineCapacityRequest(isl=2, osl=32))

    assert capacity is not None
    rows = [call[0]["scheduled_requests"] for call in forward.estimate_calls]
    assert max(row["num_decode_requests"] for row in rows) == 16
    assert all(
        not row["sum_prefill_tokens"] or not row["num_decode_requests"] for row in rows
    )


@pytest.mark.parametrize(
    ("rank_count", "second_decode", "expected"),
    [(1, 64, 3), (1, 16, 2), (2, 64, 3), (2, 16, 2)],
)
def test_agg_load_scaling_keeps_budget_saturated_prefill_vote(
    monkeypatch, rank_count, second_decode, expected
):
    monkeypatch.setattr(
        engine_query.AISForwardPassPerfModel,
        "best_available",
        lambda _: _FakeForwardPassModel(lambda _metrics: 10.0),
    )
    config = PlannerConfig.model_construct(
        mode="agg",
        backend="vllm",
        optimization_target="sla",
        ttft_ms=30.0,
        itl_ms=50.0,
        enable_throughput_scaling=False,
        min_endpoint=1,
    )
    state = PlannerScalingState(
        config,
        WorkerCapabilities(
            decode=EngineCapabilities(
                max_num_batched_tokens=64, max_num_seqs=128, max_kv_tokens=100_000
            )
        ),
    )
    state._num_d_workers = 2
    fpms = {
        (str(worker), rank): ForwardPassMetrics(
            worker_id=str(worker),
            dp_rank=rank,
            scheduled_requests=ScheduledRequestMetrics(
                num_decode_requests=decodes, sum_decode_kv_tokens=decodes * 10
            ),
            queued_requests=QueuedRequestMetrics(
                num_prefill_requests=1, sum_prefill_tokens=256 if decodes == 64 else 8
            ),
        )
        for worker, decodes in enumerate((64, second_decode))
        for rank in range(rank_count)
    }

    result = state._advance_load_agg(FpmObservations(decode=fpms))

    if expected == 2:
        assert result is None
        assert state._diag_load_reason == "scale_down_refused_consolidation"
    else:
        assert result is not None
        assert result.num_decode == expected


@pytest.mark.parametrize(
    "ranks,decode_requests,tokens,expected_s",
    [
        (1, 6, 9, 0.05),
        (1, 8, 9, None),
        (1, 8, 0, 0.0),
        (2, 6, 9, 0.02),
        (2, 13, 5, 0.03),
        (2, 16, 1, None),
        (2, 0, 9, 0.02),
    ],
)
def test_aggregate_capacity_ttft_keeps_each_synthetic_rank_within_budget(
    ranks, decode_requests, tokens, expected_s
):
    model, forward_pass_model = _model(
        worker_type="aggregated",
        estimate=lambda _metrics: 10.0,
        max_num_batched_tokens=8,
        max_num_seqs=8,
        attention_dp_size=ranks,
    )

    estimate_s = model._aggregated_ttft(tokens, decode_requests, decode_requests * 10)

    if expected_s is None:
        assert estimate_s is None
    else:
        assert estimate_s == pytest.approx(expected_s)
    for call in forward_pass_model.estimate_calls:
        for rank in call:
            scheduled = rank["scheduled_requests"]
            assert (
                scheduled["sum_prefill_tokens"] + scheduled["num_decode_requests"] <= 8
            )


def test_aggregate_queue_drain_groups_identical_rounds():
    model, forward_pass_model = _model(
        worker_type="aggregated",
        estimate=lambda _metrics: 1.0,
        max_num_batched_tokens=8,
    )
    snapshot = ForwardPassMetrics(
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=7, sum_decode_kv_tokens=70
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=1, sum_prefill_tokens=1_000_000
        ),
    )

    assert model.get_queued_prefill_time([snapshot]) == pytest.approx(1000.0)
    assert len(forward_pass_model.estimate_calls) == 1


def test_aggregate_capacity_applies_ttft_after_reserving_decode_tokens():
    model, forward_pass_model = _model(
        worker_type="aggregated",
        estimate=lambda _metrics: 10.0,
        max_num_batched_tokens=8,
        max_num_seqs=8,
    )

    capacity = model.find_engine_capacity_rps(
        EngineCapacityRequest(isl=6, osl=32, ttft_sla_s=0.015, itl_sla_s=0.1)
    )

    assert capacity is not None and capacity.eligible
    assert capacity.rps == pytest.approx(3.125)
    for call in forward_pass_model.estimate_calls:
        scheduled = call[0]["scheduled_requests"]
        assert scheduled["sum_prefill_tokens"] + scheduled["num_decode_requests"] <= 8


def test_aggregated_decode_uses_learned_prefill_mix():
    model, forward_pass_model = _model(
        worker_type="aggregated",
        estimate=lambda _metrics: 12.0,
    )
    model.tune_with_fpms(
        [
            [
                ForwardPassMetrics(
                    scheduled_requests=ScheduledRequestMetrics(
                        num_prefill_requests=2,
                        sum_prefill_tokens=120,
                        num_decode_requests=4,
                        sum_decode_kv_tokens=800,
                    ),
                    wall_time=0.012,
                )
            ]
        ]
    )

    estimate_s = model.get_scheduled_decode_itl(
        [
            ForwardPassMetrics(
                scheduled_requests=ScheduledRequestMetrics(
                    num_decode_requests=5,
                    sum_decode_kv_tokens=1_000,
                )
            )
        ]
    )

    assert estimate_s == pytest.approx(0.012)
    scheduled = forward_pass_model.estimate_calls[-1][0]["scheduled_requests"]
    assert scheduled["num_prefill_requests"] == 2
    assert scheduled["sum_prefill_tokens"] == 120


def test_decode_capacity_applies_accept_length():
    model, _forward_pass_model = _model(
        worker_type="decode",
        estimate=lambda _metrics: 20.0,
        max_num_seqs=4,
    )

    capacity = model.find_engine_capacity_rps(
        EngineCapacityRequest(
            isl=100,
            osl=10,
            itl_sla_s=0.015,
            accept_length=2.0,
        )
    )

    assert capacity is not None
    assert capacity.rps == pytest.approx(40.0)
    assert capacity.itl_s == pytest.approx(0.01)
    assert capacity.eligible


def test_aggregated_capacity_derives_ttft_itl_and_e2e():
    model, _forward_pass_model = _model(
        worker_type="aggregated",
        estimate=_sum_prefill_ms,
        max_num_batched_tokens=8,
        max_num_seqs=8,
    )

    capacity = model.find_engine_capacity_rps(
        EngineCapacityRequest(
            isl=2,
            osl=2,
            ttft_sla_s=0.02,
            itl_sla_s=0.02,
            e2e_latency_sla_s=0.03,
        )
    )

    assert capacity is not None
    assert capacity.rps == pytest.approx(500.0)
    assert capacity.ttft_s == pytest.approx(0.003)
    assert capacity.itl_s == pytest.approx(0.001)
    assert capacity.e2e_latency_s == pytest.approx(0.004)
    assert capacity.eligible


def test_attention_dp_synthetic_work_is_split_across_ranks():
    model, forward_pass_model = _model(
        worker_type="decode",
        estimate=lambda _metrics: 10.0,
        max_num_seqs=2,
        attention_dp_size=2,
    )

    capacity = model.find_engine_capacity_rps(EngineCapacityRequest(isl=10, osl=2))

    assert capacity is not None
    metrics_by_rank = forward_pass_model.estimate_calls[-1]
    assert [metrics["dp_rank"] for metrics in metrics_by_rank] == [0, 1]
    assert (
        sum(
            metrics["scheduled_requests"]["num_decode_requests"]
            for metrics in metrics_by_rank
        )
        == 2
    )


def test_attention_dp_rejects_duplicate_ranks():
    model, _forward_pass_model = _model(
        worker_type="decode",
        estimate=lambda _metrics: 10.0,
        attention_dp_size=2,
    )
    duplicate_rank = ForwardPassMetrics(
        dp_rank=0,
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=1,
            sum_decode_kv_tokens=10,
        ),
    )

    with pytest.raises(ValueError, match="duplicate dp_rank"):
        model.get_scheduled_decode_itl([duplicate_rank, duplicate_rank])


def test_worker_specific_helpers_ignore_unrelated_work():
    def estimate(metrics):
        return float(
            sum(
                item["scheduled_requests"]["sum_prefill_tokens"]
                + item["scheduled_requests"]["num_decode_requests"]
                + item["scheduled_requests"]["sum_decode_kv_tokens"]
                for item in metrics
            )
        )

    prefill, prefill_forward_pass = _model(
        worker_type="prefill",
        estimate=estimate,
    )
    decode, decode_forward_pass = _model(
        worker_type="decode",
        estimate=estimate,
    )
    noisy = ForwardPassMetrics(
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=1,
            sum_prefill_tokens=900,
            num_decode_requests=3,
            sum_decode_kv_tokens=300,
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=1,
            sum_prefill_tokens=20,
            num_decode_requests=7,
            sum_decode_kv_tokens=700,
        ),
    )

    assert prefill.get_scheduled_decode_itl([noisy]) is None
    assert prefill.get_queued_prefill_time([noisy]) == pytest.approx(0.02)
    assert decode.get_queued_prefill_time([noisy]) is None
    assert decode.get_scheduled_decode_itl([noisy]) == pytest.approx(0.303)

    prefill_scheduled = prefill_forward_pass.estimate_calls[-1][0]["scheduled_requests"]
    decode_scheduled = decode_forward_pass.estimate_calls[-1][0]["scheduled_requests"]
    assert prefill_scheduled["sum_prefill_tokens"] == 20
    assert prefill_scheduled["num_decode_requests"] == 0
    assert decode_scheduled["sum_prefill_tokens"] == 0
    assert decode_scheduled["num_decode_requests"] == 3


def test_prefill_capacity_kv_hit_rate_discounts_compute():
    model, _forward_pass_model = _model(
        worker_type="prefill",
        estimate=_sum_prefill_ms,
        max_num_batched_tokens=8192,
        max_num_seqs=1,
    )

    base = model.find_engine_capacity_rps(
        EngineCapacityRequest(isl=400, osl=10, kv_hit_rate=0.0)
    )
    hit = model.find_engine_capacity_rps(
        EngineCapacityRequest(isl=400, osl=10, kv_hit_rate=0.5)
    )

    assert base is not None
    assert hit is not None
    assert base.ttft_s == pytest.approx(0.4)
    assert hit.ttft_s == pytest.approx(0.2)
    assert hit.rps == pytest.approx(2.0 * base.rps)


def test_aggregated_capacity_kv_hit_keeps_raw_context_for_kv_limit():
    request = EngineCapacityRequest(
        isl=1000,
        osl=100,
        kv_hit_rate=0.9,
    )
    one_seq, _ = _model(
        worker_type="aggregated",
        estimate=lambda _metrics: 10.0,
        max_num_batched_tokens=8192,
        max_num_seqs=1,
        max_kv_tokens=1100,
    )
    many_seqs, _ = _model(
        worker_type="aggregated",
        estimate=lambda _metrics: 10.0,
        max_num_batched_tokens=8192,
        max_num_seqs=100,
        max_kv_tokens=1100,
    )

    one_seq_capacity = one_seq.find_engine_capacity_rps(request)
    many_seq_capacity = many_seqs.find_engine_capacity_rps(request)

    assert one_seq_capacity is not None
    assert many_seq_capacity is not None
    assert one_seq_capacity.rps == pytest.approx(many_seq_capacity.rps)


def test_capacity_boundaries_preserve_none_and_ineligible_semantics():
    decode, _ = _model(
        worker_type="decode",
        estimate=lambda _metrics: 10.0,
        max_num_seqs=4,
        max_kv_tokens=50,
    )
    assert (
        decode.find_engine_capacity_rps(EngineCapacityRequest(isl=100, osl=10)) is None
    )

    prefill, _ = _model(
        worker_type="prefill",
        estimate=lambda _metrics: 10.0,
    )
    prefill_capacity = prefill.find_engine_capacity_rps(
        EngineCapacityRequest(isl=100, osl=10, itl_sla_s=1.0)
    )
    assert prefill_capacity is not None
    assert not prefill_capacity.eligible

    decode_with_capacity, _ = _model(
        worker_type="decode",
        estimate=lambda _metrics: 10.0,
    )
    decode_capacity = decode_with_capacity.find_engine_capacity_rps(
        EngineCapacityRequest(isl=100, osl=10, ttft_sla_s=1.0)
    )
    assert decode_capacity is not None
    assert not decode_capacity.eligible


def test_capacity_sampling_is_bounded_and_includes_endpoints():
    candidates = engine_query._capacity_batch_sizes(10_000)

    assert len(candidates) == engine_query.MAX_CAPACITY_SEARCH_CANDIDATES
    assert candidates[0] == 1
    assert candidates[-1] == 10_000
    assert candidates == sorted(set(candidates))


@pytest.mark.parametrize(
    "limits",
    [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        ((1 << 32), 1, 1),
    ],
)
def test_engine_limits_reject_values_outside_u32(limits):
    with pytest.raises(ValueError):
        EnginePerfLimits(*limits)


def test_aggregated_capacity_rejects_duration_overflow():
    model, _forward_pass_model = _model(
        worker_type="aggregated",
        estimate=lambda _metrics: 1.0e15,
        max_num_batched_tokens=8192,
        max_num_seqs=1,
        max_kv_tokens=(1 << 32) - 1,
    )

    with pytest.raises(ValueError, match="aggregate E2E latency overflow"):
        model.find_engine_capacity_rps(
            EngineCapacityRequest(
                isl=1,
                osl=(1 << 32) - 1,
            )
        )
