# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Planner-owned engine queries over AISimulate forward-pass estimates."""

from __future__ import annotations

from typing import Any, Callable, Optional

import pytest

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.core.perf_model import engine_query
from dynamo.planner.core.perf_model.engine_query import (
    AISCoreEnginePerfModel,
    EngineCapacityRequest,
    EngineLoadStatus,
    EnginePerfLimits,
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
) -> tuple[AISCoreEnginePerfModel, _FakeForwardPassModel]:
    forward_pass_model = _FakeForwardPassModel(estimate)
    model = AISCoreEnginePerfModel(
        model=forward_pass_model,
        worker_type=worker_type,  # type: ignore[arg-type]
        limits=EnginePerfLimits(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_kv_tokens=max_kv_tokens,
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


def _configured_spec_model(
    monkeypatch,
    *,
    mode="op_level",
    backend="vllm",
    worker_type="aggregated",
    ranks=1,
    budget=16,
    sequences=16,
    width=4,
):
    config = {
        "model": "fixture",
        "backend": backend,
        "worker_type": worker_type,
        "estimation_mode": "auto",
        "attention_dp": ranks,
        "nextn": width - 1,
    }
    if mode == "op_level" and backend == "vllm":
        config["nextn"] = 0
        config["speculation"] = {
            "kind": "ngram",
            "params": {"num_speculative_tokens": width - 1},
        }
    native = _FakeForwardPassModel(lambda _metrics: 10.0)
    native.diagnostics = lambda: {
        "source": "aic",
        "readiness": "ready",
        "provenance": {"selected_estimation_mode": mode, "config": config},
    }

    class Facade:
        @classmethod
        def best_available(cls, _config):
            return native

    monkeypatch.setattr(engine_query, "AISForwardPassPerfModel", Facade)
    model = AISCoreEnginePerfModel.best_available(
        ais_config=config,
        worker_type=worker_type,
        limits=EnginePerfLimits(budget, sequences, 1_000_000),
        attention_dp_size=ranks,
        max_observations=16,
    )
    return model, native


def _decode_fpm(count=1, *, rank=0, queued=0):
    return ForwardPassMetrics(
        dp_rank=rank,
        wall_time=0.01,
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=count,
            sum_decode_kv_tokens=count * 128,
            var_decode_kv_tokens=9,
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=int(queued > 0), sum_prefill_tokens=queued
        ),
    )


def _prefill_fpm(tokens, *, requests=1):
    return ForwardPassMetrics(
        wall_time=0.01,
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=requests, sum_prefill_tokens=tokens
        ),
    )


@pytest.mark.parametrize("backend,width", [("vllm", 4), ("sglang", 4), ("vllm", 8)])
def test_speculative_op_queries_pack_each_rank_without_changing_logical_kv(
    monkeypatch, backend, width
):
    model, native = _configured_spec_model(
        monkeypatch, backend=backend, width=width, ranks=2, budget=100
    )
    snapshots = [_decode_fpm(count, rank=rank) for rank, count in enumerate((2, 1))]
    assert model.get_scheduled_decode_itl(snapshots) == pytest.approx(0.01)
    scheduled = [x["scheduled_requests"] for x in native.estimate_calls[-1]]
    assert [x["num_decode_requests"] for x in scheduled] == [2 * width, width]
    assert [x["sum_decode_kv_tokens"] for x in scheduled] == [256 * width, 128 * width]
    assert [x["var_decode_kv_tokens"] for x in scheduled] == [9, 9]
    assert snapshots[0].scheduled_requests.num_decode_requests == 2
    assert snapshots[0].scheduled_requests.sum_decode_kv_tokens == 256


@pytest.mark.parametrize("mode", ["fpm_regression", "fpm_interpolation"])
def test_selected_fpm_modes_keep_speculative_queries_and_training_logical(
    monkeypatch, mode
):
    model, native = _configured_spec_model(monkeypatch, mode=mode)
    fpm = _decode_fpm()
    model.tune_with_fpms([[fpm]])
    assert model.get_scheduled_decode_itl([fpm]) == pytest.approx(0.01)
    assert (
        native.tuned_iterations[0][0]["scheduled_requests"]["num_decode_requests"] == 1
    )
    assert (
        native.estimate_calls[-1][0]["scheduled_requests"]["sum_decode_kv_tokens"]
        == 128
    )


def test_speculative_op_withholds_decode_iterations_but_keeps_raw_load(monkeypatch):
    model, native = _configured_spec_model(monkeypatch, ranks=2, budget=200)
    pure = [
        ForwardPassMetrics(
            dp_rank=rank,
            scheduled_requests=ScheduledRequestMetrics(
                num_prefill_requests=1, sum_prefill_tokens=8
            ),
            queued_requests=QueuedRequestMetrics(num_decode_requests=1),
            wall_time=0.01,
        )
        for rank in range(2)
    ]
    mixed = [
        _prefill_fpm(100),
        ForwardPassMetrics(
            dp_rank=1,
            scheduled_requests=ScheduledRequestMetrics(
                num_prefill_requests=1,
                sum_prefill_tokens=1,
                num_decode_requests=1,
                sum_decode_kv_tokens=30,
            ),
            wall_time=0.01,
        ),
    ]
    model.tune_with_fpms([pure, mixed])
    assert len(native.tuned_iterations) == 1
    assert len(native.tuned_iterations[0]) == 2
    assert all(
        x["scheduled_requests"]["num_decode_requests"] == 0
        for x in native.tuned_iterations[0]
    )
    model.get_scheduled_decode_itl([_decode_fpm(rank=rank) for rank in range(2)])
    # The busiest-rank samples have 8 and 100 prefill tokens. Their mean,
    # (8 + 100) / 2 = 54, is synthesized for each rank, not divided by DP.
    assert [
        x["scheduled_requests"]["sum_prefill_tokens"] for x in native.estimate_calls[-1]
    ] == [54, 54]


@pytest.mark.parametrize(
    "loads,expected_chunks",
    [([(3, 10), (1, 10)], [[4, 10], [4, 0], [2, 0]]), ([(4, 0), (1, 10)], [[0, 10]])],
)
def test_speculative_queue_drain_uses_rank_local_headroom(
    monkeypatch, loads, expected_chunks
):
    model, native = _configured_spec_model(monkeypatch, ranks=2)
    snapshots = [
        _decode_fpm(decode, rank=rank, queued=queued)
        for rank, (decode, queued) in enumerate(loads)
    ]
    assert model.get_queued_prefill_time(snapshots) == pytest.approx(
        0.01 * len(expected_chunks)
    )
    assert [
        [x["scheduled_requests"]["sum_prefill_tokens"] for x in call]
        for call in native.estimate_calls
    ] == expected_chunks


def test_speculative_queue_reports_no_headroom_but_empty_queue_is_zero(monkeypatch):
    model, native = _configured_spec_model(monkeypatch, budget=8)
    assert model.get_queued_prefill_time([_decode_fpm(2)]) == 0
    assert (
        model.get_queued_prefill_time([_decode_fpm(2, queued=1)])
        is EngineLoadStatus.SATURATED
    )
    assert not native.estimate_calls


@pytest.mark.parametrize(
    "role,engine_request,expected_rps",
    [
        ("decode", EngineCapacityRequest(isl=100, osl=10, accept_length=4), 160),
        ("aggregated", EngineCapacityRequest(isl=1, osl=10), 20),
    ],
)
def test_speculative_capacity_obeys_rank_budget_and_logical_request_count(
    monkeypatch, role, engine_request, expected_rps
):
    model, native = _configured_spec_model(
        monkeypatch, worker_type=role, ranks=2, budget=8, sequences=100
    )
    capacity = model.find_engine_capacity_rps(engine_request)
    assert capacity is not None and capacity.rps == pytest.approx(expected_rps)
    assert all(
        x["scheduled_requests"]["sum_prefill_tokens"]
        + x["scheduled_requests"]["num_decode_requests"]
        <= 8
        for call in native.estimate_calls
        for x in call
    )


@pytest.mark.parametrize("field", ["num_decode_requests", "sum_decode_kv_tokens"])
def test_speculative_packing_rejects_u32_overflow_before_native_call(
    monkeypatch, field
):
    model, native = _configured_spec_model(monkeypatch, backend="sglang")
    values = {"num_decode_requests": 1, "sum_decode_kv_tokens": 128}
    values[field] = (1 << 32) - 1
    fpm = ForwardPassMetrics(scheduled_requests=ScheduledRequestMetrics(**values))
    with pytest.raises(ValueError, match="overflow"):
        model.get_scheduled_decode_itl([fpm])
    assert not native.estimate_calls


@pytest.mark.parametrize(
    "provenance", [{}, {"selected_estimation_mode": "op_level", "config": None}]
)
def test_speculation_requires_resolved_mode_and_identity(monkeypatch, provenance):
    native = _FakeForwardPassModel(lambda _metrics: 10.0)
    native.diagnostics = lambda: {"provenance": provenance}

    class Facade:
        @classmethod
        def best_available(cls, _config):
            return native

    monkeypatch.setattr(engine_query, "AISForwardPassPerfModel", Facade)
    with pytest.raises(ValueError, match="resolved estimator provenance"):
        AISCoreEnginePerfModel.best_available(
            ais_config={"worker_type": "decode", "nextn": 3},
            worker_type="decode",
            limits=EnginePerfLimits(16, 16, 1000),
            max_observations=16,
            attention_dp_size=1,
        )


@pytest.mark.parametrize(
    "budget,decode,learned,expected_tokens,expected_requests",
    [(16, 3, 16, 4, 2), (9, 2, 8, 1, 1)],
)
def test_speculative_itl_crops_only_the_learned_prefill_mix(
    monkeypatch, budget, decode, learned, expected_tokens, expected_requests
):
    model, native = _configured_spec_model(monkeypatch, budget=budget)
    model.tune_with_fpms([[_prefill_fpm(learned, requests=2)]])
    assert model.get_scheduled_decode_itl([_decode_fpm(decode)]) == pytest.approx(0.01)
    payload = native.estimate_calls[-1][0]["scheduled_requests"]
    assert (payload["sum_prefill_tokens"], payload["num_prefill_requests"]) == (
        expected_tokens,
        expected_requests,
    )
    count = len(native.estimate_calls)
    actual = ForwardPassMetrics(
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=2,
            sum_prefill_tokens=8,
            num_decode_requests=decode,
            sum_decode_kv_tokens=decode * 128,
        )
    )
    assert model.get_scheduled_decode_itl([actual]) is EngineLoadStatus.SATURATED
    assert len(native.estimate_calls) == count


@pytest.mark.parametrize("role", ["decode", "aggregated", "prefill"])
def test_speculative_width_must_fit_one_verification_round(monkeypatch, role):
    if role != "prefill":
        with pytest.raises(ValueError, match="width exceeds the per-rank token budget"):
            _configured_spec_model(monkeypatch, budget=3, worker_type=role)
    else:
        # A prefill-only worker has no decode verification round to reserve.
        model, _ = _configured_spec_model(monkeypatch, budget=3, worker_type=role)
        assert model.get_queued_prefill_time(
            [_decode_fpm(0, queued=8)]
        ) == pytest.approx(0.03)


def test_other_backends_do_not_inherit_vllm_verification_budget(monkeypatch):
    model, native = _configured_spec_model(monkeypatch, backend="sglang")
    assert model.get_queued_prefill_time([_decode_fpm(3, queued=10)]) == pytest.approx(
        0.01
    )
    payload = native.estimate_calls[-1][0]["scheduled_requests"]
    assert payload["sum_prefill_tokens"] == 10
    assert payload["num_decode_requests"] == 12


@pytest.mark.parametrize("decode, expected", [(0, 0.02), (3, 0.05)])
def test_speculative_ttft_reserves_full_verification_work(
    monkeypatch, decode, expected
):
    model, native = _configured_spec_model(monkeypatch)
    # B16: no decode leaves chunks 16+4; three W4 decodes leave five chunks of 4.
    assert model.get_queued_prefill_time(
        [_decode_fpm(decode, queued=20)]
    ) == pytest.approx(expected)
    assert all(
        x["scheduled_requests"]["sum_prefill_tokens"]
        + x["scheduled_requests"]["num_decode_requests"]
        <= 16
        for call in native.estimate_calls
        for x in call
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
