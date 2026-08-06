# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Planner-owned engine queries over AIC forward-pass estimates."""

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
    AicCoreEnginePerfModel,
    EngineCapacityRequest,
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
        *,
        source: str = "fallback_regression",
        max_correction_factor: Optional[float] = None,
    ) -> None:
        self._estimate = estimate
        self._source = source
        self._max_correction_factor = max_correction_factor
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
            "source": self._source,
            "readiness": "ready",
            "retained_observations": len(self.tuned_iterations),
        }

    def get_max_correction_factor(self) -> Optional[float]:
        return self._max_correction_factor


def _model(
    *,
    worker_type: str,
    estimate: Callable[[list[dict[str, Any]]], Optional[float]],
    max_num_batched_tokens: int = 100,
    max_num_seqs: int = 16,
    max_kv_tokens: int = 100_000,
    attention_dp_size: int = 1,
) -> tuple[AicCoreEnginePerfModel, _FakeForwardPassModel]:
    forward_pass_model = _FakeForwardPassModel(estimate)
    model = AicCoreEnginePerfModel(
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


def test_best_available_uses_aic_core_wheel_facade(monkeypatch):
    sentinel = _FakeForwardPassModel(lambda _metrics: 1.0)

    class _FakeAicFacade:
        last_config = None
        last_options = None

        @classmethod
        def best_available(cls, config, options):
            cls.last_config = config
            cls.last_options = options
            return sentinel

    monkeypatch.setattr(engine_query, "AicForwardPassPerfModel", _FakeAicFacade)
    limits = EnginePerfLimits(128, 16, 10_000)
    options = {
        "max_observations": 8,
        "min_observations": 2,
        "bucket_count": 4,
        "max_num_tokens": 128,
        "max_batch_size": 16,
        "max_kv_tokens": 10_000,
    }
    config = {"schema_version": 1, "model_name": "Qwen/Qwen3-0.6B"}

    model = AicCoreEnginePerfModel.best_available(
        aic_config=config,
        worker_type="prefill",
        limits=limits,
        options=options,
        attention_dp_size=1,
    )

    assert _FakeAicFacade.last_config is config
    assert _FakeAicFacade.last_options is options
    assert model.diagnostics()["readiness"] == "ready"


def test_best_available_builds_never_tuned_native_baseline_only_when_cap_enabled(
    monkeypatch,
):
    corrected = _FakeForwardPassModel(lambda _metrics: 80.0, source="aic")
    native = _FakeForwardPassModel(lambda _metrics: 10.0, source="aic")

    class _FakeAicFacade:
        best_available_calls = 0
        from_native_calls = 0

        @classmethod
        def best_available(cls, _config, _options):
            cls.best_available_calls += 1
            return corrected

        @classmethod
        def from_native(cls, _config, _options):
            cls.from_native_calls += 1
            return native

    monkeypatch.setattr(engine_query, "AicForwardPassPerfModel", _FakeAicFacade)
    model = AicCoreEnginePerfModel.best_available(
        aic_config={"schema_version": 1},
        worker_type="decode",
        limits=EnginePerfLimits(128, 16, 10_000),
        options={
            "max_observations": 8,
            "min_observations": 2,
            "bucket_count": 4,
            "max_num_tokens": 128,
            "max_batch_size": 16,
            "max_kv_tokens": 10_000,
        },
        attention_dp_size=1,
        max_correction_factor=2.0,
    )
    outlier = ForwardPassMetrics(
        wall_time=1.0,
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=1,
            sum_decode_kv_tokens=100,
        ),
    )
    model.tune_with_fpms([[outlier]])
    native_tuning_calls = len(native.estimate_calls)
    estimate_s = model.get_scheduled_decode_itl([outlier])

    assert corrected.tuned_iterations[0][0]["wall_time"] == pytest.approx(0.02)
    assert estimate_s == pytest.approx(0.08)
    assert len(native.estimate_calls) == native_tuning_calls
    assert _FakeAicFacade.best_available_calls == 1
    assert _FakeAicFacade.from_native_calls == 1
    assert model.diagnostics()["planner_correction_cap"] == {
        "configured_max_factor": 2.0,
        "enabled": True,
        "scope": "aic_tuning_input",
        "observed_aic_max_factor": None,
        "stored_aic_corrections_capped": True,
    }


def test_best_available_does_not_build_native_baseline_for_regression_fallback(
    monkeypatch,
):
    class _FallbackWithoutCorrectionGetter:
        def estimate_forward_pass_time_ms(self, _metrics):
            return 80.0

        def tune_with_fpms(self, _iterations):
            return None

        def diagnostics(self):
            return {
                "source": "fallback_regression",
                "readiness": "ready",
                "retained_observations": 0,
            }

    fallback = _FallbackWithoutCorrectionGetter()

    class _FakeAicFacade:
        @classmethod
        def best_available(cls, _config, _options):
            return fallback

        @classmethod
        def from_native(cls, _config, _options):
            raise AssertionError("regression fallback must not build a native baseline")

    monkeypatch.setattr(engine_query, "AicForwardPassPerfModel", _FakeAicFacade)
    model = AicCoreEnginePerfModel.best_available(
        aic_config={"schema_version": 1},
        worker_type="decode",
        limits=EnginePerfLimits(128, 16, 10_000),
        options={
            "max_observations": 8,
            "min_observations": 2,
            "bucket_count": 4,
            "max_num_tokens": 128,
            "max_batch_size": 16,
            "max_kv_tokens": 10_000,
        },
        attention_dp_size=1,
        max_correction_factor=2.0,
    )

    assert model.diagnostics()["planner_correction_cap"]["enabled"] is False


def test_native_correction_cap_fails_closed_when_baseline_construction_fails(
    monkeypatch,
):
    corrected = _FakeForwardPassModel(lambda _metrics: 80.0, source="aic")

    class _FakeAicFacade:
        @classmethod
        def best_available(cls, _config, _options):
            return corrected

        @classmethod
        def from_native(cls, _config, _options):
            raise RuntimeError("native baseline unavailable")

    monkeypatch.setattr(engine_query, "AicForwardPassPerfModel", _FakeAicFacade)

    with pytest.raises(RuntimeError, match="native baseline unavailable"):
        AicCoreEnginePerfModel.best_available(
            aic_config={"schema_version": 1},
            worker_type="decode",
            limits=EnginePerfLimits(128, 16, 10_000),
            options={
                "max_observations": 8,
                "min_observations": 2,
                "bucket_count": 4,
                "max_num_tokens": 128,
                "max_batch_size": 16,
                "max_kv_tokens": 10_000,
            },
            attention_dp_size=1,
            max_correction_factor=2.0,
        )


def test_native_correction_cap_is_absolute_across_repeated_outliers():
    corrected = _FakeForwardPassModel(lambda _metrics: 10.0)
    native = _FakeForwardPassModel(lambda _metrics: 10.0)
    model = AicCoreEnginePerfModel(
        model=corrected,
        uncorrected_native_model=native,
        max_correction_factor=2.0,
        worker_type="decode",
        limits=EnginePerfLimits(128, 16, 10_000),
        attention_dp_size=1,
        max_observations=8,
    )
    for wall_time in (0.02, 0.04, 0.08):
        model.tune_with_fpms(
            [
                [
                    ForwardPassMetrics(
                        wall_time=wall_time,
                        scheduled_requests=ScheduledRequestMetrics(
                            num_decode_requests=1,
                            sum_decode_kv_tokens=100,
                        ),
                    )
                ]
            ]
        )

    tuned_wall_times = [
        iteration[0]["wall_time"] for iteration in corrected.tuned_iterations
    ]
    assert tuned_wall_times == pytest.approx([0.02, 0.02, 0.02])


def test_native_correction_cap_preserves_attention_dp_max_wall_semantics():
    corrected = _FakeForwardPassModel(lambda _metrics: 10.0)
    model = AicCoreEnginePerfModel(
        model=corrected,
        uncorrected_native_model=_FakeForwardPassModel(lambda _metrics: 10.0),
        max_correction_factor=2.0,
        worker_type="decode",
        limits=EnginePerfLimits(128, 16, 10_000),
        attention_dp_size=2,
        max_observations=8,
    )

    model.tune_with_fpms(
        [
            [
                ForwardPassMetrics(
                    dp_rank=0,
                    wall_time=0.01,
                    scheduled_requests=ScheduledRequestMetrics(
                        num_decode_requests=1,
                        sum_decode_kv_tokens=100,
                    ),
                ),
                ForwardPassMetrics(
                    dp_rank=1,
                    wall_time=1.0,
                    scheduled_requests=ScheduledRequestMetrics(
                        num_decode_requests=1,
                        sum_decode_kv_tokens=100,
                    ),
                ),
            ]
        ]
    )

    tuned = corrected.tuned_iterations[0]
    assert [metrics["wall_time"] for metrics in tuned] == pytest.approx([0.01, 0.02])


def test_native_correction_cap_does_not_clamp_downward_correction():
    corrected = _FakeForwardPassModel(lambda _metrics: 10.0)
    model = AicCoreEnginePerfModel(
        model=corrected,
        uncorrected_native_model=_FakeForwardPassModel(lambda _metrics: 10.0),
        max_correction_factor=2.0,
        worker_type="decode",
        limits=EnginePerfLimits(128, 16, 10_000),
        attention_dp_size=1,
        max_observations=8,
    )

    model.tune_with_fpms(
        [
            [
                ForwardPassMetrics(
                    wall_time=0.005,
                    scheduled_requests=ScheduledRequestMetrics(
                        num_decode_requests=1,
                        sum_decode_kv_tokens=100,
                    ),
                )
            ]
        ]
    )

    assert corrected.tuned_iterations[0][0]["wall_time"] == pytest.approx(0.005)


def test_native_correction_cap_disabled_preserves_corrected_estimate():
    corrected = _FakeForwardPassModel(lambda _metrics: 80.0)
    model = AicCoreEnginePerfModel(
        model=corrected,
        worker_type="decode",
        limits=EnginePerfLimits(128, 16, 10_000),
        attention_dp_size=1,
        max_observations=8,
    )

    observation = ForwardPassMetrics(
        wall_time=1.0,
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=1,
            sum_decode_kv_tokens=100,
        ),
    )
    model.tune_with_fpms([[observation]])
    estimate_s = model.get_scheduled_decode_itl([observation])

    assert corrected.tuned_iterations[0][0]["wall_time"] == 1.0
    assert estimate_s == pytest.approx(0.08)


def test_native_correction_cap_fails_closed_when_baseline_estimate_is_unavailable():
    model = AicCoreEnginePerfModel(
        model=_FakeForwardPassModel(lambda _metrics: 80.0),
        uncorrected_native_model=_FakeForwardPassModel(lambda _metrics: None),
        max_correction_factor=2.0,
        worker_type="decode",
        limits=EnginePerfLimits(128, 16, 10_000),
        attention_dp_size=1,
        max_observations=8,
    )

    with pytest.raises(ValueError, match="native AIC.*unavailable"):
        model.tune_with_fpms(
            [
                [
                    ForwardPassMetrics(
                        wall_time=1.0,
                        scheduled_requests=ScheduledRequestMetrics(
                            num_decode_requests=1,
                            sum_decode_kv_tokens=100,
                        ),
                    )
                ]
            ]
        )


@pytest.mark.parametrize("factor", [0.0, 0.5, float("inf"), float("nan")])
def test_native_correction_cap_rejects_invalid_values(factor):
    with pytest.raises(ValueError, match="must be finite and >= 1.0"):
        AicCoreEnginePerfModel(
            model=_FakeForwardPassModel(lambda _metrics: 10.0),
            max_correction_factor=factor,
            worker_type="decode",
            limits=EnginePerfLimits(128, 16, 10_000),
            attention_dp_size=1,
            max_observations=8,
        )


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
