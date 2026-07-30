# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Planner's AIC core performance-model adapter."""

import pytest

try:
    from dynamo.common.forward_pass_metrics import (
        ForwardPassMetrics,
        QueuedRequestMetrics,
        ScheduledRequestMetrics,
    )
except ImportError:
    pytest.skip("forward_pass_metrics not available", allow_module_level=True)

from dynamo.planner.config.parallelization import PickedParallelConfig
from dynamo.planner.config.planner_config import AICPerfModelSpec, PlannerConfig
from dynamo.planner.core.perf_model import aic_adapter
from dynamo.planner.core.perf_model.aic_adapter import PlannerEnginePerfModel
from dynamo.planner.core.types import EngineCapabilities

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _FakeEngineFactory:
    next_model = None
    last_kwargs = None

    @classmethod
    def best_available(cls, **kwargs):
        assert cls.next_model is not None
        cls.last_kwargs = kwargs
        return cls.next_model


class _FakeEngineModel:
    def __init__(self, *, diagnostics, queued_prefill_time=None, capacity=None):
        self._diagnostics = diagnostics
        self._queued_prefill_time = queued_prefill_time
        self._capacity = capacity
        self.tuned_iterations = []
        self.capacity_requests = []

    def diagnostics(self):
        return self._diagnostics

    def tune_with_fpms(self, iterations):
        self.tuned_iterations.extend(iterations)

    def get_queued_prefill_time(self, _metrics_by_rank):
        return self._queued_prefill_time

    def find_engine_capacity_rps(self, request):
        self.capacity_requests.append(request)
        return self._capacity


class _FakeCapacity:
    def __init__(
        self,
        *,
        rps,
        itl_s=None,
        ttft_s=None,
        e2e_latency_s=None,
        eligible=True,
    ):
        self.rps = rps
        self.itl_s = itl_s
        self.ttft_s = ttft_s
        self.e2e_latency_s = e2e_latency_s
        self.eligible = eligible


def _install_fake_engine(monkeypatch, fake_model: _FakeEngineModel) -> None:
    _FakeEngineFactory.next_model = fake_model
    _FakeEngineFactory.last_kwargs = None
    monkeypatch.setattr(aic_adapter, "AicCoreEnginePerfModel", _FakeEngineFactory)


def _config(
    *,
    dp: int = 1,
    min_observations: int = 5,
    speculative_nextn: int = 0,
) -> PlannerConfig:
    pick = PickedParallelConfig(dp=dp)
    return PlannerConfig.model_construct(
        aic_perf_model=AICPerfModelSpec.model_construct(
            hf_id="Qwen/Qwen3-0.6B",
            system="h200_sxm",
            backend="vllm",
            prefill_pick=pick,
            decode_pick=pick,
        ),
        max_num_fpm_samples=16,
        load_min_observations=min_observations,
        fpm_sample_bucket_size=16,
        ttft_ms=500.0,
        itl_ms=50.0,
        speculative_nextn=speculative_nextn,
    )


def _caps(*, speculative_nextn: int | None = None) -> EngineCapabilities:
    return EngineCapabilities(
        max_num_batched_tokens=1024,
        max_num_seqs=128,
        max_kv_tokens=100_000,
        kv_cache_block_size=16,
        speculative_nextn=speculative_nextn,
    )


def _prefill_fpm(
    *,
    scheduled_tokens: int = 0,
    queued_tokens: int = 0,
    wall_time: float = 0.01,
) -> ForwardPassMetrics:
    return ForwardPassMetrics(
        worker_id="w0",
        dp_rank=0,
        wall_time=wall_time,
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=1 if scheduled_tokens else 0,
            sum_prefill_tokens=scheduled_tokens,
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=1 if queued_tokens else 0,
            sum_prefill_tokens=queued_tokens,
        ),
    )


def _decode_fpm(
    *,
    requests: int = 1,
    kv_tokens: int = 128,
    wall_time: float = 0.01,
) -> ForwardPassMetrics:
    return ForwardPassMetrics(
        worker_id="w0",
        dp_rank=0,
        wall_time=wall_time,
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=requests,
            sum_decode_kv_tokens=kv_tokens,
        ),
    )


def test_aic_diagnostics_gate_sufficient_data(monkeypatch):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "fallback_regression",
            "readiness": "insufficient_data",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _install_fake_engine(monkeypatch, fake)
    model = PlannerEnginePerfModel(
        worker_type="prefill",
        config=_config(),
        capabilities=_caps(),
    )

    assert not model.has_sufficient_data()

    fake._diagnostics["readiness"] = "ready"
    assert model.has_sufficient_data()


def test_missing_capability_fields_use_engine_query_defaults(monkeypatch):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "fallback_regression",
            "readiness": "insufficient_data",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _install_fake_engine(monkeypatch, fake)

    PlannerEnginePerfModel(
        worker_type="prefill",
        config=_config(),
        capabilities=EngineCapabilities(max_num_batched_tokens=2048),
    )

    assert _FakeEngineFactory.last_kwargs is not None
    limits = _FakeEngineFactory.last_kwargs["limits"]
    assert limits.max_num_batched_tokens == 2048
    assert limits.max_num_seqs == aic_adapter.DEFAULT_MAX_NUM_SEQS
    assert limits.max_kv_tokens == aic_adapter.DEFAULT_MAX_KV_TOKENS


@pytest.mark.parametrize(
    ("capability_nextn", "config_nextn", "expected_nextn"),
    [
        (3, 5, "3"),
        (None, 4, "4"),
    ],
)
def test_aic_config_requests_raw_spec_decode_iteration_time(
    monkeypatch,
    capability_nextn,
    config_nextn,
    expected_nextn,
):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _install_fake_engine(monkeypatch, fake)

    PlannerEnginePerfModel(
        worker_type="decode",
        config=_config(speculative_nextn=config_nextn),
        capabilities=_caps(speculative_nextn=capability_nextn),
    )

    assert _FakeEngineFactory.last_kwargs is not None
    aic_config = _FakeEngineFactory.last_kwargs["aic_config"]
    assert aic_config["nextn"] == int(expected_nextn)


def test_capability_nextn_update_rebuilds_aic_model_and_replays_fpms(monkeypatch):
    first = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _install_fake_engine(monkeypatch, first)
    model = PlannerEnginePerfModel(
        worker_type="decode",
        config=_config(),
        capabilities=_caps(),
    )
    fpm = _decode_fpm(requests=2, kv_tokens=256, wall_time=0.02)
    model.add_observations({("w0", 0): fpm})
    assert first.tuned_iterations == [[fpm]]

    second = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _FakeEngineFactory.next_model = second

    model.update_capabilities(_caps(speculative_nextn=3))

    assert _FakeEngineFactory.last_kwargs is not None
    aic_config = _FakeEngineFactory.last_kwargs["aic_config"]
    assert aic_config["nextn"] == 3
    assert second.tuned_iterations == [[fpm]]


def test_aic_none_result_remains_unavailable(monkeypatch):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        },
        queued_prefill_time=None,
    )
    _install_fake_engine(monkeypatch, fake)
    model = PlannerEnginePerfModel(
        worker_type="prefill",
        config=_config(min_observations=1),
        capabilities=_caps(),
    )
    model.load_benchmark_fpms([_prefill_fpm(scheduled_tokens=256, wall_time=0.02)])

    result = model.estimate_queued_prefill_time(
        [_prefill_fpm(queued_tokens=512)],
        max_num_batched_tokens=1024,
    )

    assert result is None


def test_flat_bootstrap_fpms_skip_aic_tuning_for_attention_dp(monkeypatch):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _install_fake_engine(monkeypatch, fake)
    model = PlannerEnginePerfModel(
        worker_type="prefill",
        config=_config(dp=2, min_observations=1),
        capabilities=_caps(),
    )

    model.load_benchmark_fpms([_prefill_fpm(scheduled_tokens=256, wall_time=0.02)])

    assert fake.tuned_iterations == []
    assert model.num_observations == 0
    assert model.avg_isl == 256


def test_capacity_request_passes_kv_hit_rate(monkeypatch):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _install_fake_engine(monkeypatch, fake)
    model = PlannerEnginePerfModel(
        worker_type="aggregated",
        config=_config(min_observations=1),
        capabilities=_caps(),
    )

    assert model.find_engine_capacity_rps(isl=1000, osl=100, kv_hit_rate=0.4) is None
    assert fake.capacity_requests
    assert fake.capacity_requests[0].kv_hit_rate == 0.4


def test_prefill_capacity_does_not_pass_accept_length(monkeypatch):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        },
        capacity=_FakeCapacity(rps=10.0, ttft_s=0.1),
    )
    _install_fake_engine(monkeypatch, fake)
    model = PlannerEnginePerfModel(
        worker_type="prefill",
        config=_config(min_observations=1),
        capabilities=_caps(),
    )

    capacity = model.find_engine_capacity_rps(
        isl=1000,
        osl=100,
        ttft_sla_ms=500.0,
        accept_length=2.0,
    )

    assert fake.capacity_requests[0].accept_length == 1.0
    assert capacity is not None
    assert capacity.rps == 10.0


def test_decode_capacity_passes_accept_length_to_engine_model(monkeypatch):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        },
        capacity=_FakeCapacity(rps=200.0, itl_s=0.025),
    )
    _install_fake_engine(monkeypatch, fake)
    model = PlannerEnginePerfModel(
        worker_type="decode",
        config=_config(min_observations=1),
        capabilities=_caps(),
    )

    capacity = model.find_engine_capacity_rps(
        isl=1000,
        osl=100,
        itl_sla_ms=30.0,
        accept_length=2.0,
    )

    assert fake.capacity_requests[0].itl_sla_s == 0.03
    assert fake.capacity_requests[0].accept_length == 2.0
    assert capacity is not None
    assert capacity.rps == 200.0
    assert capacity.itl_ms == 25.0


def test_agg_capacity_passes_accept_length_to_engine_model(monkeypatch):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        },
        capacity=_FakeCapacity(rps=100.0, ttft_s=0.01, itl_s=0.025),
    )
    _install_fake_engine(monkeypatch, fake)
    model = PlannerEnginePerfModel(
        worker_type="aggregated",
        config=_config(min_observations=1),
        capabilities=EngineCapabilities(
            max_num_batched_tokens=1000,
            max_num_seqs=1000,
            max_kv_tokens=100_000,
            kv_cache_block_size=16,
        ),
    )

    capacity = model.find_engine_capacity_rps(
        isl=100,
        osl=100,
        ttft_sla_ms=1000.0,
        itl_sla_ms=30.0,
        accept_length=2.0,
    )

    assert fake.capacity_requests[0].itl_sla_s == 0.03
    assert fake.capacity_requests[0].accept_length == 2.0
    assert capacity is not None
    assert capacity.rps == 100.0
    assert capacity.itl_ms == 25.0


def test_adapter_does_not_expose_legacy_prediction_passthroughs():
    for name in (
        "add_observation",
        "estimate_next_ttft",
        "estimate_next_itl",
        "find_best_engine_prefill_rps",
        "find_best_engine_decode_rps",
        "find_best_engine_agg_rps",
    ):
        assert not hasattr(PlannerEnginePerfModel, name)
