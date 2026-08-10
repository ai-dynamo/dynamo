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


class _FakeEngineModel:
    def __init__(
        self,
        *,
        diagnostics,
        queued_prefill_time=None,
        capacity=None,
        tune_error=None,
        tune_fail_after=0,
    ):
        self._diagnostics = diagnostics
        self._queued_prefill_time = queued_prefill_time
        self._capacity = capacity
        self._tune_error = tune_error
        self._tune_fail_after = tune_fail_after
        self.tuned_iterations = []
        self.capacity_requests = []

    def diagnostics(self):
        return self._diagnostics

    def tune_with_fpms(self, iterations):
        if self._tune_error is not None:
            self.tuned_iterations.extend(iterations[: self._tune_fail_after])
            raise self._tune_error
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


@pytest.fixture
def fake_engine_factory(monkeypatch):
    class FakeEngineFactory:
        next_model = None
        last_kwargs = None

        @classmethod
        def best_available(cls, **kwargs):
            assert cls.next_model is not None
            cls.last_kwargs = kwargs
            return cls.next_model

    monkeypatch.setattr(aic_adapter, "AicCoreEnginePerfModel", FakeEngineFactory)
    return FakeEngineFactory


def _install_fake_engine(fake_engine_factory, fake_model: _FakeEngineModel) -> None:
    fake_engine_factory.next_model = fake_model
    fake_engine_factory.last_kwargs = None


# These tests replace the engine factory, so the value is never used against a
# real database -- it only has to be non-None so that version resolution
# short-circuits and the tests stay independent of the installed wheel.
_STUB_BACKEND_VERSION = "test-backend-version"


def _config(
    *,
    dp: int = 1,
    min_observations: int = 5,
    speculative_nextn: int = 0,
    backend_version: str | None = _STUB_BACKEND_VERSION,
) -> PlannerConfig:
    pick = PickedParallelConfig(dp=dp)
    return PlannerConfig.model_construct(
        aic_perf_model=AICPerfModelSpec.model_construct(
            hf_id="Qwen/Qwen3-0.6B",
            system="h200_sxm",
            backend="vllm",
            backend_version=backend_version,
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
    worker_id: str = "w0",
    requests: int = 1,
    kv_tokens: int = 128,
    wall_time: float = 0.01,
) -> ForwardPassMetrics:
    return ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=0,
        wall_time=wall_time,
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=requests,
            sum_decode_kv_tokens=kv_tokens,
        ),
    )


def test_aic_diagnostics_gate_sufficient_data(fake_engine_factory):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "fallback_regression",
            "readiness": "insufficient_data",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _install_fake_engine(fake_engine_factory, fake)
    model = PlannerEnginePerfModel(
        worker_type="prefill",
        config=_config(),
        capabilities=_caps(),
    )

    assert not model.has_sufficient_data()

    fake._diagnostics["readiness"] = "ready"
    assert model.has_sufficient_data()


def test_missing_capability_fields_use_engine_query_defaults(fake_engine_factory):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "fallback_regression",
            "readiness": "insufficient_data",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _install_fake_engine(fake_engine_factory, fake)

    PlannerEnginePerfModel(
        worker_type="prefill",
        config=_config(),
        capabilities=EngineCapabilities(max_num_batched_tokens=2048),
    )

    assert fake_engine_factory.last_kwargs is not None
    limits = fake_engine_factory.last_kwargs["limits"]
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
    fake_engine_factory,
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
    _install_fake_engine(fake_engine_factory, fake)

    PlannerEnginePerfModel(
        worker_type="decode",
        config=_config(speculative_nextn=config_nextn),
        capabilities=_caps(speculative_nextn=capability_nextn),
    )

    assert fake_engine_factory.last_kwargs is not None
    aic_config = fake_engine_factory.last_kwargs["aic_config"]
    assert aic_config["nextn"] == int(expected_nextn)


def test_capability_nextn_update_rebuilds_aic_model_and_replays_fpms(
    fake_engine_factory,
):
    first = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _install_fake_engine(fake_engine_factory, first)
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
    fake_engine_factory.next_model = second

    model.update_capabilities(_caps(speculative_nextn=3))

    assert fake_engine_factory.last_kwargs is not None
    aic_config = fake_engine_factory.last_kwargs["aic_config"]
    assert aic_config["nextn"] == 3
    assert second.tuned_iterations == [[fpm]]


def test_partial_replay_failure_waits_for_fresh_model_before_full_retry(
    fake_engine_factory,
):
    first = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        },
        tune_error=ValueError("invalid retained observation"),
        tune_fail_after=1,
    )
    _install_fake_engine(fake_engine_factory, first)
    model = PlannerEnginePerfModel(
        worker_type="decode",
        config=_config(),
        capabilities=None,
    )
    first_fpm = _decode_fpm(worker_id="w0", requests=2, kv_tokens=256, wall_time=0.02)
    invalid_fpm = _decode_fpm(worker_id="w1", requests=3, kv_tokens=384, wall_time=0.03)
    model.add_observations(
        {
            ("w0", 0): first_fpm,
            ("w1", 0): invalid_fpm,
        }
    )

    model.update_capabilities(_caps())

    assert model._engine_model is first
    assert first.tuned_iterations == [[first_fpm]]
    assert model._pending_iterations == [[first_fpm], [invalid_fpm]]
    assert model.has_sufficient_data()

    first._tune_error = None
    later_fpm = _decode_fpm(worker_id="w2", requests=4, kv_tokens=512, wall_time=0.04)
    model.add_observations({("w2", 0): later_fpm})

    assert first.tuned_iterations == [[first_fpm], [later_fpm]]
    assert model._pending_iterations == [[first_fpm], [invalid_fpm]]

    second = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    fake_engine_factory.next_model = second

    model.update_capabilities(_caps(speculative_nextn=3))

    assert second.tuned_iterations == [[first_fpm], [invalid_fpm], [later_fpm]]
    assert model._pending_iterations == []


def test_aic_none_result_remains_unavailable(fake_engine_factory):
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
    _install_fake_engine(fake_engine_factory, fake)
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


def test_flat_bootstrap_fpms_skip_aic_tuning_for_attention_dp(fake_engine_factory):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _install_fake_engine(fake_engine_factory, fake)
    model = PlannerEnginePerfModel(
        worker_type="prefill",
        config=_config(dp=2, min_observations=1),
        capabilities=_caps(),
    )

    model.load_benchmark_fpms([_prefill_fpm(scheduled_tokens=256, wall_time=0.02)])

    assert fake.tuned_iterations == []
    assert model.num_observations == 0
    assert model.avg_isl == 256


def test_capacity_request_passes_kv_hit_rate(fake_engine_factory):
    fake = _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )
    _install_fake_engine(fake_engine_factory, fake)
    model = PlannerEnginePerfModel(
        worker_type="aggregated",
        config=_config(min_observations=1),
        capabilities=_caps(),
    )

    assert model.find_engine_capacity_rps(isl=1000, osl=100, kv_hit_rate=0.4) is None
    assert fake.capacity_requests
    assert fake.capacity_requests[0].kv_hit_rate == 0.4


def test_prefill_capacity_does_not_pass_accept_length(fake_engine_factory):
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
    _install_fake_engine(fake_engine_factory, fake)
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


def test_decode_capacity_passes_accept_length_to_engine_model(fake_engine_factory):
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
    _install_fake_engine(fake_engine_factory, fake)
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


def test_agg_capacity_passes_accept_length_to_engine_model(fake_engine_factory):
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
    _install_fake_engine(fake_engine_factory, fake)
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


def _ready_engine() -> _FakeEngineModel:
    return _FakeEngineModel(
        diagnostics={
            "source": "aic",
            "readiness": "ready",
            "retained_observations": 0,
            "correction_ready_buckets": 0,
            "last_warning": None,
        }
    )


@pytest.fixture
def stub_latest_database_version(monkeypatch):
    """Stub the packaged AIC perf-database version lookup."""

    def install(version):
        lookups = []

        def fake(system, backend):
            lookups.append((system, backend))
            return version

        monkeypatch.setattr(
            aic_adapter,
            "get_latest_database_version",
            fake,
        )
        return lookups

    return install


def test_missing_backend_version_resolves_packaged_database_version(
    fake_engine_factory, stub_latest_database_version
):
    """A hand-written config without backend_version must still query natively."""
    lookups = stub_latest_database_version("0.24.0")
    _install_fake_engine(fake_engine_factory, _ready_engine())

    PlannerEnginePerfModel(
        worker_type="decode",
        config=_config(backend_version=None),
        capabilities=_caps(),
    )

    assert fake_engine_factory.last_kwargs is not None
    aic_config = fake_engine_factory.last_kwargs["aic_config"]
    assert aic_config is not None
    assert aic_config["backend_version"] == "0.24.0"
    assert lookups == [("h200_sxm", "vllm")]


def test_unresolvable_backend_version_falls_back_to_regression(
    fake_engine_factory, stub_latest_database_version
):
    """No packaged database means FPM regression, not a hard native failure."""
    stub_latest_database_version(None)
    _install_fake_engine(fake_engine_factory, _ready_engine())

    PlannerEnginePerfModel(
        worker_type="decode",
        config=_config(backend_version=None),
        capabilities=_caps(),
    )

    assert fake_engine_factory.last_kwargs is not None
    assert fake_engine_factory.last_kwargs["aic_config"] is None


def test_explicit_backend_version_takes_precedence_over_lookup(
    fake_engine_factory, stub_latest_database_version
):
    stub_latest_database_version("9.9.9")
    _install_fake_engine(fake_engine_factory, _ready_engine())

    PlannerEnginePerfModel(
        worker_type="decode",
        config=_config(backend_version="0.19.0"),
        capabilities=_caps(),
    )

    assert fake_engine_factory.last_kwargs is not None
    aic_config = fake_engine_factory.last_kwargs["aic_config"]
    assert aic_config is not None
    assert aic_config["backend_version"] == "0.19.0"


def _real_engine_config(backend_version):
    """Engine config in the shape `_build_aic_config` emits."""
    return {
        "schema_version": 1,
        "model_name": "nvidia/Llama-3.1-8B-Instruct-FP8",
        "system_name": "h200_sxm",
        "backend": "vllm",
        "backend_version": backend_version,
        "kv_block_size": 16,
        "tp_size": 1,
        "pp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
        "attention_dp_size": 1,
        "cp_size": None,
        "weight_dtype": None,
        "moe_dtype": None,
        "activation_dtype": None,
        "kv_cache_dtype": None,
        "nextn": None,
        "extra": {},
    }


_REAL_ENGINE_OPTIONS = {
    "max_observations": 16,
    "min_observations": 5,
    "bucket_count": 16,
    "max_num_tokens": 8192,
    "max_batch_size": 512,
    "max_kv_tokens": 2_000_000,
}


def test_aic_core_rejects_missing_backend_version():
    """Pin the upstream contract the adapter's version resolution works around.

    aiconfigurator-core raises ``InvalidEngineConfig`` for a missing
    ``backend_version``, and that variant is not fallback-safe, so the wheel
    does not degrade to regression on its own.
    """
    aic_sdk = pytest.importorskip("aiconfigurator_core.sdk")

    with pytest.raises(ValueError, match="backend_version is required"):
        aic_sdk.RustForwardPassPerfModel.best_available(
            _real_engine_config(None), _REAL_ENGINE_OPTIONS
        )


def test_aic_core_accepts_resolved_backend_version():
    """The same config succeeds once the packaged version is supplied."""
    aic_sdk = pytest.importorskip("aiconfigurator_core.sdk")
    perf_database = pytest.importorskip("aiconfigurator_core.sdk.perf_database")

    version = perf_database.get_latest_database_version("h200_sxm", "vllm")
    if version is None:
        pytest.skip("wheel ships no h200_sxm/vllm performance database")

    model = aic_sdk.RustForwardPassPerfModel.best_available(
        _real_engine_config(version), _REAL_ENGINE_OPTIONS
    )

    assert model.diagnostics()["source"] == "aic"


def test_config_without_backend_version_initializes_native_model():
    """End-to-end guard against silent SLA-autoscaling failure.

    Uses the real wheel: a hand-written config that omits ``backend_version``
    must still produce a usable native model instead of leaving the planner
    permanently unable to estimate.
    """
    perf_database = pytest.importorskip("aiconfigurator_core.sdk.perf_database")
    if perf_database.get_latest_database_version("h200_sxm", "vllm") is None:
        pytest.skip("wheel ships no h200_sxm/vllm performance database")

    pick = PickedParallelConfig(dp=1)
    config = PlannerConfig.model_construct(
        aic_perf_model=AICPerfModelSpec.model_construct(
            hf_id="nvidia/Llama-3.1-8B-Instruct-FP8",
            system="h200_sxm",
            backend="vllm",
            backend_version=None,
            prefill_pick=pick,
            decode_pick=pick,
        ),
        max_num_fpm_samples=16,
        load_min_observations=5,
        fpm_sample_bucket_size=16,
        ttft_ms=500.0,
        itl_ms=50.0,
        speculative_nextn=0,
    )

    model = PlannerEnginePerfModel(
        worker_type="decode",
        config=config,
        capabilities=EngineCapabilities(
            max_num_batched_tokens=8192,
            max_num_seqs=512,
            max_kv_tokens=2_000_000,
            kv_cache_block_size=16,
        ),
    )

    capacity = model.find_engine_capacity_rps(
        isl=1000, osl=200, ttft_sla_ms=500.0, itl_sla_ms=50.0
    )

    assert capacity is not None
    assert capacity.rps > 0
