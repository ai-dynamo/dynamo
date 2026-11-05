# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gzip
import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from dynamo.common.forward_pass_metrics import ForwardPassMetrics, QueuedRequestMetrics
from dynamo.planner import reconfiguration
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.engine_protocol import (
    EngineProtocol,
    ReconfigurableEngineProtocol,
)
from dynamo.planner.core.load.predictors import ARIMAPredictor, KalmanPredictor
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    ScheduledTick,
    TickInput,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.builtins.local_planner import BuiltinLoadPredict
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter
from dynamo.planner.reconfiguration import (
    ConfigurationConflict,
    ReconfigureRequest,
    candidate_config,
    prepare_predictor,
    runtime_candidate_errors,
    runtime_configuration_errors,
)
from dynamo.planner.tests.unit.test_control_api import _planner

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
]


def sla_planner(**overrides):
    return _planner(
        "disagg", load_predictor="constant", optimization_target="sla", **overrides
    )


def request(planner, changes, **kwargs):
    return ReconfigureRequest(
        instance_id=planner.reconfiguration.instance_id,
        expected_revision=planner._config_generation,
        request_id="update-1",
        config=changes,
        **kwargs,
    )


async def finish(planner):
    await asyncio.gather(planner.reconfiguration._task, return_exceptions=True)
    return planner.reconfiguration.get_operation("update-1")


@pytest.mark.parametrize(
    ("change", "update_at"),
    [
        ("sla", 200),
        ("sla", 355),
        ("throughput_limit", 355),
        ("predictor", 355),
        ("predictor_warmup", 355),
        ("load", 355),
        ("budget", 355),
        ("noop", 355),
        ("legacy", 355),
    ],
)
async def test_reconfiguration_preserves_complete_prediction_windows(
    change, update_at, tmp_path
):
    planner = sla_planner(
        enable_load_scaling=True, runtime_warmup_trace_dir=str(tmp_path)
    )
    clock = VirtualClock()
    engine = OrchestratorEngineAdapter(
        planner.config, planner._build_worker_capabilities(), clock=clock
    )
    planner._engine = engine
    tick = engine.initial_tick(0.0)

    async def advance(tick):
        clock.advance(tick.at_s - clock.now())
        traffic = (
            TrafficObservation(duration_s=180, num_req=10, isl=100, osl=20)
            if tick.need_traffic_metrics
            else None
        )
        effects = await engine.tick(
            tick,
            TickInput(
                now_s=clock.now(),
                traffic=traffic,
                worker_counts=WorkerCounts(
                    ready_num_prefill=1,
                    expected_num_prefill=1,
                    ready_num_decode=1,
                    expected_num_decode=1,
                ),
            ),
        )
        return effects.next_tick

    while tick.at_s <= update_at:
        tick = await advance(tick)
    previous = engine._builtins["load_predict"]._num_req_predictor
    assert len(previous.data_buffer) == 1
    if change == "legacy":
        assert not planner.config.enable_runtime_reconfiguration
        await planner.patch_min_endpoints({"prefill_min_endpoint": 2})
    else:
        updates = {
            "sla": {"ttft_ms": 250.0},
            "throughput_limit": {"max_throughput_scaling_replicas": 4},
            "predictor": {"load_predictor": "kalman"},
            "predictor_warmup": {"load_predictor": "kalman"},
            "load": {"load_scaling_down_sensitivity": 70},
            "budget": {"max_gpu_budget": 24, "prefill_min_endpoint": 2},
            "noop": {"ttft_ms": planner.config.ttft_ms},
        }
        await planner.reconfiguration.submit(
            request(
                planner,
                updates[change],
                cold_start=change == "predictor",
                warmup_trace=trace(tmp_path) if change == "predictor_warmup" else None,
            )
        )
        assert (await finish(planner))["state"] == "applied"
    active = engine._builtins["load_predict"]._num_req_predictor
    assert (active is not previous) == change.startswith("predictor")
    count = len(active.data_buffer)
    for expected_prediction in (360, 540):
        while tick.at_s < expected_prediction:
            tick = await advance(tick)
            assert len(active.data_buffer) == count
        assert tick.at_s == expected_prediction
        assert tick.run_throughput_scaling
        assert tick.need_traffic_metrics and tick.traffic_metrics_duration_s == 180
        tick = await advance(tick)
        count += 1
        assert len(active.data_buffer) == count
    await planner._shutdown_runtime()


def trace(tmp_path, interval=180):
    rows = []
    for i, count in enumerate([2, 4, 3, 6, 4, 7, 5, 9]):
        for j in range(count):
            rows.append(
                {
                    "timestamp": i * interval * 1000 + j,
                    "input_length": 100 + i * 5,
                    "output_length": 20 + i,
                }
            )
    path = tmp_path / "warmup.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return str(path)


async def test_scalar_update_is_conditional_idempotent_and_preserves_live_state():
    planner = sla_planner()
    engine = planner._ensure_engine()
    assert isinstance(engine, ReconfigurableEngineProtocol)
    state = engine._scaling_state
    predictor = engine._builtins["load_predict"]
    state._pending_num_p = 3
    state._num_p_workers = 2
    state._throughput_lower_bound_p = 7
    update = request(planner, {"ttft_ms": 250.0, "max_gpu_budget": 24})
    caller_owned = update.model_copy(deep=True)
    accepted = await planner.reconfiguration.submit(caller_owned)
    caller_owned.config["ttft_ms"] = 10.0
    assert accepted["state"] == "preparing"
    operation = await finish(planner)
    assert operation["state"] == "applied"
    assert operation["applied_revision"] == 1
    assert (planner.config.ttft_ms, planner.config.max_gpu_budget) == (250.0, 24)
    assert engine._scaling_state is state
    assert (state._pending_num_p, state._num_p_workers) == (3, 2)
    assert engine._builtins["load_predict"] is predictor
    assert state._throughput_lower_bound_p == 7
    assert await planner.reconfiguration.submit(update) == operation
    reordered = update.model_copy(
        update={"config": {"max_gpu_budget": 24, "ttft_ms": 250.0}}
    )
    assert await planner.reconfiguration.submit(reordered) == operation
    with pytest.raises(ConfigurationConflict):
        await planner.reconfiguration.submit(
            update.model_copy(update={"config": {"ttft_ms": 200.0}})
        )
    with pytest.raises(ConfigurationConflict):
        await planner.reconfiguration.submit(
            update.model_copy(update={"request_id": "stale"})
        )
    planner.prometheus_metrics.sla_target_ttft_ms.set.assert_called_with(250.0)
    await planner._shutdown_runtime()


async def test_tick_only_engine_rejects_reconfiguration_and_keeps_legacy_tick_signature(
    monkeypatch,
):
    class TickOnlyEngine:
        def __init__(self):
            self.scheduled = 0
            self.stopped = False

        def initial_tick(self, start_s):
            self.scheduled += 1
            return ScheduledTick(at_s=0.0)

        async def tick(self, scheduled_tick, tick_input):
            raise AssertionError("test intercepts the tick before decision execution")

        async def shutdown(self):
            self.stopped = True

    planner = _planner("agg")
    engine = TickOnlyEngine()
    planner._engine = engine
    assert isinstance(engine, EngineProtocol)
    result = await planner.reconfiguration.handle_request(
        {
            "method": "PATCH",
            "path": "config",
            "body": request(planner, {"max_gpu_budget": 24}).model_dump(),
        }
    )
    assert result["status"] == 422
    assert "does not support runtime reconfiguration" in result["body"]["error"]
    assert planner._config_generation == 0

    async def tick_then_stop(active, tick):
        if planner._config_generation == 0:
            await planner.patch_min_endpoints({"max_gpu_budget": 24})
            return ScheduledTick(at_s=0.0)
        raise asyncio.CancelledError

    monkeypatch.setattr(planner, "_run_one_tick", tick_then_stop)
    with pytest.raises(asyncio.CancelledError):
        await planner.run()
    assert engine.scheduled == 1
    assert engine.stopped


@pytest.mark.parametrize("cold_start", [False, True])
async def test_warmup_replaces_all_three_predictors_without_replacing_perf_models(
    tmp_path,
    cold_start,
):
    planner = sla_planner(runtime_warmup_trace_dir=str(tmp_path))
    engine = planner._ensure_engine()
    active = engine._builtins["load_predict"]
    regressions = (
        engine._scaling_state._prefill_regression,
        engine._scaling_state._decode_regression,
    )
    active._last_kv_hit_rate = 0.6
    preparation = (
        {"cold_start": True} if cold_start else {"warmup_trace": trace(tmp_path)}
    )
    update = request(planner, {"load_predictor": "kalman"}, **preparation)
    await planner.reconfiguration.submit(update)
    assert (await finish(planner))["state"] == "applied"
    assert planner.config.load_predictor == "kalman"
    for predictor in (
        active._num_req_predictor,
        active._isl_predictor,
        active._osl_predictor,
    ):
        assert isinstance(predictor, KalmanPredictor)
        if cold_start:
            assert predictor.data_buffer == []
        else:
            assert len(predictor.data_buffer) >= predictor.minimum_data_points
            assert predictor._has_cached_pred
    assert active._last_kv_hit_rate == 0.6
    assert (
        engine._scaling_state._prefill_regression,
        engine._scaling_state._decode_regression,
    ) == regressions
    await planner._shutdown_runtime()


async def test_insufficient_warmup_does_not_activate_fallback_prediction(tmp_path):
    planner = sla_planner(runtime_warmup_trace_dir=str(tmp_path))
    await planner.reconfiguration.submit(
        request(
            planner,
            {"load_predictor": "kalman", "kalman_min_points": 100},
            warmup_trace=trace(tmp_path),
        )
    )
    operation = await finish(planner)
    assert operation["state"] == "failed"
    assert "effective warmup points" in operation["error"]
    assert planner.config.load_predictor == "constant"
    assert planner._config_generation == 0
    await planner._shutdown_runtime()


async def test_runtime_warmup_accepts_a_bounded_gzip_file(tmp_path):
    planner = sla_planner(runtime_warmup_trace_dir=str(tmp_path))
    source = Path(trace(tmp_path))
    compressed = source.with_suffix(".jsonl.gz")
    compressed.write_bytes(gzip.compress(source.read_bytes()))
    await planner.reconfiguration.submit(
        request(
            planner,
            {"load_predictor": "kalman"},
            warmup_trace=str(compressed),
        )
    )
    assert (await finish(planner))["state"] == "applied"
    assert planner.config.load_predictor == "kalman"
    await planner._shutdown_runtime()


async def test_runtime_warmup_is_confined_to_configured_data_directory(
    tmp_path, monkeypatch
):
    root = tmp_path / "allowed"
    root.mkdir()
    planner = sla_planner(runtime_warmup_trace_dir=str(root))
    outside = Path(trace(tmp_path))
    (root / "escape.jsonl").symlink_to(outside)
    (root / "escape-dir").symlink_to(tmp_path, target_is_directory=True)
    unreadable = [
        str(outside),
        str(tmp_path / "absent.jsonl"),
        "../warmup.jsonl",
        "escape.jsonl",
        "escape-dir/warmup.jsonl",
        "absent.jsonl",
    ]
    errors = []
    for index, source in enumerate(unreadable):
        update = request(
            planner, {"load_predictor": "kalman"}, warmup_trace=source
        ).model_copy(update={"request_id": f"rejected-{index}"})
        response = await planner.reconfiguration.handle_request(
            {"method": "PATCH", "path": "config", "body": update.model_dump()}
        )
        assert response["status"] == 202
        await asyncio.gather(planner.reconfiguration._task, return_exceptions=True)
        outcome = planner.reconfiguration.get_operation(update.request_id)
        assert outcome["state"] == "failed"
        assert str(tmp_path) not in outcome["error"]
        errors.append(outcome["error"])
        assert (
            planner.config.load_predictor == "constant"
            and planner._config_generation == 0
        )
    assert errors[0] == errors[1]
    assert errors[3] == errors[4] == errors[5]
    # Size rejection is also a failed preparation, with the previous policy usable.
    monkeypatch.setattr(reconfiguration, "MAX_WARMUP_TRACE_BYTES", 256)
    (root / "large.jsonl.gz").write_bytes(gzip.compress(b"x" * 257))
    await planner.reconfiguration.submit(
        request(planner, {"load_predictor": "kalman"}, warmup_trace="large.jsonl.gz")
    )
    assert (await finish(planner))["state"] == "failed"
    update = request(planner, {"max_gpu_budget": 24}).model_copy(
        update={"request_id": "recovery"}
    )
    await planner.reconfiguration.submit(update)
    await planner.reconfiguration._task
    assert planner.config.max_gpu_budget == 24
    await planner._shutdown_runtime()


async def test_unknown_operation_is_404_but_unexpected_key_errors_propagate(
    monkeypatch,
):
    planner = sla_planner()
    controller = planner.reconfiguration
    response = await controller.handle_request(
        {"method": "GET", "path": "operations/missing", "body": None}
    )
    assert response["status"] == 404

    async def broken_get_config():
        raise KeyError("internal failure")

    monkeypatch.setattr(controller, "get_config", broken_get_config)
    with pytest.raises(KeyError, match="internal failure"):
        await controller.handle_request(
            {"method": "GET", "path": "config", "body": None}
        )
    await planner._shutdown_runtime()


@pytest.mark.parametrize(
    "changes",
    [
        {"itl_ms": 0},
        {"ttft_ms": float("inf")},
        {"kalman_r": -1},
        {"kalman_q_level": float("nan")},
        {"kalman_min_points": 0},
        {"max_gpu_budget": -2},
    ],
)
async def test_startup_and_runtime_share_configuration_constraints(changes):
    with pytest.raises(ValidationError):
        PlannerConfig(**changes)
    planner = sla_planner()
    response = await planner.reconfiguration.handle_request(
        {
            "method": "PATCH",
            "path": "config",
            "body": request(planner, changes).model_dump(),
        }
    )
    assert response["status"] == 422
    assert planner._config_generation == 0
    await planner._shutdown_runtime()


async def test_runtime_threshold_changes_the_next_scaling_decision():
    planner = _planner(
        "prefill",
        optimization_target="load",
        prefill_scale_up_queue_tokens=1000,
        prefill_scale_down_queue_tokens=100,
    )
    state = planner._ensure_engine()._scaling_state
    state.observe_worker_counts(
        WorkerCounts(ready_num_prefill=1, expected_num_prefill=1)
    )
    observations = FpmObservations(
        prefill={
            ("worker", 0): ForwardPassMetrics(
                worker_id="worker",
                queued_requests=QueuedRequestMetrics(
                    num_prefill_requests=1, sum_prefill_tokens=500
                ),
            )
        }
    )
    before = state.advance_load(observations)
    assert before is None or before.num_prefill == 1
    await planner.reconfiguration.submit(
        request(planner, {"prefill_scale_up_queue_tokens": 300})
    )
    assert (await finish(planner))["state"] == "applied"
    after = state.advance_load(observations)
    assert after is not None and after.num_prefill == 2
    await planner._shutdown_runtime()


async def test_fit_failure_and_missing_warmup_leave_old_configuration_active(
    tmp_path, monkeypatch
):
    planner = sla_planner(runtime_warmup_trace_dir=str(tmp_path))
    engine = planner._ensure_engine()
    active = engine._builtins["load_predict"]
    original = active._num_req_predictor
    with pytest.raises(ValueError, match="requires warmup_trace"):
        await planner.reconfiguration.submit(
            request(planner, {"load_predictor": "arima"})
        )

    def failed_fit(*args, **kwargs):
        raise ValueError("fit failed for test")

    monkeypatch.setattr(
        "dynamo.planner.core.load.predictors.pmdarima.auto_arima", failed_fit
    )
    await planner.reconfiguration.submit(
        request(planner, {"load_predictor": "arima"}, warmup_trace=trace(tmp_path))
    )
    result = await finish(planner)
    assert result["state"] == "failed"
    assert "fit failed" in result["error"]
    assert planner._config_generation == 0
    assert planner.config.load_predictor == "constant"
    assert active._num_req_predictor is original
    # Normal live prediction retains the existing last-value fallback.
    fallback = ARIMAPredictor(planner.config)
    for value in [1.0, 2.0, 3.0, 2.0, 4.0]:
        fallback.add_data_point(value)
    assert fallback.predict_next() == 4.0
    await planner._shutdown_runtime()


async def test_legacy_write_invalidates_a_preparation_without_blocking_it(
    tmp_path, monkeypatch
):
    planner = sla_planner(runtime_warmup_trace_dir=str(tmp_path))
    entered, release = threading.Event(), threading.Event()
    original = reconfiguration._prepare_predictor

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(reconfiguration, "_prepare_predictor", blocked)
    try:
        update = request(
            planner, {"load_predictor": "kalman"}, warmup_trace=trace(tmp_path)
        )
        await planner.reconfiguration.submit(update)
        assert await asyncio.to_thread(entered.wait, 5)
        assert (await planner.reconfiguration.get_config())["config"][
            "load_predictor"
        ] == "constant"
        assert (await planner.reconfiguration.submit(update))["state"] == "warming"
        with pytest.raises(ConfigurationConflict, match="still preparing"):
            await planner.reconfiguration.submit(
                update.model_copy(update={"request_id": "second"})
            )
        await planner.patch_min_endpoints({"max_gpu_budget": 24})
        release.set()
        operation = await finish(planner)
        assert operation["state"] == "failed"
        assert "revision changed" in operation["error"]
        assert planner.config.load_predictor == "constant"
        assert planner.config.max_gpu_budget == 24
        assert planner._config_generation == 1
    finally:
        release.set()
        await planner._shutdown_runtime()


async def test_timeout_bounds_preparation_and_shutdown_prevents_late_activation(
    tmp_path, monkeypatch
):
    planner = sla_planner(runtime_warmup_trace_dir=str(tmp_path))
    entered, release = threading.Event(), threading.Event()

    def blocked(config, capabilities, _trace):
        entered.set()
        assert release.wait(5)
        return BuiltinLoadPredict(config, capabilities)

    monkeypatch.setattr(reconfiguration, "_prepare_predictor", blocked)
    monkeypatch.setattr(reconfiguration, "PREPARATION_TIMEOUT_SECONDS", 0.01)
    try:
        update = request(planner, {"load_predictor": "kalman"}, cold_start=True)
        await planner.reconfiguration.submit(update)
        assert await asyncio.to_thread(entered.wait, 5)
        assert (await finish(planner))["state"] == "failed"
        budget_update = request(
            planner, {"max_gpu_budget": 24, "ttft_ms": 250.0}
        ).model_copy(update={"request_id": "budget"})
        await planner.reconfiguration.submit(budget_update)
        await planner.reconfiguration._task
        assert planner.reconfiguration.get_operation("budget")["state"] == "applied"
        with pytest.raises(ConfigurationConflict, match="still preparing"):
            await planner.reconfiguration.submit(
                request(
                    planner, {"load_predictor": "kalman"}, cold_start=True
                ).model_copy(update={"request_id": "second"})
            )
        await asyncio.wait_for(planner._shutdown_runtime(), 1)
        release.set()
        await planner.reconfiguration._preparation
        assert planner.config.load_predictor == "constant"
        assert planner._config_generation == 1
        assert planner.config.max_gpu_budget == 24
    finally:
        release.set()


@pytest.mark.parametrize(
    "changes",
    [
        {"throughput_adjustment_interval_seconds": 60},
        {"ais_perf_model": None},
        {"ttft_ms": -1.0},
        {"ttft_ms": True},
        {"ttft_ms": "50"},
        {"prefill_min_endpoint": 30},
        {"min_gpu_budget": 30, "max_gpu_budget": 20},
    ],
)
async def test_http_contract_rejects_unsupported_or_invalid_updates(changes):
    planner = _planner("disagg")
    controller = planner.reconfiguration
    initial = await controller.handle_request(
        {"method": "GET", "path": "config", "body": None}
    )
    result = await controller.handle_request(
        {
            "method": "PATCH",
            "path": "config",
            "body": request(planner, changes).model_dump(),
        }
    )
    assert result["status"] == 422
    assert await controller.get_config() == initial["body"]
    await planner._shutdown_runtime()


def test_runtime_configuration_errors_skip_unknown_power():
    capabilities = WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=1), decode=EngineCapabilities(num_gpu=1)
    )
    assert runtime_configuration_errors(100, 100, -1, -1, 1, capabilities) == []


def test_candidate_config_keeps_the_fields_excluded_from_the_dump(monkeypatch):
    # The Redis URL comes from the config file, not the environment; the dump
    # excludes it, and non-advisory batch scheduling requires it.
    monkeypatch.delenv("DYN_PLANNER_BATCH_REDIS_URL", raising=False)
    config = PlannerConfig(
        namespace="test-ns",
        environment="kubernetes",
        mode="agg",
        metric_pulling_prometheus_endpoint="http://prometheus.test:9090",
        batch_scheduling={
            "enabled": True,
            "gateway": {"base_url": "http://gateway:8000", "tenant": "tenant"},
            "metrics": {
                "frontend_metrics_url": "http://frontend:8000/metrics",
                "dispatcher_metrics_url": "http://dispatcher:9090/metrics",
                "online_match_labels": {"model": "test-model"},
            },
            "redis": {"url": "redis://valkey:6379/0", "control_key": "drain-limit"},
            "pool": {
                "pool_id": "pool",
                "work_class": "work",
                "safe_rps_per_ready_replica": 1.0,
                "max_replicas": 4,
                "drain_lease_duration_seconds": 600.0,
            },
        },
    )
    dumped = config.model_dump()
    assert "url" not in dumped["batch_scheduling"]["redis"]
    assert "metric_pulling_prometheus_endpoint" not in dumped

    candidate = candidate_config(config, {"min_endpoint": 2})

    assert candidate.min_endpoint == 2
    redis = candidate.batch_scheduling.redis
    assert redis is not None and redis.url is not None
    assert redis.url.get_secret_value() == "redis://valkey:6379/0"
    assert candidate.metric_pulling_prometheus_endpoint == "http://prometheus.test:9090"
    assert candidate.namespace == "test-ns"


async def test_runtime_candidate_errors_match_runtime_validation():
    planner = _planner(
        "disagg", prefill_gpu_cost=2, decode_gpu_cost=1, total_gpu_power_limit=2000
    )
    capabilities = planner._build_worker_capabilities()
    candidate = candidate_config(
        planner.config, {"prefill_min_endpoint": 8, "max_gpu_budget": 16}
    )
    errors = runtime_candidate_errors(candidate, capabilities)
    assert errors == [
        "minimum endpoint footprint requires 17 GPUs, exceeding max_gpu_budget=16"
    ]
    with pytest.raises(ValueError) as error:
        planner.validate_runtime_candidate(candidate)
    assert str(error.value) == errors[0]
    fitting = candidate_config(planner.config, {"max_gpu_budget": 24})
    assert runtime_candidate_errors(fitting, capabilities) == []
    await planner._shutdown_runtime()


async def test_prepare_predictor_warms_from_observations_like_the_trace_path(tmp_path):
    planner = sla_planner(runtime_warmup_trace_dir=str(tmp_path))
    candidate = candidate_config(planner.config, {"load_predictor": "kalman"})
    capabilities = planner._build_worker_capabilities()
    source = trace(tmp_path)
    observations = reconfiguration._load_runtime_warmup(
        source,
        candidate.throughput_adjustment_interval_seconds,
        candidate.runtime_warmup_trace_dir,
    )

    from_trace = reconfiguration._prepare_predictor(candidate, capabilities, source)
    warmed = prepare_predictor(candidate, capabilities, tuple(observations))
    for name in ("_num_req_predictor", "_isl_predictor", "_osl_predictor"):
        predictor = getattr(warmed, name)
        assert isinstance(predictor, KalmanPredictor)
        assert predictor.data_buffer == getattr(from_trace, name).data_buffer
        assert len(predictor.data_buffer) >= predictor.minimum_data_points
    assert warmed._last_observed_isl == from_trace._last_observed_isl

    cold = prepare_predictor(candidate, capabilities, None)
    assert cold._num_req_predictor.data_buffer == []
    with pytest.raises(ValueError, match="effective warmup points"):
        prepare_predictor(candidate, capabilities, observations[:1])
    await planner._shutdown_runtime()


def test_native_planner_rejects_retune_config():
    with pytest.raises(ValueError, match="planner.retune is replay-only"):
        _planner(
            "disagg",
            enable_load_scaling=False,
            enable_throughput_scaling=False,
            retune={
                "interval_seconds": 600,
                "router": {"overlap_score_credit": {"choices": [0.0, 1.0]}},
            },
        )
