# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading

import pytest

from dynamo.common.forward_pass_metrics import ForwardPassMetrics, QueuedRequestMetrics
from dynamo.planner import reconfiguration
from dynamo.planner.core.load.predictors import ARIMAPredictor, KalmanPredictor
from dynamo.planner.core.types import FpmObservations, WorkerCounts
from dynamo.planner.plugins.builtins.local_planner import BuiltinLoadPredict
from dynamo.planner.reconfiguration import ConfigurationConflict, ReconfigureRequest
from dynamo.planner.tests.unit.test_control_api import _planner

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
]


def sla_planner():
    return _planner("disagg", load_predictor="constant", optimization_target="sla")


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
    next_tick = engine.initial_tick(100.0, force_due=True)
    assert next_tick.need_traffic_metrics and next_tick.run_throughput_scaling
    assert (
        next_tick.traffic_metrics_duration_s
        == planner.config.throughput_adjustment_interval_seconds
    )
    assert (
        state._throughput_lower_bound_p == planner.config.effective_prefill_min_endpoint
    )
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


@pytest.mark.parametrize("cold_start", [False, True])
async def test_warmup_replaces_all_three_predictors_without_replacing_perf_models(
    tmp_path,
    cold_start,
):
    planner = sla_planner()
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
    planner = sla_planner()
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
    planner = sla_planner()
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
    planner = sla_planner()
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
    planner = sla_planner()
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
        with pytest.raises(ConfigurationConflict, match="still preparing"):
            await planner.reconfiguration.submit(
                update.model_copy(update={"request_id": "second"})
            )
        await asyncio.wait_for(planner._shutdown_runtime(), 1)
        release.set()
        await planner.reconfiguration._preparation
        assert planner.config.load_predictor == "constant"
        assert planner._config_generation == 0
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
