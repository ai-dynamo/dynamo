# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for worker_factory.py"""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.worker_factory import (
    ENGINE_PROBE_TIMEOUT_SECONDS,
    EngineSetupResult,
    SnapshotEngineSetupResult,
    WorkerFactory,
    _attach_engine_resolved,
    _await_benchmark_then_restore_workers,
    _benchmark_engine_identity,
    _DecodeWorkerLifecycle,
    _make_engine_probe,
    _merge_benchmark_rank_results,
    _register_request_cache_metrics,
    _stop_worker_gc_policy,
    _wait_and_load_benchmark,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.xpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
    pytest.mark.pre_merge,
]


def _make_config(**overrides) -> Mock:
    """Create a mock Config with canonical worker settings."""
    defaults = {
        "enable_multimodal": False,
        "omni": False,
        "route_to_encoder": False,
        "disaggregation_mode": DisaggregationMode.AGGREGATED,
        "embedding_worker": False,
        "embedding_frontend_tokenization": False,
        # Pin to the real Config default: an auto-created Mock attribute is
        # truthy, which enables the GMS shadow-mode path and imports the
        # optional gpu_memory_service package (absent in some test images).
        "gms_shadow_mode": False,
        "realtime": False,
        "classify_worker": False,
    }
    defaults.update(overrides)
    return Mock(**defaults)


def _make_factory(**overrides) -> WorkerFactory:
    defaults = {
        "setup_vllm_engine_fn": Mock(),
        "setup_kv_event_publisher_fn": Mock(return_value=None),
        "register_vllm_model_fn": AsyncMock(),
        "setup_fpm_relay_fn": Mock(return_value=None),
        "setup_metrics_collection_fn": Mock(),
    }
    defaults.update(overrides)
    return WorkerFactory(**defaults)


def test_register_request_cache_metrics_includes_multimodal_image_loader():
    endpoint = Mock()
    embedding_cache = object()
    image_loader = object()
    handler = SimpleNamespace(
        embedding_cache_manager=embedding_cache,
        _multimodal_request_processor=SimpleNamespace(image_loader=image_loader),
    )
    config = SimpleNamespace(
        enable_multimodal=True,
        served_model_name="served-model",
        model="source-model",
        component="backend",
    )

    with (
        patch(
            "dynamo.vllm.worker_factory.register_embedding_cache_metrics"
        ) as register_embedding,
        patch(
            "dynamo.vllm.worker_factory.register_image_loader_metrics"
        ) as register_image,
    ):
        _register_request_cache_metrics(endpoint, handler, config)

    register_embedding.assert_called_once_with(
        endpoint=endpoint,
        cache=embedding_cache,
        model_name="served-model",
        component_name="backend",
    )
    register_image.assert_called_once_with(
        endpoint=endpoint,
        loader=image_loader,
        model_name="served-model",
        component_name="backend",
    )


def test_decode_worker_lifecycle_cleanup_in_reverse_construction_order():
    calls = []
    shutdown_event = asyncio.Event()
    handler = Mock()
    handler.cleanup.side_effect = lambda: calls.append("handler")
    engine_client = Mock()
    engine_client.shutdown.side_effect = lambda **_kwargs: calls.append("engine")
    resources = _DecodeWorkerLifecycle(
        engine_client=engine_client,
        vllm_config=SimpleNamespace(shutdown_timeout=7.0),
        handler=handler,
        shutdown_event=shutdown_event,
    )

    resources.cleanup()

    assert calls == ["handler", "engine"]
    assert shutdown_event.is_set()
    engine_client.shutdown.assert_called_once_with(timeout=7.0)


def test_decode_worker_lifecycle_shutdown_engine_when_handler_cleanup_fails():
    handler = Mock()
    handler.cleanup.side_effect = RuntimeError("handler cleanup failed")
    engine_client = Mock()
    resources = _DecodeWorkerLifecycle(
        engine_client=engine_client,
        vllm_config=SimpleNamespace(shutdown_timeout=5.0),
        handler=handler,
    )

    with pytest.raises(RuntimeError, match="handler cleanup failed"):
        resources.cleanup()

    engine_client.shutdown.assert_called_once_with(timeout=5.0)


def test_decode_worker_lifecycle_chains_handler_and_engine_cleanup_failures():
    handler_error = RuntimeError("handler cleanup failed")
    engine_error = RuntimeError("engine shutdown failed")
    handler = Mock()
    handler.cleanup.side_effect = handler_error
    engine_client = Mock()
    engine_client.shutdown.side_effect = engine_error
    lifecycle = _DecodeWorkerLifecycle(
        engine_client=engine_client,
        vllm_config=SimpleNamespace(shutdown_timeout=5.0),
        handler=handler,
    )

    with pytest.raises(RuntimeError, match="engine shutdown failed") as exc_info:
        lifecycle.cleanup()

    assert exc_info.value is engine_error
    assert exc_info.value.__context__ is handler_error
    engine_client.shutdown.assert_called_once_with(timeout=5.0)


@pytest.mark.asyncio
async def test_custom_encoder_preserves_primary_error_when_cleanup_fails(caplog):
    factory = _make_factory()
    engine_client = Mock()
    handler = Mock()
    handler.cleanup.side_effect = RuntimeError("handler cleanup failed")
    startup_error = ValueError("decode worker startup failed")

    async def fail_after_resource_creation(*_args, lifecycle, **_kwargs):
        lifecycle.engine_client = engine_client
        lifecycle.vllm_config = SimpleNamespace(shutdown_timeout=5.0)
        lifecycle.handler = handler
        raise startup_error

    factory._run_decode_worker = fail_after_resource_creation  # type: ignore[method-assign]
    caplog.set_level(logging.ERROR)
    config = SimpleNamespace(custom_encoder_class="encoder.Backend")

    with pytest.raises(ValueError, match="decode worker startup failed") as exc_info:
        await factory._create_decode_worker(Mock(), config, asyncio.Event(), [])

    assert exc_info.value is startup_error
    engine_client.shutdown.assert_called_once_with(timeout=5.0)
    assert "Failed to clean up decode worker after an earlier failure" in caplog.text


@pytest.mark.asyncio
async def test_decode_worker_without_custom_encoder_uses_lifecycle():
    factory = _make_factory()
    engine_client = Mock()
    startup_error = ValueError("decode worker startup failed")

    async def fail_after_engine_creation(*_args, lifecycle, **_kwargs):
        lifecycle.engine_client = engine_client
        lifecycle.vllm_config = SimpleNamespace(shutdown_timeout=5.0)
        raise startup_error

    factory._run_decode_worker = fail_after_engine_creation  # type: ignore[method-assign]
    config = SimpleNamespace(custom_encoder_class=None)

    with pytest.raises(ValueError, match="decode worker startup failed") as exc_info:
        await factory._create_decode_worker(Mock(), config, asyncio.Event(), [])

    assert exc_info.value is startup_error
    engine_client.shutdown.assert_called_once_with(timeout=5.0)


@pytest.mark.asyncio
async def test_decode_failure_withdraws_state_agent_before_engine_cleanup():
    calls = []
    state_agent_lifecycle = SimpleNamespace(
        close=AsyncMock(side_effect=lambda: calls.append("state-agent"))
    )
    factory = _make_factory(state_agent_lifecycle=state_agent_lifecycle)
    engine_client = Mock()
    engine_client.shutdown.side_effect = lambda **_kwargs: calls.append("engine")
    handler = Mock()
    handler.cleanup.side_effect = lambda: calls.append("handler")

    async def fail_after_setup(*_args, lifecycle, **_kwargs):
        lifecycle.engine_client = engine_client
        lifecycle.vllm_config = SimpleNamespace(shutdown_timeout=5.0)
        lifecycle.handler = handler
        raise RuntimeError("startup failed")

    factory._run_decode_worker = fail_after_setup  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="startup failed"):
        await factory._create_decode_worker(
            Mock(), SimpleNamespace(custom_encoder_class=None), asyncio.Event(), []
        )

    assert calls == ["state-agent", "handler", "engine"]


@pytest.mark.asyncio
async def test_prefill_startup_failure_withdraws_state_agent_owner():
    lifecycle = SimpleNamespace(close=AsyncMock())
    factory = _make_factory(state_agent_lifecycle=lifecycle)
    factory._run_prefill_worker = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("prefill startup failed")
    )

    with pytest.raises(RuntimeError, match="prefill startup failed"):
        await factory._create_prefill_worker(Mock(), Mock(), asyncio.Event(), [])

    lifecycle.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_state_agent_setup_failure_never_falls_back_to_legacy(monkeypatch):
    from dynamo.vllm import worker_factory

    monkeypatch.setattr(
        worker_factory, "state_agent_settings", lambda _config: object()
    )
    setup_legacy = Mock(return_value=["legacy"])
    setup_owner = AsyncMock(side_effect=RuntimeError("host is unavailable"))
    factory = _make_factory(
        setup_kv_event_publisher_fn=setup_legacy,
        setup_kv_state_attachment_owner_fn=setup_owner,
    )

    result = await factory._setup_kv_routing(
        Mock(),
        Mock(),
        Mock(),
        consolidator_enabled=False,
        consolidator_port=0,
    )

    assert result is None
    setup_owner.assert_awaited_once()
    setup_legacy.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["configure", "handler"])
async def test_custom_encoder_shutdown_engine_on_startup_failure(
    monkeypatch, failure_stage, tmp_path
):
    engine_client = Mock()
    vllm_config = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(num_gpu_blocks=1),
        model_config=SimpleNamespace(max_model_len=1024),
        shutdown_timeout=5.0,
    )
    engine_setup: EngineSetupResult = (
        engine_client,
        vllm_config,
        Mock(),
        str(tmp_path / "prometheus"),
        Mock(),
    )
    factory = _make_factory(setup_vllm_engine_fn=Mock(return_value=engine_setup))
    factory._maybe_create_failover_metrics = Mock(return_value=None)  # type: ignore[method-assign]
    factory._maybe_get_encode_worker_client = AsyncMock(return_value=None)  # type: ignore[method-assign]

    stat_logger = Mock()
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.StatLoggerFactory",
        Mock(return_value=stat_logger),
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (0, 1)
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.per_rank_kv_blocks",
        lambda _num_blocks, _dp_size: 1,
    )

    async def configure_block_size(*_args, **_kwargs):
        if failure_stage == "configure":
            raise ValueError("decode startup rejected")
        return None

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.configure_kv_event_block_size",
        configure_block_size,
    )
    handler_constructor = Mock(
        side_effect=(
            ValueError("decode startup rejected")
            if failure_stage == "handler"
            else None
        )
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.DecodeWorkerHandler",
        handler_constructor,
    )

    endpoint = Mock(connection_id=Mock(return_value="worker-id"))
    runtime = Mock()
    runtime.endpoint.return_value = endpoint
    config = SimpleNamespace(
        namespace="dynamo",
        component="backend",
        endpoint="generate",
        enable_rl=False,
        engine_args=SimpleNamespace(enable_lora=False),
        enable_multimodal=True,
        custom_encoder_class="encoder.Backend",
        use_vllm_tokenizer=False,
        frontend_decoding=False,
    )

    with pytest.raises(ValueError, match="decode startup rejected"):
        await factory._create_decode_worker(runtime, config, asyncio.Event(), [])

    engine_client.shutdown.assert_called_once_with(timeout=5.0)
    if failure_stage == "configure":
        handler_constructor.assert_not_called()


def _single_rank_benchmark_payload(
    *,
    status: str = "complete",
    expected_points: int = 1,
) -> dict:
    point = {"benchmark_id": 1, "point_type": "decode"}
    fpm = {"counter_id": 1, "dp_rank": 0, "wall_time": 0.01}
    partial = status == "partial"
    return {
        "status": status,
        "valid": not partial,
        "usable": True,
        "stop_reason": "timeout" if partial else None,
        "run_id": "run-1",
        "grid_digest": "grid-1",
        "timing": {
            "started_at": "2026-07-13T12:00:00Z",
            "completed_at": "2026-07-13T12:00:01Z",
            "benchmark_elapsed_seconds": 1.0,
            "measured_iteration_seconds": 0.01,
        },
        "dp": {"rank": 0, "size": 1},
        "coverage": {
            "expected_points": expected_points,
            "completed_points": 1,
            "skipped_points": 0,
        },
        "results": [{"point": point, "fpms": [fpm]}],
        "iteration_groups": [
            {
                "benchmark_id": 1,
                "point": point,
                "expected_dp_ranks": [0],
                "complete": True,
                "wall_time": 0.01,
                "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
            }
        ],
        "skipped_points": [],
    }


@pytest.mark.asyncio
async def test_wait_and_load_benchmark_rejects_invalid_results(monkeypatch, tmp_path):
    output_path = tmp_path / "benchmark.json"
    output_path.write_text(
        json.dumps(
            {
                "valid": False,
                "coverage": {
                    "expected_points": 2,
                    "completed_points": 1,
                    "skipped_points": 1,
                },
                "skipped_points": [{"reason": "seed_cache_validation_failed"}],
                "missing_phases": ["decode"],
            }
        )
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (0, 1)
    )

    with pytest.raises(RuntimeError, match="incomplete results") as exc_info:
        await _wait_and_load_benchmark(
            {"output_path": str(output_path), "timeout": 1}, Mock()
        )
    assert "missing_phases=['decode']" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stop_worker_gc_policy_rpcs_every_worker(monkeypatch):
    """After the benchmark wait, the launcher must stop the GC policy in
    every model worker (workers auto-start it on extension import) and hold
    serving until the RPC completes."""
    monkeypatch.setenv("DYN_FPM_GC_POLICY", "freeze")
    engine_client = SimpleNamespace(collective_rpc=AsyncMock())

    await _stop_worker_gc_policy(engine_client)

    engine_client.collective_rpc.assert_awaited_once_with("fpm_gc_stop")


@pytest.mark.asyncio
async def test_stop_worker_gc_policy_noop_when_policy_disabled(monkeypatch):
    monkeypatch.delenv("DYN_FPM_GC_POLICY", raising=False)
    engine_client = SimpleNamespace(collective_rpc=AsyncMock())

    await _stop_worker_gc_policy(engine_client)

    engine_client.collective_rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_worker_gc_policy_failure_propagates(monkeypatch):
    """Serving on workers that are not GC-equivalent to never-benchmarked
    ones must not silently proceed."""
    monkeypatch.setenv("DYN_FPM_GC_POLICY", "freeze")
    engine_client = SimpleNamespace(
        collective_rpc=AsyncMock(side_effect=RuntimeError("worker died"))
    )

    with pytest.raises(RuntimeError, match="worker died"):
        await _stop_worker_gc_policy(engine_client)


@pytest.mark.asyncio
async def test_wait_and_load_benchmark_accepts_timeout_partial(monkeypatch, tmp_path):
    output_path = tmp_path / "benchmark.json"
    output_path.write_text(
        json.dumps(_single_rank_benchmark_payload(status="partial", expected_points=2))
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (0, 1)
    )

    merged = await _wait_and_load_benchmark(
        {"output_path": str(output_path), "timeout": 1}, Mock()
    )

    assert merged["status"] == "partial"
    assert merged["valid"] is False
    assert merged["usable"] is True
    assert merged["stop_reason"] == "timeout"
    assert merged["coverage"] == {
        "expected_points": 2,
        "completed_points": 1,
        "skipped_points": 0,
    }


@pytest.mark.asyncio
async def test_wait_and_load_benchmark_warns_then_waits_for_partial(
    monkeypatch, tmp_path, caplog
):
    output_path = tmp_path / "benchmark.json"
    payload = _single_rank_benchmark_payload(status="partial", expected_points=2)
    monotonic_times = iter([0.0, 2.0])

    async def finish_current_iteration(_delay):
        output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (0, 1)
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._time.monotonic",
        lambda: next(monotonic_times, 2.0),
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.asyncio.sleep", finish_current_iteration
    )
    caplog.set_level(logging.WARNING)

    merged = await _wait_and_load_benchmark(
        {"output_path": str(output_path), "timeout": 1}, Mock()
    )

    assert merged["status"] == "partial"
    assert "for the current profiling iteration" in caplog.text
    assert "Engine startup will continue" in caplog.text


@pytest.mark.asyncio
async def test_wait_and_load_benchmark_bounds_post_timeout_grace(
    monkeypatch, tmp_path, caplog
):
    output_path = tmp_path / "benchmark.json"
    monotonic_times = iter([0.0, 2.0, 3.0])

    async def no_result(_delay):
        return None

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (0, 1)
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.BENCHMARK_SOFT_TIMEOUT_GRACE_SECONDS", 1
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._time.monotonic",
        lambda: next(monotonic_times, 3.0),
    )
    monkeypatch.setattr("dynamo.vllm.worker_factory.asyncio.sleep", no_result)
    caplog.set_level(logging.WARNING)

    with pytest.raises(TimeoutError, match="cleanup grace"):
        await _wait_and_load_benchmark(
            {"output_path": str(output_path), "timeout": 1}, Mock()
        )

    assert "waiting up to 1s" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["partial_valid", "partial_skipped", "complete_invalid", "complete_unusable"],
)
async def test_wait_and_load_benchmark_rejects_inconsistent_status(
    monkeypatch, tmp_path, mutation
):
    output_path = tmp_path / "benchmark.json"
    if mutation in {"complete_invalid", "complete_unusable"}:
        payload = _single_rank_benchmark_payload()
        if mutation == "complete_invalid":
            payload["valid"] = False
        else:
            payload["usable"] = False
    else:
        payload = _single_rank_benchmark_payload(status="partial", expected_points=3)
        if mutation == "partial_valid":
            payload["valid"] = True
        else:
            payload["coverage"]["skipped_points"] = 1
            payload["skipped_points"] = [{"reason": "shape mismatch"}]
    output_path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (0, 1)
    )

    with pytest.raises(RuntimeError, match="incomplete results"):
        await _wait_and_load_benchmark(
            {"output_path": str(output_path), "timeout": 1}, Mock()
        )


@pytest.mark.asyncio
async def test_wait_and_load_benchmark_aggregates_dp_coverage(monkeypatch, tmp_path):
    base_path = tmp_path / "benchmark.json"
    point = {"benchmark_id": 1, "point_type": "prefill"}
    rank_results = [
        {
            "dp_rank": 0,
            "fpms": [{"counter_id": 1, "dp_rank": 0, "wall_time": 0.01}],
        },
        {
            "dp_rank": 1,
            "fpms": [{"counter_id": 1, "dp_rank": 1, "wall_time": 0.02}],
        },
    ]
    iteration_groups = [
        {
            "benchmark_id": 1,
            "point": point,
            "expected_dp_ranks": [0, 1],
            "complete": True,
            "wall_time": 0.02,
            "rank_results": rank_results,
        }
    ]

    def rank_payload(dp_rank: int, wall_time: float) -> dict:
        return {
            "valid": True,
            "run_id": "run-1",
            "grid_digest": "grid-1",
            "timing": {
                "started_at": f"2026-07-10T12:00:0{dp_rank}Z",
                "completed_at": f"2026-07-10T12:00:1{dp_rank}Z",
                "benchmark_elapsed_seconds": 10.0 + dp_rank,
                "measured_iteration_seconds": 0.02,
            },
            "dp": {"rank": dp_rank, "size": 2},
            "coverage": {
                "expected_points": 1,
                "completed_points": 1,
                "skipped_points": 0,
            },
            "results": [
                {
                    "point": point,
                    "fpms": [
                        {
                            "counter_id": 1,
                            "dp_rank": dp_rank,
                            "wall_time": wall_time,
                        }
                    ],
                }
            ],
            "iteration_groups": iteration_groups,
            "skipped_points": [],
        }

    base_path.write_text(json.dumps(rank_payload(0, 0.01)))
    (tmp_path / "benchmark_dp1.json").write_text(json.dumps(rank_payload(1, 0.02)))
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (0, 2)
    )

    merged = await _wait_and_load_benchmark(
        {"output_path": str(base_path), "timeout": 1}, Mock()
    )

    assert merged["coverage"] == {
        "expected_points": 2,
        "completed_points": 2,
        "skipped_points": 0,
    }
    assert merged["timing"] == {
        "started_at": "2026-07-10T12:00:01Z",
        "completed_at": "2026-07-10T12:00:11Z",
        "benchmark_elapsed_seconds": 11.0,
        "measured_iteration_seconds": 0.02,
        "rank_benchmark_elapsed_seconds": {"0": 10.0, "1": 11.0},
    }
    assert [result["point"]["dp_rank"] for result in merged["results"]] == [0, 1]
    assert merged["iteration_groups"] == [
        {
            "benchmark_id": 1,
            "point": point,
            "expected_dp_ranks": [0, 1],
            "complete": True,
            "wall_time": 0.02,
            "rank_results": [
                {
                    "dp_rank": 0,
                    "fpms": [{"counter_id": 1, "dp_rank": 0, "wall_time": 0.01}],
                },
                {
                    "dp_rank": 1,
                    "fpms": [{"counter_id": 1, "dp_rank": 1, "wall_time": 0.02}],
                },
            ],
        }
    ]
    merged_path = tmp_path / "benchmark_merged.json"
    assert merged_path.exists()
    assert json.loads(merged_path.read_text()) == merged

    bad_rank = rank_payload(1, 0.02)
    bad_rank["results"][0]["fpms"][0]["counter_id"] = 2
    (tmp_path / "benchmark_dp1.json").write_text(json.dumps(bad_rank))
    with pytest.raises(RuntimeError, match="FPM counter mismatch"):
        await _wait_and_load_benchmark(
            {"output_path": str(base_path), "timeout": 1}, Mock()
        )

    partial_ranks = [rank_payload(0, 0.01), rank_payload(1, 0.02)]
    for data in partial_ranks:
        data.update(
            {
                "status": "partial",
                "valid": False,
                "usable": True,
                "stop_reason": "timeout",
            }
        )
        data["coverage"]["expected_points"] = 2
    base_path.write_text(json.dumps(partial_ranks[0]))
    (tmp_path / "benchmark_dp1.json").write_text(json.dumps(partial_ranks[1]))

    partial_merged = await _wait_and_load_benchmark(
        {"output_path": str(base_path), "timeout": 1}, Mock()
    )

    assert partial_merged["status"] == "partial"
    assert partial_merged["coverage"] == {
        "expected_points": 4,
        "completed_points": 2,
        "skipped_points": 0,
    }


@pytest.mark.asyncio
async def test_wait_and_load_benchmark_external_dp_keeps_global_group(
    monkeypatch, tmp_path
):
    base_path = tmp_path / "benchmark.json"
    point = {"benchmark_id": 1, "point_type": "decode"}
    rank_results = [
        {
            "dp_rank": 0,
            "fpms": [{"counter_id": 1, "dp_rank": 0, "wall_time": 0.01}],
        },
        {
            "dp_rank": 1,
            "fpms": [{"counter_id": 1, "dp_rank": 1, "wall_time": 0.02}],
        },
    ]
    base_path.write_text(
        json.dumps(
            {
                "valid": True,
                "run_id": "run-1",
                "grid_digest": "grid-1",
                "timing": {
                    "started_at": "2026-07-10T12:00:00Z",
                    "completed_at": "2026-07-10T12:00:10Z",
                    "benchmark_elapsed_seconds": 10.0,
                    "measured_iteration_seconds": 0.02,
                },
                "dp": {"rank": 0, "size": 2},
                "coverage": {
                    "expected_points": 1,
                    "completed_points": 1,
                    "skipped_points": 0,
                },
                "results": [
                    {
                        "point": point,
                        "fpms": rank_results[0]["fpms"],
                    }
                ],
                "iteration_groups": [
                    {
                        "benchmark_id": 1,
                        "point": point,
                        "expected_dp_ranks": [0, 1],
                        "complete": True,
                        "wall_time": 0.02,
                        "rank_results": rank_results,
                    }
                ],
                "skipped_points": [],
            }
        )
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (0, 1)
    )

    merged = await _wait_and_load_benchmark(
        {"output_path": str(base_path), "timeout": 1}, Mock()
    )

    assert merged["dp"] == {
        "ranks": [0, 1],
        "source_ranks": [0],
        "managed_size": 1,
        "global_size": 2,
    }
    assert [result["point"]["dp_rank"] for result in merged["results"]] == [0, 1]
    assert merged["coverage"] == {
        "expected_points": 2,
        "completed_points": 2,
        "skipped_points": 0,
    }


@pytest.mark.asyncio
class TestCreate:
    """Test WorkerFactory.create() routing."""

    @pytest.fixture
    def factory(self) -> WorkerFactory:
        factory = WorkerFactory(
            setup_vllm_engine_fn=Mock(),
            setup_kv_event_publisher_fn=Mock(),
            register_vllm_model_fn=AsyncMock(),
            setup_fpm_relay_fn=Mock(),
            setup_metrics_collection_fn=Mock(),
        )
        factory._create_multimodal_encode_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_prefill_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_decode_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_embedding_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_realtime_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_classify_worker = AsyncMock()  # type: ignore[assignment]
        return factory

    # Tests for non-legacy worker config, 'route_to_encode' is worker internal config
    # so either case should hit creation function.
    @pytest.mark.parametrize("route_to_encode", [True, False])
    async def test_aggregated(
        self, factory: WorkerFactory, route_to_encode: bool
    ) -> None:
        config = _make_config(route_to_encoder=route_to_encode)
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event, [])

        factory._create_decode_worker.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.parametrize("route_to_encode", [True, False])
    async def test_prefill(self, factory: WorkerFactory, route_to_encode: bool) -> None:
        config = _make_config(
            disaggregation_mode=DisaggregationMode.PREFILL,
            route_to_encoder=route_to_encode,
        )
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event, [])

        factory._create_prefill_worker.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.parametrize("route_to_encode", [True, False])
    async def test_decode(self, factory: WorkerFactory, route_to_encode: bool) -> None:
        config = _make_config(
            disaggregation_mode=DisaggregationMode.DECODE,
            route_to_encoder=route_to_encode,
        )
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event, [])

        factory._create_decode_worker.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.parametrize("route_to_encode", [True, False])
    async def test_encode(self, factory: WorkerFactory, route_to_encode: bool) -> None:
        config = _make_config(
            disaggregation_mode=DisaggregationMode.ENCODE,
            route_to_encoder=route_to_encode,
        )
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event, [])

        factory._create_multimodal_encode_worker.assert_called_once()  # type: ignore[union-attr]

    async def test_embedding_worker_takes_priority(
        self, factory: WorkerFactory
    ) -> None:
        config = _make_config(embedding_worker=True)
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event, [])

        factory._create_embedding_worker.assert_called_once()  # type: ignore[union-attr]
        factory._create_realtime_worker.assert_not_called()  # type: ignore[union-attr]
        factory._create_decode_worker.assert_not_called()  # type: ignore[union-attr]
        factory._create_prefill_worker.assert_not_called()  # type: ignore[union-attr]
        factory._create_multimodal_encode_worker.assert_not_called()  # type: ignore[union-attr]

    async def test_realtime_worker_takes_priority(self, factory: WorkerFactory) -> None:
        config = _make_config(realtime=True)
        runtime = Mock()
        shutdown_event = asyncio.Event()
        shutdown_endpoints = []
        snapshot_engine = Mock()

        await factory.create(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            snapshot_engine=snapshot_engine,
        )

        factory._create_realtime_worker.assert_awaited_once_with(  # type: ignore[union-attr]
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            snapshot_engine=snapshot_engine,
        )
        factory._create_embedding_worker.assert_not_called()  # type: ignore[union-attr]
        factory._create_decode_worker.assert_not_called()  # type: ignore[union-attr]
        factory._create_prefill_worker.assert_not_called()  # type: ignore[union-attr]
        factory._create_multimodal_encode_worker.assert_not_called()  # type: ignore[union-attr]

    async def test_classify_worker_takes_priority(self, factory: WorkerFactory) -> None:
        config = _make_config(classify_worker=True)
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event, [])

        factory._create_classify_worker.assert_called_once()  # type: ignore[union-attr]
        factory._create_decode_worker.assert_not_called()  # type: ignore[union-attr]
        factory._create_prefill_worker.assert_not_called()  # type: ignore[union-attr]
        factory._create_multimodal_encode_worker.assert_not_called()  # type: ignore[union-attr]

    async def test_passes_snapshot_engine(self, factory: WorkerFactory) -> None:
        config = _make_config(enable_multimodal=True)
        runtime = Mock()
        shutdown_event = asyncio.Event()
        shutdown_endpoints: list = []
        snapshot_engine: SnapshotEngineSetupResult = (
            (
                Mock(),
                Mock(),
                Mock(),
                "/tmp/prometheus",
                Mock(),
            ),
            Mock(),
        )

        await factory.create(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            snapshot_engine=snapshot_engine,
        )

        factory._create_decode_worker.assert_called_once_with(  # type: ignore[union-attr]
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            snapshot_engine=snapshot_engine,
        )


@pytest.mark.asyncio
async def test_classify_worker_registers_classify_and_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = Mock()
    endpoint.connection_id.return_value = "worker-1"
    endpoint.serve_endpoint = AsyncMock()
    runtime = Mock()
    runtime.endpoint.return_value = endpoint

    engine_client = Mock()
    vllm_config = Mock(model_config=Mock())
    engine_tuple: EngineSetupResult = (
        engine_client,
        vllm_config,
        Mock(),
        "/tmp/prom",
        None,
    )
    register_model = AsyncMock()
    handler = Mock()
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.ClassifyWorkerHandler",
        Mock(return_value=handler),
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.StatLoggerFactory",
        Mock(return_value=Mock()),
    )

    factory = WorkerFactory(
        setup_vllm_engine_fn=Mock(return_value=engine_tuple),
        setup_kv_event_publisher_fn=Mock(),
        register_vllm_model_fn=register_model,
        setup_fpm_relay_fn=Mock(),
        setup_metrics_collection_fn=Mock(),
    )
    config = _make_config(
        classify_worker=True,
        namespace="dynamo",
        component="worker",
        endpoint="generate",
        served_model_name="model",
        model="model",
    )
    shutdown_endpoints: list = []

    await factory._create_classify_worker(
        runtime,
        config,
        asyncio.Event(),
        shutdown_endpoints,
    )

    register_model.assert_awaited_once()
    register_args = register_model.await_args.args
    registered_model_type = register_args[1]
    assert register_args[0] == ModelInput.Text
    assert registered_model_type.supports_classify()
    assert registered_model_type.supports_pooling()
    assert not registered_model_type.supports_embedding()
    assert register_args[2:] == (
        endpoint,
        config,
        engine_client,
        vllm_config,
    )
    assert register_model.await_args.kwargs == {
        "worker_type": WorkerType.Aggregated,
        "needs": [],
    }
    assert shutdown_endpoints == [endpoint]
    handler.cleanup.assert_called_once()


@pytest.mark.asyncio
class TestPrefillRegistrationContract:
    """The ModelInput on a prefill `register_model` call is the inter-worker
    contract, not an engine-local tokenization preference. Prefill workers only
    ever receive token IDs from their decode peer, so this must be Tokens
    regardless of `config.use_vllm_tokenizer` — that flag only swaps the
    frontend↔decode boundary and the engine-local health-check payload shape.

    Registering Text + WorkerType.Prefill is rejected by the Rust binding
    guard (lib/bindings/python/rust/lib.rs), so the wrong choice here means
    prefill workers fail to register at startup.
    """

    @pytest.mark.parametrize("use_vllm_tokenizer", [True, False])
    @pytest.mark.parametrize("route_to_encoder", [True, False])
    async def test_prefill_registers_with_tokens(
        self,
        monkeypatch: pytest.MonkeyPatch,
        use_vllm_tokenizer: bool,
        route_to_encoder: bool,
    ) -> None:
        captured: dict = {}
        stop_after_register = RuntimeError("stop-after-register")

        async def fake_register_vllm_model(
            model_input,
            model_type,
            endpoint,
            config,
            engine_client,
            vllm_config,
            worker_type,
            needs,
        ) -> None:
            captured["model_input"] = model_input
            captured["model_type"] = model_type
            captured["worker_type"] = worker_type
            captured["needs"] = needs
            raise stop_after_register

        engine_client = Mock()
        vllm_config = Mock()
        vllm_config.additional_config = {}
        engine_tuple: EngineSetupResult = (
            engine_client,
            vllm_config,
            Mock(),
            "/tmp/prom",
            Mock(),
        )

        factory = WorkerFactory(
            setup_vllm_engine_fn=Mock(return_value=engine_tuple),
            setup_kv_event_publisher_fn=Mock(return_value=None),
            register_vllm_model_fn=fake_register_vllm_model,
            setup_fpm_relay_fn=Mock(return_value=None),
            setup_metrics_collection_fn=Mock(),
        )
        factory._maybe_get_encode_worker_client = AsyncMock(return_value=None)  # type: ignore[assignment]
        factory._maybe_wait_for_failover_lock = AsyncMock()  # type: ignore[assignment]
        factory.register_engine_routes = Mock()  # type: ignore[assignment]

        # embedding_cache_manager=None skips register_embedding_cache_metrics.
        mock_handler = Mock(embedding_cache_manager=None)
        monkeypatch.setattr(
            "dynamo.vllm.worker_factory.PrefillWorkerHandler",
            Mock(return_value=mock_handler),
        )

        async def _noop(*_args, **_kwargs) -> None:
            return None

        monkeypatch.setattr(
            "dynamo.vllm.worker_factory.configure_kv_event_block_size", _noop
        )

        config = _make_config(
            disaggregation_mode=DisaggregationMode.PREFILL,
            route_to_encoder=route_to_encoder,
            use_vllm_tokenizer=use_vllm_tokenizer,
            namespace="dyn",
            component="prefill",
            endpoint="generate",
            served_model_name="m",
            model="m",
            frontend_decoding=False,
            enable_multimodal=False,
            enable_rl=False,
            engine_args=SimpleNamespace(enable_lora=True),
        )

        runtime = Mock()
        runtime.endpoint.return_value = Mock(connection_id=Mock(return_value="cid"))
        shutdown_endpoints: list = []

        with pytest.raises(RuntimeError, match="stop-after-register"):
            await factory._create_prefill_worker(
                runtime,
                config,
                asyncio.Event(),
                shutdown_endpoints,
            )

        assert captured["model_input"] == ModelInput.Tokens
        assert captured["worker_type"] == WorkerType.Prefill
        # Dual-emit: prefill registers the legacy ModelType.Prefill marker bit
        # (no OpenAI surface) so an old frontend still detects it.
        assert captured["model_type"] == ModelType.Prefill
        expected_needs_set = [WorkerType.Decode]
        if route_to_encoder:
            expected_needs_set.append(WorkerType.Encode)
        assert captured["needs"] == [expected_needs_set]
        endpoint_names = [call.args[0] for call in runtime.endpoint.call_args_list]
        assert "dyn.prefill.load_lora" in endpoint_names
        assert "dyn.prefill.unload_lora" in endpoint_names
        assert "dyn.prefill.list_loras" in endpoint_names
        # generate, clear, perf, and all three LoRA lifecycle endpoints.
        assert len(shutdown_endpoints) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("lora_enabled", [True, False])
@pytest.mark.parametrize("snapshot_mode", [True, False])
async def test_prefill_initializes_metrics_and_serves_lora_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    lora_enabled: bool,
    snapshot_mode: bool,
) -> None:
    engine_client = Mock()
    vllm_config = Mock(
        additional_config={},
        cache_config=SimpleNamespace(num_gpu_blocks=96),
    )
    engine_tuple: EngineSetupResult = (
        engine_client,
        vllm_config,
        Mock(),
        str(tmp_path / "prometheus"),
        Mock(),
    )
    stat_logger = Mock()
    snapshot_engine: SnapshotEngineSetupResult | None = (
        (engine_tuple, stat_logger) if snapshot_mode else None
    )
    setup_vllm_engine = Mock(return_value=engine_tuple)
    factory = WorkerFactory(
        setup_vllm_engine_fn=setup_vllm_engine,
        setup_kv_event_publisher_fn=Mock(return_value=None),
        register_vllm_model_fn=AsyncMock(),
        setup_fpm_relay_fn=Mock(return_value=None),
        setup_metrics_collection_fn=Mock(),
    )
    factory._maybe_get_encode_worker_client = AsyncMock(return_value=None)  # type: ignore[assignment]
    factory._maybe_wait_for_failover_lock = AsyncMock()  # type: ignore[assignment]
    factory.register_engine_routes = Mock()  # type: ignore[assignment]

    handler = Mock(embedding_cache_manager=None)
    handler.cleanup = Mock()
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.PrefillWorkerHandler",
        Mock(return_value=handler),
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.VllmPrefillHealthCheckPayload",
        Mock(return_value=Mock(to_dict=Mock(return_value={}))),
    )

    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.configure_kv_event_block_size", _noop
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (3, 2)
    )
    stat_logger_factory = Mock(return_value=stat_logger)
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.StatLoggerFactory", stat_logger_factory
    )

    endpoints: dict[str, Mock] = {}

    def _endpoint(name: str) -> Mock:
        endpoint = Mock(connection_id=Mock(return_value="cid"))
        endpoint.serve_endpoint = AsyncMock(return_value=None)
        endpoints[name] = endpoint
        return endpoint

    runtime = Mock()
    runtime.endpoint.side_effect = _endpoint
    config = _make_config(
        disaggregation_mode=DisaggregationMode.PREFILL,
        route_to_encoder=False,
        use_vllm_tokenizer=False,
        namespace="dyn",
        component="prefill",
        endpoint="generate",
        served_model_name="m",
        model="m",
        frontend_decoding=False,
        enable_multimodal=False,
        enable_rl=False,
        engine_args=SimpleNamespace(enable_lora=lora_enabled),
    )
    shutdown_endpoints: list = []

    await factory._create_prefill_worker(
        runtime,
        config,
        asyncio.Event(),
        shutdown_endpoints,
        snapshot_engine=snapshot_engine,
    )

    if snapshot_mode:
        stat_logger_factory.assert_not_called()
        setup_vllm_engine.assert_not_called()
        stat_logger.bind_endpoint.assert_called_once_with(
            endpoints["dyn.prefill.generate"]
        )
    else:
        stat_logger_factory.assert_called_once_with(
            endpoint=endpoints["dyn.prefill.generate"]
        )
        setup_vllm_engine.assert_called_once_with(
            config, stat_logger, fpm_worker_id="cid"
        )
        stat_logger.bind_endpoint.assert_not_called()
    stat_logger.set_num_gpu_blocks_all.assert_called_once_with(48)
    stat_logger.init_publish.assert_called_once_with()

    lifecycle_names = {
        "dyn.prefill.load_lora",
        "dyn.prefill.unload_lora",
        "dyn.prefill.list_loras",
    }
    if lora_enabled:
        assert lifecycle_names <= endpoints.keys()
        for name in lifecycle_names:
            endpoints[name].serve_endpoint.assert_awaited_once()
        assert len(shutdown_endpoints) == 6
    else:
        assert lifecycle_names.isdisjoint(endpoints)
        assert len(shutdown_endpoints) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("embedding_frontend_tokenization", "expected_model_input"),
    [(False, ModelInput.Text), (True, ModelInput.Tokens)],
)
async def test_embedding_worker_registration_and_cleanup(
    embedding_frontend_tokenization: bool,
    expected_model_input: ModelInput,
) -> None:
    cleanup_order: list[str] = []
    endpoint = Mock()
    endpoint.connection_id.return_value = "embedding-worker-id"
    endpoint.serve_endpoint = AsyncMock(return_value=None)
    runtime = Mock()
    runtime.endpoint.return_value = endpoint

    engine_client = Mock()
    engine_client.shutdown.side_effect = lambda: cleanup_order.append("client")
    engine_cleanup_resource = Mock()
    engine_cleanup_resource.cleanup.side_effect = lambda: cleanup_order.append(
        "resource"
    )
    setup_vllm_engine = Mock(
        return_value=(
            engine_client,
            Mock(),
            Mock(),
            engine_cleanup_resource,
            Mock(),
        )
    )
    register_vllm_model = AsyncMock(return_value=None)
    factory = WorkerFactory(
        setup_vllm_engine_fn=setup_vllm_engine,
        setup_kv_event_publisher_fn=Mock(),
        register_vllm_model_fn=register_vllm_model,
        setup_fpm_relay_fn=Mock(),
        setup_metrics_collection_fn=Mock(),
    )
    handler = Mock()
    handler.cleanup.side_effect = lambda: cleanup_order.append("handler")
    config = _make_config(
        embedding_worker=True,
        embedding_frontend_tokenization=embedding_frontend_tokenization,
        namespace="dynamo",
        component="backend",
        endpoint="generate",
        model="test-model",
    )
    shutdown_event = asyncio.Event()
    shutdown_endpoints: list = []

    with patch(
        "dynamo.vllm.worker_factory.EmbeddingWorkerHandler",
        return_value=handler,
    ):
        await factory._create_embedding_worker(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
        )

    register_vllm_model.assert_awaited_once()
    assert register_vllm_model.await_args.args[0] == expected_model_input
    assert cleanup_order == ["handler", "client", "resource"]
    assert shutdown_endpoints == [endpoint]


def _rank_artifact(dp_rank: int, regime: str) -> dict:
    point = {
        "point_type": "decode",
        "benchmark_id": 1,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 64,
        "batch_size": 2,
        "expected_cudagraph_mode": "FULL",
        "expected_capture_size": 2,
        "padding_tokens": 0,
        "sample_reasons": ["explicit"],
        "partition": None,
        "rows": None,
    }
    fpm = {
        "version": 1,
        "worker_id": "w",
        "dp_rank": dp_rank,
        "counter_id": 1,
        "wall_time": 0.01,
        "scheduled_requests": {
            "num_prefill_requests": 0,
            "sum_prefill_tokens": 0,
            "var_prefill_length": 0.0,
            "sum_prefill_kv_tokens": 0,
            "num_decode_requests": 2,
            "sum_decode_kv_tokens": 64,
            "var_decode_kv_tokens": 0.0,
        },
        "queued_requests": {
            "num_prefill_requests": 0,
            "sum_prefill_tokens": 0,
            "var_prefill_length": 0.0,
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
            "var_decode_kv_tokens": 0.0,
        },
    }
    return {
        "schema_version": 2,
        "artifact_type": "rank",
        "status": "complete",
        "valid": True,
        "usable": True,
        "stop_reason": None,
        "timing_valid": True,
        "run_id": "run",
        "grid_digest": "g" * 64,
        "timing": {
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:00:01Z",
            "benchmark_elapsed_seconds": 1.0,
            "measured_iteration_seconds": 0.01,
        },
        "dp": {"rank": dp_rank, "size": 1},
        "coverage": {"expected_points": 1, "completed_points": 1, "skipped_points": 0},
        "results": [{"point": point, "kv_seed_regime": regime, "fpms": [fpm]}],
        "iteration_groups": [
            {
                "benchmark_id": 1,
                "point": point,
                "expected_dp_ranks": [0],
                "complete": True,
                "wall_time": 0.01,
                "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
            }
        ],
        "skipped_points": [],
        "missing_phases": [],
        "error": None,
    }


def test_merge_benchmark_rank_results_carries_kv_seed_regime(tmp_path):
    artifact = _rank_artifact(0, "fake_fallback")
    merged = _merge_benchmark_rank_results(
        [(0, tmp_path / "rank0.json", artifact)], tmp_path / "merged.json"
    )
    assert merged["results"][0]["kv_seed_regime"] == "fake_fallback"
    assert merged["results"][0]["point"]["dp_rank"] == 0


@pytest.mark.asyncio
async def test_benchmark_wait_success_stops_workers_then_returns(monkeypatch):
    calls = []

    async def fake_wait(_cfg, _vllm_config):
        calls.append("wait")
        return {"status": "complete"}

    async def fake_stop(_client):
        calls.append("stop")

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._wait_and_load_benchmark", fake_wait
    )
    monkeypatch.setattr("dynamo.vllm.worker_factory._stop_worker_gc_policy", fake_stop)

    results = await _await_benchmark_then_restore_workers({}, Mock(), Mock())

    assert results == {"status": "complete"}
    assert calls == ["wait", "stop"]


@pytest.mark.asyncio
async def test_benchmark_wait_failure_still_stops_workers_and_preserves_error(
    monkeypatch,
):
    """An aborted benchmark publishes status=failed, so the wait raises during
    validation; the worker GC stop must still run and the original error must
    reach the caller unchanged."""
    calls = []
    original = RuntimeError("Self-benchmark produced incomplete results")

    async def fake_wait(_cfg, _vllm_config):
        raise original

    async def fake_stop(_client):
        calls.append("stop")

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._wait_and_load_benchmark", fake_wait
    )
    monkeypatch.setattr("dynamo.vllm.worker_factory._stop_worker_gc_policy", fake_stop)

    with pytest.raises(RuntimeError) as exc_info:
        await _await_benchmark_then_restore_workers({}, Mock(), Mock())

    assert exc_info.value is original
    assert calls == ["stop"]


@pytest.mark.asyncio
async def test_benchmark_wait_failure_logs_stop_failure_and_keeps_original(
    monkeypatch, caplog
):
    original = TimeoutError("Self-benchmark did not publish results")

    async def fake_wait(_cfg, _vllm_config):
        raise original

    async def fake_stop(_client):
        raise RuntimeError("worker died")

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._wait_and_load_benchmark", fake_wait
    )
    monkeypatch.setattr("dynamo.vllm.worker_factory._stop_worker_gc_policy", fake_stop)
    caplog.set_level(logging.ERROR)

    with pytest.raises(TimeoutError) as exc_info:
        await _await_benchmark_then_restore_workers({}, Mock(), Mock())

    assert exc_info.value is original
    assert "Failed to restore model workers" in caplog.text


@pytest.mark.asyncio
async def test_benchmark_wait_success_fails_closed_when_stop_fails(monkeypatch):
    async def fake_wait(_cfg, _vllm_config):
        return {"status": "complete"}

    async def fake_stop(_client):
        raise RuntimeError("worker died")

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._wait_and_load_benchmark", fake_wait
    )
    monkeypatch.setattr("dynamo.vllm.worker_factory._stop_worker_gc_policy", fake_stop)

    with pytest.raises(RuntimeError, match="worker died"):
        await _await_benchmark_then_restore_workers({}, Mock(), Mock())


@pytest.mark.asyncio
async def test_worker_gc_restore_completes_before_model_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serving must not start on workers that still run the benchmark GC
    policy: the launcher awaits the worker stop RPC after the benchmark wait
    and before register_model publishes the worker as ready."""
    order: list[str] = []
    stop_after_register = RuntimeError("stop-after-register")

    async def fake_wait(_cfg, _vllm_config):
        order.append("wait")
        return {"status": "complete"}

    async def fake_stop(_client):
        order.append("stop")

    async def fake_register_vllm_model(*_args, **_kwargs) -> None:
        order.append("register")
        raise stop_after_register

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._wait_and_load_benchmark", fake_wait
    )
    monkeypatch.setattr("dynamo.vllm.worker_factory._stop_worker_gc_policy", fake_stop)

    engine_client = Mock()
    vllm_config = Mock()
    vllm_config.additional_config = {
        "benchmark": {"output_path": "/tmp/bench.json", "timeout": 1}
    }
    engine_tuple: EngineSetupResult = (
        engine_client,
        vllm_config,
        Mock(),
        "/tmp/prom",
        Mock(),
    )
    factory = WorkerFactory(
        setup_vllm_engine_fn=Mock(return_value=engine_tuple),
        setup_kv_event_publisher_fn=Mock(return_value=None),
        register_vllm_model_fn=fake_register_vllm_model,
        setup_fpm_relay_fn=Mock(return_value=None),
        setup_metrics_collection_fn=Mock(),
    )
    factory._maybe_get_encode_worker_client = AsyncMock(return_value=None)  # type: ignore[assignment]
    factory._maybe_wait_for_failover_lock = AsyncMock()  # type: ignore[assignment]
    factory.register_engine_routes = Mock()  # type: ignore[assignment]
    mock_handler = Mock(embedding_cache_manager=None)
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.PrefillWorkerHandler",
        Mock(return_value=mock_handler),
    )

    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.configure_kv_event_block_size", _noop
    )
    config = _make_config(
        disaggregation_mode=DisaggregationMode.PREFILL,
        use_vllm_tokenizer=False,
        namespace="dyn",
        component="prefill",
        endpoint="generate",
        served_model_name="m",
        model="m",
        frontend_decoding=False,
        enable_multimodal=False,
        enable_rl=False,
        engine_args=SimpleNamespace(enable_lora=False),
    )
    runtime = Mock()
    runtime.endpoint.return_value = Mock(connection_id=Mock(return_value="cid"))

    with pytest.raises(RuntimeError, match="stop-after-register"):
        await factory._create_prefill_worker(runtime, config, asyncio.Event(), [])

    assert order == ["wait", "stop", "register"]
    assert mock_handler._benchmark_results == {"status": "complete"}


@pytest.mark.asyncio
async def test_aborted_benchmark_artifact_still_stops_workers(monkeypatch, tmp_path):
    """The real abort chain: ``_bench_abort`` publishes a ``status="failed"``
    rank artifact, ``_wait_and_load_benchmark`` rejects it during validation,
    and the worker GC stop must still be issued with the validation error
    propagating unchanged. Only the RPC is faked; the wait and validation
    are the production code paths."""
    output_path = tmp_path / "benchmark.json"
    output_path.write_text(
        json.dumps({"status": "failed", "valid": False, "error": "benchmark aborted"})
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (0, 1)
    )
    stop = AsyncMock()
    monkeypatch.setattr("dynamo.vllm.worker_factory._stop_worker_gc_policy", stop)

    with pytest.raises(RuntimeError, match="incomplete results"):
        await _await_benchmark_then_restore_workers(
            {"output_path": str(output_path), "timeout": 1}, Mock(), Mock()
        )

    stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_failure_path_stop_is_time_boxed_and_original_error_wins(
    monkeypatch, caplog
):
    """If the engine died, the stop RPC never answers; the launcher must not
    hang on the error path -- the wait times out, is logged, and the original
    benchmark error propagates."""
    original = RuntimeError("Self-benchmark produced incomplete results")

    async def fake_wait(_cfg, _vllm_config):
        raise original

    async def hanging_stop(_client):
        await asyncio.sleep(3600)

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._wait_and_load_benchmark", fake_wait
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._stop_worker_gc_policy", hanging_stop
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.WORKER_GC_STOP_TIMEOUT_SECONDS", 0.05
    )
    caplog.set_level(logging.ERROR)

    with pytest.raises(RuntimeError) as exc_info:
        await _await_benchmark_then_restore_workers({}, Mock(), Mock())

    assert exc_info.value is original
    assert "Failed to restore model workers" in caplog.text


@pytest.mark.asyncio
async def test_success_path_stop_timeout_fails_closed(monkeypatch):
    async def fake_wait(_cfg, _vllm_config):
        return {"status": "complete"}

    async def hanging_stop(_client):
        await asyncio.sleep(3600)

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._wait_and_load_benchmark", fake_wait
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._stop_worker_gc_policy", hanging_stop
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.WORKER_GC_STOP_TIMEOUT_SECONDS", 0.05
    )

    with pytest.raises(asyncio.TimeoutError):
        await _await_benchmark_then_restore_workers({}, Mock(), Mock())


@pytest.mark.asyncio
async def test_prefill_call_site_stops_workers_when_benchmark_wait_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the call site itself: with the old two-line
    form (wait, then stop) a raising wait skips the stop. Drive the real
    prefill startup path with a raising wait and require the stop."""
    order: list[str] = []
    benchmark_error = RuntimeError("Self-benchmark produced incomplete results")

    async def fake_wait(_cfg, _vllm_config):
        order.append("wait")
        raise benchmark_error

    async def fake_stop(_client):
        order.append("stop")

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._wait_and_load_benchmark", fake_wait
    )
    monkeypatch.setattr("dynamo.vllm.worker_factory._stop_worker_gc_policy", fake_stop)

    engine_client = Mock()
    vllm_config = Mock()
    vllm_config.additional_config = {
        "benchmark": {"output_path": "/tmp/bench.json", "timeout": 1}
    }
    engine_tuple: EngineSetupResult = (
        engine_client,
        vllm_config,
        Mock(),
        "/tmp/prom",
        Mock(),
    )
    register = AsyncMock()
    factory = WorkerFactory(
        setup_vllm_engine_fn=Mock(return_value=engine_tuple),
        setup_kv_event_publisher_fn=Mock(return_value=None),
        register_vllm_model_fn=register,
        setup_fpm_relay_fn=Mock(return_value=None),
        setup_metrics_collection_fn=Mock(),
    )
    factory._maybe_get_encode_worker_client = AsyncMock(return_value=None)  # type: ignore[assignment]
    factory._maybe_wait_for_failover_lock = AsyncMock()  # type: ignore[assignment]
    factory.register_engine_routes = Mock()  # type: ignore[assignment]
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.PrefillWorkerHandler",
        Mock(return_value=Mock(embedding_cache_manager=None)),
    )

    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.configure_kv_event_block_size", _noop
    )
    config = _make_config(
        disaggregation_mode=DisaggregationMode.PREFILL,
        use_vllm_tokenizer=False,
        namespace="dyn",
        component="prefill",
        endpoint="generate",
        served_model_name="m",
        model="m",
        frontend_decoding=False,
        enable_multimodal=False,
        enable_rl=False,
        engine_args=SimpleNamespace(enable_lora=False),
    )
    runtime = Mock()
    runtime.endpoint.return_value = Mock(connection_id=Mock(return_value="cid"))

    with pytest.raises(RuntimeError) as exc_info:
        await factory._create_prefill_worker(runtime, config, asyncio.Event(), [])

    assert exc_info.value is benchmark_error
    assert order == ["wait", "stop"]
    register.assert_not_awaited()


def _decode_benchmark_factory(monkeypatch, tmp_path, register_fn):
    """Decode-path harness with a benchmark config: engine setup, handler,
    stat logger, KV sizing, and encode/failover collaborators are stubbed so
    the run reaches the benchmark wait (the stop-before-registration ordering
    is pinned on the prefill path, which shares the same wrapper call)."""
    engine_client = Mock()
    vllm_config = SimpleNamespace(
        additional_config={
            "benchmark": {"output_path": str(tmp_path / "bench.json"), "timeout": 1}
        },
        cache_config=SimpleNamespace(num_gpu_blocks=1),
        model_config=SimpleNamespace(max_model_len=1024),
        shutdown_timeout=5.0,
    )
    engine_setup: EngineSetupResult = (
        engine_client,
        vllm_config,
        Mock(),
        str(tmp_path / "prometheus"),
        Mock(),
    )
    factory = _make_factory(
        setup_vllm_engine_fn=Mock(return_value=engine_setup),
        register_vllm_model_fn=register_fn,
    )
    factory._maybe_create_failover_metrics = Mock(return_value=None)  # type: ignore[method-assign]
    factory._maybe_get_encode_worker_client = AsyncMock(return_value=None)  # type: ignore[method-assign]
    factory._maybe_wait_for_failover_lock = AsyncMock()  # type: ignore[method-assign]
    factory.register_engine_routes = Mock()  # type: ignore[method-assign]
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.StatLoggerFactory", Mock(return_value=Mock())
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (0, 1)
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.per_rank_kv_blocks",
        lambda _num_blocks, _dp_size: 1,
    )

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.configure_kv_event_block_size", _noop
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.DecodeWorkerHandler",
        Mock(return_value=Mock(embedding_cache_manager=None)),
    )
    runtime = Mock()
    runtime.endpoint.return_value = Mock(connection_id=Mock(return_value="worker-id"))
    # Mock-backed config (as in the prefill registration tests) so attributes
    # read after the benchmark wait resolve instead of raising AttributeError.
    config = _make_config(
        namespace="dynamo",
        component="backend",
        endpoint="generate",
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        enable_rl=False,
        engine_args=SimpleNamespace(enable_lora=False),
        enable_multimodal=False,
        route_to_encoder=False,
        custom_encoder_class=None,
        use_vllm_tokenizer=False,
        frontend_decoding=False,
        served_model_name="m",
        model="m",
        endpoint_types="chat,completions",
        dyn_endpoint_types="chat,completions",
        custom_jinja_template=None,
    )
    return factory, runtime, config, engine_client


@pytest.mark.asyncio
async def test_decode_call_site_stops_workers_when_benchmark_wait_raises(
    monkeypatch, tmp_path
):
    """Regression guard on the decode call site: a raising wait must still
    reach the worker stop (the old two-line form skipped it), the original
    error must propagate, registration must not happen, and the lifecycle
    must still tear the engine down afterwards."""
    order: list[str] = []
    benchmark_error = RuntimeError("Self-benchmark produced incomplete results")

    async def fake_wait(_cfg, _vllm_config):
        order.append("wait")
        raise benchmark_error

    async def fake_stop(_client):
        order.append("stop")

    register = AsyncMock()
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._wait_and_load_benchmark", fake_wait
    )
    monkeypatch.setattr("dynamo.vllm.worker_factory._stop_worker_gc_policy", fake_stop)
    factory, runtime, config, engine_client = _decode_benchmark_factory(
        monkeypatch, tmp_path, register
    )

    with pytest.raises(RuntimeError) as exc_info:
        await factory._create_decode_worker(runtime, config, asyncio.Event(), [])

    assert exc_info.value is benchmark_error
    assert order == ["wait", "stop"]
    register.assert_not_awaited()
    engine_client.shutdown.assert_called_once_with(timeout=5.0)


@pytest.mark.asyncio
class TestEncodeWorkerEmbeddingCacheCapacity:
    """The encode worker gets its cache capacity from the configured flag.

    ``--multimodal-embedding-cache-capacity-gb`` defaults to 0, which disables
    the cache, so the disabled case is the stock deployment rather than an edge
    case. The handler decides whether to build a cache at all; the factory's
    part is to hand it the configured value unchanged.
    """

    @staticmethod
    async def _create_encode_worker(capacity_gb):
        """Run ``_create_multimodal_encode_worker`` with everything external stubbed.

        Returns the patched handler constructor.
        """
        endpoint = Mock()
        endpoint.serve_endpoint = AsyncMock()
        runtime = Mock()
        runtime.endpoint = Mock(return_value=endpoint)

        handler = Mock()
        handler.async_init = AsyncMock()

        config = _make_config(
            namespace="dynamo",
            component="encoder",
            endpoint="generate",
            model="/models/qwen-vl",
            served_model_name="qwen-vl",
            frontend_decoding=False,
            multimodal_embedding_cache_capacity_gb=capacity_gb,
        )

        with patch(
            "dynamo.vllm.worker_factory.EncodeWorkerHandler", return_value=handler
        ) as handler_cls, patch(
            "dynamo.vllm.worker_factory.register_model", AsyncMock()
        ), patch(
            "dynamo.vllm.worker_factory.register_model_taint_route"
        ):
            await _make_factory()._create_multimodal_encode_worker(
                runtime, config, asyncio.Event(), []
            )

        return handler_cls

    async def test_passes_the_configured_capacity_to_the_handler(self) -> None:
        handler_cls = await self._create_encode_worker(4.0)

        assert handler_cls.call_args.kwargs["embedding_cache_capacity_gb"] == 4.0

    async def test_passes_the_disabling_default_through_unchanged(self) -> None:
        handler_cls = await self._create_encode_worker(0.0)

        assert handler_cls.call_args.kwargs["embedding_cache_capacity_gb"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_random_state_worker_is_disabled_after_benchmark(monkeypatch, failed):
    async def wait(_config, _vllm_config):
        if failed:
            raise RuntimeError("benchmark failed")
        return {"status": "complete"}

    monkeypatch.setattr("dynamo.vllm.worker_factory._wait_and_load_benchmark", wait)
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._stop_worker_gc_policy", AsyncMock()
    )
    client = SimpleNamespace(collective_rpc=AsyncMock())
    if failed:
        with pytest.raises(RuntimeError, match="benchmark failed"):
            await _await_benchmark_then_restore_workers(
                {"randomize_kda_state": True}, Mock(), client
            )
    else:
        await _await_benchmark_then_restore_workers(
            {"randomize_kda_state": True}, Mock(), client
        )
    client.collective_rpc.assert_awaited_once_with("finish_benchmark_kda_state")


@pytest.mark.asyncio
async def test_random_state_stop_failure_still_restores_gc(monkeypatch):
    wait = AsyncMock(return_value={"status": "complete"})
    stop_gc = AsyncMock()
    monkeypatch.setattr("dynamo.vllm.worker_factory._wait_and_load_benchmark", wait)
    monkeypatch.setattr("dynamo.vllm.worker_factory._stop_worker_gc_policy", stop_gc)
    client = SimpleNamespace(
        collective_rpc=AsyncMock(side_effect=RuntimeError("state stop failed"))
    )
    with pytest.raises(RuntimeError, match="state stop failed"):
        await _await_benchmark_then_restore_workers(
            {"randomize_kda_state": True}, Mock(), client
        )
    stop_gc.assert_awaited_once_with(client)


# --------------------------------------------------------------------------
# Engine provenance (AIC-1950)
# --------------------------------------------------------------------------


def _engine_block(dp_rank: int, backend: str | None = None) -> dict:
    """Minimal engine block shaped like the one the scheduler writes."""
    return {
        "attention": {
            "backend_requested": backend,
            "backend_resolved": None,
            "resolution": "pending_worker_probe",
        },
        "parallel": {"data_parallel_rank": dp_rank, "tensor_parallel_size": 4},
        "versions": {"vllm": "0.28.0"},
        "resolved": None,
        "resolution": "pending_worker_probe",
    }


def _engine_rank_payload(dp_rank: int, engine: dict | None) -> dict:
    """A two-rank-group rank artifact whose only variable is ``engine``."""
    point = {"benchmark_id": 1, "point_type": "prefill"}
    rank_results = [
        {"dp_rank": 0, "fpms": [{"counter_id": 1, "dp_rank": 0, "wall_time": 0.01}]},
        {"dp_rank": 1, "fpms": [{"counter_id": 1, "dp_rank": 1, "wall_time": 0.02}]},
    ]
    payload: dict = {
        "schema_version": 2,
        "artifact_type": "rank",
        "valid": True,
        "run_id": "run-1",
        "grid_digest": "grid-1",
        "timing": {
            "started_at": "2026-09-19T12:00:00Z",
            "completed_at": "2026-09-19T12:00:10Z",
            "benchmark_elapsed_seconds": 10.0,
            "measured_iteration_seconds": 0.02,
        },
        "dp": {"rank": dp_rank, "size": 2},
        "coverage": {
            "expected_points": 1,
            "completed_points": 1,
            "skipped_points": 0,
        },
        "results": [
            {
                "point": point,
                "fpms": [
                    {
                        "counter_id": 1,
                        "dp_rank": dp_rank,
                        "wall_time": 0.01 * (dp_rank + 1),
                    }
                ],
            }
        ],
        "iteration_groups": [
            {
                "benchmark_id": 1,
                "point": point,
                "expected_dp_ranks": [0, 1],
                "complete": True,
                "wall_time": 0.02,
                "rank_results": rank_results,
            }
        ],
        "skipped_points": [],
    }
    if engine is not None:
        payload["engine"] = engine
    return payload


def _engine_rank_payload_group(dp_rank: int, engine: dict | None, size: int) -> dict:
    """An N-rank-group rank artifact whose only variable is ``engine``.

    ``_engine_rank_payload``'s fixed 2-rank group cannot exercise a third,
    unrelated degraded rank alongside two ranks that genuinely disagree.
    """
    point = {"benchmark_id": 1, "point_type": "prefill"}
    # Scaled the same way as results[0].fpms[0] below, so a rank's own local
    # result matches its own copy of the synchronized group at every rank,
    # not just rank 0.
    rank_results = [
        {
            "dp_rank": r,
            "fpms": [{"counter_id": 1, "dp_rank": r, "wall_time": 0.01 * (r + 1)}],
        }
        for r in range(size)
    ]
    group_wall_time = 0.01 * size
    payload: dict = {
        "schema_version": 2,
        "artifact_type": "rank",
        "valid": True,
        "run_id": "run-1",
        "grid_digest": "grid-1",
        "timing": {
            "started_at": "2026-09-19T12:00:00Z",
            "completed_at": "2026-09-19T12:00:10Z",
            "benchmark_elapsed_seconds": 10.0,
            "measured_iteration_seconds": group_wall_time,
        },
        "dp": {"rank": dp_rank, "size": size},
        "coverage": {
            "expected_points": 1,
            "completed_points": 1,
            "skipped_points": 0,
        },
        "results": [
            {
                "point": point,
                "fpms": [
                    {
                        "counter_id": 1,
                        "dp_rank": dp_rank,
                        "wall_time": 0.01 * (dp_rank + 1),
                    }
                ],
            }
        ],
        "iteration_groups": [
            {
                "benchmark_id": 1,
                "point": point,
                "expected_dp_ranks": list(range(size)),
                "complete": True,
                "wall_time": group_wall_time,
                "rank_results": rank_results,
            }
        ],
        "skipped_points": [],
    }
    if engine is not None:
        payload["engine"] = engine
    return payload


def test_benchmark_engine_identity_strips_rank_and_probe_fields():
    identity = _benchmark_engine_identity({"engine": _engine_block(3, "FLASH_ATTN")})

    assert identity == {
        "attention": {
            "backend_requested": "FLASH_ATTN",
            "backend_resolved": None,
            "resolution": "pending_worker_probe",
        },
        "parallel": {"tensor_parallel_size": 4},
        "versions": {"vllm": "0.28.0"},
    }
    assert _benchmark_engine_identity({}) is None


def test_merge_copies_engine_provenance_and_clears_the_rank(tmp_path):
    merged = _merge_benchmark_rank_results(
        [
            (0, tmp_path / "rank0.json", _engine_rank_payload(0, _engine_block(0))),
            (1, tmp_path / "rank1.json", _engine_rank_payload(1, _engine_block(1))),
        ],
        tmp_path / "merged.json",
    )

    assert merged["engine"]["parallel"]["tensor_parallel_size"] == 4
    # Per-rank in a document that describes every rank would be a lie.
    assert merged["engine"]["parallel"]["data_parallel_rank"] is None
    assert merged["engine"]["versions"] == {"vllm": "0.28.0"}
    assert "capture_errors" not in merged["engine"]


def test_merge_rejects_engine_provenance_mismatch(tmp_path):
    with pytest.raises(RuntimeError, match="engine provenance mismatch"):
        _merge_benchmark_rank_results(
            [
                (
                    0,
                    tmp_path / "rank0.json",
                    _engine_rank_payload(0, _engine_block(0, "FLASH_ATTN")),
                ),
                (
                    1,
                    tmp_path / "rank1.json",
                    _engine_rank_payload(1, _engine_block(1, "FLASHINFER")),
                ),
            ],
            tmp_path / "merged.json",
        )


def test_merge_engine_provenance_mismatch_with_absent_vs_none_key_still_raises(
    tmp_path,
):
    """A top-level engine key present-with-None on one rank and absent on
    the other makes every key's .get() comparison agree (None either way),
    even though the two identity dicts are still != as a whole (one has the
    key, one doesn't): the generator that looks for the differing field is
    empty. Before the fix this raised IndexError instead of the descriptive
    RuntimeError (M1)."""
    engine_0 = _engine_block(0, "FLASH_ATTN")
    engine_0["extra"] = None
    engine_1 = _engine_block(1, "FLASH_ATTN")  # no "extra" key at all

    with pytest.raises(RuntimeError, match="engine provenance mismatch"):
        _merge_benchmark_rank_results(
            [
                (0, tmp_path / "rank0.json", _engine_rank_payload(0, engine_0)),
                (1, tmp_path / "rank1.json", _engine_rank_payload(1, engine_1)),
            ],
            tmp_path / "merged.json",
        )


def test_merge_carries_engine_capture_error_without_raising(tmp_path, caplog):
    """A rank whose engine capture raised (``_bench_capture_engine`` fails
    soft into ``capture_error``, AIC-1950 Task 1) must not fail the whole
    merge: its FPM numbers are still trustworthy even though its engine
    provenance is not. The failure is carried into the merged document
    instead of being compared for identity."""
    caplog.set_level(logging.WARNING)
    # A real capture failure yields a partially filled block -- whatever
    # sub-blocks _bench_capture_engine built before the exception -- plus
    # capture_error, not just the marker in isolation.
    partial_capture = {
        "attention": {
            "backend_requested": "FLASHINFER",
            "backend_resolved": None,
            "resolution": "pending_worker_probe",
        },
        "capture_error": "hf_config missing",
    }

    merged = _merge_benchmark_rank_results(
        [
            (0, tmp_path / "rank0.json", _engine_rank_payload(0, _engine_block(0))),
            (1, tmp_path / "rank1.json", _engine_rank_payload(1, partial_capture)),
        ],
        tmp_path / "merged.json",
    )

    assert merged["engine"]["capture_errors"] == {"1": "hf_config missing"}
    # The reference rank (0) captured fine; its fields still come through.
    assert merged["engine"]["parallel"]["tensor_parallel_size"] == 4
    assert "engine provenance" in caplog.text


def test_merge_carries_engine_provenance_missing_on_one_rank_only(tmp_path, caplog):
    """A rank with no ``engine`` block at all (e.g. an older worker image)
    is handled the same way as a capture failure: recorded, not raised."""
    caplog.set_level(logging.WARNING)

    merged = _merge_benchmark_rank_results(
        [
            (0, tmp_path / "rank0.json", _engine_rank_payload(0, _engine_block(0))),
            (1, tmp_path / "rank1.json", _engine_rank_payload(1, None)),
        ],
        tmp_path / "merged.json",
    )

    assert merged["engine"]["capture_errors"] == {"1": "missing engine block"}
    assert "engine provenance" in caplog.text


def test_merge_ignores_probe_filled_engine_fields(tmp_path):
    """resolved/resolution are written after the merge by the worker probe;
    a rank that already carries one must not fail the merge."""
    probed = _engine_block(1)
    probed["resolved"] = {"attention_backends": {"layer.0": "FLASHINFER_MLA"}}
    probed["resolution"] = "worker_probe"

    merged = _merge_benchmark_rank_results(
        [
            (0, tmp_path / "rank0.json", _engine_rank_payload(0, _engine_block(0))),
            (1, tmp_path / "rank1.json", _engine_rank_payload(1, probed)),
        ],
        tmp_path / "merged.json",
    )

    assert merged["engine"]["resolved"] is None


def test_merge_still_rejects_mismatch_with_one_degraded_rank_present(tmp_path):
    """One degraded rank must not switch the identity check off for the
    ranks that did capture: two clean ranks on different attention backends
    must still raise even though a third rank's capture failed."""
    with pytest.raises(RuntimeError, match="engine provenance mismatch"):
        _merge_benchmark_rank_results(
            [
                (
                    0,
                    tmp_path / "rank0.json",
                    _engine_rank_payload_group(0, _engine_block(0, "FLASH_ATTN"), 3),
                ),
                (
                    1,
                    tmp_path / "rank1.json",
                    _engine_rank_payload_group(1, _engine_block(1, "FLASHINFER"), 3),
                ),
                (
                    2,
                    tmp_path / "rank2.json",
                    _engine_rank_payload_group(2, {"capture_error": "boom"}, 3),
                ),
            ],
            tmp_path / "merged.json",
        )


def test_merge_reseeds_engine_from_first_clean_rank_when_reference_degraded(
    tmp_path, caplog
):
    """When the merge's reference rank (``rank_data[0]``) is itself
    degraded, the provenance another rank did capture must not be lost: the
    merged engine block is reseeded from the first rank whose capture
    succeeded, not just left as the degraded reference's own (missing or
    partial) block."""
    caplog.set_level(logging.WARNING)

    merged = _merge_benchmark_rank_results(
        [
            (0, tmp_path / "rank0.json", _engine_rank_payload(0, None)),
            (1, tmp_path / "rank1.json", _engine_rank_payload(1, _engine_block(1))),
        ],
        tmp_path / "merged.json",
    )

    assert merged["engine"]["parallel"]["tensor_parallel_size"] == 4
    assert merged["engine"]["parallel"]["data_parallel_rank"] is None
    assert merged["engine"]["versions"] == {"vllm": "0.28.0"}
    assert merged["engine"]["capture_errors"] == {"0": "missing engine block"}
    assert "engine provenance" in caplog.text


def test_merge_without_engine_provenance_is_unchanged(tmp_path, caplog):
    """No rank capturing an ``engine`` block at all (e.g. a pre-Task-1
    artifact) is not a degraded run: the merge stays byte-for-byte backward
    compatible, with no ``engine`` key and no warning."""
    caplog.set_level(logging.WARNING)

    merged = _merge_benchmark_rank_results(
        [
            (0, tmp_path / "rank0.json", _engine_rank_payload(0, None)),
            (1, tmp_path / "rank1.json", _engine_rank_payload(1, None)),
        ],
        tmp_path / "merged.json",
    )

    assert "engine" not in merged
    assert "engine provenance" not in caplog.text


# --------------------------------------------------------------------------
# Engine provenance worker probe (AIC-1950, Task 3)
# --------------------------------------------------------------------------


class _FakeAttentionBackend:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


class _FakePrefillBackend:
    """A stand-in prefill backend class. vLLM actually stores an *instance*
    of it on MLAAttention.prefill_backend, but the probe tolerates a bare
    class too -- see test_engine_probe_reads_backends_off_the_attention_layers."""


class _FakeAttentionLayer:
    def __init__(self, backend_name: str, prefill_backend=None) -> None:
        self._backend = _FakeAttentionBackend(backend_name)
        if prefill_backend is not None:
            self.prefill_backend = prefill_backend

    def get_attn_backend(self):
        return self._backend


def _fake_worker(
    layers: dict,
    capture_sizes=(1, 2, 8),
    rank: int = 2,
    data_parallel_rank=5,
    data_parallel_index=None,
    tensor_parallel_size=None,
):
    parallel_kwargs = {"data_parallel_rank": data_parallel_rank}
    if data_parallel_index is not None:
        parallel_kwargs["data_parallel_index"] = data_parallel_index
    if tensor_parallel_size is not None:
        parallel_kwargs["tensor_parallel_size"] = tensor_parallel_size
    return SimpleNamespace(
        rank=rank,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context=layers,
                cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
                cudagraph_capture_sizes=list(capture_sizes),
            ),
            parallel_config=SimpleNamespace(**parallel_kwargs),
        ),
    )


def _merged_with_engine(tmp_path, rank_count: int = 1) -> dict:
    """A merged document plus the rank files it points at, all on disk."""
    rank_files = []
    for dp_rank in range(rank_count):
        path = tmp_path / f"rank{dp_rank}.json"
        path.write_text(json.dumps({"engine": _engine_block(dp_rank)}))
        rank_files.append(str(path))
    merged_path = tmp_path / "merged.json"
    merged = {
        "engine": _engine_block(0),
        "rank_files": rank_files,
        "merged_output_path": str(merged_path),
    }
    merged["engine"]["parallel"]["data_parallel_rank"] = None
    merged_path.write_text(json.dumps(merged))
    return merged


class _ExplodingAttentionLayer:
    """A layer whose backend accessor raises, e.g. a half-initialised layer."""

    prefill_backend = _FakePrefillBackend

    def get_attn_backend(self):
        raise RuntimeError("backend selector exploded")


def test_engine_probe_reads_backends_off_the_attention_layers():
    probe = _make_engine_probe()
    layers = {
        # vLLM stores an *instance* on MLAAttention.prefill_backend (layer
        # 1); the probe must also tolerate a bare class (layer 0).
        "model.layers.0.self_attn.attn": _FakeAttentionLayer(
            "FLASHINFER_MLA", _FakePrefillBackend
        ),
        "model.layers.1.self_attn.attn": _FakeAttentionLayer(
            "FLASHINFER_MLA", _FakePrefillBackend()
        ),
        # A standard (non-MLA) attention layer: it has no prefill_backend
        # attribute at all, the same as real vLLM's non-MLA Attention layers.
        "model.layers.2.self_attn.attn": _FakeAttentionLayer("FLASH_ATTN"),
        # A half-initialised layer whose backend accessor raises: it must be
        # skipped entirely (no entry, no prefill contribution) and must not
        # fail the probe for the healthy layers.
        "model.layers.3.self_attn.attn": _ExplodingAttentionLayer(),
        "model.layers.1.mlp": SimpleNamespace(),
    }

    result = probe(_fake_worker(layers, rank=6, tensor_parallel_size=4))

    assert result["attention_backends"] == {
        "model.layers.0.self_attn.attn": "FLASHINFER_MLA",
        "model.layers.1.self_attn.attn": "FLASHINFER_MLA",
        "model.layers.2.self_attn.attn": "FLASH_ATTN",
    }
    assert "model.layers.3.self_attn.attn" not in result["attention_backends"]
    # Only the MLA layers contribute a prefill backend; the standard layer
    # does not dilute or clear it, and the exploding layer never reaches its
    # own prefill_backend attribute. Both the class (layer 0) and the
    # instance (layer 1) resolve to the same name.
    assert result["mla_prefill_backend"] == "_FakePrefillBackend"
    assert result["cudagraph_mode_resolved"] == "FULL_DECODE_ONLY"
    assert result["cudagraph_capture_sizes_resolved"] == [1, 2, 8]
    assert result["dp_rank"] == 5
    assert result["worker_rank"] == 6
    # tp_rank is derived from worker_rank % tensor_parallel_size, never from
    # the raw global rank directly -- 6 % 4 == 2, distinct from both inputs,
    # so this cannot pass by coincidentally echoing one of them.
    assert result["tp_rank"] == 2


def test_engine_probe_prefers_data_parallel_index_over_zeroed_rank():
    """vLLM zeroes parallel_config.data_parallel_rank on every engine for a
    dense (non-MoE) model under external DP, keeping the true rank only in
    data_parallel_index -- the same trap instrumented_scheduler.py already
    routes around. The probe must not silently report dp_rank=0 here."""
    probe = _make_engine_probe()
    worker = _fake_worker(
        {}, data_parallel_rank=0, data_parallel_index=3, tensor_parallel_size=4
    )

    result = probe(worker)

    assert result["dp_rank"] == 3


def test_engine_probe_tp_rank_is_none_without_tensor_parallel_size():
    """No tensor_parallel_size on the config means the TP-local rank cannot
    be derived: tp_rank must be None, never the raw global rank (the bug
    this derivation replaces)."""
    probe = _make_engine_probe()
    worker = _fake_worker({}, rank=7)

    result = probe(worker)

    assert result["tp_rank"] is None
    assert result["worker_rank"] == 7


def test_attach_engine_resolved_updates_merged_and_rank_files(tmp_path):
    merged = _merged_with_engine(tmp_path, rank_count=2)
    probe_result = {
        "tp_rank": 0,
        "dp_rank": 0,
        "attention_backends": {"model.layers.0.self_attn.attn": "FLASHINFER_MLA"},
        "mla_prefill_backend": "TrtllmRaggedMLAPrefill",
        "cudagraph_mode_resolved": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes_resolved": [1, 2],
    }
    engine_client = SimpleNamespace(
        collective_rpc=AsyncMock(return_value=[probe_result, probe_result])
    )

    asyncio.run(_attach_engine_resolved(merged, engine_client))

    engine_client.collective_rpc.assert_awaited_once()
    assert (
        engine_client.collective_rpc.call_args.kwargs["timeout"]
        == ENGINE_PROBE_TIMEOUT_SECONDS
    )
    assert merged["engine"]["resolved"] == probe_result
    assert merged["engine"]["resolution"] == "worker_probe"
    assert merged["engine"]["attention"]["backend_resolved"] == "FLASHINFER_MLA"
    assert (
        merged["engine"]["attention"]["mla_prefill_backend_resolved"]
        == "TrtllmRaggedMLAPrefill"
    )
    assert merged["engine"]["attention"]["resolution"] == "worker_probe"
    on_disk = json.loads((tmp_path / "merged.json").read_text())
    assert on_disk["engine"]["resolved"] == probe_result
    for dp_rank in (0, 1):
        rank_doc = json.loads((tmp_path / f"rank{dp_rank}.json").read_text())
        assert rank_doc["engine"]["resolved"] == probe_result
        assert rank_doc["engine"]["resolution"] == "worker_probe"
        # The rank's own provenance is untouched.
        assert rank_doc["engine"]["parallel"]["data_parallel_rank"] == dp_rank


def test_attach_engine_resolved_marks_mixed_backends(tmp_path):
    merged = _merged_with_engine(tmp_path)
    engine_client = SimpleNamespace(
        collective_rpc=AsyncMock(
            return_value=[
                {
                    "attention_backends": {
                        "layer.0": "FLASHINFER_MLA",
                        "layer.1": "TRITON_ATTN",
                    },
                    "mla_prefill_backend": None,
                }
            ]
        )
    )

    asyncio.run(_attach_engine_resolved(merged, engine_client))

    assert merged["engine"]["attention"]["backend_resolved"] is None
    assert merged["engine"]["attention"]["resolution"] == "worker_probe_mixed"
    assert merged["engine"]["resolved"]["attention_backends"]["layer.1"] == (
        "TRITON_ATTN"
    )


def test_attach_engine_resolved_records_probe_failure(tmp_path):
    merged = _merged_with_engine(tmp_path)
    engine_client = SimpleNamespace(
        collective_rpc=AsyncMock(side_effect=RuntimeError("worker died"))
    )

    asyncio.run(_attach_engine_resolved(merged, engine_client))

    assert merged["engine"]["resolved"] is None
    assert merged["engine"]["resolution"] == "probe_failed: RuntimeError: worker died"
    assert merged["engine"]["attention"]["backend_resolved"] is None
    rank_doc = json.loads((tmp_path / "rank0.json").read_text())
    assert rank_doc["engine"]["resolution"] == "probe_failed: RuntimeError: worker died"


def test_attach_engine_resolved_handles_an_empty_rpc_result(tmp_path):
    """No worker answered at all (e.g. zero ranks in this DP group's RPC
    fan-out); treated the same as a probe failure, not a crash."""
    merged = _merged_with_engine(tmp_path)
    engine_client = SimpleNamespace(collective_rpc=AsyncMock(return_value=[]))

    asyncio.run(_attach_engine_resolved(merged, engine_client))

    assert merged["engine"]["resolved"] is None
    assert merged["engine"]["resolution"].startswith("probe_failed:")
    assert merged["engine"]["attention"]["backend_resolved"] is None


def test_attach_engine_resolved_handles_a_non_dict_rpc_result(tmp_path):
    """A worker answered with something that isn't the probe's dict shape
    (e.g. it raised inside the RPC handler and vLLM surfaced ``None``);
    treated the same as a probe failure, not a crash."""
    merged = _merged_with_engine(tmp_path)
    engine_client = SimpleNamespace(collective_rpc=AsyncMock(return_value=[None]))

    asyncio.run(_attach_engine_resolved(merged, engine_client))

    assert merged["engine"]["resolved"] is None
    assert merged["engine"]["resolution"].startswith("probe_failed:")
    assert merged["engine"]["attention"]["backend_resolved"] is None


def test_attach_engine_resolved_survives_a_malformed_dict_shaped_result(tmp_path):
    """A dict-shaped RPC result whose ``attention_backends`` is not a
    ``dict[str, str]`` (a malformed or future-incompatible probe payload)
    used to raise ``AttributeError`` out of ``_apply_engine_resolved`` and
    escape the launcher: applying the result and writing it out now run
    inside the same fail-soft try as the RPC itself, so this is recorded as
    a probe failure instead."""
    merged = _merged_with_engine(tmp_path)
    engine_client = SimpleNamespace(
        collective_rpc=AsyncMock(
            return_value=[
                {"attention_backends": ["FLASH_ATTN"], "mla_prefill_backend": None}
            ]
        )
    )

    asyncio.run(_attach_engine_resolved(merged, engine_client))  # must not raise

    assert merged["engine"]["resolved"] is None
    assert merged["engine"]["resolution"].startswith("probe_failed:")
    assert "AttributeError" in merged["engine"]["resolution"]
    assert merged["engine"]["attention"]["backend_resolved"] is None
    on_disk = json.loads((tmp_path / "merged.json").read_text())
    assert on_disk["engine"]["resolution"] == merged["engine"]["resolution"]
    rank_doc = json.loads((tmp_path / "rank0.json").read_text())
    assert rank_doc["engine"]["resolution"] == merged["engine"]["resolution"]


def test_attach_engine_resolved_records_the_exception_type_on_a_timeout(
    monkeypatch, tmp_path
):
    """``asyncio.wait_for`` raises a bare ``TimeoutError``, whose ``str()``
    is empty; the recorded reason must still name the exception type so a
    hung probe -- the single most likely production failure -- does not
    degrade into a useless, reason-free ``"probe_failed: "``."""
    monkeypatch.setattr("dynamo.vllm.worker_factory.ENGINE_PROBE_TIMEOUT_SECONDS", 0.05)
    merged = _merged_with_engine(tmp_path)

    async def hangs(*_args, **_kwargs):
        await asyncio.sleep(2.0)
        return [{}]

    engine_client = SimpleNamespace(collective_rpc=hangs)

    asyncio.run(_attach_engine_resolved(merged, engine_client))

    assert merged["engine"]["resolved"] is None
    assert merged["engine"]["resolution"] == "probe_failed: TimeoutError: "


def test_attach_engine_resolved_preserves_non_engine_rank_file_content(tmp_path):
    """Only the ``engine`` block changes; everything else in a rank file --
    results, timing, coverage, key order, exact floats -- must round-trip
    the rewrite unchanged."""
    rank_path = tmp_path / "rank0.json"
    original = {
        "schema_version": 2,
        "artifact_type": "rank",
        "engine": _engine_block(0),
        "results": [
            {
                "point": {"benchmark_id": 1, "point_type": "prefill"},
                "fpms": [{"counter_id": 1, "wall_time": 0.123456789012345}],
            }
        ],
        "timing": {"benchmark_elapsed_seconds": 12.5},
        "coverage": {"expected_points": 1, "completed_points": 1, "skipped_points": 0},
    }
    rank_path.write_text(json.dumps(original))
    merged_path = tmp_path / "merged.json"
    merged = {
        "engine": _engine_block(0),
        "rank_files": [str(rank_path)],
        "merged_output_path": str(merged_path),
    }
    merged_path.write_text(json.dumps(merged))
    engine_client = SimpleNamespace(
        collective_rpc=AsyncMock(
            return_value=[
                {
                    "attention_backends": {"layer.0": "FLASH_ATTN"},
                    "mla_prefill_backend": None,
                }
            ]
        )
    )

    asyncio.run(_attach_engine_resolved(merged, engine_client))

    rewritten = json.loads(rank_path.read_text())
    assert rewritten["engine"]["resolution"] == "worker_probe"
    for key in ("schema_version", "artifact_type", "results", "timing", "coverage"):
        assert rewritten[key] == original[key]


def test_attach_engine_resolved_is_a_noop_without_an_engine_block():
    engine_client = SimpleNamespace(collective_rpc=AsyncMock())

    asyncio.run(_attach_engine_resolved({"status": "complete"}, engine_client))

    engine_client.collective_rpc.assert_not_awaited()


def test_benchmark_wait_probes_the_engine_before_restoring_workers(monkeypatch):
    calls = []

    async def fake_wait(_cfg, _vllm_config):
        calls.append("wait")
        return {"status": "complete"}

    async def fake_attach(_merged, _client):
        calls.append("probe")

    async def fake_stop(_client):
        calls.append("stop")

    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._wait_and_load_benchmark", fake_wait
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory._attach_engine_resolved", fake_attach
    )
    monkeypatch.setattr("dynamo.vllm.worker_factory._stop_worker_gc_policy", fake_stop)

    results = asyncio.run(_await_benchmark_then_restore_workers({}, Mock(), Mock()))

    assert results == {"status": "complete"}
    assert calls == ["wait", "probe", "stop"]
