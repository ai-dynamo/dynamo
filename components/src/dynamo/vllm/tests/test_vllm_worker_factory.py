# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for worker_factory.py"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.worker_factory import (
    EngineSetupResult,
    WorkerFactory,
    _wait_and_load_benchmark,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    # gpu_1 not gpu_0: vLLM DeviceConfig(device='auto') fails on CPU-only arm64
    # runners with "Failed to infer device type" even for mock tests.
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
    pytest.mark.pre_merge,
]


def _make_config(**overrides) -> Mock:
    """Create a mock Config with all multimodal flags defaulting to False."""
    defaults = {
        "multimodal_encode_worker": False,
        "multimodal_worker": False,
        "multimodal_decode_worker": False,
        "omni": False,
        "route_to_encoder": False,
        "disaggregation_mode": DisaggregationMode.AGGREGATED,
        "embedding_worker": False,
        "transcription_worker": False,
    }
    defaults.update(overrides)
    return Mock(**defaults)


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
        factory._create_multimodal_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_prefill_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_decode_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_embedding_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_transcription_worker = AsyncMock()  # type: ignore[assignment]
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

    async def test_transcription_worker_takes_priority(
        self, factory: WorkerFactory
    ) -> None:
        config = _make_config(transcription_worker=True)
        await factory.create(Mock(), config, asyncio.Event(), [])

        factory._create_transcription_worker.assert_called_once()  # type: ignore[union-attr]
        factory._create_decode_worker.assert_not_called()  # type: ignore[union-attr]

    async def test_embedding_worker_takes_priority(
        self, factory: WorkerFactory
    ) -> None:
        """--embedding-worker is checked first; disaggregation_mode is ignored."""
        config = _make_config(embedding_worker=True)
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event, [])

        factory._create_embedding_worker.assert_called_once()  # type: ignore[union-attr]
        factory._create_decode_worker.assert_not_called()  # type: ignore[union-attr]
        factory._create_prefill_worker.assert_not_called()  # type: ignore[union-attr]
        factory._create_multimodal_encode_worker.assert_not_called()  # type: ignore[union-attr]

    async def test_passes_snapshot_engine(self, factory: WorkerFactory) -> None:
        config = _make_config(multimodal_worker=True)
        runtime = Mock()
        shutdown_event = asyncio.Event()
        shutdown_endpoints: list = []
        snapshot_engine: EngineSetupResult = (
            Mock(),
            Mock(),
            Mock(),
            "/tmp/prometheus",
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
        )

        runtime = Mock()
        runtime.endpoint.return_value = Mock(connection_id=Mock(return_value="cid"))

        with pytest.raises(RuntimeError, match="stop-after-register"):
            await factory._create_prefill_worker(
                runtime,
                config,
                asyncio.Event(),
                [],
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
