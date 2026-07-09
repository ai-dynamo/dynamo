# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for worker_factory.py"""

import asyncio
import json
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
    }
    defaults.update(overrides)
    return Mock(**defaults)


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
async def test_wait_and_load_benchmark_aggregates_dp_coverage(monkeypatch, tmp_path):
    base_path = tmp_path / "benchmark.json"
    rank_payload = {
        "valid": True,
        "coverage": {
            "expected_points": 1,
            "completed_points": 1,
            "skipped_points": 0,
        },
        "results": [{"point": {"point_type": "prefill"}, "fpms": [{}]}],
        "skipped_points": [],
    }
    base_path.write_text(json.dumps(rank_payload))
    (tmp_path / "benchmark_dp1.json").write_text(json.dumps(rank_payload))
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
    assert [result["point"]["dp_rank"] for result in merged["results"]] == [0, 1]


@pytest.mark.asyncio
class TestCreate:
    """Test WorkerFactory.create() routing."""

    @pytest.fixture
    def factory(self) -> WorkerFactory:
        factory = WorkerFactory(
            setup_vllm_engine_fn=Mock(),
            setup_kv_event_publisher_fn=AsyncMock(),
            setup_valkey_worker_registration_fn=AsyncMock(return_value=None),
            register_vllm_model_fn=AsyncMock(),
            setup_fpm_relay_fn=Mock(),
            setup_metrics_collection_fn=Mock(),
        )
        factory._create_multimodal_encode_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_multimodal_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_prefill_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_decode_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_embedding_worker = AsyncMock()  # type: ignore[assignment]
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
            setup_kv_event_publisher_fn=AsyncMock(return_value=None),
            setup_valkey_worker_registration_fn=AsyncMock(return_value=None),
            register_vllm_model_fn=fake_register_vllm_model,
            setup_fpm_relay_fn=Mock(return_value=None),
            setup_metrics_collection_fn=Mock(),
        )
        factory._maybe_get_encode_worker_client = AsyncMock(return_value=None)  # type: ignore[assignment]
        factory._maybe_wait_for_failover_lock = AsyncMock()  # type: ignore[assignment]
        factory.register_engine_routes = Mock()  # type: ignore[assignment]

        # embedding_cache_manager=None skips register_embedding_cache_metrics.
        mock_handler = Mock(
            embedding_cache_manager=None,
            shutdown_valkey_worker_registration=AsyncMock(),
            cleanup=Mock(),
        )
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

        factory.setup_valkey_worker_registration.assert_awaited_once_with(
            config,
            runtime.endpoint.return_value,
            vllm_config,
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
