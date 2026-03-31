# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for worker_factory.py"""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import dynamo.vllm.worker_factory as worker_factory_module
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.worker_factory import EngineSetupResult, WorkerFactory

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
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
    }
    defaults.update(overrides)
    return Mock(**defaults)


class TestHandles:
    """Test WorkerFactory.handles() config detection."""

    # Legacy worker config
    @pytest.mark.parametrize("route_to_encode", [True, False])
    def test_multimodal_encode_worker(self, route_to_encode: bool) -> None:
        # 'route_to_encoder' can be passed, the worker creation may ignore it.
        config = _make_config(
            multimodal_encode_worker=True, route_to_encoder=route_to_encode
        )
        assert WorkerFactory.handles(config)

    @pytest.mark.parametrize("route_to_encode", [True, False])
    def test_multimodal_worker(self, route_to_encode: bool) -> None:
        config = _make_config(multimodal_worker=True, route_to_encoder=route_to_encode)
        assert WorkerFactory.handles(config)

    @pytest.mark.parametrize("route_to_encode", [True, False])
    def test_multimodal_decode_worker(self, route_to_encode: bool) -> None:
        config = _make_config(
            multimodal_decode_worker=True, route_to_encoder=route_to_encode
        )
        assert WorkerFactory.handles(config)

    # Tests for no standalone encode worker setting
    @pytest.mark.parametrize("route_to_encode", [True, False])
    def test_no_multimodal_flags(self, route_to_encode: bool) -> None:
        config = _make_config(route_to_encoder=route_to_encode)
        assert WorkerFactory.handles(config)

    @pytest.mark.parametrize("route_to_encode", [True, False])
    def test_prefill(self, route_to_encode: bool) -> None:
        config = _make_config(
            disaggregation_mode=DisaggregationMode.PREFILL,
            route_to_encoder=route_to_encode,
        )
        assert WorkerFactory.handles(config)

    @pytest.mark.parametrize("route_to_encode", [True, False])
    def test_decode(self, route_to_encode: bool) -> None:
        config = _make_config(
            disaggregation_mode=DisaggregationMode.DECODE,
            route_to_encoder=route_to_encode,
        )
        assert WorkerFactory.handles(config)


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

    # Tests with legacy worker config.
    @pytest.mark.parametrize("route_to_encode", [True, False])
    async def test_routes_to_multimodal_encode(
        self, factory: WorkerFactory, route_to_encode: bool
    ) -> None:
        config = _make_config(
            multimodal_encode_worker=True, route_to_encoder=route_to_encode
        )
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event, [])

        factory._create_multimodal_encode_worker.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.parametrize("route_to_encode", [True, False])
    async def test_routes_to_multimodal_worker(
        self, factory: WorkerFactory, route_to_encode: bool
    ) -> None:
        config = _make_config(multimodal_worker=True, route_to_encoder=route_to_encode)
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event, [])

        factory._create_multimodal_worker.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.parametrize("route_to_encode", [True, False])
    async def test_routes_multimodal_decode_worker(
        self, factory: WorkerFactory, route_to_encode: bool
    ) -> None:
        config = _make_config(
            multimodal_decode_worker=True, route_to_encoder=route_to_encode
        )
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event, [])

        factory._create_multimodal_worker.assert_called_once()  # type: ignore[union-attr]

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

        factory._create_multimodal_worker.assert_called_once_with(  # type: ignore[union-attr]
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            snapshot_engine=snapshot_engine,
        )

    async def test_shadow_mode_wakes_with_resume_only(
        self, factory: WorkerFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _make_config(
            namespace="test-ns",
            component="decode",
            endpoint="generate",
            endpoint_types="generate",
            gms_shadow_mode=True,
            enable_multimodal=False,
            use_vllm_tokenizer=False,
            frontend_decoding=False,
            custom_jinja_template=None,
            engine_args=Mock(enable_lora=False),
        )
        runtime = Mock()
        generate_endpoint = Mock()
        generate_endpoint.connection_id.return_value = 123
        clear_endpoint = Mock()
        runtime.endpoint.side_effect = [generate_endpoint, clear_endpoint]
        shutdown_event = asyncio.Event()
        shutdown_endpoints: list = []

        engine_client = Mock()
        vllm_config = Mock()
        vllm_config.cache_config.num_gpu_blocks = 8
        vllm_config.additional_config = {}
        vllm_config.model_config = Mock(max_model_len=128)
        factory.setup_vllm_engine.return_value = (
            engine_client,
            vllm_config,
            Mock(),
            Mock(),
            Mock(),
        )
        factory.setup_kv_event_publisher.return_value = None
        factory.setup_fpm_relay.return_value = None
        factory.setup_metrics_collection = Mock()
        factory._maybe_get_encode_worker_client = AsyncMock(return_value=None)  # type: ignore[method-assign]
        factory.register_vllm_model = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("stop after shadow wake")
        )

        lock = SimpleNamespace(acquire=AsyncMock())
        flock_module = types.ModuleType("gpu_memory_service.failover_lock.flock")
        flock_module.FlockFailoverLock = Mock(return_value=lock)
        monkeypatch.setitem(sys.modules, "gpu_memory_service", types.ModuleType("gms"))
        monkeypatch.setitem(
            sys.modules,
            "gpu_memory_service.failover_lock",
            types.ModuleType("gpu_memory_service.failover_lock"),
        )
        monkeypatch.setitem(
            sys.modules,
            "gpu_memory_service.failover_lock.flock",
            flock_module,
        )

        quiesce_controller = SimpleNamespace(
            quiesce=AsyncMock(),
            resume=AsyncMock(),
        )
        handler = Mock()
        handler._quiesce_controller = quiesce_controller
        handler.add_temp_dir = Mock()
        handler.sleep = AsyncMock()
        handler.wake_up = AsyncMock()
        handler.scale_elastic_ep = AsyncMock()
        handler.cleanup = Mock()

        monkeypatch.setattr(
            worker_factory_module,
            "DecodeWorkerHandler",
            Mock(return_value=handler),
        )
        monkeypatch.setattr(
            worker_factory_module,
            "parse_endpoint_types",
            Mock(return_value=Mock()),
        )

        with pytest.raises(RuntimeError, match="stop after shadow wake"):
            await factory._create_decode_worker(
                runtime,
                config,
                shutdown_event,
                shutdown_endpoints,
            )

        quiesce_controller.quiesce.assert_awaited_once_with(1)
        quiesce_controller.resume.assert_awaited_once_with()
        runtime.set_health_status.assert_called_once_with(True)
        lock.acquire.assert_awaited_once_with(engine_id="engine-0")
