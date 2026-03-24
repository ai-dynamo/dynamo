# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for worker_factory.py"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

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
