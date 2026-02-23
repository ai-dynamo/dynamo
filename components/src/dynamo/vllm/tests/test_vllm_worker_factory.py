# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for worker_factory.py"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm.worker_factory import EngineSetupResult, WorkerFactory


def _make_config(**overrides) -> Mock:
    """Create a mock Config with all multimodal flags defaulting to False."""
    defaults = {
        "multimodal_encode_worker": False,
        "multimodal_worker": False,
        "multimodal_decode_worker": False,
        "omni": False,
        "is_prefill_worker": False,
    }
    defaults.update(overrides)
    return Mock(**defaults)


class TestHandles:
    """Test WorkerFactory.handles() config detection."""

    def test_multimodal_encode_worker(self) -> None:
        config = _make_config(multimodal_encode_worker=True)
        assert WorkerFactory.handles(config)

    def test_multimodal_worker(self) -> None:
        config = _make_config(multimodal_worker=True)
        assert WorkerFactory.handles(config)

    def test_multimodal_decode_worker(self) -> None:
        config = _make_config(multimodal_decode_worker=True)
        assert WorkerFactory.handles(config)

    def test_no_multimodal_flags(self) -> None:
        config = _make_config()
        assert not WorkerFactory.handles(config)

    def test_omni_not_handled(self) -> None:
        config = _make_config(omni=True)
        assert not WorkerFactory.handles(config)

    def test_prefill_only_not_handled(self) -> None:
        config = _make_config(is_prefill_worker=True)
        assert not WorkerFactory.handles(config)


class TestCreate:
    """Test WorkerFactory.create() routing."""

    @pytest.fixture
    def factory(self) -> WorkerFactory:
        factory = WorkerFactory(
            setup_vllm_engine_fn=Mock(),
            setup_kv_event_publisher_fn=Mock(),
            register_vllm_model_fn=AsyncMock(),
        )
        factory._create_multimodal_encode_worker = AsyncMock()  # type: ignore[assignment]
        factory._create_multimodal_worker = AsyncMock()  # type: ignore[assignment]
        return factory

    @pytest.mark.asyncio
    async def test_routes_to_multimodal_encode(self, factory: WorkerFactory) -> None:
        config = _make_config(multimodal_encode_worker=True)
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event)

        factory._create_multimodal_encode_worker.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_routes_to_multimodal_worker(self, factory: WorkerFactory) -> None:
        config = _make_config(multimodal_worker=True)
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event)

        factory._create_multimodal_worker.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_routes_multimodal_decode_worker(
        self, factory: WorkerFactory
    ) -> None:
        config = _make_config(multimodal_decode_worker=True)
        shutdown_event = asyncio.Event()

        await factory.create(Mock(), config, shutdown_event)

        factory._create_multimodal_worker.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_passes_pre_created_engine(self, factory: WorkerFactory) -> None:
        config = _make_config(multimodal_worker=True)
        runtime = Mock()
        shutdown_event = asyncio.Event()
        pre_created_engine: EngineSetupResult = (
            Mock(),
            Mock(),
            Mock(),
            "/tmp/prometheus",
            Mock(),
        )

        await factory.create(
            runtime, config, shutdown_event, pre_created_engine=pre_created_engine
        )

        factory._create_multimodal_worker.assert_called_once_with(  # type: ignore[union-attr]
            runtime, config, shutdown_event, pre_created_engine=pre_created_engine
        )

    @pytest.mark.asyncio
    async def test_raises_when_no_multimodal_flag(self, factory: WorkerFactory) -> None:
        config = _make_config()
        with pytest.raises(ValueError, match="no multimodal worker type set"):
            await factory.create(Mock(), config, asyncio.Event())
