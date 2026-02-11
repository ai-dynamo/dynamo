# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for worker_factory.py"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm.worker_factory import EngineSetupResult, WorkerFactory

# TODO(DIS-1468): turn on embedding cache support in vLLM agg node when v0.17 is released.
# from vllm.config import ECTransferConfig


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


class TestEcTransferConfig:
    """Test that ec_transfer_config is set correctly for embedding cache."""

    @pytest.mark.asyncio
    async def test_ec_transfer_config_set_when_cache_enabled(self) -> None:
        """When multimodal_embedding_cache_capacity_gb > 0 and no encode worker,
        _create_multimodal_worker should set ec_transfer_config on engine_args
        BEFORE calling setup_vllm_engine so that vLLM sees the config."""
        config = _make_config(
            multimodal_worker=True,
            multimodal_embedding_cache_capacity_gb=4.0,
            route_to_encoder=False,
        )
        config.namespace = "dynamo"
        config.component = "backend"
        config.endpoint = "generate"
        config.engine_args = Mock()
        config.engine_args.ec_transfer_config = None

        # setup_vllm_engine captures the ec_transfer_config at the moment it's called
        # to verify it was set BEFORE engine creation
        captured_ec_config = {}

        def fake_setup_vllm_engine(cfg):
            captured_ec_config["value"] = cfg.engine_args.ec_transfer_config
            return (Mock(), Mock(), Mock(), None, Mock())

        factory = WorkerFactory(
            setup_vllm_engine_fn=fake_setup_vllm_engine,
            setup_kv_event_publisher_fn=Mock(return_value=None),
            register_vllm_model_fn=AsyncMock(),
        )

        # Call _create_multimodal_worker directly — it will call setup_vllm_engine
        # which captures the ec_transfer_config, then fail later on endpoint setup
        with pytest.raises(Exception):
            await factory._create_multimodal_worker(Mock(), config, asyncio.Event())

        ec_cfg = captured_ec_config.get("value")
        assert ec_cfg is None

        # TODO(DIS-1468): turn embedding cache support in vLLM agg node when v0.17 is
        # released.
        # assert isinstance(
        #     ec_cfg, ECTransferConfig
        # ), f"setup_vllm_engine should see ECTransferConfig, got {ec_cfg!r}. "
        # assert ec_cfg.ec_role == "ec_both"
        # assert ec_cfg.ec_connector == "DynamoMultimodalEmbeddingCacheConnector"
        # assert ec_cfg.ec_connector_module_path == (
        #     "dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector"
        # )
        # assert (
        #     ec_cfg.ec_connector_extra_config["multimodal_embedding_cache_capacity_gb"]
        #     == 4.0
        # )
