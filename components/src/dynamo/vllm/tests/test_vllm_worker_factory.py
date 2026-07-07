# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for worker_factory.py"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.worker_factory import EngineSetupResult, WorkerFactory

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


@pytest.mark.asyncio
class TestEmbeddingRegistrationContract:
    """The embedding worker's ModelInput mirrors the chat/completion split:

    - `--use-vllm-tokenizer` (True) registers `ModelInput.Text`, so the Rust
      frontend forwards the raw OpenAI `/v1/embeddings` request and vLLM
      tokenizes internally (the historical behavior).
    - The default (False) registers `ModelInput.Tokens`, routing through the
      Rust `OpenAIPreprocessor` so tokenization happens off the engine
      process.

    Either way the surface is `ModelType.Embedding` with
    `WorkerType.Aggregated` and no peer dependencies.
    """

    @pytest.mark.parametrize(
        "use_vllm_tokenizer,expected_input",
        [(True, ModelInput.Text), (False, ModelInput.Tokens)],
    )
    async def test_embedding_registers_input_from_tokenizer_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
        use_vllm_tokenizer: bool,
        expected_input: ModelInput,
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

        # Avoid the real handler (spawns VllmEngineMonitor background tasks)
        # and StatLoggerFactory wiring; neither is under test here.
        monkeypatch.setattr(
            "dynamo.vllm.worker_factory.EmbeddingWorkerHandler",
            Mock(return_value=Mock()),
        )
        monkeypatch.setattr(
            "dynamo.vllm.worker_factory.StatLoggerFactory",
            Mock(return_value=Mock()),
        )

        config = _make_config(
            disaggregation_mode=DisaggregationMode.AGGREGATED,
            embedding_worker=True,
            use_vllm_tokenizer=use_vllm_tokenizer,
            namespace="dyn",
            component="embed",
            endpoint="generate",
            served_model_name="m",
            model="m",
        )

        # generate_endpoint.serve_endpoint is awaited concurrently with the
        # register call via asyncio.gather, so it must be awaitable; the
        # register failure is what stops the flow.
        generate_endpoint = Mock()
        generate_endpoint.connection_id = Mock(return_value="cid")
        generate_endpoint.serve_endpoint = AsyncMock()
        runtime = Mock()
        runtime.endpoint.return_value = generate_endpoint

        with pytest.raises(RuntimeError, match="stop-after-register"):
            await factory._create_embedding_worker(
                runtime,
                config,
                asyncio.Event(),
                [],
            )

        assert captured["model_input"] == expected_input
        assert captured["model_type"] == ModelType.Embedding
        assert captured["worker_type"] == WorkerType.Aggregated
        assert captured["needs"] == []
