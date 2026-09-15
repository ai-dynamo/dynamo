# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapter selection on the pooling-family workers."""

import asyncio
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from vllm.lora.request import LoRARequest

import dynamo.vllm.handlers as base_handlers
import dynamo.vllm.pooling_handlers as pooling_mod
from dynamo.common.lora.manager import LoRAInfo
from dynamo.vllm.lora_handler import LoRAHandlerMixin
from dynamo.vllm.lora_state import LoRAState

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]

ADAPTER = "my-adapter"
BASE_MODEL = "test-model"


def _config() -> MagicMock:
    return MagicMock(
        served_model_name=BASE_MODEL,
        model=BASE_MODEL,
        served_model_aliases=(),
        engine_args=SimpleNamespace(enable_lora=True, max_loras=4, model=BASE_MODEL),
    )


def _context() -> MagicMock:
    context = MagicMock()
    context.id.return_value = "engine-request"
    context.async_killed_or_stopped.side_effect = (
        lambda: asyncio.get_running_loop().create_future()
    )
    return context


def _pooling_output(data: Any, prompt_token_ids: list[int]) -> MagicMock:
    output = MagicMock()
    output.outputs.data = torch.tensor(data)
    output.outputs.embedding = list(data)
    output.prompt_token_ids = prompt_token_ids
    return output


def _capture_encode(handler, output) -> list[dict[str, Any]]:
    """Record every encode_kwargs the handler hands to vLLM."""
    seen: list[dict[str, Any]] = []

    async def _encode(**kwargs):
        seen.append(kwargs)
        yield output

    handler.engine_client.encode = _encode
    return seen


def _enable_lora(handler) -> None:
    """Turn on adapter resolution. _lora_enabled is a cached_property that also
    requires a process-wide LoRA manager, which unit tests do not start."""
    handler.__dict__["_lora_enabled"] = True


def _install_adapter(handler, name: str = ADAPTER) -> None:
    """Register an adapter as already loaded, as load_lora would."""
    _enable_lora(handler)
    handler.loaded_loras = {name: LoRAInfo(id=7, path="/tmp/adapter")}


def _classify_handler() -> pooling_mod.ClassifyWorkerHandler:
    model_config = MagicMock()
    model_config.hf_config = SimpleNamespace(id2label={0: "a", 1: "b"})
    model_config.get_pooling_task.return_value = "classify"
    with patch.object(base_handlers, "VllmEngineMonitor"):
        handler = pooling_mod.ClassifyWorkerHandler(
            runtime=MagicMock(),
            engine=MagicMock(),
            config=_config(),
            model_config=model_config,
            shutdown_event=None,
        )
    from vllm.renderers import TokenizeParams

    handler.engine_client = MagicMock()
    handler.engine_client.abort = AsyncMock()
    handler.engine_client.get_supported_tasks = AsyncMock(return_value=("classify",))
    handler.engine_client.renderer.default_cmpl_tok_params = TokenizeParams(
        max_total_tokens=None
    )
    handler.engine_client.renderer.tokenizer = None
    handler.engine_client.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8)
    )
    return handler


def _embedding_handler() -> base_handlers.EmbeddingWorkerHandler:
    with patch.object(base_handlers, "VllmEngineMonitor"):
        handler = base_handlers.EmbeddingWorkerHandler(
            runtime=MagicMock(),
            engine=MagicMock(),
            config=_config(),
            shutdown_event=None,
        )
    handler.engine_client = MagicMock()
    handler.engine_client.abort = AsyncMock()
    return handler


async def _drain(agen) -> list[Any]:
    return [item async for item in agen]


async def _settle(turns: int = 50) -> None:
    """Let every runnable coroutine advance to its next blocking await."""
    for _ in range(turns):
        await asyncio.sleep(0)


# --- both roles inherit the shared machinery ---------------------------------


def test_both_pooling_roles_are_lora_capable():
    assert issubclass(base_handlers.EmbeddingWorkerHandler, LoRAHandlerMixin)
    assert issubclass(pooling_mod.ClassifyWorkerHandler, LoRAHandlerMixin)
    for name in ("load_lora", "unload_lora", "list_loras", "_resolve_lora_request"):
        assert hasattr(pooling_mod.ClassifyWorkerHandler, name), name


def test_adapter_cards_carry_the_pooling_model_types():
    """An adapter must register as the same model type as its base card, or
    the frontend routes adapter requests to the wrong endpoint."""
    # ModelType has no __eq__ (PyO3 identity comparison), so assert on the
    # capability bits the frontend actually routes on.
    embedding = base_handlers.EmbeddingWorkerHandler._lora_model_type(MagicMock())
    assert embedding.supports_embedding()

    classify = pooling_mod.ClassifyWorkerHandler._lora_model_type(MagicMock())
    assert classify.supports_classify() and classify.supports_pooling()


# --- classify / pooling ------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_forwards_lora_request_for_adapter_model():
    handler = _classify_handler()
    _install_adapter(handler)
    seen = _capture_encode(handler, _pooling_output([0.25, 0.75], [1, 2]))

    await _drain(handler.generate({"model": ADAPTER, "input": "hello"}, _context()))

    assert len(seen) == 1
    lora = seen[0].get("lora_request")
    assert isinstance(lora, LoRARequest), "adapter request pooled with base weights"
    assert lora.lora_name == ADAPTER


@pytest.mark.asyncio
async def test_classify_omits_lora_request_for_base_model():
    handler = _classify_handler()
    _install_adapter(handler)
    seen = _capture_encode(handler, _pooling_output([0.25, 0.75], [1, 2]))

    await _drain(handler.generate({"model": BASE_MODEL, "input": "hello"}, _context()))

    assert "lora_request" not in seen[0]


@pytest.mark.asyncio
async def test_classify_rejects_unknown_adapter_name():
    """An unloaded adapter name must fail rather than silently fall back."""
    handler = _classify_handler()
    _enable_lora(handler)
    _capture_encode(handler, _pooling_output([0.25, 0.75], [1, 2]))

    with pytest.raises(ValueError):
        await _drain(
            handler.generate({"model": "never-loaded", "input": "hello"}, _context())
        )


@pytest.mark.asyncio
async def test_classify_batch_resolves_adapter_once_for_every_prompt():
    """Every prompt in a batch names the same model, so all of them must
    carry the adapter — not just the first."""
    handler = _classify_handler()
    _install_adapter(handler)
    seen = _capture_encode(handler, _pooling_output([0.25, 0.75], [1, 2]))

    await _drain(
        handler.generate(
            {"model": ADAPTER, "input": ["one", "two", "three"]}, _context()
        )
    )

    assert len(seen) == 3
    assert all(k.get("lora_request") is not None for k in seen)
    assert {k["lora_request"].lora_name for k in seen} == {ADAPTER}


# --- embeddings --------------------------------------------------------------


@pytest.mark.asyncio
async def test_embeddings_forwards_lora_request_for_adapter_model():
    handler = _embedding_handler()
    _install_adapter(handler)
    seen = _capture_encode(handler, _pooling_output([0.1, 0.2, 0.3], [1, 2]))

    await _drain(handler.generate({"model": ADAPTER, "input": "hello"}, _context()))

    assert len(seen) == 1
    lora = seen[0].get("lora_request")
    assert isinstance(lora, LoRARequest), "adapter request pooled with base weights"
    assert lora.lora_name == ADAPTER


@pytest.mark.asyncio
async def test_embeddings_omits_lora_request_for_base_model():
    handler = _embedding_handler()
    _install_adapter(handler)
    seen = _capture_encode(handler, _pooling_output([0.1, 0.2, 0.3], [1, 2]))

    await _drain(handler.generate({"model": BASE_MODEL, "input": "hello"}, _context()))

    assert "lora_request" not in seen[0]


@pytest.mark.asyncio
async def test_embeddings_rejects_unknown_adapter_name():
    handler = _embedding_handler()
    _enable_lora(handler)
    _capture_encode(handler, _pooling_output([0.1, 0.2, 0.3], [1, 2]))

    with pytest.raises(ValueError):
        await _drain(
            handler.generate({"model": "never-loaded", "input": "hello"}, _context())
        )


# --- adapter lifecycle races against an in-flight batch ----------------------


@pytest.mark.asyncio
async def test_unload_waits_for_an_in_flight_pooling_batch():
    """A batch larger than max_num_seqs is admitted in waves, so it spans many
    event-loop turns. An unload landing between waves must wait: otherwise the
    later waves submit against an adapter vLLM has already removed."""
    handler = _classify_handler()
    handler.engine_client.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2)
    )
    _install_adapter(handler)
    handler._engine_loaded_loras.add(ADAPTER)
    handler.engine_client.remove_lora = AsyncMock()

    seen_ids: list[int | None] = []
    first_wave = asyncio.Event()
    finish_encodes = asyncio.Event()

    async def _encode(**kwargs):
        lora = kwargs.get("lora_request")
        seen_ids.append(lora.lora_int_id if lora is not None else None)
        first_wave.set()
        await finish_encodes.wait()
        yield _pooling_output([0.25, 0.75], [1, 2])

    handler.engine_client.encode = _encode

    prompts = [f"prompt-{i}" for i in range(6)]
    batch = asyncio.create_task(
        _drain(handler.generate({"model": ADAPTER, "input": prompts}, _context()))
    )
    await first_wave.wait()

    unload = asyncio.create_task(_drain(handler.unload_lora({"lora_name": ADAPTER})))
    await _settle()
    assert not unload.done(), "unload must drain the in-flight batch first"
    assert ADAPTER in handler.loaded_loras
    handler.engine_client.remove_lora.assert_not_awaited()

    finish_encodes.set()
    await batch
    await unload

    assert len(seen_ids) == len(prompts), "every prompt must reach the engine"
    assert set(seen_ids) == {7}, "the whole batch must use one adapter version"
    assert ADAPTER not in handler.loaded_loras


@pytest.mark.asyncio
async def test_hot_swap_waits_for_an_in_flight_pooling_batch():
    """An adapter id is a stable hash of its name, so a hot swap replaces the
    weights behind the same id. A batch admitted before the swap and still
    submitting after it would be answered from two different adapters under one
    id, with nothing in the response to show it. The swap must wait."""
    handler = _classify_handler()
    handler.engine_client.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2)
    )
    _install_adapter(handler)
    handler.generate_endpoint = MagicMock()
    handler._engine_loaded_loras.add(ADAPTER)
    handler.engine_client.add_lora = AsyncMock()
    handler.engine_client.remove_lora = AsyncMock()
    handler.engine_client.reset_prefix_cache = AsyncMock()
    handler._register_lora_discovery = AsyncMock()
    handler._resolve_lora_source_path = AsyncMock(return_value=(True, "/tmp/new"))

    encodes = 0
    first_wave = asyncio.Event()
    finish_encodes = asyncio.Event()

    async def _encode(**kwargs):
        nonlocal encodes
        encodes += 1
        first_wave.set()
        await finish_encodes.wait()
        yield _pooling_output([0.25, 0.75], [1, 2])

    handler.engine_client.encode = _encode

    prompts = [f"prompt-{i}" for i in range(6)]
    batch = asyncio.create_task(
        _drain(handler.generate({"model": ADAPTER, "input": prompts}, _context()))
    )
    await first_wave.wait()

    with patch.dict(os.environ, {"DYN_LORA_HOTSWAP_ENABLED": "true"}):
        swap = asyncio.create_task(
            _drain(
                handler.load_lora(
                    {"lora_name": ADAPTER, "source": {"uri": "s3://bucket/new"}}
                )
            )
        )
        await _settle()
        assert not swap.done(), "hot swap must drain the in-flight batch first"
        handler.engine_client.add_lora.assert_not_awaited()

        finish_encodes.set()
        await batch
        await swap

    assert encodes == len(prompts), "every prompt must reach the engine"
    handler.engine_client.add_lora.assert_awaited()


@pytest.mark.asyncio
async def test_batch_reservation_blocks_a_lifecycle_drain():
    """The drain primitive the lifecycle ops rely on."""
    state = LoRAState()
    state.reserve_batch(ADAPTER)

    drain = asyncio.create_task(state.wait_for_batch_drain(ADAPTER))
    await _settle()
    assert not drain.done()

    state.release_batch(ADAPTER)
    await drain


@pytest.mark.asyncio
async def test_failed_engine_removal_restores_the_discovery_card():
    """Unload withdraws the adapter from discovery before touching the engine.
    If the engine call fails the card has to come back: otherwise the adapter
    stays in loaded_loras with nothing routing to it, and the idempotent branch
    of load_lora reports success without ever republishing it."""
    handler = _classify_handler()
    _install_adapter(handler)
    handler.generate_endpoint = MagicMock()
    handler._engine_loaded_loras.add(ADAPTER)
    handler.engine_client.remove_lora = AsyncMock(
        side_effect=RuntimeError("engine unreachable")
    )
    handler._unregister_lora_discovery = AsyncMock()
    handler._register_lora_discovery = AsyncMock()

    results = await _drain(handler.unload_lora({"lora_name": ADAPTER}))

    handler._unregister_lora_discovery.assert_awaited_once()
    handler._register_lora_discovery.assert_awaited_once_with(ADAPTER, 7)
    assert results[-1]["status"] == "error"
    assert ADAPTER in handler.loaded_loras, "tracking must survive a failed unload"
