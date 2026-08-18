# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapter selection on the pooling-family workers.

The defect these cover: the handlers accepted ``--enable-lora`` but never put
``lora_request`` in ``encode_kwargs``, so a request naming an adapter was
pooled with the base weights and returned a plausible wrong answer with no
error. Both pooling roles are checked — ``ClassifyWorkerHandler`` extends
``EmbeddingWorkerHandler``, so a fix applied to only one leaves the sibling
silently broken.
"""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from vllm.lora.request import LoRARequest

import dynamo.vllm.handlers as base_handlers
import dynamo.vllm.pooling_handlers as pooling_mod
from dynamo.common.lora.manager import LoRAInfo

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


# --- both roles inherit the shared machinery ---------------------------------


def test_both_pooling_roles_are_lora_capable():
    from dynamo.vllm.lora_handler import LoRAHandlerMixin

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
    """This is the path that was silently wrong on main."""
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
