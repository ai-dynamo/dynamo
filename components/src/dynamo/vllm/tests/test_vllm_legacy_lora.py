# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy worker-factory LoRA lifecycle tests.

The unified vLLM engine has separate lifecycle coverage. These tests protect
the still-supported ``BaseWorkerHandler`` path used by release images.
"""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.common.lora.manager import LoRAInfo  # noqa: E402
from dynamo.llm import ModelType, WorkerType  # noqa: E402
from dynamo.vllm import handlers as handlers_mod  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_prefill_handler():
    handler = handlers_mod.PrefillWorkerHandler.__new__(
        handlers_mod.PrefillWorkerHandler
    )
    handler.config = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.PREFILL,
        route_to_encoder=False,
        model="/models/base",
        dyn_tool_call_parser=None,
        dyn_reasoning_parser=None,
        engine_args=SimpleNamespace(block_size=16, max_loras=4),
    )
    handler.engine_client = SimpleNamespace(
        add_lora=AsyncMock(),
        remove_lora=AsyncMock(),
        reset_prefix_cache=AsyncMock(),
    )
    handler.generate_endpoint = object()
    handler.model_max_len = 8192
    handler.loaded_loras = {}
    handler._lora_load_locks = {}
    handler._lora_load_locks_guard = threading.Lock()
    return handler


@pytest.mark.asyncio
async def test_prefill_load_records_and_publishes_without_eager_engine_add(
    monkeypatch,
):
    handler = _make_prefill_handler()
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/adapter"}
        )
    )
    register = AsyncMock()
    monkeypatch.delenv("DYN_LORA_HOTSWAP_ENABLED", raising=False)
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(handlers_mod, "lora_name_to_id", lambda _name: 123)
    monkeypatch.setattr(handlers_mod, "register_model", register)

    results = [
        result
        async for result in handler.load_lora(
            {"lora_name": "adapterA", "source": {"uri": "file:///adapter"}}
        )
    ]

    assert results[-1]["status"] == "success"
    handler.engine_client.add_lora.assert_not_awaited()
    assert handler.loaded_loras["adapterA"] == LoRAInfo(id=123, path="/cache/adapter")
    register.assert_awaited_once()
    kwargs = register.await_args.kwargs
    assert str(kwargs["model_type"]) == str(ModelType.Prefill)
    assert kwargs["worker_type"] == WorkerType.Prefill
    assert kwargs["needs"] == [[WorkerType.Decode]]


@pytest.mark.asyncio
async def test_decode_load_still_eagerly_adds_to_engine(monkeypatch):
    handler = _make_prefill_handler()
    handler.config.disaggregation_mode = DisaggregationMode.DECODE
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/adapter"}
        )
    )
    monkeypatch.delenv("DYN_LORA_HOTSWAP_ENABLED", raising=False)
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(handlers_mod, "lora_name_to_id", lambda _name: 123)
    monkeypatch.setattr(handlers_mod, "register_model", AsyncMock())

    results = [
        result
        async for result in handler.load_lora(
            {"lora_name": "adapterA", "source": {"uri": "file:///adapter"}}
        )
    ]

    assert results[-1]["status"] == "success"
    handler.engine_client.add_lora.assert_awaited_once()


@pytest.mark.asyncio
async def test_prefill_publish_failure_rolls_back_metadata_only(monkeypatch):
    handler = _make_prefill_handler()
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/adapter"}
        )
    )
    register = AsyncMock(side_effect=RuntimeError("discovery is down"))
    monkeypatch.delenv("DYN_LORA_HOTSWAP_ENABLED", raising=False)
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(handlers_mod, "lora_name_to_id", lambda _name: 123)
    monkeypatch.setattr(handlers_mod, "register_model", register)

    results = [
        result
        async for result in handler.load_lora(
            {"lora_name": "adapterA", "source": {"uri": "file:///adapter"}}
        )
    ]

    assert results[-1]["status"] == "error"
    assert "adapterA" not in handler.loaded_loras
    handler.engine_client.add_lora.assert_not_awaited()
    handler.engine_client.remove_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_unload_unregisters_before_engine_removal(monkeypatch):
    handler = _make_prefill_handler()
    handler.loaded_loras = {"adapterA": LoRAInfo(id=123, path="/cache/adapter")}
    order: list[str] = []
    unregister = AsyncMock(side_effect=lambda **_kwargs: order.append("unregister"))
    handler.engine_client.remove_lora.side_effect = lambda _id: order.append("remove")
    monkeypatch.setattr(handlers_mod, "unregister_model", unregister)

    results = [
        result async for result in handler.unload_lora({"lora_name": "adapterA"})
    ]

    assert results[-1]["status"] == "success"
    assert order == ["unregister", "remove"]
    assert "adapterA" not in handler.loaded_loras


@pytest.mark.asyncio
async def test_legacy_unload_unregister_failure_preserves_engine_state(monkeypatch):
    handler = _make_prefill_handler()
    original = LoRAInfo(id=123, path="/cache/adapter")
    handler.loaded_loras = {"adapterA": original}
    monkeypatch.setattr(
        handlers_mod,
        "unregister_model",
        AsyncMock(side_effect=RuntimeError("discovery is down")),
    )

    results = [
        result async for result in handler.unload_lora({"lora_name": "adapterA"})
    ]

    assert results[-1]["status"] == "error"
    handler.engine_client.remove_lora.assert_not_awaited()
    assert handler.loaded_loras["adapterA"] == original
