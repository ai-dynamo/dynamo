# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM worker-factory LoRA lifecycle tests."""

import asyncio
import gc
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.constants import (  # noqa: E402
    KV_HINT_TRANSFER_CAPABILITY_KEY,
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    DisaggregationMode,
)
from dynamo.common.lora.manager import LoRAInfo  # noqa: E402
from dynamo.llm import ModelType, WorkerType  # noqa: E402
from dynamo.vllm import handlers as handlers_mod  # noqa: E402
from dynamo.vllm.cache_info import DYNAMO_KV_EVENT_BLOCK_SIZE_KEY  # noqa: E402

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
        engine_args=SimpleNamespace(block_size=16, max_loras=4, model="/models/base"),
        use_kv_events=True,
    )
    handler.engine_client = SimpleNamespace(
        add_lora=AsyncMock(),
        remove_lora=AsyncMock(),
        reset_prefix_cache=AsyncMock(),
        # LoRA MDC registration reads the engine-actual main-attention block
        # size from here (hybrid-attention models inflate it past the CLI's
        # engine_args.block_size=16 above).
        vllm_config=SimpleNamespace(
            additional_config={DYNAMO_KV_EVENT_BLOCK_SIZE_KEY: 1056},
            cache_config=SimpleNamespace(block_size=16),
        ),
    )
    handler.generate_endpoint = object()
    handler.model_max_len = 8192
    # Initialize LoRA state
    from dynamo.vllm.lora_state import LoRAState

    handler.engine_args = handler.config.engine_args
    handler.dp_range = (0, 1)
    handler._served_model_name = "llama2-7b"
    handler._served_model_aliases = ("llama2-7b-alias",)
    handler._lora_state = LoRAState()
    handler._engine_loaded_loras = set()
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
    assert handler._lora_state.loaded_loras["adapterA"] == LoRAInfo(
        id=123, path="/cache/adapter"
    )
    register.assert_awaited_once()
    kwargs = register.await_args.kwargs
    assert str(kwargs["model_type"]) == str(ModelType.Prefill)
    assert kwargs["worker_type"] == WorkerType.Prefill
    assert kwargs["needs"] == [[WorkerType.Decode]]
    runtime_config = kwargs["runtime_config"]
    assert runtime_config.context_length == 8192
    assert json.loads(runtime_config.runtime_data["token_budget"]) == {
        "combined_limit": 8192,
        "reject_prompt_overflow": True,
        "reject_total_overflow": True,
    }
    assert runtime_config.kv_event_publishing_enabled is True
    # The adapter card must carry the engine-actual main-attention block size,
    # not engine_args.block_size (16) — see #11866.
    assert kwargs["kv_cache_block_size"] == 1056


@pytest.mark.asyncio
async def test_prefill_lora_registration_preserves_worker_dp_range(monkeypatch):
    # LoRA registration builds a fresh ModelRuntimeConfig for the adapter card.
    # This checks that a parent worker owning global DP ranks 4 and 5 publishes
    # the same range on the LoRA card, with router-hint endpoints keyed as
    # {"4": "tcp://...:23280", "5": "tcp://...:23281"}.
    handler = _make_prefill_handler()
    handler.dp_range = (4, 2)
    handler.config.engine_args.kv_transfer_config = SimpleNamespace(
        kv_connector_extra_config={
            "secondary_tiers": [
                {
                    "type": "custom",
                    "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                    "control_advertise_host": "worker-a",
                    "control_port": "23280",
                }
            ]
        }
    )
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
    runtime_config = register.await_args.kwargs["runtime_config"]
    assert runtime_config.data_parallel_start_rank == 4
    assert runtime_config.data_parallel_size == 2
    assert json.loads(
        runtime_config.runtime_data[
            KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY
        ]
    ) == {"4": "tcp://worker-a:23280", "5": "tcp://worker-a:23281"}


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
    assert "adapterA" not in handler._lora_state.loaded_loras
    handler.engine_client.add_lora.assert_not_awaited()
    handler.engine_client.remove_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_unload_unregisters_before_engine_removal(monkeypatch):
    handler = _make_prefill_handler()
    handler._lora_state.loaded_loras = {
        "adapterA": LoRAInfo(id=123, path="/cache/adapter")
    }
    handler._engine_loaded_loras = {"adapterA"}
    order: list[str] = []
    unregister = AsyncMock(side_effect=lambda **_kwargs: order.append("unregister"))
    handler.engine_client.remove_lora.side_effect = lambda _id: order.append("remove")
    monkeypatch.setattr(handlers_mod, "unregister_model", unregister)

    results = [
        result async for result in handler.unload_lora({"lora_name": "adapterA"})
    ]

    assert results[-1]["status"] == "success"
    assert order == ["unregister", "remove"]
    assert "adapterA" not in handler._lora_state.loaded_loras


@pytest.mark.asyncio
async def test_legacy_prefill_unload_skips_engine_removal_for_metadata_only_adapter(
    monkeypatch,
):
    handler = _make_prefill_handler()
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/adapter"}
        )
    )
    unregister = AsyncMock()
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(handlers_mod, "lora_name_to_id", lambda _name: 123)
    monkeypatch.setattr(handlers_mod, "register_model", AsyncMock())
    monkeypatch.setattr(handlers_mod, "unregister_model", unregister)

    load_results = [
        result
        async for result in handler.load_lora(
            {"lora_name": "adapterA", "source": {"uri": "file:///adapter"}}
        )
    ]
    unload_results = [
        result async for result in handler.unload_lora({"lora_name": "adapterA"})
    ]

    assert load_results[-1]["status"] == "success"
    assert unload_results[-1]["status"] == "success"
    unregister.assert_awaited_once()
    handler.engine_client.add_lora.assert_not_awaited()
    handler.engine_client.remove_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_prefill_unload_removes_request_activated_adapter(monkeypatch):
    handler = _make_prefill_handler()
    handler._lora_state.loaded_loras = {
        "adapterA": LoRAInfo(id=123, path="/cache/adapter")
    }
    handler._track_lora_request_activation(handler._resolve_lora_request("adapterA"))
    monkeypatch.setattr(handlers_mod, "unregister_model", AsyncMock())

    results = [
        result async for result in handler.unload_lora({"lora_name": "adapterA"})
    ]

    assert results[-1]["status"] == "success"
    handler.engine_client.remove_lora.assert_awaited_once_with(123)


@pytest.mark.asyncio
async def test_legacy_prefill_request_admission_serializes_with_unload(monkeypatch):
    handler = _make_prefill_handler()
    handler._lora_state.loaded_loras = {
        "adapterA": LoRAInfo(id=123, path="/cache/adapter")
    }
    monkeypatch.setattr(handlers_mod, "unregister_model", AsyncMock())

    admission_started = asyncio.Event()
    allow_admission = asyncio.Event()
    admitted = asyncio.Event()

    async def _blocked_generate(_lora_request):
        admission_started.set()
        await allow_admission.wait()
        admitted.set()
        yield SimpleNamespace()

    async def _remove_lora(lora_id):
        assert lora_id == 123
        assert admitted.is_set()

    handler.engine_client.remove_lora = AsyncMock(side_effect=_remove_lora)
    admission = handler._generate_with_lora_admission_lock(
        handler._resolve_lora_request("adapterA"),
        _blocked_generate,
    )
    admission_task = asyncio.create_task(anext(admission))
    await admission_started.wait()

    async def _unload():
        return [
            result async for result in handler.unload_lora({"lora_name": "adapterA"})
        ]

    unload_task = asyncio.create_task(_unload())
    await asyncio.sleep(0)
    assert not unload_task.done()
    assert "adapterA" in handler._lora_state.loaded_loras

    allow_admission.set()
    await admission_task
    results = await unload_task
    await admission.aclose()

    assert results[-1]["status"] == "success"
    handler.engine_client.remove_lora.assert_awaited_once_with(123)


@pytest.mark.asyncio
async def test_legacy_prefill_request_rejects_adapter_unloaded_before_admission(
    monkeypatch, caplog
):
    handler = _make_prefill_handler()
    handler._lora_state.loaded_loras = {
        "adapterA": LoRAInfo(id=123, path="/cache/adapter")
    }
    monkeypatch.setattr(handlers_mod, "unregister_model", AsyncMock())
    stale_request = handler._resolve_lora_request("adapterA")

    results = [
        result async for result in handler.unload_lora({"lora_name": "adapterA"})
    ]

    assert results[-1]["status"] == "success"

    async def _must_not_generate(_lora_request):
        raise AssertionError("stale request must not reach vLLM")
        yield  # pragma: no cover - marks this as an async generator

    with pytest.raises(ValueError, match="unknown model or LoRA adapter"):
        await anext(
            handler._generate_with_lora_admission_lock(
                stale_request,
                _must_not_generate,
            )
        )

    assert "adapterA was unloaded before vLLM admission" in caplog.text


def test_resolve_lora_request_treats_served_alias_as_base_model_when_enabled():
    handler = _make_prefill_handler()
    handler.config.engine_args.enable_lora = True

    assert handler._resolve_lora_request("llama2-7b-alias") is None


def test_lora_lock_table_does_not_retain_transient_adapter_names():
    handler = _make_prefill_handler()

    # Create many distinct lock names without holding on to lock references.
    for idx in range(200):
        handler._get_lora_lock(f"transient-adapter-{idx}")

    # Weak lock entries should be reclaimable once references drop.
    gc.collect()
    assert len(handler._lora_state.lora_load_locks) == 0


@pytest.mark.asyncio
async def test_load_lora_cancellation_releases_capacity_placeholder(monkeypatch):
    handler = _make_prefill_handler()
    handler._lora_capacity = 1
    handler._lora_capacity_guard = asyncio.Lock()

    gate = asyncio.Event()
    download_started = asyncio.Event()

    async def _blocked_download(_uri):
        download_started.set()
        await gate.wait()
        return {"status": "success", "local_path": "/cache/adapter"}

    manager = SimpleNamespace(download_lora=_blocked_download)
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(handlers_mod, "unregister_model", AsyncMock())

    async def _run_load():
        return [
            result
            async for result in handler.load_lora(
                {"lora_name": "adapterA", "source": {"uri": "file:///adapter"}}
            )
        ]

    task = asyncio.create_task(_run_load())
    await download_started.wait()

    # Placeholder reservation is inserted while download is in flight.
    assert handler._lora_state.loaded_loras["adapterA"].id == -1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Cancellation must not leave ghost placeholder capacity entries.
    assert "adapterA" not in handler._lora_state.loaded_loras


@pytest.mark.asyncio
async def test_legacy_prefill_unload_treats_missing_request_adapter_as_idempotent(
    monkeypatch,
):
    handler = _make_prefill_handler()
    handler._lora_state.loaded_loras = {
        "adapterA": LoRAInfo(id=123, path="/cache/adapter")
    }
    handler._engine_loaded_loras = {"adapterA"}
    handler.engine_client.remove_lora.side_effect = RuntimeError("adapter not found")
    monkeypatch.setattr(handlers_mod, "unregister_model", AsyncMock())

    results = [
        result async for result in handler.unload_lora({"lora_name": "adapterA"})
    ]

    assert results[-1]["status"] == "success"
    assert "adapterA" not in handler._lora_state.loaded_loras


@pytest.mark.asyncio
async def test_legacy_unload_unregister_failure_preserves_engine_state(monkeypatch):
    handler = _make_prefill_handler()
    original = LoRAInfo(id=123, path="/cache/adapter")
    handler._lora_state.loaded_loras = {"adapterA": original}
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
    assert handler._lora_state.loaded_loras["adapterA"] == original
