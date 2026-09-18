# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM worker-factory LoRA lifecycle tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.common.lora.manager import LoRAInfo  # noqa: E402
from dynamo.vllm import handlers as handlers_mod  # noqa: E402
from dynamo.vllm.cache_info import DYNAMO_KV_EVENT_BLOCK_SIZE_KEY  # noqa: E402
from dynamo.vllm.lora_state import RuntimeLoRAInfo  # noqa: E402

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
        add_lora=AsyncMock(return_value=True),
        remove_lora=AsyncMock(return_value=True),
        reset_prefix_cache=AsyncMock(return_value=True),
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

    handler._lora_capacity = None
    handler._lora_capacity_guard = asyncio.Lock()
    handler.engine_args = handler.config.engine_args
    handler.dp_range = (0, 1)
    handler._served_model_name = "llama2-7b"
    handler._served_model_aliases = ("llama2-7b-alias",)
    handler._lora_state = LoRAState()
    handler._engine_loaded_loras = set()
    return handler


@pytest.mark.asyncio
async def test_decode_load_rolls_back_engine_when_add_raises_after_applying(
    monkeypatch,
):
    handler = _make_prefill_handler()
    handler.config.disaggregation_mode = DisaggregationMode.DECODE
    engine_ids = set()

    async def add_lora(request):
        engine_ids.add(request.lora_int_id)
        raise RuntimeError("engine response lost")

    async def remove_lora(lora_id):
        engine_ids.discard(lora_id)
        return True

    handler.engine_client.add_lora.side_effect = add_lora
    handler.engine_client.remove_lora.side_effect = remove_lora
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

    assert results[-1]["status"] == "error"
    assert not engine_ids
    assert "adapterA" not in handler._lora_state.loaded_loras
    assert "adapterA" not in handler._engine_loaded_loras
    handler.engine_client.remove_lora.assert_awaited_once_with(123)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation_before_block", [False, True])
async def test_decode_load_cancellation_finishes_discovery_registration(
    monkeypatch, mutation_before_block
):
    handler = _make_prefill_handler()
    handler.config.disaggregation_mode = DisaggregationMode.DECODE
    published = set()
    registration_started = asyncio.Event()
    finish_registration = asyncio.Event()

    async def register(lora_name, _lora_id):
        if mutation_before_block:
            published.add(lora_name)
        registration_started.set()
        await finish_registration.wait()
        if not mutation_before_block:
            published.add(lora_name)

    handler._register_lora_discovery = register
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/adapter"}
        )
    )
    monkeypatch.delenv("DYN_LORA_HOTSWAP_ENABLED", raising=False)
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(handlers_mod, "lora_name_to_id", lambda _name: 123)

    async def load():
        return [
            result
            async for result in handler.load_lora(
                {"lora_name": "adapterA", "source": {"uri": "file:///adapter"}}
            )
        ]

    task = asyncio.create_task(load())
    await registration_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish_registration.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert published == {"adapterA"}
    assert handler._lora_state.loaded_loras["adapterA"] == LoRAInfo(
        id=123, path="/cache/adapter"
    )
    assert "adapterA" in handler._engine_loaded_loras


@pytest.mark.asyncio
async def test_decode_add_cancellation_finishes_engine_cleanup(monkeypatch):
    handler = _make_prefill_handler()
    handler.config.disaggregation_mode = DisaggregationMode.DECODE
    add_started = asyncio.Event()
    finish_add = asyncio.Event()
    engine_ids = set()

    async def add_lora(request):
        add_started.set()
        await finish_add.wait()
        engine_ids.add(request.lora_int_id)
        return True

    async def remove_lora(lora_id):
        engine_ids.discard(lora_id)
        return True

    handler.engine_client.add_lora.side_effect = add_lora
    handler.engine_client.remove_lora.side_effect = remove_lora
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/adapter"}
        )
    )
    monkeypatch.delenv("DYN_LORA_HOTSWAP_ENABLED", raising=False)
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(handlers_mod, "lora_name_to_id", lambda _name: 123)
    monkeypatch.setattr(handlers_mod, "register_model", AsyncMock())

    async def load():
        return [
            result
            async for result in handler.load_lora(
                {"lora_name": "adapterA", "source": {"uri": "file:///adapter"}}
            )
        ]

    task = asyncio.create_task(load())
    await add_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish_add.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not engine_ids
    assert "adapterA" not in handler._lora_state.loaded_loras
    assert "adapterA" not in handler._engine_loaded_loras
    handler.engine_client.remove_lora.assert_awaited_once_with(123)


@pytest.mark.asyncio
async def test_decode_registration_failure_compensates_engine_and_discovery(
    monkeypatch,
):
    handler = _make_prefill_handler()
    handler.config.disaggregation_mode = DisaggregationMode.DECODE
    published = set()

    async def register(lora_name, _lora_id):
        published.add(lora_name)
        raise RuntimeError("registration response lost")

    async def unregister(lora_name):
        published.discard(lora_name)

    handler._register_lora_discovery = register
    handler._unregister_lora_discovery = unregister
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/adapter"}
        )
    )
    monkeypatch.delenv("DYN_LORA_HOTSWAP_ENABLED", raising=False)
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(handlers_mod, "lora_name_to_id", lambda _name: 123)

    results = [
        result
        async for result in handler.load_lora(
            {"lora_name": "adapterA", "source": {"uri": "file:///adapter"}}
        )
    ]

    assert results[-1]["status"] == "error"
    assert not published
    assert "adapterA" not in handler._lora_state.loaded_loras
    assert "adapterA" not in handler._engine_loaded_loras
    handler.engine_client.remove_lora.assert_awaited_once_with(123)


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_mutates_before_error", [False, True])
async def test_admin_hot_swap_remove_error_restores_old_adapter(
    monkeypatch, remove_mutates_before_error
):
    handler = _make_prefill_handler()
    old_info = LoRAInfo(id=123, path="/cache/old")
    handler._lora_state.loaded_loras["adapterA"] = old_info
    handler._engine_loaded_loras.add("adapterA")
    engine_paths = {123: old_info.path}

    async def remove_lora(lora_id):
        if remove_mutates_before_error:
            engine_paths.pop(lora_id, None)
        raise RuntimeError("remove response lost")

    async def add_lora(request):
        engine_paths[request.lora_int_id] = request.lora_path
        return True

    handler.engine_client.remove_lora.side_effect = remove_lora
    handler.engine_client.add_lora.side_effect = add_lora
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/new"}
        )
    )
    monkeypatch.setenv("DYN_LORA_HOTSWAP_ENABLED", "true")
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)

    results = [
        result
        async for result in handler.load_lora(
            {"lora_name": "adapterA", "source": {"uri": "file:///new"}}
        )
    ]

    assert results[-1]["status"] == "error"
    assert handler._lora_state.loaded_loras["adapterA"] == old_info
    assert handler._engine_loaded_loras == {"adapterA"}
    assert engine_paths == {123: old_info.path}


@pytest.mark.asyncio
async def test_admin_load_counts_quarantined_engine_id_against_capacity(monkeypatch):
    handler = _make_prefill_handler()
    handler._lora_capacity = 1
    handler._lora_state.uncertain_engine_lora_ids.add(123)
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/adapter"}
        )
    )
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)

    results = [
        result
        async for result in handler.load_lora(
            {"lora_name": "adapterA", "source": {"uri": "file:///adapter"}}
        )
    ]

    assert results[-1]["status"] == "error"
    assert "capacity exceeded" in results[-1]["message"]
    manager.download_lora.assert_not_awaited()
    handler.engine_client.add_lora.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("unregister_mutates_before_error", [False, True])
async def test_admin_load_retries_indeterminate_discovery_registration(
    monkeypatch, unregister_mutates_before_error
):
    handler = _make_prefill_handler()
    handler.config.disaggregation_mode = DisaggregationMode.DECODE
    published = set()
    register_count = 0

    async def register(lora_name, _lora_id):
        nonlocal register_count
        register_count += 1
        published.add(lora_name)
        if register_count == 1:
            raise RuntimeError("register response lost")

    async def unregister(lora_name):
        if unregister_mutates_before_error:
            published.discard(lora_name)
        raise RuntimeError("unregister response lost")

    handler._register_lora_discovery = register
    handler._unregister_lora_discovery = unregister
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/adapter"}
        )
    )
    monkeypatch.delenv("DYN_LORA_HOTSWAP_ENABLED", raising=False)
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(handlers_mod, "lora_name_to_id", lambda _name: 123)
    request = {"lora_name": "adapterA", "source": {"uri": "file:///adapter"}}

    first = [result async for result in handler.load_lora(request)]

    assert first[-1]["status"] == "error"
    assert "adapterA" in handler._lora_state.discovery_uncertain_loras
    assert "adapterA" in handler._lora_state.loaded_loras

    second = [result async for result in handler.load_lora(request)]

    assert second[-1]["status"] == "success"
    assert published == {"adapterA"}
    assert "adapterA" not in handler._lora_state.discovery_uncertain_loras
    handler.engine_client.add_lora.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("register_mutates_before_error", [False, True])
async def test_admin_unload_retries_indeterminate_discovery_restoration(
    register_mutates_before_error,
):
    handler = _make_prefill_handler()
    original = LoRAInfo(id=123, path="/cache/adapter")
    handler._lora_state.loaded_loras["adapterA"] = original
    handler._engine_loaded_loras.add("adapterA")
    published = {"adapterA"}
    unregister_count = 0

    async def unregister(lora_name):
        nonlocal unregister_count
        unregister_count += 1
        published.discard(lora_name)
        if unregister_count == 1:
            raise RuntimeError("unregister response lost")

    async def register(lora_name, _lora_id):
        if register_mutates_before_error:
            published.add(lora_name)
        raise RuntimeError("register response lost")

    handler._unregister_lora_discovery = unregister
    handler._register_lora_discovery = register

    first = [result async for result in handler.unload_lora({"lora_name": "adapterA"})]

    assert first[-1]["status"] == "error"
    assert "adapterA" in handler._lora_state.discovery_uncertain_loras
    assert handler._lora_state.loaded_loras["adapterA"] == original

    second = [result async for result in handler.unload_lora({"lora_name": "adapterA"})]

    assert second[-1]["status"] == "success"
    assert not published
    assert "adapterA" not in handler._lora_state.discovery_uncertain_loras
    assert "adapterA" not in handler._lora_state.loaded_loras


@pytest.mark.asyncio
async def test_runtime_lease_cleanup_error_preserves_cancellation(caplog):
    handler = _make_prefill_handler()
    handler._lora_state.loaded_loras["adapterA"] = LoRAInfo(
        id=123, path="/cache/adapter"
    )
    lora_request = handler._resolve_lora_request("adapterA")
    generation_started = asyncio.Event()
    release_started = asyncio.Event()
    finish_release = asyncio.Event()

    async def activate(_request_id, request):
        return request

    async def release(_adapter_key):
        release_started.set()
        await finish_release.wait()
        raise RuntimeError("lease release failed")

    handler._runtime_lora_coordinator = SimpleNamespace(
        activate_pending_admission=activate,
        release_active_admission=release,
    )

    async def generate(_request):
        generation_started.set()
        await asyncio.Event().wait()
        yield SimpleNamespace()

    admission = handler._generate_with_lora_admission_lock(
        lora_request,
        generate,
        request_id="request-1",
        runtime_lora=True,
    )
    task = asyncio.create_task(anext(admission))
    await generation_started.wait()
    task.cancel()
    await release_started.wait()
    task.cancel()
    finish_release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert "Failed to release active LoRA lease during cancellation" in caplog.text


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


@pytest.mark.asyncio
async def test_admin_load_rejects_runtime_owned_adapter(monkeypatch):
    handler = _make_prefill_handler()
    adapter_key = "dyn-lora-runtime"
    runtime_info = RuntimeLoRAInfo(
        adapter_key=adapter_key,
        full_identity_digest=bytes(32),
        base_model_name="base",
        source_revision="revision",
        id=123,
        path="/cache/runtime",
    )
    handler._lora_state.runtime_loras[adapter_key] = runtime_info
    handler._lora_state.loaded_loras[adapter_key] = LoRAInfo(
        id=123, path="/cache/runtime"
    )
    manager = SimpleNamespace(download_lora=AsyncMock())
    monkeypatch.setenv("DYN_LORA_HOTSWAP_ENABLED", "true")
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)

    results = [
        result
        async for result in handler.load_lora(
            {"lora_name": adapter_key, "source": {"uri": "file:///replacement"}}
        )
    ]

    assert results[-1]["status"] == "error"
    assert "cannot be replaced" in results[-1]["message"]
    manager.download_lora.assert_not_awaited()
    handler.engine_client.remove_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_load_skips_runtime_reserved_integer_id(monkeypatch):
    handler = _make_prefill_handler()
    handler._lora_state.runtime_reserved_ids["dyn-lora-runtime"] = 123
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/admin"}
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
    assert results[-1]["lora_id"] == 124
    assert handler._lora_state.loaded_loras["adapterA"].id == 124
    assert not handler._lora_state.admin_reserved_ids


@pytest.mark.asyncio
async def test_admin_initial_load_rejects_false_engine_result(monkeypatch):
    handler = _make_prefill_handler()
    handler.config.disaggregation_mode = DisaggregationMode.DECODE
    handler.engine_client.add_lora.return_value = False
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/adapter"}
        )
    )
    monkeypatch.delenv("DYN_LORA_HOTSWAP_ENABLED", raising=False)
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(handlers_mod, "lora_name_to_id", lambda _name: 123)

    results = [
        result
        async for result in handler.load_lora(
            {"lora_name": "adapterA", "source": {"uri": "file:///adapter"}}
        )
    ]

    assert results[-1]["status"] == "error"
    assert "vLLM rejected" in results[-1]["message"]
    assert "adapterA" not in handler._lora_state.loaded_loras
    assert "adapterA" not in handler._engine_loaded_loras


@pytest.mark.asyncio
async def test_admin_hot_swap_false_add_restores_old_adapter(monkeypatch):
    handler = _make_prefill_handler()
    original = LoRAInfo(id=123, path="/cache/old")
    handler._lora_state.loaded_loras["adapterA"] = original
    handler._engine_loaded_loras.add("adapterA")
    handler.engine_client.add_lora.side_effect = [False, True]
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/new"}
        )
    )
    monkeypatch.setenv("DYN_LORA_HOTSWAP_ENABLED", "true")
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)

    results = [
        result
        async for result in handler.load_lora(
            {"lora_name": "adapterA", "source": {"uri": "file:///new"}}
        )
    ]

    assert results[-1]["status"] == "error"
    assert handler._lora_state.loaded_loras["adapterA"] == original
    assert "adapterA" in handler._engine_loaded_loras
    assert handler.engine_client.add_lora.await_count == 2
    handler.engine_client.reset_prefix_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_hot_swap_false_restore_drops_tracking(monkeypatch):
    handler = _make_prefill_handler()
    handler._lora_state.loaded_loras["adapterA"] = LoRAInfo(id=123, path="/cache/old")
    handler._engine_loaded_loras.add("adapterA")
    handler.engine_client.add_lora.side_effect = [False, False]
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/new"}
        )
    )
    monkeypatch.setenv("DYN_LORA_HOTSWAP_ENABLED", "true")
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)

    results = [
        result
        async for result in handler.load_lora(
            {"lora_name": "adapterA", "source": {"uri": "file:///new"}}
        )
    ]

    assert results[-1]["status"] == "error"
    assert "adapterA" not in handler._lora_state.loaded_loras
    assert "adapterA" not in handler._engine_loaded_loras
    assert handler.engine_client.add_lora.await_count == 2


@pytest.mark.asyncio
async def test_admin_hot_swap_false_prefix_reset_restores_old_adapter(monkeypatch):
    handler = _make_prefill_handler()
    original = LoRAInfo(id=123, path="/cache/old")
    handler._lora_state.loaded_loras["adapterA"] = original
    handler._engine_loaded_loras.add("adapterA")
    handler.engine_client.reset_prefix_cache.return_value = False
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/new"}
        )
    )
    monkeypatch.setenv("DYN_LORA_HOTSWAP_ENABLED", "true")
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)

    results = [
        result
        async for result in handler.load_lora(
            {"lora_name": "adapterA", "source": {"uri": "file:///new"}}
        )
    ]

    assert results[-1]["status"] == "error"
    assert "prefix cache reset failed" in results[-1]["message"]
    assert handler._lora_state.loaded_loras["adapterA"] == original
    assert "adapterA" in handler._engine_loaded_loras
    assert handler.engine_client.remove_lora.await_count == 2
    assert handler.engine_client.add_lora.await_count == 2


@pytest.mark.asyncio
async def test_admin_hot_swap_cancellation_during_remove_restores_old_adapter(
    monkeypatch,
):
    handler = _make_prefill_handler()
    original = LoRAInfo(id=123, path="/cache/old")
    handler._lora_state.loaded_loras["adapterA"] = original
    handler._engine_loaded_loras.add("adapterA")
    remove_started = asyncio.Event()
    finish_remove = asyncio.Event()

    async def remove_lora(_lora_id):
        remove_started.set()
        await finish_remove.wait()
        return True

    handler.engine_client.remove_lora.side_effect = remove_lora
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/new"}
        )
    )
    monkeypatch.setenv("DYN_LORA_HOTSWAP_ENABLED", "true")
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)

    async def load():
        return [
            result
            async for result in handler.load_lora(
                {"lora_name": "adapterA", "source": {"uri": "file:///new"}}
            )
        ]

    task = asyncio.create_task(load())
    await remove_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish_remove.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert handler._lora_state.loaded_loras["adapterA"] == original
    assert "adapterA" in handler._engine_loaded_loras
    handler.engine_client.add_lora.assert_awaited_once()


@pytest.mark.asyncio
async def test_admin_hot_swap_cancellation_during_add_restores_old_adapter(
    monkeypatch,
):
    handler = _make_prefill_handler()
    original = LoRAInfo(id=123, path="/cache/old")
    handler._lora_state.loaded_loras["adapterA"] = original
    handler._engine_loaded_loras.add("adapterA")
    add_started = asyncio.Event()
    finish_add = asyncio.Event()
    add_count = 0

    async def add_lora(_request):
        nonlocal add_count
        add_count += 1
        if add_count == 1:
            add_started.set()
            await finish_add.wait()
        return True

    handler.engine_client.add_lora.side_effect = add_lora
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/new"}
        )
    )
    monkeypatch.setenv("DYN_LORA_HOTSWAP_ENABLED", "true")
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)

    async def load():
        return [
            result
            async for result in handler.load_lora(
                {"lora_name": "adapterA", "source": {"uri": "file:///new"}}
            )
        ]

    task = asyncio.create_task(load())
    await add_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish_add.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert handler._lora_state.loaded_loras["adapterA"] == original
    assert "adapterA" in handler._engine_loaded_loras
    assert handler.engine_client.remove_lora.await_count == 2
    assert handler.engine_client.add_lora.await_count == 2


@pytest.mark.asyncio
async def test_admin_hot_swap_cancellation_during_restore_finishes_rollback(
    monkeypatch,
):
    handler = _make_prefill_handler()
    original = LoRAInfo(id=123, path="/cache/old")
    handler._lora_state.loaded_loras["adapterA"] = original
    handler._engine_loaded_loras.add("adapterA")
    restore_started = asyncio.Event()
    finish_restore = asyncio.Event()
    add_count = 0

    async def add_lora(_request):
        nonlocal add_count
        add_count += 1
        if add_count == 1:
            return False
        restore_started.set()
        await finish_restore.wait()
        return True

    handler.engine_client.add_lora.side_effect = add_lora
    manager = SimpleNamespace(
        download_lora=AsyncMock(
            return_value={"status": "success", "local_path": "/cache/new"}
        )
    )
    monkeypatch.setenv("DYN_LORA_HOTSWAP_ENABLED", "true")
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)

    async def load():
        return [
            result
            async for result in handler.load_lora(
                {"lora_name": "adapterA", "source": {"uri": "file:///new"}}
            )
        ]

    task = asyncio.create_task(load())
    await restore_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish_restore.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert handler._lora_state.loaded_loras["adapterA"] == original
    assert "adapterA" in handler._engine_loaded_loras
    assert handler.engine_client.add_lora.await_count == 2


@pytest.mark.asyncio
async def test_admin_unload_cancellation_finishes_discovery_and_engine_cleanup():
    handler = _make_prefill_handler()
    handler._lora_state.loaded_loras["adapterA"] = LoRAInfo(
        id=123, path="/cache/adapter"
    )
    handler._engine_loaded_loras.add("adapterA")
    unregister_started = asyncio.Event()
    finish_unregister = asyncio.Event()
    unregister_finished = False

    async def unregister(_lora_name):
        nonlocal unregister_finished
        unregister_started.set()
        await finish_unregister.wait()
        unregister_finished = True

    handler._unregister_lora_discovery = unregister

    async def unload():
        return [
            result async for result in handler.unload_lora({"lora_name": "adapterA"})
        ]

    task = asyncio.create_task(unload())
    await unregister_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish_unregister.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert unregister_finished
    handler.engine_client.remove_lora.assert_awaited_once_with(123)
    assert "adapterA" not in handler._lora_state.loaded_loras
    assert "adapterA" not in handler._engine_loaded_loras


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation_before_block", [False, True])
@pytest.mark.parametrize("cancel_requested", [False, True])
async def test_admin_unload_discovery_error_restores_publication(
    mutation_before_block, cancel_requested
):
    handler = _make_prefill_handler()
    handler._lora_state.loaded_loras["adapterA"] = LoRAInfo(
        id=123, path="/cache/adapter"
    )
    handler._engine_loaded_loras.add("adapterA")
    published = {"adapterA"}
    unregister_started = asyncio.Event()
    finish_unregister = asyncio.Event()

    async def unregister(lora_name):
        if mutation_before_block:
            published.discard(lora_name)
        unregister_started.set()
        await finish_unregister.wait()
        if not mutation_before_block:
            published.discard(lora_name)
        raise RuntimeError("unregister response lost")

    async def register(lora_name, _lora_id):
        published.add(lora_name)

    handler._unregister_lora_discovery = unregister
    handler._register_lora_discovery = register

    async def unload():
        return [
            result async for result in handler.unload_lora({"lora_name": "adapterA"})
        ]

    task = asyncio.create_task(unload())
    await unregister_started.wait()
    if cancel_requested:
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finish_unregister.set()

    if cancel_requested:
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        results = await task
        assert results[-1]["status"] == "error"

    assert published == {"adapterA"}
    assert handler._lora_state.loaded_loras["adapterA"] == LoRAInfo(
        id=123, path="/cache/adapter"
    )
    assert "adapterA" in handler._engine_loaded_loras
    handler.engine_client.remove_lora.assert_not_awaited()
