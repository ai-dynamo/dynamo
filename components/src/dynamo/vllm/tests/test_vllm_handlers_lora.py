# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseWorkerHandler's dynamic-LoRA lifecycle (handlers.py).

Covers the load/unload publish couplings and, in particular, the
loaded-but-unpublished reconcile: a LoRA whose engine add succeeded but whose
discovery publish failed (and whose engine-side rollback also failed) must be
re-published by a retried load instead of short-circuiting as "already
loaded" with no card — the state that previously left an adapter permanently
invisible to the frontend while every caller was told the load succeeded.

Mirrors the coverage test_vllm_lora.py provides for VllmLLMEngine.
Everything is mocked: no GPU, no real AsyncLLM, no real discovery.
"""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("vllm.lora.request")
pytest.importorskip("vllm.v1.engine.async_llm")

from dynamo.common.lora.manager import LoRAInfo  # noqa: E402
from dynamo.vllm import handlers as handlers_mod  # noqa: E402
from dynamo.vllm.constants import DisaggregationMode  # noqa: E402
from dynamo.vllm.handlers import BaseWorkerHandler  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _StubHandler(BaseWorkerHandler):
    """Concrete subclass so the ABC can be instantiated without __init__."""

    def generate(self, request, context):  # pragma: no cover - never called
        raise NotImplementedError


def _make_handler(endpoint=object()) -> BaseWorkerHandler:
    """Build a BaseWorkerHandler with only the LoRA-relevant state populated.

    Skips __init__ (it constructs a VllmEngineMonitor against a live runtime);
    sets exactly the attributes the LoRA lifecycle paths read.
    """
    handler = _StubHandler.__new__(_StubHandler)
    handler.loaded_loras = {}
    handler._published_loras = set()
    handler._lora_load_locks = {}
    handler._lora_load_locks_guard = threading.Lock()
    handler.engine_client = SimpleNamespace(
        add_lora=AsyncMock(),
        remove_lora=AsyncMock(),
    )
    handler.generate_endpoint = endpoint
    handler.model_max_len = 4096
    handler.config = SimpleNamespace(
        model="/models/base",
        dyn_tool_call_parser=None,
        dyn_reasoning_parser=None,
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        route_to_encoder=False,
        engine_args=SimpleNamespace(block_size=16, max_loras=4),
    )
    return handler


def _patch_discovery(monkeypatch, *, manager=None):
    """Patch discovery + LoRA-manager symbols imported into handlers.

    Returns the (register_model, unregister_model) AsyncMocks for assertions.
    """
    if manager is None:
        manager = SimpleNamespace(
            download_lora=AsyncMock(
                return_value={"status": "success", "local_path": "/cache/adapter"}
            )
        )
    register = AsyncMock()
    unregister = AsyncMock()
    monkeypatch.setattr(handlers_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(handlers_mod, "register_model", register)
    monkeypatch.setattr(handlers_mod, "unregister_model", unregister)
    monkeypatch.setattr(handlers_mod, "lora_name_to_id", lambda name: 123)
    monkeypatch.setattr(handlers_mod, "ModelRuntimeConfig", MagicMock())
    return register, unregister


async def _drain(agen):
    return [chunk async for chunk in agen]


LOAD_REQ = {"lora_name": "adapterA", "source": {"uri": "file:///x"}}


# --------------------------------------------------------------------------- #
# load_lora
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_load_lora_happy_path_publishes_and_tracks(monkeypatch):
    handler = _make_handler()
    register, _ = _patch_discovery(monkeypatch)

    results = await _drain(handler.load_lora(LOAD_REQ))

    assert results[-1]["status"] == "success"
    handler.engine_client.add_lora.assert_awaited_once()
    register.assert_awaited_once()
    assert register.await_args.kwargs["lora_name"] == "adapterA"
    # The engine already owns the base weights; the card needs only
    # config/tokenizer files, so the publish must never fetch weights.
    assert register.await_args.kwargs["ignore_weights"] is True
    assert handler.loaded_loras["adapterA"].id == 123
    assert "adapterA" in handler._published_loras


@pytest.mark.asyncio
async def test_load_lora_already_loaded_and_published_skips_publish(monkeypatch):
    handler = _make_handler()
    register, _ = _patch_discovery(monkeypatch)
    handler.loaded_loras["adapterA"] = LoRAInfo(id=123, path="/cache/adapter")
    handler._published_loras.add("adapterA")

    results = await _drain(handler.load_lora(LOAD_REQ))

    assert results[-1]["status"] == "success"
    assert "already loaded" in results[-1]["message"]
    register.assert_not_awaited()
    handler.engine_client.add_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_lora_reconciles_loaded_but_unpublished(monkeypatch):
    # A prior publish failed and its engine-side rollback also failed: the
    # adapter is loaded but has no discovery card. A retried load must
    # re-publish instead of short-circuiting as "already loaded".
    handler = _make_handler()
    register, _ = _patch_discovery(monkeypatch)
    handler.loaded_loras["adapterA"] = LoRAInfo(id=123, path="/cache/adapter")

    results = await _drain(handler.load_lora(LOAD_REQ))

    assert results[-1]["status"] == "success"
    register.assert_awaited_once()
    assert register.await_args.kwargs["lora_name"] == "adapterA"
    assert "adapterA" in handler._published_loras
    # No re-download / re-add churn for an adapter the engine already has.
    handler.engine_client.add_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_lora_reconcile_publish_failure_surfaces_error(monkeypatch):
    handler = _make_handler()
    register, _ = _patch_discovery(monkeypatch)
    register.side_effect = RuntimeError("discovery down")
    handler.loaded_loras["adapterA"] = LoRAInfo(id=123, path="/cache/adapter")

    results = await _drain(handler.load_lora(LOAD_REQ))

    assert results[-1]["status"] == "error"
    assert "publish failed" in results[-1]["message"]
    assert "adapterA" not in handler._published_loras
    # The adapter stays loaded so the next attempt reconciles again.
    assert "adapterA" in handler.loaded_loras


@pytest.mark.asyncio
async def test_load_lora_publish_failure_with_failed_rollback_stays_reconcilable(
    monkeypatch,
):
    # Publish fails AND the engine-side rollback fails: the adapter must stay
    # in loaded_loras and out of _published_loras — the exact state the
    # already-loaded reconcile heals on the next load attempt.
    handler = _make_handler()
    register, _ = _patch_discovery(monkeypatch)
    register.side_effect = RuntimeError("discovery down")
    handler.engine_client.remove_lora.side_effect = RuntimeError("engine stuck")

    results = await _drain(handler.load_lora(LOAD_REQ))

    assert results[-1]["status"] == "error"
    assert "adapterA" in handler.loaded_loras
    assert "adapterA" not in handler._published_loras


@pytest.mark.asyncio
async def test_load_lora_publish_failure_with_successful_rollback(monkeypatch):
    handler = _make_handler()
    register, _ = _patch_discovery(monkeypatch)
    register.side_effect = RuntimeError("discovery down")

    results = await _drain(handler.load_lora(LOAD_REQ))

    assert results[-1]["status"] == "error"
    handler.engine_client.remove_lora.assert_awaited_once()
    assert "adapterA" not in handler.loaded_loras
    assert "adapterA" not in handler._published_loras


# --------------------------------------------------------------------------- #
# unload_lora
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unload_lora_unregisters_before_engine_removal(monkeypatch):
    handler = _make_handler()
    _, unregister = _patch_discovery(monkeypatch)
    handler.loaded_loras["adapterA"] = LoRAInfo(id=123, path="/cache/adapter")
    handler._published_loras.add("adapterA")

    order: list[str] = []
    unregister.side_effect = lambda **kw: order.append("unregister")
    handler.engine_client.remove_lora = AsyncMock(
        side_effect=lambda *a: order.append("remove")
    )

    results = await _drain(handler.unload_lora({"lora_name": "adapterA"}))

    assert results[-1]["status"] == "success"
    assert order == ["unregister", "remove"]
    assert "adapterA" not in handler.loaded_loras
    assert "adapterA" not in handler._published_loras


@pytest.mark.asyncio
async def test_unload_lora_unregister_failure_leaves_state_intact(monkeypatch):
    # If the unregister fails nothing has mutated: the adapter is still
    # routable and still loaded. No engine-side re-add rollback is attempted
    # (re-adding after a partial remove can assert inside vLLM's fused-MoE
    # LoRA layers and take down the whole engine).
    handler = _make_handler()
    _, unregister = _patch_discovery(monkeypatch)
    unregister.side_effect = RuntimeError("discovery down")
    handler.loaded_loras["adapterA"] = LoRAInfo(id=123, path="/cache/adapter")
    handler._published_loras.add("adapterA")

    results = await _drain(handler.unload_lora({"lora_name": "adapterA"}))

    assert results[-1]["status"] == "error"
    handler.engine_client.remove_lora.assert_not_awaited()
    handler.engine_client.add_lora.assert_not_awaited()
    assert "adapterA" in handler.loaded_loras
    assert "adapterA" in handler._published_loras


@pytest.mark.asyncio
async def test_unload_lora_engine_removal_failure_keeps_tracking(monkeypatch):
    handler = _make_handler()
    _, unregister = _patch_discovery(monkeypatch)
    handler.loaded_loras["adapterA"] = LoRAInfo(id=123, path="/cache/adapter")
    handler._published_loras.add("adapterA")
    handler.engine_client.remove_lora.side_effect = RuntimeError("engine stuck")

    results = await _drain(handler.unload_lora({"lora_name": "adapterA"}))

    assert results[-1]["status"] == "error"
    unregister.assert_awaited_once()
    # Card gone, engine still holds the adapter: keep it in loaded_loras so a
    # retried unload skips the unregister and retries only the removal.
    assert "adapterA" in handler.loaded_loras
    assert "adapterA" not in handler._published_loras


@pytest.mark.asyncio
async def test_unload_lora_reconciles_stale_card(monkeypatch):
    # Not loaded but still published (a prior unload unregister failed after
    # the engine removal): retry the unregister so discovery converges.
    handler = _make_handler()
    _, unregister = _patch_discovery(monkeypatch)
    handler._published_loras.add("adapterA")

    results = await _drain(handler.unload_lora({"lora_name": "adapterA"}))

    assert results[-1]["status"] == "success"
    unregister.assert_awaited_once()
    assert "adapterA" not in handler._published_loras


@pytest.mark.asyncio
async def test_unload_lora_unknown_adapter_errors(monkeypatch):
    handler = _make_handler()
    _, unregister = _patch_discovery(monkeypatch)

    results = await _drain(handler.unload_lora({"lora_name": "nope"}))

    assert results[-1]["status"] == "error"
    assert "not found" in results[-1]["message"]
    unregister.assert_not_awaited()
