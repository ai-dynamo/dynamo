# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for unified-vLLM dynamic-LoRA support.

Covers engine-control gating, per-request adapter resolution, the
load/unload/list lifecycle (including discovery publish/unpublish and the
add_lora<->register_model rollback couplings), and the on_endpoint_ready
handoff. Everything is mocked: no GPU, no real AsyncLLM, no real discovery.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("vllm.lora.request")
pytest.importorskip("vllm.usage.usage_lib")
pytest.importorskip("vllm.v1.engine.async_llm")

from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.common.lora.manager import LoRAInfo  # noqa: E402
from dynamo.llm import ModelType, WorkerType  # noqa: E402
from dynamo.vllm import llm_engine as llm_engine_mod  # noqa: E402
from dynamo.vllm.llm_engine import VllmLLMEngine  # noqa: E402
from dynamo.vllm.multimodal_utils.request_processor import (  # noqa: E402
    VllmMultimodalRequestProcessor,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_lora_engine(enable_lora: bool = True, endpoint=None) -> VllmLLMEngine:
    """Build a VllmLLMEngine with only the LoRA-relevant state populated.

    Calls the real ``__init__`` (side-effect-free: only attribute assignment
    and lock creation; AsyncLLM is built later in ``start()``), so new engine
    attributes get their real defaults automatically and this helper does not
    silently drift from the constructor. Only the fields ``__init__`` leaves
    None/unset that the LoRA paths read are overridden below.
    """
    engine = VllmLLMEngine(
        engine_args=SimpleNamespace(
            enable_lora=enable_lora,
            model="/models/base",
            block_size=16,
            max_loras=4,
        ),
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        served_model_name="base-model",
        component="test",
    )
    engine.engine_client = SimpleNamespace(
        add_lora=AsyncMock(),
        remove_lora=AsyncMock(),
    )
    engine._kv_event_block_size = 16
    engine._endpoint = endpoint
    return engine


def _patch_discovery(monkeypatch, *, manager=None, name_to_id=None):
    """Patch the discovery + LoRA-manager symbols imported into llm_engine.

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
    monkeypatch.setattr(llm_engine_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(llm_engine_mod, "register_model", register)
    monkeypatch.setattr(llm_engine_mod, "unregister_model", unregister)
    monkeypatch.setattr(
        llm_engine_mod, "lora_name_to_id", name_to_id or (lambda name: 123)
    )
    monkeypatch.setattr(llm_engine_mod, "ModelRuntimeConfig", MagicMock())
    return register, unregister


# --------------------------------------------------------------------------- #
# Engine-update gating
#
# LoRA lifecycle ops ride the engine-*update* surface (supported_updates /
# engine_update), not engine controls, so they don't inflate the control set.
# --------------------------------------------------------------------------- #


def test_lora_updates_not_advertised_without_manager(monkeypatch):
    monkeypatch.setattr(llm_engine_mod, "get_lora_manager", lambda: None)
    engine = _make_lora_engine(enable_lora=True)

    updates = engine.supported_updates()

    assert "load_lora" not in updates
    assert "unload_lora" not in updates
    assert "list_loras" not in updates
    # LoRA must never leak back into the control surface.
    assert {"load_lora", "unload_lora", "list_loras"}.isdisjoint(
        engine.supported_controls()
    )


def test_lora_updates_not_advertised_without_enable_lora(monkeypatch):
    monkeypatch.setattr(llm_engine_mod, "get_lora_manager", lambda: MagicMock())
    engine = _make_lora_engine(enable_lora=False)

    updates = engine.supported_updates()

    assert {"load_lora", "unload_lora", "list_loras"}.isdisjoint(updates)


@pytest.mark.asyncio
async def test_lora_updates_advertised_and_dispatchable(monkeypatch):
    monkeypatch.setattr(llm_engine_mod, "get_lora_manager", lambda: MagicMock())
    engine = _make_lora_engine(enable_lora=True)

    updates = engine.supported_updates()
    assert {"load_lora", "unload_lora", "list_loras"} <= updates
    # LoRA must not appear among controls.
    assert {"load_lora", "unload_lora", "list_loras"}.isdisjoint(
        engine.supported_controls()
    )

    # The dispatcher routes the update to the real method.
    result = await engine.engine_update("list_loras", {})
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_disabled_lora_update_is_rejected_by_dispatcher(monkeypatch):
    monkeypatch.setattr(llm_engine_mod, "get_lora_manager", lambda: None)
    engine = _make_lora_engine(enable_lora=True)

    result = await engine.engine_update("load_lora", {"lora_name": "x"})

    assert result["status"] == "error"
    assert "unsupported engine update" in result["message"]
    engine.engine_client.add_lora.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Per-request adapter resolution
# --------------------------------------------------------------------------- #


def test_resolve_lora_request_for_loaded_adapter():
    engine = _make_lora_engine()
    engine.loaded_loras = {"adapterA": LoRAInfo(id=7, path="/path/a")}

    lora_request = engine._resolve_lora_request("adapterA")

    assert lora_request is not None
    assert lora_request.lora_name == "adapterA"
    assert lora_request.lora_int_id == 7
    assert lora_request.lora_path == "/path/a"


def test_resolve_lora_request_for_base_is_none(monkeypatch):
    monkeypatch.setattr(llm_engine_mod, "get_lora_manager", lambda: MagicMock())
    engine = _make_lora_engine()
    engine.loaded_loras = {"adapterA": LoRAInfo(id=7, path="/path/a")}

    # Base-model name (served name and the engine_args.model path) and the
    # absent-model case both resolve to the base model (None), even with LoRA
    # enabled.
    assert engine._resolve_lora_request("base-model") is None
    assert engine._resolve_lora_request("/models/base") is None
    assert engine._resolve_lora_request(None) is None


def test_resolve_lora_request_unknown_adapter_raises_when_lora_enabled(monkeypatch):
    monkeypatch.setattr(llm_engine_mod, "get_lora_manager", lambda: MagicMock())
    engine = _make_lora_engine(enable_lora=True)
    engine.loaded_loras = {"adapterA": LoRAInfo(id=7, path="/path/a")}

    # An unknown or just-unloaded adapter name must fail loudly rather than be
    # silently served by the base model.
    with pytest.raises(ValueError, match="unknown model or LoRA adapter"):
        engine._resolve_lora_request("ghost-adapter")


def test_resolve_lora_request_unknown_adapter_is_none_when_lora_disabled(monkeypatch):
    monkeypatch.setattr(llm_engine_mod, "get_lora_manager", lambda: None)
    engine = _make_lora_engine(enable_lora=False)

    # With LoRA disabled there are no adapters, so a non-base name is left to
    # the engine instead of being rejected here.
    assert engine._resolve_lora_request("ghost-adapter") is None


@pytest.mark.asyncio
async def test_generate_passes_resolved_lora_request(monkeypatch):
    engine = _make_lora_engine()
    engine.loaded_loras = {"adapterA": LoRAInfo(id=9, path="/p/a")}
    engine._default_sampling_params = SimpleNamespace()
    engine._model_max_len = None
    engine._dp_range = None
    engine._multimodal_request_processor = VllmMultimodalRequestProcessor(
        model=engine.engine_args.model,
        enable_multimodal=False,
    )

    captured: dict = {}

    async def _empty_gen():
        return
        yield  # pragma: no cover - marks this as an async generator

    def _fake_generate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _empty_gen()

    engine.engine_client.generate = _fake_generate
    monkeypatch.setattr(
        llm_engine_mod,
        "build_sampling_params",
        lambda *a, **k: SimpleNamespace(extra_args=None, max_tokens=10, min_tokens=0),
    )
    monkeypatch.setattr(
        llm_engine_mod.telemetry, "engine_trace_kwargs", lambda context: {}
    )

    context = SimpleNamespace(id=lambda: "req-1")

    # Adapter request -> resolved LoRARequest.
    _ = [
        c
        async for c in engine.generate(
            {"token_ids": [1, 2], "model": "adapterA"}, context
        )
    ]
    assert captured["kwargs"]["lora_request"] is not None
    assert captured["kwargs"]["lora_request"].lora_name == "adapterA"

    # Base-model request -> None.
    _ = [
        c
        async for c in engine.generate({"token_ids": [1, 2], "model": "base"}, context)
    ]
    assert captured["kwargs"]["lora_request"] is None


# --------------------------------------------------------------------------- #
# load_lora
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_load_lora_happy_path(monkeypatch):
    engine = _make_lora_engine(endpoint=object())
    register, _ = _patch_discovery(monkeypatch)

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    assert result["status"] == "success"
    assert result["lora_id"] == 123
    engine.engine_client.add_lora.assert_awaited_once()
    register.assert_awaited_once()
    assert register.await_args.kwargs["lora_name"] == "adapterA"
    assert engine.loaded_loras["adapterA"].id == 123


@pytest.mark.asyncio
async def test_prefill_load_lora_defers_engine_activation(monkeypatch):
    """Prefill records and advertises the adapter without eagerly loading it.

    The first inference supplies the cached path through ``LoRARequest`` and
    lets vLLM activate the adapter on demand. Decode and aggregated workers
    continue to preload so they are ready to generate immediately.
    """
    engine = _make_lora_engine(endpoint=object())
    engine.disaggregation_mode = DisaggregationMode.PREFILL
    register, _ = _patch_discovery(monkeypatch)

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    assert result["status"] == "success"
    engine.engine_client.add_lora.assert_not_awaited()
    assert engine.loaded_loras["adapterA"] == LoRAInfo(id=123, path="/cache/adapter")
    register.assert_awaited_once()
    assert register.await_args.kwargs["worker_type"] == WorkerType.Prefill
    assert register.await_args.kwargs["needs"] == [[WorkerType.Decode]]
    assert register.await_args.kwargs["max_gpu_lora_count"] == 4


@pytest.mark.asyncio
async def test_prefill_publish_failure_rolls_back_metadata_only(monkeypatch):
    engine = _make_lora_engine(endpoint=object())
    engine.disaggregation_mode = DisaggregationMode.PREFILL
    register, _ = _patch_discovery(monkeypatch)
    register.side_effect = RuntimeError("discovery is down")

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    assert result["status"] == "error"
    assert "adapterA" not in engine.loaded_loras
    engine.engine_client.add_lora.assert_not_awaited()
    engine.engine_client.remove_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_prefill_inference_uses_lifecycle_registered_lora(monkeypatch):
    engine = _make_lora_engine(endpoint=object())
    engine.disaggregation_mode = DisaggregationMode.PREFILL
    engine._default_sampling_params = SimpleNamespace()
    engine._model_max_len = None
    engine._dp_range = None
    engine._multimodal_request_processor = VllmMultimodalRequestProcessor(
        model=engine.engine_args.model,
        enable_multimodal=False,
    )
    _patch_discovery(monkeypatch)

    load_result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )
    assert load_result["status"] == "success"
    engine.engine_client.add_lora.assert_not_awaited()

    captured: dict = {}

    async def _empty_gen():
        return
        yield  # pragma: no cover

    def _fake_generate(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _empty_gen()

    engine.engine_client.generate = _fake_generate
    monkeypatch.setattr(
        llm_engine_mod,
        "build_sampling_params",
        lambda *a, **k: SimpleNamespace(extra_args=None, max_tokens=10, min_tokens=0),
    )
    monkeypatch.setattr(
        llm_engine_mod.telemetry, "engine_trace_kwargs", lambda context: {}
    )

    _ = [
        chunk
        async for chunk in engine.generate(
            {"token_ids": [1, 2], "model": "adapterA"},
            SimpleNamespace(id=lambda: "req-1"),
        )
    ]

    lora_request = captured["kwargs"]["lora_request"]
    assert lora_request.lora_name == "adapterA"
    assert lora_request.lora_path == "/cache/adapter"


@pytest.mark.asyncio
@pytest.mark.parametrize("collision", ["base-model", "/models/base"])
async def test_load_lora_rejects_base_model_name_collision(monkeypatch, collision):
    # An adapter named after the base model would shadow its frontend model key
    # and route plain base-model requests through the adapter. Reject it before
    # touching the download/engine/discovery path.
    engine = _make_lora_engine(endpoint=object())
    manager = SimpleNamespace(download_lora=AsyncMock())
    register, _ = _patch_discovery(monkeypatch, manager=manager)

    result = await engine.load_lora(
        {"lora_name": collision, "source": {"uri": "file:///x"}}
    )

    assert result["status"] == "error"
    assert "collides" in result["message"]
    manager.download_lora.assert_not_awaited()
    engine.engine_client.add_lora.assert_not_awaited()
    register.assert_not_awaited()
    assert collision not in engine.loaded_loras


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, expected_model_type, expected_worker_type, expected_needs",
    [
        (
            DisaggregationMode.AGGREGATED,
            ModelType.Chat | ModelType.Completions,
            WorkerType.Aggregated,
            [],
        ),
        (
            DisaggregationMode.DECODE,
            ModelType.Chat | ModelType.Completions,
            WorkerType.Decode,
            [[WorkerType.Prefill]],
        ),
        (
            DisaggregationMode.PREFILL,
            ModelType.Prefill,
            WorkerType.Prefill,
            [[WorkerType.Decode]],
        ),
    ],
)
async def test_load_lora_publishes_disagg_topology(
    monkeypatch, mode, expected_model_type, expected_worker_type, expected_needs
):
    # The LoRA MDC must match the base-model registration topology so the
    # frontend builds the pipeline against the right component. A prefill worker
    # publishing the adapter as decode-capable chat/completions would make the
    # frontend route chat traffic straight to prefill.
    engine = _make_lora_engine(endpoint=object())
    engine.disaggregation_mode = mode
    engine._dyn_tool_call_parser = "hermes"
    engine._dyn_reasoning_parser = "deepseek_r1"
    register, _ = _patch_discovery(monkeypatch)

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    assert result["status"] == "success"
    kwargs = register.await_args.kwargs
    # ModelType is a Rust bitflags pyclass without Python __eq__, so combined
    # flags (Chat | Completions) compare by identity and never match a freshly
    # built copy. Compare the deterministic str() form instead.
    assert str(kwargs["model_type"]) == str(expected_model_type)
    assert kwargs["worker_type"] == expected_worker_type
    assert kwargs["needs"] == expected_needs


@pytest.mark.asyncio
async def test_load_lora_publishes_dp_and_capacity_metadata(monkeypatch):
    # The LoRA MDC must carry the worker's real DP-rank range and capacity (the
    # same effective vLLM metadata the base-model card publishes), so multi-DP
    # LoRA requests are routed/attributed per rank instead of as if every worker
    # only served rank 0.
    engine = _make_lora_engine(endpoint=object())
    engine._vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=256,
            max_num_batched_tokens=8192,
        )
    )
    engine._dp_range = (2, 4)
    engine._total_kv_blocks = 1024
    register, _ = _patch_discovery(monkeypatch)

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    assert result["status"] == "success"
    runtime_config = register.await_args.kwargs["runtime_config"]
    assert runtime_config.total_kv_blocks == 1024
    assert runtime_config.max_num_seqs == 256
    assert runtime_config.max_num_batched_tokens == 8192
    assert runtime_config.data_parallel_start_rank == 2
    assert runtime_config.data_parallel_size == 4


@pytest.mark.asyncio
async def test_load_lora_idempotent_reload(monkeypatch):
    engine = _make_lora_engine(endpoint=object())
    manager = SimpleNamespace(download_lora=AsyncMock())
    register, _ = _patch_discovery(monkeypatch, manager=manager)
    # A healthy adapter is both loaded into the engine AND published to
    # discovery; re-loading it must short-circuit without re-publishing.
    engine.loaded_loras = {"adapterA": LoRAInfo(id=123, path="/cache/adapter")}
    engine._published_loras = {"adapterA"}

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    assert result["status"] == "success"
    assert "already loaded" in result["message"]
    manager.download_lora.assert_not_awaited()
    engine.engine_client.add_lora.assert_not_awaited()
    register.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_lora_reconciles_loaded_but_unpublished(monkeypatch):
    # Simulate a sticky partial-failure state: the adapter is loaded into the
    # engine but its discovery card was never published (a prior publish + its
    # rollback both failed). A retried load must re-publish rather than report
    # early success.
    engine = _make_lora_engine(endpoint=object())
    manager = SimpleNamespace(download_lora=AsyncMock())
    register, _ = _patch_discovery(monkeypatch, manager=manager)
    engine.loaded_loras = {"adapterA": LoRAInfo(id=123, path="/cache/adapter")}
    # _published_loras intentionally left empty -> loaded-but-unpublished.

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    assert result["status"] == "success"
    # Engine load is not repeated, but discovery publication is reconciled.
    manager.download_lora.assert_not_awaited()
    engine.engine_client.add_lora.assert_not_awaited()
    register.assert_awaited_once()
    assert "adapterA" in engine._published_loras


@pytest.mark.asyncio
async def test_load_lora_reconcile_publish_failure_surfaces_error(monkeypatch):
    engine = _make_lora_engine(endpoint=object())
    register, _ = _patch_discovery(monkeypatch)
    register.side_effect = RuntimeError("discovery is down")
    engine.loaded_loras = {"adapterA": LoRAInfo(id=123, path="/cache/adapter")}

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    # Reconciliation tried and failed: report error (not a false "already
    # loaded" success) and leave the adapter unpublished so a later retry can
    # reconcile again.
    assert result["status"] == "error"
    assert "discovery publish failed" in result["message"]
    assert "adapterA" not in engine._published_loras


@pytest.mark.asyncio
async def test_load_lora_marks_adapter_published(monkeypatch):
    engine = _make_lora_engine(endpoint=object())
    _patch_discovery(monkeypatch)

    await engine.load_lora({"lora_name": "adapterA", "source": {"uri": "file:///x"}})

    assert "adapterA" in engine._published_loras


@pytest.mark.asyncio
async def test_load_lora_rollback_failure_leaves_adapter_unpublished(monkeypatch):
    # register fails AND the engine-side remove_lora rollback also fails: the
    # adapter stays in loaded_loras but must NOT be marked published, so a
    # subsequent load reconciles instead of short-circuiting.
    engine = _make_lora_engine(endpoint=object())
    register, _ = _patch_discovery(monkeypatch)
    register.side_effect = RuntimeError("discovery is down")
    engine.engine_client.remove_lora.side_effect = RuntimeError("engine wedged")

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    assert result["status"] == "error"
    assert "adapterA" not in engine._published_loras


@pytest.mark.asyncio
async def test_load_lora_errors_when_manager_missing(monkeypatch):
    engine = _make_lora_engine(endpoint=object())
    monkeypatch.setattr(llm_engine_mod, "get_lora_manager", lambda: None)

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    assert result["status"] == "error"
    assert "LoRAManager not initialized" in result["message"]
    engine.engine_client.add_lora.assert_not_awaited()
    assert "adapterA" not in engine.loaded_loras


@pytest.mark.asyncio
async def test_load_lora_rolls_back_when_register_fails(monkeypatch):
    engine = _make_lora_engine(endpoint=object())
    register, _ = _patch_discovery(monkeypatch)
    register.side_effect = RuntimeError("discovery is down")

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    assert result["status"] == "error"
    assert "Failed to register" in result["message"]
    # Rollback removes the adapter from the engine and tracking.
    engine.engine_client.remove_lora.assert_awaited_once_with(123)
    assert "adapterA" not in engine.loaded_loras


# --------------------------------------------------------------------------- #
# unload_lora
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unload_lora_happy_path(monkeypatch):
    engine = _make_lora_engine(endpoint=object())
    _, unregister = _patch_discovery(monkeypatch)
    # Healthy adapter: loaded into the engine AND published to discovery.
    engine.loaded_loras = {"adapterA": LoRAInfo(id=123, path="/cache/adapter")}
    engine._published_loras = {"adapterA"}

    result = await engine.unload_lora({"lora_name": "adapterA"})

    assert result["status"] == "success"
    engine.engine_client.remove_lora.assert_awaited_once_with(123)
    unregister.assert_awaited_once()
    assert "adapterA" not in engine.loaded_loras
    assert "adapterA" not in engine._published_loras


@pytest.mark.asyncio
async def test_unload_lora_unregisters_before_engine_removal(monkeypatch):
    # Discovery must be unpublished before the engine drops the adapter, so the
    # frontend stops routing LoRA traffic before the adapter disappears.
    engine = _make_lora_engine(endpoint=object())
    _, unregister = _patch_discovery(monkeypatch)
    engine.loaded_loras = {"adapterA": LoRAInfo(id=123, path="/cache/adapter")}
    engine._published_loras = {"adapterA"}

    order: list[str] = []
    unregister.side_effect = lambda **_: order.append("unregister")
    engine.engine_client.remove_lora.side_effect = lambda *_: order.append("remove")

    result = await engine.unload_lora({"lora_name": "adapterA"})

    assert result["status"] == "success"
    assert order == ["unregister", "remove"]


@pytest.mark.asyncio
async def test_unload_lora_loaded_but_unpublished_skips_unregister(monkeypatch):
    # Loaded-but-unpublished adapter (a prior load's publish failed): unload
    # should not attempt an unregister, just drop it from the engine.
    engine = _make_lora_engine(endpoint=object())
    _, unregister = _patch_discovery(monkeypatch)
    engine.loaded_loras = {"adapterA": LoRAInfo(id=123, path="/cache/adapter")}
    # _published_loras intentionally empty.

    result = await engine.unload_lora({"lora_name": "adapterA"})

    assert result["status"] == "success"
    unregister.assert_not_awaited()
    engine.engine_client.remove_lora.assert_awaited_once_with(123)
    assert "adapterA" not in engine.loaded_loras


@pytest.mark.asyncio
async def test_unload_lora_not_found(monkeypatch):
    engine = _make_lora_engine(endpoint=object())
    _patch_discovery(monkeypatch)

    result = await engine.unload_lora({"lora_name": "nope"})

    assert result["status"] == "error"
    assert "not found" in result["message"]
    engine.engine_client.remove_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_unload_lora_happy_path_clears_published(monkeypatch):
    engine = _make_lora_engine(endpoint=object())
    _, unregister = _patch_discovery(monkeypatch)
    engine.loaded_loras = {"adapterA": LoRAInfo(id=123, path="/cache/adapter")}
    engine._published_loras = {"adapterA"}

    result = await engine.unload_lora({"lora_name": "adapterA"})

    assert result["status"] == "success"
    unregister.assert_awaited_once()
    assert "adapterA" not in engine._published_loras


@pytest.mark.asyncio
async def test_unload_lora_reconciles_stale_published_card(monkeypatch):
    # Adapter is gone from the engine but still has a stale discovery card (a
    # prior unload's unregister + re-add rollback both failed). A retried
    # unload must retry the unregister so discovery converges.
    engine = _make_lora_engine(endpoint=object())
    _, unregister = _patch_discovery(monkeypatch)
    engine.loaded_loras = {}
    engine._published_loras = {"adapterA"}

    result = await engine.unload_lora({"lora_name": "adapterA"})

    assert result["status"] == "success"
    unregister.assert_awaited_once()
    engine.engine_client.remove_lora.assert_not_awaited()
    assert "adapterA" not in engine._published_loras


@pytest.mark.asyncio
async def test_unload_lora_unregister_failure_leaves_state_intact(monkeypatch):
    # Unregister runs first; if it fails, nothing has been mutated yet, so the
    # adapter stays both loaded and published (consistent and still routable)
    # and the engine is never touched.
    engine = _make_lora_engine(endpoint=object())
    _, unregister = _patch_discovery(monkeypatch)
    unregister.side_effect = RuntimeError("discovery is down")
    engine.loaded_loras = {"adapterA": LoRAInfo(id=123, path="/cache/adapter")}
    engine._published_loras = {"adapterA"}

    result = await engine.unload_lora({"lora_name": "adapterA"})

    assert result["status"] == "error"
    assert "Failed to unregister" in result["message"]
    engine.engine_client.remove_lora.assert_not_awaited()
    assert "adapterA" in engine.loaded_loras
    assert "adapterA" in engine._published_loras


@pytest.mark.asyncio
async def test_unload_lora_engine_removal_failure_after_unregister(monkeypatch):
    # Unregister succeeds (card removed, discarded from published) but the
    # engine removal fails: the adapter stays loaded-but-unpublished so a
    # retried unload skips the unregister and retries only the engine removal.
    engine = _make_lora_engine(endpoint=object())
    _, unregister = _patch_discovery(monkeypatch)
    engine.engine_client.remove_lora.side_effect = RuntimeError("engine wedged")
    engine.loaded_loras = {"adapterA": LoRAInfo(id=123, path="/cache/adapter")}
    engine._published_loras = {"adapterA"}

    result = await engine.unload_lora({"lora_name": "adapterA"})

    assert result["status"] == "error"
    assert "Failed to remove" in result["message"]
    unregister.assert_awaited_once()
    assert "adapterA" not in engine._published_loras
    assert "adapterA" in engine.loaded_loras


# --------------------------------------------------------------------------- #
# list_loras
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list_loras_reports_loaded_adapters():
    engine = _make_lora_engine()
    engine.loaded_loras = {
        "a": LoRAInfo(id=1, path="/a"),
        "b": LoRAInfo(id=2, path="/b"),
    }

    result = await engine.list_loras({})

    assert result["status"] == "success"
    assert result["loras"] == {"a": 1, "b": 2}
    assert result["count"] == 2


# --------------------------------------------------------------------------- #
# on_endpoint_ready handoff
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_on_endpoint_ready_stashes_endpoint_for_publishing(monkeypatch):
    engine = _make_lora_engine(endpoint=None)
    register, _ = _patch_discovery(monkeypatch)

    sentinel = object()
    await engine.on_endpoint_ready(sentinel)
    assert engine._endpoint is sentinel

    # load_lora now publishes against the stashed endpoint.
    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )
    assert result["status"] == "success"
    register.assert_awaited_once()
    assert register.await_args.kwargs["endpoint"] is sentinel


@pytest.mark.asyncio
async def test_load_lora_skips_publish_without_endpoint(monkeypatch):
    engine = _make_lora_engine(endpoint=None)
    register, _ = _patch_discovery(monkeypatch)

    result = await engine.load_lora(
        {"lora_name": "adapterA", "source": {"uri": "file:///x"}}
    )

    # Adapter still loads into the engine; discovery publish is skipped.
    assert result["status"] == "success"
    engine.engine_client.add_lora.assert_awaited_once()
    register.assert_not_awaited()
    assert engine.loaded_loras["adapterA"].id == 123


@pytest.mark.asyncio
async def test_cleanup_clears_endpoint_and_lora_state():
    engine = _make_lora_engine(endpoint=object())
    engine.engine_client.shutdown = MagicMock()
    engine.loaded_loras = {"adapterA": LoRAInfo(id=7, path="/path/a")}
    engine._published_loras = {"adapterA"}

    await engine.cleanup()

    # A shut-down engine must not retain a dangling endpoint reference or any
    # stale adapter bookkeeping. The fixed stripe locks are process state, not
    # per-adapter, so they are left intact.
    assert engine._endpoint is None
    assert engine.loaded_loras == {}
    assert engine._published_loras == set()


def test_get_lora_lock_is_stable_and_bounded():
    from dynamo.vllm.llm_engine import _LORA_LOCK_STRIPES

    engine = _make_lora_engine()

    # The same name always maps to the same stripe lock: this is the
    # serialization invariant load/unload depends on.
    assert engine._get_lora_lock("adapterA") is engine._get_lora_lock("adapterA")

    # The lock store is a fixed set of stripes, so it does not grow per distinct
    # adapter name (bounded memory, no eviction needed).
    for i in range(_LORA_LOCK_STRIPES * 4):
        engine._get_lora_lock(f"adapter-{i}")
    assert len(engine._lora_load_locks) == _LORA_LOCK_STRIPES
