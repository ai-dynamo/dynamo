# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.constants import DisaggregationMode
from dynamo.common.lora.runtime import ResolvedLoRA
from dynamo.llm import WorkerType
from dynamo.vllm import handlers as handlers_mod
from dynamo.vllm import runtime_lora as runtime_lora_mod
from dynamo.vllm.lora_state import LoRAState
from dynamo.vllm.runtime_lora import (
    RuntimeLoRACoordinator,
    publish_runtime_lora_capability,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _identity(base_model: str, source_uri: str) -> str:
    digest = hashlib.sha256(
        b"dynamo-runtime-lora-v1\0" + base_model.encode() + b"\0" + source_uri.encode()
    ).hexdigest()
    return f"dyn-lora-{digest[:32]}"


def _snapshot(tmp_path, target_modules=None):
    snapshot = tmp_path / "adapter"
    snapshot.mkdir()
    (snapshot / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "target_modules": target_modules or ["q_proj"],
                "base_model_name_or_path": "base",
            }
        )
    )
    header = json.dumps(
        {
            "base_model.model.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [1, 2],
                "data_offsets": [0, 8],
            }
        }
    ).encode()
    (snapshot / "adapter_model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + bytes(8)
    )
    return snapshot


def _handler():
    return SimpleNamespace(
        config=SimpleNamespace(
            disaggregation_mode=DisaggregationMode.AGGREGATED,
            engine_args=SimpleNamespace(
                enable_lora=True,
                max_loras=4,
                max_lora_rank=64,
                model="base",
            ),
        ),
        engine_args=SimpleNamespace(
            enable_lora=True,
            max_loras=4,
            max_lora_rank=64,
            model="base",
        ),
        _served_model_name="base",
        _served_model_aliases=(),
        _lora_capacity=4,
        _lora_capacity_guard=asyncio.Lock(),
        _lora_state=LoRAState(),
        _engine_loaded_loras=set(),
        engine_client=SimpleNamespace(
            add_lora=AsyncMock(return_value=True),
            remove_lora=AsyncMock(return_value=True),
        ),
    )


def _request(source_uri: str, *, adapter_key: str | None = None):
    key = adapter_key or _identity("base", source_uri)
    return {
        "model": "base",
        "routing": {
            "lora_name": key,
            "base_model_name": "base",
            "lora_source_uri": source_uri,
            "lora_resolution_version": 2,
        },
    }


@pytest.mark.asyncio
async def test_runtime_capacity_evicts_only_runtime_adapter(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    coordinator = RuntimeLoRACoordinator(handler)
    first_source = "wandb-artifact:///entity/project/first:v1"
    second_source = "wandb-artifact:///entity/project/second:v1"

    first = await coordinator.ensure_from_request(_request(first_source), "request-1")
    await coordinator.release_pending_admission("request-1")
    second = await coordinator.ensure_from_request(_request(second_source), "request-2")

    handler.engine_client.remove_lora.assert_awaited_once_with(first.lora_int_id)
    assert set(handler._lora_state.runtime_loras) == {second.lora_name}
    assert first.lora_name not in handler._lora_state.loaded_loras


@pytest.mark.asyncio
async def test_eviction_after_load_before_lease_returns_capacity_error(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    coordinator = RuntimeLoRACoordinator(handler)
    first_source = "wandb-artifact:///entity/project/first:v1"
    first_key = _identity("base", first_source)
    post_load = asyncio.Event()
    allow_first_lease = asyncio.Event()
    original_lease = coordinator._lease_resident
    first_lease_calls = 0

    async def gated_lease(metadata, request_id):
        nonlocal first_lease_calls
        if metadata.adapter_key == first_key and request_id == "request-1":
            first_lease_calls += 1
            if first_lease_calls == 2:
                post_load.set()
                await allow_first_lease.wait()
        return await original_lease(metadata, request_id)

    coordinator._lease_resident = gated_lease
    first_task = asyncio.create_task(
        coordinator.ensure_from_request(_request(first_source), "request-1")
    )
    await post_load.wait()

    second = await coordinator.ensure_from_request(
        _request("wandb-artifact:///entity/project/second:v1"), "request-2"
    )
    allow_first_lease.set()

    with pytest.raises(runtime_lora_mod.HttpError) as error:
        await first_task

    assert error.value.code == 429
    assert error.value.message == "lora_capacity_exceeded"
    assert second is not None


@pytest.mark.asyncio
async def test_symlinked_snapshot_root_is_rejected(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    symlink = tmp_path / "adapter-link"
    symlink.symlink_to(snapshot, target_is_directory=True)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(symlink, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()

    with pytest.raises(runtime_lora_mod.HttpError, match="invalid_lora_adapter"):
        await RuntimeLoRACoordinator(handler).ensure_from_request(
            _request(source_uri), "request-id"
        )

    handler.engine_client.add_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_lora_rejects_worker_base_mismatch(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    handler._served_model_name = "different-base"
    handler.engine_args.model = "different-base"

    with pytest.raises(runtime_lora_mod.HttpError, match="identity_mismatch"):
        await RuntimeLoRACoordinator(handler).ensure_from_request(
            _request(source_uri), "request-id"
        )

    manager.resolve_runtime_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_evicted_snapshot_reloads_when_cache_is_exactly_full(
    monkeypatch, tmp_path
):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    cache_bytes = sum(
        path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()
    )
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact")
    monkeypatch.setenv("DYN_LORA_MAX_CACHE_BYTES", str(cache_bytes))
    monkeypatch.setenv("DYN_LORA_MAX_DOWNLOAD_BYTES", str(cache_bytes))
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()

    result = await RuntimeLoRACoordinator(handler).ensure_from_request(
        _request(source_uri), "request-id"
    )

    assert result is not None
    assert result.lora_path == str(snapshot)
    assert (
        manager.resolve_runtime_lora.await_args.kwargs["context"].max_download_bytes
        == 0
    )
    handler.engine_client.add_lora.assert_awaited_once()


def test_opaque_source_is_preserved_for_plugin_policy():
    source_uri = "custom:opaque?api_key=provider-owned#fragment"
    metadata = runtime_lora_mod._parse_request(
        _request(source_uri, adapter_key=_identity("base", source_uri))
    )
    assert metadata is not None
    assert metadata.source_uri == source_uri


def test_runtime_capability_is_published_only_for_valid_aggregated_worker(
    monkeypatch,
    tmp_path,
):
    class RuntimeConfig:
        def __init__(self):
            self.values = {}
            self.taints = {"existing"}

        def set_engine_specific(self, key, value):
            self.values[key] = json.loads(value)

    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    runtime_config = RuntimeConfig()
    handler_config = SimpleNamespace(
        engine_args=SimpleNamespace(enable_lora=True, max_loras=4, max_cpu_loras=8)
    )

    publish_runtime_lora_capability(
        runtime_config,
        handler_config,
        WorkerType.Aggregated,
    )

    assert runtime_config.values["supports_runtime_lora_resolution"] is True
    assert runtime_config.values["runtime_lora_protocol_versions"] == [2]
    assert runtime_config.values["runtime_lora_schemes"] == ["wandb-artifact"]
    assert runtime_config.taints == {
        "existing",
        "dynamo.runtime-lora/v2",
    }
    assert handler_config.runtime_lora_settings.max_registered_loras == 8
    assert handler_config.runtime_lora_settings.max_resident_runtime_loras == 8


def test_runtime_capability_rejects_invalid_resource_bounds(monkeypatch, tmp_path):
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "5")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)

    with pytest.raises(
        runtime_lora_mod.RuntimeLoRAConfigurationError,
        match="cannot exceed",
    ):
        publish_runtime_lora_capability(
            SimpleNamespace(),
            SimpleNamespace(engine_args=SimpleNamespace(enable_lora=True, max_loras=4)),
            WorkerType.Aggregated,
        )


def test_disabled_runtime_does_not_parse_stale_resource_limits(monkeypatch):
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "false")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "not-an-int")

    coordinator = RuntimeLoRACoordinator(_handler())
    assert coordinator._settings is None


@pytest.mark.asyncio
async def test_active_runtime_adapter_is_not_evicted(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    coordinator = RuntimeLoRACoordinator(handler)
    first_source = "wandb-artifact:///entity/project/first:v1"
    first = await coordinator.ensure_from_request(_request(first_source), "request-1")
    await coordinator.release_pending_admission("request-1")
    handler._lora_state.runtime_active_leases[first.lora_name] = 1

    with pytest.raises(runtime_lora_mod.HttpError, match="lora_capacity_exceeded"):
        await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/second:v1"), "request-2"
        )

    handler.engine_client.remove_lora.assert_not_awaited()
    assert set(handler._lora_state.runtime_loras) == {first.lora_name}


@pytest.mark.asyncio
async def test_inflight_load_reserves_runtime_residency(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def add_lora(_request):
        started.set()
        await release.wait()
        return True

    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    handler.engine_client.add_lora.side_effect = add_lora
    coordinator = RuntimeLoRACoordinator(handler)
    first = asyncio.create_task(
        coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/first:v1"), "request-1"
        )
    )
    await started.wait()

    with pytest.raises(runtime_lora_mod.HttpError, match="lora_capacity_exceeded"):
        await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/second:v1"), "request-2"
        )

    release.set()
    first_result = await first
    assert first_result is not None
    assert len(handler._lora_state.runtime_loras) == 1


@pytest.mark.asyncio
async def test_pending_runtime_admission_prevents_eviction(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    coordinator = RuntimeLoRACoordinator(handler)
    first = await coordinator.ensure_from_request(
        _request("wandb-artifact:///entity/project/first:v1"), "request-1"
    )

    with pytest.raises(runtime_lora_mod.HttpError, match="lora_capacity_exceeded"):
        await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/second:v1"), "request-2"
        )

    handler.engine_client.remove_lora.assert_not_awaited()
    assert handler._lora_state.runtime_pending_leases == {first.lora_name: 1}
    await coordinator.release_pending_admission("request-1")
    await asyncio.sleep(0)
    second = await coordinator.ensure_from_request(
        _request("wandb-artifact:///entity/project/second:v1"), "request-3"
    )
    assert set(handler._lora_state.runtime_loras) == {second.lora_name}


@pytest.mark.asyncio
async def test_failed_replacement_restores_evicted_runtime_adapter(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    coordinator = RuntimeLoRACoordinator(handler)
    first = await coordinator.ensure_from_request(
        _request("wandb-artifact:///entity/project/first:v1"), "request-1"
    )
    await coordinator.release_pending_admission("request-1")
    handler.engine_client.add_lora.reset_mock()
    handler.engine_client.add_lora.side_effect = [
        RuntimeError("replacement failed"),
        True,
    ]

    second_source = "wandb-artifact:///entity/project/second:v1"
    with pytest.raises(runtime_lora_mod.HttpError, match="lora_load_failed"):
        await coordinator.ensure_from_request(_request(second_source), "request-2")

    assert set(handler._lora_state.runtime_loras) == {first.lora_name}
    assert handler._lora_state.loaded_loras[first.lora_name].id == first.lora_int_id
    assert _identity("base", second_source) not in handler._lora_state.loaded_loras
    assert not handler._lora_state.rollback_reserved_ids
    assert handler.engine_client.add_lora.await_count == 2


@pytest.mark.asyncio
async def test_cancelled_engine_add_is_removed_and_untracked(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def add_lora(_request):
        started.set()
        await release.wait()
        return True

    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    handler.engine_client.add_lora.side_effect = add_lora
    coordinator = RuntimeLoRACoordinator(handler)
    source = "wandb-artifact:///entity/project/cancelled:v1"
    adapter_key = _identity("base", source)
    waiter = asyncio.create_task(
        coordinator.ensure_from_request(_request(source), "request-1")
    )
    await started.wait()

    handler._lora_state.runtime_load_tasks[adapter_key].cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    handler.engine_client.remove_lora.assert_awaited_once()
    assert adapter_key not in handler._lora_state.loaded_loras
    assert adapter_key not in handler._lora_state.runtime_loras
    assert adapter_key not in handler._engine_loaded_loras


def test_lora_id_allocator_skips_rollback_reserved_id(monkeypatch):
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setattr(runtime_lora_mod, "lora_name_to_id", lambda _name: 7)
    handler = _handler()
    handler._lora_state.rollback_reserved_ids.add(7)

    assert RuntimeLoRACoordinator(handler)._allocate_lora_id("adapter") == 8


@pytest.mark.asyncio
async def test_cancelled_eviction_finishes_outside_capacity_guard(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(tmp_path)
    remove_started = asyncio.Event()
    allow_remove = asyncio.Event()

    async def remove_lora(_lora_id):
        remove_started.set()
        await allow_remove.wait()
        return True

    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    coordinator = RuntimeLoRACoordinator(handler)
    first = await coordinator.ensure_from_request(
        _request("wandb-artifact:///entity/project/first:v1"), "request-1"
    )
    await coordinator.release_pending_admission("request-1")
    handler.engine_client.remove_lora.side_effect = remove_lora

    second_source = "wandb-artifact:///entity/project/second:v1"
    second_key = _identity("base", second_source)
    waiter = asyncio.create_task(
        coordinator.ensure_from_request(_request(second_source), "request-2")
    )
    await remove_started.wait()
    handler._lora_state.runtime_load_tasks[second_key].cancel()

    await asyncio.wait_for(handler._lora_capacity_guard.acquire(), timeout=0.2)
    handler._lora_capacity_guard.release()
    allow_remove.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert set(handler._lora_state.runtime_loras) == {first.lora_name}
    assert handler._lora_state.loaded_loras[first.lora_name].id == first.lora_int_id
    assert second_key not in handler._lora_state.loaded_loras
    assert not handler._lora_state.rollback_reserved_ids


@pytest.mark.asyncio
async def test_false_engine_add_is_rolled_back_and_untracked(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/rejected:v1"
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    handler.engine_client.add_lora.return_value = False
    coordinator = RuntimeLoRACoordinator(handler)
    adapter_key = _identity("base", source_uri)

    with pytest.raises(runtime_lora_mod.HttpError, match="lora_load_failed"):
        await coordinator.ensure_from_request(_request(source_uri), "request-1")

    handler.engine_client.remove_lora.assert_awaited_once()
    assert adapter_key not in handler._lora_state.loaded_loras
    assert adapter_key not in handler._lora_state.runtime_loras
    assert adapter_key not in handler._engine_loaded_loras
    assert not handler._lora_state.runtime_reserved_ids


@pytest.mark.asyncio
async def test_false_rollback_restore_keeps_victim_id_uncertain(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    coordinator = RuntimeLoRACoordinator(handler)
    first = await coordinator.ensure_from_request(
        _request("wandb-artifact:///entity/project/first:v1"), "request-1"
    )
    await coordinator.release_pending_admission("request-1")
    handler.engine_client.add_lora.side_effect = [
        RuntimeError("replacement failed"),
        False,
    ]

    with pytest.raises(runtime_lora_mod.HttpError, match="lora_load_failed"):
        await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/second:v1"), "request-2"
        )

    assert first.lora_name not in handler._lora_state.runtime_loras
    assert first.lora_name not in handler._lora_state.loaded_loras
    assert first.lora_int_id in handler._lora_state.rollback_reserved_ids
    assert first.lora_int_id in handler._lora_state.uncertain_engine_lora_ids
    assert not handler._lora_state.runtime_eviction_events


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_requested", [False, True])
async def test_eviction_remove_error_restores_victim(
    monkeypatch, tmp_path, cancel_requested
):
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    engine_ids: set[int] = set()

    async def add_lora(request):
        engine_ids.add(request.lora_int_id)
        return True

    remove_started = asyncio.Event()
    finish_remove = asyncio.Event()

    async def remove_lora(lora_id):
        engine_ids.discard(lora_id)
        remove_started.set()
        await finish_remove.wait()
        raise RuntimeError("remove response lost")

    handler.engine_client.add_lora.side_effect = add_lora
    handler.engine_client.remove_lora.side_effect = remove_lora
    coordinator = RuntimeLoRACoordinator(handler)
    first = await coordinator.ensure_from_request(
        _request("wandb-artifact:///entity/project/first:v1"), "request-1"
    )
    await coordinator.release_pending_admission("request-1")

    task = asyncio.create_task(
        coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/second:v1"), "request-2"
        )
    )
    await remove_started.wait()
    if cancel_requested:
        second_key = _identity("base", "wandb-artifact:///entity/project/second:v1")
        handler._lora_state.runtime_load_tasks[second_key].cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finish_remove.set()

    if cancel_requested:
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(runtime_lora_mod.HttpError, match="lora_capacity_exceeded"):
            await task

    assert engine_ids == {first.lora_int_id}
    assert handler._lora_state.runtime_loras[first.lora_name].id == first.lora_int_id
    assert handler._lora_state.loaded_loras[first.lora_name].id == first.lora_int_id
    assert not handler._lora_state.uncertain_engine_lora_ids
    assert not handler._lora_state.runtime_eviction_events


@pytest.mark.asyncio
async def test_victim_request_waits_for_failed_replacement_rollback(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    replacement_started = asyncio.Event()
    fail_replacement = asyncio.Event()
    engine_ids: set[int] = set()
    second_key = _identity("base", "wandb-artifact:///entity/project/second:v1")

    async def add_lora(request):
        if request.lora_name == second_key:
            replacement_started.set()
            await fail_replacement.wait()
            raise RuntimeError("replacement failed")
        if request.lora_int_id in engine_ids:
            return False
        engine_ids.add(request.lora_int_id)
        return True

    async def remove_lora(lora_id):
        if lora_id not in engine_ids:
            return False
        engine_ids.remove(lora_id)
        return True

    handler.engine_client.add_lora.side_effect = add_lora
    handler.engine_client.remove_lora.side_effect = remove_lora
    coordinator = RuntimeLoRACoordinator(handler)
    first_source = "wandb-artifact:///entity/project/first:v1"
    first = await coordinator.ensure_from_request(_request(first_source), "request-1")
    await coordinator.release_pending_admission("request-1")

    replacement = asyncio.create_task(
        coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/second:v1"), "request-2"
        )
    )
    await replacement_started.wait()
    victim_request = asyncio.create_task(
        coordinator.ensure_from_request(_request(first_source), "request-3")
    )
    await asyncio.sleep(0)

    assert not victim_request.done()
    assert manager.resolve_runtime_lora.await_count == 2
    fail_replacement.set()
    with pytest.raises(runtime_lora_mod.HttpError, match="lora_load_failed"):
        await replacement
    recovered = await victim_request

    assert recovered is not None
    assert recovered.lora_int_id == first.lora_int_id
    assert engine_ids == {first.lora_int_id}
    assert set(handler._lora_state.runtime_loras) == {first.lora_name}
    assert handler._lora_state.loaded_loras[first.lora_name].id == first.lora_int_id
    assert not handler._lora_state.runtime_eviction_events


@pytest.mark.asyncio
async def test_admin_mutations_reject_adapter_during_runtime_eviction(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    coordinator = RuntimeLoRACoordinator(handler)
    first_source = "wandb-artifact:///entity/project/first:v1"
    first = await coordinator.ensure_from_request(_request(first_source), "request-1")
    await coordinator.release_pending_admission("request-1")
    remove_started = asyncio.Event()
    allow_remove = asyncio.Event()

    async def remove_lora(_lora_id):
        remove_started.set()
        await allow_remove.wait()
        return True

    handler.engine_client.remove_lora.side_effect = remove_lora
    replacement = asyncio.create_task(
        coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/second:v1"), "request-2"
        )
    )
    await remove_started.wait()
    assert first.lora_name in handler._lora_state.runtime_eviction_events

    handler._get_lora_lock = handler._lora_state.get_lock
    handler._parse_lora_unload_request = lambda request: request["lora_name"]
    load_results = [
        result
        async for result in handlers_mod.BaseWorkerHandler.load_lora(
            handler,
            {
                "lora_name": first.lora_name,
                "source": {"uri": "file:///replacement"},
            },
        )
    ]
    unload_results = [
        result
        async for result in handlers_mod.BaseWorkerHandler.unload_lora(
            handler, {"lora_name": first.lora_name}
        )
    ]

    assert load_results[-1]["status"] == "error"
    assert "reserved for request-time adapters" in load_results[-1]["message"]
    assert unload_results[-1]["status"] == "error"
    assert "cannot be unloaded" in unload_results[-1]["message"]
    assert handler.engine_client.add_lora.await_count == 1

    allow_remove.set()
    second = await replacement
    assert second is not None
    assert set(handler._lora_state.runtime_loras) == {second.lora_name}


@pytest.mark.asyncio
async def test_cancellation_during_rollback_runs_cleanup_once(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/rollback-cancel:v1"
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    rollback_started = asyncio.Event()
    finish_rollback = asyncio.Event()

    async def remove_lora(_lora_id):
        rollback_started.set()
        await finish_rollback.wait()
        return True

    handler.engine_client.add_lora.side_effect = RuntimeError("engine rejected")
    handler.engine_client.remove_lora.side_effect = remove_lora
    coordinator = RuntimeLoRACoordinator(handler)
    adapter_key = _identity("base", source_uri)
    waiter = asyncio.create_task(
        coordinator.ensure_from_request(_request(source_uri), "request-1")
    )
    await rollback_started.wait()

    handler._lora_state.runtime_load_tasks[adapter_key].cancel()
    await asyncio.sleep(0)
    assert not waiter.done()
    assert handler.engine_client.remove_lora.await_count == 1

    finish_rollback.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert handler.engine_client.remove_lora.await_count == 1
    assert adapter_key not in handler._lora_state.loaded_loras
    assert adapter_key not in handler._lora_state.runtime_loras
    assert not handler._lora_state.runtime_reserved_ids


@pytest.mark.asyncio
async def test_partial_eviction_selection_is_rolled_back(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path)
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(
            return_value=ResolvedLoRA(snapshot, "digest:immutable")
        ),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "1")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    coordinator = RuntimeLoRACoordinator(handler)
    first = await coordinator.ensure_from_request(
        _request("wandb-artifact:///entity/project/first:v1"), "request-1"
    )
    await coordinator.release_pending_admission("request-1")
    handler._lora_state.uncertain_engine_lora_ids.add(999)
    handler.engine_client.add_lora.reset_mock()
    handler.engine_client.remove_lora.reset_mock()

    with pytest.raises(runtime_lora_mod.HttpError, match="lora_capacity_exceeded"):
        await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/second:v1"), "request-2"
        )

    assert set(handler._lora_state.runtime_loras) == {first.lora_name}
    assert handler._lora_state.loaded_loras[first.lora_name].id == first.lora_int_id
    assert not handler._lora_state.runtime_eviction_events
    handler.engine_client.add_lora.assert_not_awaited()
    handler.engine_client.remove_lora.assert_not_awaited()
