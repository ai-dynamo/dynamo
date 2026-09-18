# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.constants import DisaggregationMode
from dynamo.common.lora.runtime import (
    ResolvedLoRA,
    RuntimeLoRANotFoundError,
    RuntimeLoRAPluginError,
    RuntimeLoRAResolverUnavailableError,
)
from dynamo.llm import WorkerType
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


def _snapshot(tmp_path):
    snapshot = tmp_path / "adapter"
    snapshot.mkdir()
    (snapshot / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "target_modules": ["q_proj"],
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
        engine_client=SimpleNamespace(add_lora=AsyncMock()),
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
async def test_runtime_lora_miss_resolves_loads_and_returns_request(
    monkeypatch, tmp_path
):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)
    handler = _handler()

    lora_request = await RuntimeLoRACoordinator(handler).ensure_from_request(
        _request(source_uri), "request-id"
    )

    assert lora_request is not None
    assert lora_request.lora_name == _identity("base", source_uri)
    assert lora_request.lora_path == str(snapshot)
    manager.resolve_runtime_lora.assert_awaited_once()
    handler.engine_client.add_lora.assert_awaited_once()
    assert handler._lora_state.runtime_loras[
        lora_request.lora_name
    ].source_revision == ("digest:immutable")


@pytest.mark.asyncio
async def test_concurrent_same_key_resolves_and_adds_once(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def resolve(**_kwargs):
        started.set()
        await release.wait()
        return ResolvedLoRA(snapshot, "digest:immutable")

    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(side_effect=resolve),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)
    handler = _handler()
    coordinator = RuntimeLoRACoordinator(handler)

    first = asyncio.create_task(
        coordinator.ensure_from_request(_request(source_uri), "request-1")
    )
    await started.wait()
    second = asyncio.create_task(
        coordinator.ensure_from_request(_request(source_uri), "request-2")
    )
    await asyncio.sleep(0)
    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.lora_int_id == second_result.lora_int_id
    manager.resolve_runtime_lora.assert_awaited_once()
    handler.engine_client.add_lora.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_distinct_keys_reserve_distinct_integer_ids(
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
    both_loading = asyncio.Event()
    load_count = 0

    async def add_lora(_request):
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            both_loading.set()
        await both_loading.wait()

    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)
    monkeypatch.setattr(runtime_lora_mod, "lora_name_to_id", lambda _name: 7)
    handler = _handler()
    handler.engine_client.add_lora.side_effect = add_lora
    coordinator = RuntimeLoRACoordinator(handler)

    first, second = await asyncio.gather(
        coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/first:v1"), "request-1"
        ),
        coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/second:v1"), "request-2"
        ),
    )

    assert {first.lora_int_id, second.lora_int_id} == {7, 8}


@pytest.mark.asyncio
async def test_identity_mismatch_fails_before_resolver_or_engine(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)
    handler = _handler()

    with pytest.raises(
        runtime_lora_mod.HttpError, match="runtime_lora_identity_mismatch"
    ):
        await RuntimeLoRACoordinator(handler).ensure_from_request(
            _request(
                source_uri,
                adapter_key="dyn-lora-00000000000000000000000000000000",
            ),
            "request-id",
        )

    manager.resolve_runtime_lora.assert_not_awaited()
    handler.engine_client.add_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_lora_metadata_is_all_or_none():
    handler = _handler()
    with pytest.raises(runtime_lora_mod.HttpError, match="invalid_lora_model_id"):
        await RuntimeLoRACoordinator(handler).ensure_from_request(
            {
                "model": "base",
                "routing": {"lora_name": "dyn-lora-74bbe88c562d9d177d098e3ba118851a"},
            },
            "request-id",
        )


@pytest.mark.asyncio
async def test_known_prefixed_alias_without_runtime_metadata_stays_explicit():
    alias = "dyn-lora-74bbe88c562d9d177d098e3ba118851a"
    handler = _handler()
    handler._lora_state.loaded_loras[alias] = runtime_lora_mod.LoRAInfo(
        id=7,
        path="/preloaded/adapter",
    )

    result = await RuntimeLoRACoordinator(handler).ensure_from_request(
        {"model": "base", "routing": {"lora_name": alias}},
        "request-id",
    )

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolver_error", "status", "code"),
    [
        (RuntimeLoRANotFoundError("provider detail"), 404, "lora_not_found"),
        (
            RuntimeLoRAResolverUnavailableError("secret provider response"),
            503,
            "lora_resolver_unavailable",
        ),
        (RuntimeLoRAPluginError("local path detail"), 500, "lora_plugin_error"),
        (TimeoutError("signed URL detail"), 504, "runtime_lora_resolve_timeout"),
    ],
)
async def test_resolver_errors_use_redacted_http_taxonomy(
    monkeypatch, tmp_path, resolver_error, status, code
):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(side_effect=resolver_error),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)

    with pytest.raises(runtime_lora_mod.HttpError) as error:
        await RuntimeLoRACoordinator(_handler()).ensure_from_request(
            _request(source_uri), "request-id"
        )

    assert error.value.code == status
    assert error.value.message == code
    assert "detail" not in str(error.value)


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_load(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def resolve(**_kwargs):
        started.set()
        await release.wait()
        return ResolvedLoRA(snapshot, "digest:immutable")

    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(side_effect=resolve),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)
    handler = _handler()
    coordinator = RuntimeLoRACoordinator(handler)

    cancelled_waiter = asyncio.create_task(
        coordinator.ensure_from_request(_request(source_uri), "request-1")
    )
    await started.wait()
    surviving_waiter = asyncio.create_task(
        coordinator.ensure_from_request(_request(source_uri), "request-2")
    )
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    release.set()

    result = await surviving_waiter

    assert result.lora_name == _identity("base", source_uri)
    manager.resolve_runtime_lora.assert_awaited_once()
    handler.engine_client.add_lora.assert_awaited_once()


@pytest.mark.asyncio
async def test_engine_load_failure_releases_capacity_reservation(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)
    handler = _handler()
    handler.engine_client.add_lora.side_effect = RuntimeError("engine rejected")
    adapter_key = _identity("base", source_uri)

    with pytest.raises(runtime_lora_mod.HttpError, match="lora_load_failed"):
        await RuntimeLoRACoordinator(handler).ensure_from_request(
            _request(source_uri), "request-id"
        )
    await asyncio.sleep(0)

    assert adapter_key not in handler._lora_state.loaded_loras
    assert adapter_key not in handler._lora_state.runtime_loras
    assert adapter_key not in handler._lora_state.runtime_load_tasks


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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)
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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)
    handler = _handler()
    handler._served_model_name = "different-base"
    handler.engine_args.model = "different-base"

    with pytest.raises(runtime_lora_mod.HttpError, match="identity_mismatch"):
        await RuntimeLoRACoordinator(handler).ensure_from_request(
            _request(source_uri), "request-id"
        )

    manager.resolve_runtime_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_lora_enforces_total_cache_budget(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    (tmp_path / "occupied").write_bytes(bytes(64))
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=AsyncMock(),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact")
    monkeypatch.setenv("DYN_LORA_MAX_CACHE_BYTES", "64")
    monkeypatch.setenv("DYN_LORA_MAX_DOWNLOAD_BYTES", "64")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)
    handler = _handler()

    with pytest.raises(runtime_lora_mod.HttpError, match="lora_capacity_exceeded"):
        await RuntimeLoRACoordinator(handler).ensure_from_request(
            _request(source_uri), "request-id"
        )

    manager.resolve_runtime_lora.assert_not_awaited()


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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)
    runtime_config = RuntimeConfig()

    publish_runtime_lora_capability(
        runtime_config,
        SimpleNamespace(engine_args=SimpleNamespace(enable_lora=True)),
        WorkerType.Aggregated,
    )

    assert runtime_config.values["supports_runtime_lora_resolution"] is True
    assert runtime_config.values["runtime_lora_protocol_versions"] == [2]
    assert runtime_config.values["runtime_lora_schemes"] == ["wandb-artifact"]
    assert runtime_config.taints == {
        "existing",
        "dynamo.runtime-lora/v2",
    }


def test_runtime_capability_rejects_invalid_resource_bounds(monkeypatch, tmp_path):
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "5")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda: manager)

    with pytest.raises(
        runtime_lora_mod.RuntimeLoRAConfigurationError,
        match="cannot exceed",
    ):
        publish_runtime_lora_capability(
            SimpleNamespace(),
            SimpleNamespace(engine_args=SimpleNamespace(enable_lora=True, max_loras=4)),
            WorkerType.Aggregated,
        )
