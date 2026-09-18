# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

import asyncio
import hashlib
import json
from dataclasses import replace
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


@pytest.mark.parametrize(
    ("base_model", "source_uri", "expected"),
    [
        (
            "meta-llama/Llama-3.1-8B-Instruct",
            "wandb-artifact:///team/project/adapter:v7",
            "dyn-lora-c7246b154263a336c006517e4bc6d7c8",
        ),
        (
            "base",
            "custom://adapter@sha256:abc",
            "dyn-lora-e3cfcd02f2e62cb70f45999324e7d4de",
        ),
        (
            "base",
            "wandb-artifact:///entity/project/artifact:v1",
            "dyn-lora-27056eb300024fbfc91334895bba1e26",
        ),
        (
            "base",
            "wandb-artifact:///a|b:v1",
            "dyn-lora-74bbe88c562d9d177d098e3ba118851a",
        ),
    ],
)
def test_adapter_key_matches_rust_golden_vectors(base_model, source_uri, expected):
    assert _identity(base_model, source_uri) == expected


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


def _coordinator(monkeypatch, tmp_path, resolve_runtime_lora):
    resolver = (
        AsyncMock(side_effect=resolve_runtime_lora)
        if callable(resolve_runtime_lora)
        else AsyncMock(return_value=resolve_runtime_lora)
    )
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
        resolve_runtime_lora=resolver,
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
    handler = _handler()
    return handler, RuntimeLoRACoordinator(handler)


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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
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
        return True

    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)

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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
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
    assert set(handler._lora_state.runtime_pending_admissions) == {"request-2"}
    assert handler._lora_state.runtime_pending_leases == {result.lora_name: 1}
    await coordinator.release_pending_admission("request-2")
    assert not handler._lora_state.runtime_pending_admissions
    assert not handler._lora_state.runtime_pending_leases


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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
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
async def test_runtime_lora_enforces_total_cache_budget(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    (tmp_path / "occupied").write_bytes(bytes(65))
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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
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
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)
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


def test_runtime_settings_parse_pending_admission_timeout(monkeypatch):
    monkeypatch.setenv("DYN_LORA_PENDING_ADMISSION_TIMEOUT_SECONDS", "17")

    settings = runtime_lora_mod.RuntimeLoRASettings.from_engine_args(
        SimpleNamespace(max_loras=4, max_num_seqs=32)
    )

    assert settings.pending_admission_timeout_seconds == 17


@pytest.mark.asyncio
async def test_pending_admission_guard_releases_abandoned_request(
    monkeypatch, tmp_path
):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    handler, coordinator = _coordinator(
        monkeypatch,
        tmp_path,
        ResolvedLoRA(snapshot, "digest:immutable"),
    )

    await coordinator.ensure_from_request(_request(source_uri), "request-1")
    async with coordinator.pending_admission_guard("request-1"):
        pass

    assert not handler._lora_state.runtime_pending_admissions
    assert not handler._lora_state.runtime_pending_leases


@pytest.mark.asyncio
async def test_pending_request_limit_is_separate_from_distinct_key_limit(
    monkeypatch, tmp_path
):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    monkeypatch.setenv("DYN_LORA_MAX_PENDING_RUNTIME_KEYS", "1")
    monkeypatch.setenv("DYN_LORA_MAX_PENDING_ADMISSIONS", "2")
    _handler_state, coordinator = _coordinator(
        monkeypatch,
        tmp_path,
        ResolvedLoRA(snapshot, "digest:immutable"),
    )

    first = await coordinator.ensure_from_request(_request(source_uri), "request-1")
    second = await coordinator.ensure_from_request(_request(source_uri), "request-2")
    with pytest.raises(runtime_lora_mod.HttpError) as error:
        await coordinator.ensure_from_request(_request(source_uri), "request-3")

    assert first is not None and second is not None
    assert error.value.code == 429
    assert error.value.message == "lora_capacity_exceeded"


@pytest.mark.parametrize(
    "handler_type",
    [handlers_mod.DecodeWorkerHandler, handlers_mod.PrefillWorkerHandler],
)
@pytest.mark.asyncio
async def test_token_path_cancellation_after_resolution_releases_pending_admission(
    monkeypatch, tmp_path, handler_type
):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    state_handler, coordinator = _coordinator(
        monkeypatch,
        tmp_path,
        ResolvedLoRA(snapshot, "digest:immutable"),
    )
    ensure_from_request = coordinator.ensure_from_request

    async def cancelling_ensure(request, request_id):
        result = await ensure_from_request(request, request_id)
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        return result

    coordinator.ensure_from_request = cancelling_ensure
    request_handler = handler_type.__new__(handler_type)
    request_handler._runtime_lora_coordinator = coordinator

    async def request_flow():
        (
            admission_stack,
            _lora_request,
            _runtime_lora,
        ) = await request_handler._prepare_lora_admission(
            _request(source_uri), "request-1"
        )
        async with admission_stack:
            await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.create_task(request_flow())

    assert not state_handler._lora_state.runtime_pending_admissions
    assert not state_handler._lora_state.runtime_pending_leases


@pytest.mark.parametrize(
    "handler_type",
    [handlers_mod.DecodeWorkerHandler, handlers_mod.PrefillWorkerHandler],
)
@pytest.mark.asyncio
async def test_token_path_cancellation_during_shared_resolution_creates_no_admission(
    monkeypatch, tmp_path, handler_type
):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    resolution_started = asyncio.Event()
    finish_resolution = asyncio.Event()

    async def resolve(**_kwargs):
        resolution_started.set()
        await finish_resolution.wait()
        return ResolvedLoRA(snapshot, "digest:immutable")

    state_handler, coordinator = _coordinator(monkeypatch, tmp_path, resolve)
    request_handler = handler_type.__new__(handler_type)
    request_handler._runtime_lora_coordinator = coordinator

    async def request_flow():
        (
            admission_stack,
            _lora_request,
            _runtime_lora,
        ) = await request_handler._prepare_lora_admission(
            _request(source_uri), "request-1"
        )
        async with admission_stack:
            await asyncio.Event().wait()

    request_task = asyncio.create_task(request_flow())
    await resolution_started.wait()
    load_task = next(iter(state_handler._lora_state.runtime_load_tasks.values()))
    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task

    finish_resolution.set()
    await load_task
    await asyncio.sleep(0)

    assert not state_handler._lora_state.runtime_pending_admissions
    assert not state_handler._lora_state.runtime_pending_leases


@pytest.mark.asyncio
async def test_pending_admission_expiry_task_is_retained_until_release(
    monkeypatch, tmp_path
):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    handler, coordinator = _coordinator(
        monkeypatch,
        tmp_path,
        ResolvedLoRA(snapshot, "digest:immutable"),
    )
    coordinator._settings = replace(
        coordinator._active_settings,
        pending_admission_timeout_seconds=0,
    )

    lora_request = await coordinator.ensure_from_request(
        _request(source_uri), "request-1"
    )
    await handler._lora_capacity_guard.acquire()
    for _ in range(5):
        await asyncio.sleep(0)
        if handler._lora_state.runtime_expiry_tasks:
            break

    assert len(handler._lora_state.runtime_expiry_tasks) == 1
    assert "request-1" in handler._lora_state.runtime_pending_admissions

    handler._lora_capacity_guard.release()
    for _ in range(5):
        await asyncio.sleep(0)
        if not handler._lora_state.runtime_expiry_tasks:
            break

    assert not handler._lora_state.runtime_expiry_tasks
    assert not handler._lora_state.runtime_pending_admissions
    assert not handler._lora_state.runtime_pending_leases
    with pytest.raises(runtime_lora_mod.HttpError) as error:
        await coordinator.activate_pending_admission("request-1", lora_request)
    assert error.value.code == 429
    assert error.value.message == "lora_capacity_exceeded"


@pytest.mark.asyncio
async def test_close_cancels_resolution_and_rejects_new_runtime_requests(
    monkeypatch, tmp_path
):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def resolve(**_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    handler, coordinator = _coordinator(monkeypatch, tmp_path, resolve)
    waiter = asyncio.create_task(
        coordinator.ensure_from_request(_request(source_uri), "request-1")
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    await coordinator.close()

    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert cancelled.is_set()
    handler.engine_client.add_lora.assert_not_awaited()
    assert not handler._lora_state.runtime_load_tasks
    assert not handler._lora_state.runtime_load_digests
    await coordinator.close()

    with pytest.raises(runtime_lora_mod.HttpError) as error:
        await coordinator.ensure_from_request(_request(source_uri), "request-2")
    assert error.value.code == 503
    assert error.value.message == "lora_resolver_unavailable"


@pytest.mark.asyncio
async def test_close_cancels_pending_admission_timer(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    handler, coordinator = _coordinator(
        monkeypatch,
        tmp_path,
        ResolvedLoRA(snapshot, "digest:immutable"),
    )

    await coordinator.ensure_from_request(_request(source_uri), "request-1")
    _adapter_key, timer = handler._lora_state.runtime_pending_admissions["request-1"]

    await coordinator.close()
    await asyncio.sleep(0)

    assert timer.cancelled()
    assert not handler._lora_state.runtime_pending_admissions
    assert not handler._lora_state.runtime_pending_leases
    assert not handler._lora_state.runtime_expiry_tasks


@pytest.mark.asyncio
async def test_close_waits_for_engine_mutation_and_rollback(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    add_started = asyncio.Event()
    release_add = asyncio.Event()

    async def add_lora(_request):
        add_started.set()
        await release_add.wait()
        return True

    handler, coordinator = _coordinator(
        monkeypatch,
        tmp_path,
        ResolvedLoRA(snapshot, "digest:immutable"),
    )
    handler.engine_client.add_lora = AsyncMock(side_effect=add_lora)
    waiter = asyncio.create_task(
        coordinator.ensure_from_request(_request(source_uri), "request-1")
    )
    await asyncio.wait_for(add_started.wait(), timeout=1)

    close_task = asyncio.create_task(coordinator.close())
    await asyncio.sleep(0)
    assert not close_task.done()

    release_add.set()
    await close_task
    with pytest.raises(asyncio.CancelledError):
        await waiter

    adapter_key = _identity("base", source_uri)
    handler.engine_client.remove_lora.assert_awaited_once()
    assert adapter_key not in handler._lora_state.loaded_loras
    assert adapter_key not in handler._lora_state.runtime_loras
    assert adapter_key not in handler._lora_state.runtime_reserved_ids
    assert not handler._lora_state.runtime_load_tasks
    assert not handler._lora_state.runtime_load_digests


@pytest.mark.asyncio
async def test_string_target_modules_are_deferred_to_vllm(monkeypatch, tmp_path):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path, target_modules="q_proj|v_proj")
    _handler_state, coordinator = _coordinator(
        monkeypatch,
        tmp_path,
        ResolvedLoRA(snapshot, "digest:immutable"),
    )

    result = await coordinator.ensure_from_request(_request(source_uri), "request-id")

    assert result is not None


@pytest.mark.asyncio
async def test_distinct_runtime_resolutions_can_overlap(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path)
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = 0

    async def resolve(**_kwargs):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await release.wait()
        return ResolvedLoRA(snapshot, "digest:immutable")

    _handler_state, coordinator = _coordinator(monkeypatch, tmp_path, resolve)
    first = asyncio.create_task(
        coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/first:v1"), "request-1"
        )
    )
    second = asyncio.create_task(
        coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/second:v1"), "request-2"
        )
    )
    await asyncio.wait_for(both_started.wait(), timeout=1)
    release.set()
    await asyncio.gather(first, second)

    assert started == 2


@pytest.mark.asyncio
async def test_resident_runtime_key_requires_full_identity_metadata():
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    adapter_key = _identity("base", source_uri)
    handler = _handler()
    record = runtime_lora_mod.RuntimeLoRAInfo(
        adapter_key=adapter_key,
        full_identity_digest=runtime_lora_mod._identity_digest("base", source_uri),
        base_model_name="base",
        source_revision="digest:immutable",
        id=7,
        path="/runtime/adapter",
    )
    handler._lora_state.runtime_loras[adapter_key] = record
    handler._lora_state.loaded_loras[adapter_key] = runtime_lora_mod.LoRAInfo(
        id=record.id,
        path=record.path,
    )

    with pytest.raises(runtime_lora_mod.HttpError, match="invalid_lora_model_id"):
        await RuntimeLoRACoordinator(handler).ensure_from_request(
            {"model": "base", "routing": {"lora_name": adapter_key}},
            "request-id",
        )


def test_runtime_capability_rejects_vllm_tokenizer_mode(monkeypatch, tmp_path):
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setattr(runtime_lora_mod, "get_lora_manager", lambda **_kwargs: manager)

    with pytest.raises(
        runtime_lora_mod.RuntimeLoRAConfigurationError,
        match="tokenized request path",
    ):
        publish_runtime_lora_capability(
            SimpleNamespace(),
            SimpleNamespace(
                use_vllm_tokenizer=True,
                engine_args=SimpleNamespace(enable_lora=True, max_loras=4),
            ),
            WorkerType.Aggregated,
        )


@pytest.mark.asyncio
async def test_resolution_queue_exhaustion_does_not_invoke_plugin(
    monkeypatch, tmp_path
):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    handler, coordinator = _coordinator(
        monkeypatch,
        tmp_path,
        ResolvedLoRA(snapshot, "digest:immutable"),
    )
    handler._lora_state.runtime_resolution_semaphore = asyncio.Semaphore(0)
    coordinator._settings = replace(
        coordinator._active_settings,
        resolve_timeout_seconds=0.01,
    )

    with pytest.raises(runtime_lora_mod.HttpError) as error:
        await coordinator.ensure_from_request(_request(source_uri), "request-id")

    assert error.value.code == 429
    assert error.value.message == "lora_capacity_exceeded"
    runtime_lora_mod.get_lora_manager().resolve_runtime_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_false_add_is_accepted_only_when_engine_confirms_id(
    monkeypatch, tmp_path
):
    source_uri = "wandb-artifact:///entity/project/adapter:v1"
    snapshot = _snapshot(tmp_path)
    handler, coordinator = _coordinator(
        monkeypatch,
        tmp_path,
        ResolvedLoRA(snapshot, "digest:immutable"),
    )
    handler.engine_client.add_lora.return_value = False

    async def list_loras():
        request = handler.engine_client.add_lora.await_args.args[0]
        return {request.lora_int_id}

    handler.engine_client.list_loras = AsyncMock(side_effect=list_loras)

    result = await coordinator.ensure_from_request(_request(source_uri), "request-id")

    assert result is not None
    assert result.lora_name in handler._lora_state.runtime_loras
    handler.engine_client.remove_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_uncertain_engine_id_is_removed_and_released(monkeypatch, tmp_path):
    handler, coordinator = _coordinator(
        monkeypatch,
        tmp_path,
        RuntimeLoRANotFoundError("unused"),
    )
    handler._lora_state.uncertain_engine_lora_ids.add(77)
    handler._lora_state.rollback_reserved_ids.add(77)
    handler.engine_client.list_loras = AsyncMock(side_effect=[{77}, set()])

    await coordinator._reconcile_uncertain_engine_loras()

    handler.engine_client.remove_lora.assert_awaited_once_with(77)
    assert not handler._lora_state.uncertain_engine_lora_ids
    assert not handler._lora_state.rollback_reserved_ids
