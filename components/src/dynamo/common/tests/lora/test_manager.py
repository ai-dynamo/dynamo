# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.lora.manager.get_lora_manager singleton."""

import asyncio
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from dynamo.common.lora import manager as manager_module
from dynamo.common.lora.manager import get_lora_manager
from dynamo.common.lora.once import OnceLock
from dynamo.common.lora.runtime import (
    ResolveContext,
    ResolvedLoRA,
    RuntimeLoRAConfigurationError,
    RuntimeLoRAPluginError,
    RuntimeLoRAResolverChain,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def fresh_singleton(monkeypatch):
    """Reset the module-level OnceLock so each test starts with no cached manager."""
    monkeypatch.setattr(manager_module, "_lora_manager", OnceLock())


@pytest.mark.parametrize(
    "source", ["explicit", "DYN_LORA_PATH", "HOME", "USERPROFILE", "fallback"]
)
def test_cache_root_matches_downloader_without_home_lookup(
    monkeypatch, tmp_path, source
):
    for name in ("DYN_LORA_PATH", "HOME", "USERPROFILE"):
        monkeypatch.delenv(name, raising=False)
    if source in ("DYN_LORA_PATH", "HOME", "USERPROFILE"):
        monkeypatch.setenv(source, str(tmp_path))

    def unavailable_home():
        raise RuntimeError("home directory unavailable")

    monkeypatch.setattr(Path, "home", unavailable_home)
    monkeypatch.setattr(manager_module, "LoRADownloader", lambda path: object())
    manager = manager_module.LoRAManager(tmp_path if source == "explicit" else None)
    expected = (
        tmp_path
        if source in ("explicit", "DYN_LORA_PATH")
        else (Path("/tmp") if source == "fallback" else tmp_path)
        / ".cache"
        / "dynamo_loras"
    )
    assert manager.cache_root == expected


class TestGetLoraManagerSingleton:
    def test_returns_none_when_disabled(self, fresh_singleton, monkeypatch):
        monkeypatch.delenv("DYN_LORA_ENABLED", raising=False)
        assert get_lora_manager() is None

    def test_returns_none_when_env_value_not_truthy(self, fresh_singleton, monkeypatch):
        monkeypatch.setenv("DYN_LORA_ENABLED", "false")
        assert get_lora_manager() is None

    def test_returns_same_instance_on_subsequent_calls(
        self, fresh_singleton, monkeypatch
    ):
        instance_count = 0

        class CountingLoRAManager:
            def __init__(self, cache_path=None):
                nonlocal instance_count
                instance_count += 1

        monkeypatch.setattr(manager_module, "LoRAManager", CountingLoRAManager)
        monkeypatch.setenv("DYN_LORA_ENABLED", "true")

        first = get_lora_manager()
        second = get_lora_manager()
        assert first is not None
        assert first is second
        assert instance_count == 1

    def test_init_failure_propagates_and_allows_retry(
        self, fresh_singleton, monkeypatch
    ):
        """Init exceptions bubble up to the caller. OnceLock does not cache
        failures, so a subsequent call retries and can succeed."""
        attempts = 0

        class FlakyLoRAManager:
            def __init__(self, cache_path=None):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("first attempt fails")

        monkeypatch.setattr(manager_module, "LoRAManager", FlakyLoRAManager)
        monkeypatch.setenv("DYN_LORA_ENABLED", "true")

        with pytest.raises(RuntimeError, match="first attempt fails"):
            get_lora_manager()
        assert attempts == 1

        second = get_lora_manager()
        assert second is not None
        assert isinstance(second, FlakyLoRAManager)
        assert attempts == 2

    def test_explicit_manager_ignores_runtime_plugin_configuration(
        self, fresh_singleton, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("DYN_LORA_PATH", str(tmp_path))
        monkeypatch.setattr(manager_module, "LoRADownloader", lambda path: object())
        monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
        monkeypatch.setenv("DYN_LORA_ENABLED", "true")
        monkeypatch.delenv("DYN_LORA_DOWNLOADER_PLUGIN", raising=False)

        manager = get_lora_manager()

        assert manager is not None
        assert manager.runtime_lora_schemes == frozenset()

    def test_runtime_loading_requires_master_lora_switch(
        self, fresh_singleton, monkeypatch
    ):
        monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
        monkeypatch.delenv("DYN_LORA_ENABLED", raising=False)

        with pytest.raises(
            RuntimeLoRAConfigurationError,
            match="requires DYN_LORA_ENABLED",
        ):
            get_lora_manager(configure_runtime=True)

    def test_runtime_loading_imports_and_registers_plugin(
        self, fresh_singleton, monkeypatch, tmp_path
    ):
        module = types.ModuleType("test_manager_runtime_plugin")

        class Resolver:
            protocol_version = 2
            schemes = frozenset({"wandb-artifact"})

            async def resolve(self, *, source_uri, context):
                return None

        module.resolver = Resolver()
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setenv("DYN_LORA_ENABLED", "true")
        monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
        monkeypatch.setenv(
            "DYN_LORA_DOWNLOADER_PLUGIN",
            f"{module.__name__}:resolver",
        )
        monkeypatch.setenv("DYN_LORA_ALLOWED_SCHEMES", "wandb-artifact,s3")
        monkeypatch.setenv("DYN_LORA_PATH", str(tmp_path))

        manager = get_lora_manager(configure_runtime=True)

        assert manager is not None
        assert manager.runtime_lora_schemes == frozenset({"wandb-artifact"})


@pytest.mark.parametrize("cache_path", ["", "relative/dir"])
def test_runtime_cache_requires_absolute_path(
    fresh_singleton, monkeypatch, tmp_path, cache_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_PATH", cache_path)
    monkeypatch.setattr(manager_module, "LoRADownloader", lambda path: object())

    assert get_lora_manager() is not None
    with pytest.raises(RuntimeLoRAConfigurationError, match="absolute"):
        get_lora_manager(configure_runtime=True)


async def _resolve_snapshot(monkeypatch, cache_root, snapshot, context_root=None):
    class FixedResolver:
        protocol_version = 2
        schemes = frozenset({"wandb-artifact"})

        async def resolve(self, *, source_uri, context):
            assert context.cache_root == cache_root
            return ResolvedLoRA(snapshot, "immutable-revision")

    monkeypatch.setattr(manager_module, "LoRADownloader", lambda path: object())
    manager = manager_module.LoRAManager(cache_root)
    manager.configure_runtime_resolvers(
        RuntimeLoRAResolverChain([FixedResolver()], {"wandb-artifact"})
    )
    return await manager.resolve_runtime_lora(
        source_uri="wandb-artifact:///test/adapter:v1",
        context=ResolveContext(
            adapter_key="dyn-lora-0123456789abcdef0123456789abcdef",
            base_model_name="base",
            cache_root=context_root or cache_root,
            deadline_monotonic=asyncio.get_running_loop().time() + 1,
            max_download_bytes=1024,
            request_id="request-id",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["root", "relative", "symlink", "outside", "missing", "file", "loop"]
)
async def test_runtime_snapshot_rejects_invalid_root(monkeypatch, tmp_path, kind):
    root = tmp_path / "cache"
    root.mkdir()
    snapshot = root / "snapshot"
    snapshot.mkdir()
    if kind == "root":
        snapshot = root
    elif kind == "relative":
        monkeypatch.chdir(root)
        snapshot = Path("snapshot")
    elif kind == "symlink":
        link = root / "linked"
        link.symlink_to(snapshot, target_is_directory=True)
        snapshot = link
    elif kind == "outside":
        snapshot = tmp_path
    elif kind == "missing":
        snapshot = root / "missing"
    elif kind == "file":
        snapshot = root / "file"
        snapshot.write_bytes(b"not a directory")
    elif kind == "loop":
        loop = root / "loop"
        loop.symlink_to(loop)
        snapshot = loop / "snapshot"

    with pytest.raises(RuntimeLoRAPluginError) as error:
        await _resolve_snapshot(monkeypatch, root, snapshot)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
    assert str(tmp_path) not in str(error.value)


@pytest.mark.asyncio
async def test_runtime_snapshot_returns_canonical_path(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    result = await _resolve_snapshot(monkeypatch, tmp_path, snapshot / ".")

    assert result.local_path == snapshot.resolve()


@pytest.mark.asyncio
async def test_runtime_snapshot_uses_manager_cache_root(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    result = await _resolve_snapshot(
        monkeypatch, tmp_path, snapshot, context_root=tmp_path / "other"
    )

    assert result.local_path == snapshot


@pytest.mark.asyncio
async def test_runtime_snapshot_sanitizes_permission_errors(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    original = Path.is_dir

    def deny_stat(path):
        if path == snapshot:
            raise PermissionError(f"private snapshot path: {path}")
        return original(path)

    monkeypatch.setattr(Path, "is_dir", deny_stat)
    with pytest.raises(RuntimeLoRAPluginError) as error:
        await _resolve_snapshot(monkeypatch, tmp_path, snapshot)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
    assert str(tmp_path) not in str(error.value)


class TestGetLoraManagerConcurrency:
    def test_no_race_under_concurrent_init(self, fresh_singleton, monkeypatch):
        """Without the OnceLock adoption multiple worker threads would each
        observe the singleton as None and construct their own LoRAManager."""
        num_threads = 50
        instance_count = 0
        count_lock = threading.Lock()
        barrier = threading.Barrier(num_threads)

        class CountingLoRAManager:
            def __init__(self, cache_path=None):
                nonlocal instance_count
                with count_lock:
                    instance_count += 1
                time.sleep(0.01)  # widen race window

        monkeypatch.setattr(manager_module, "LoRAManager", CountingLoRAManager)
        monkeypatch.setenv("DYN_LORA_ENABLED", "true")

        results: list[object] = [None] * num_threads
        errors: list[Exception | None] = [None] * num_threads

        def worker(idx: int) -> None:
            try:
                barrier.wait()  # release all threads at once
                results[idx] = get_lora_manager()
            except Exception as exc:
                errors[idx] = exc

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for idx, exc in enumerate(errors):
            if exc is not None:
                raise AssertionError(f"worker {idx} raised") from exc

        assert instance_count == 1, (
            f"expected 1 LoRAManager instance, got {instance_count} — "
            "race condition not fixed"
        )
        assert all(r is results[0] for r in results)
        assert all(isinstance(r, CountingLoRAManager) for r in results)


def test_is_cached_passes_source_uri_to_rust_downloader(monkeypatch):
    seen: list[str] = []

    class RecordingDownloader:
        def __init__(self, cache_path=None):
            pass

        def is_cached(self, lora_uri: str) -> bool:
            seen.append(lora_uri)
            return True

    monkeypatch.setattr(manager_module, "LoRADownloader", RecordingDownloader)
    manager = manager_module.LoRAManager()

    assert manager.is_cached("hf://org/adapter")
    assert seen == ["hf://org/adapter"]
