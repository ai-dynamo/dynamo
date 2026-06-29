# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GMS TRT-LLM connector persistent-KV allocation hook.

The patched TRT-LLM engine calls ``allocate_kv_caches`` before creating its
primary KV pool. In GMS mode this must allocate inside
``gms_use_persistent_pool`` so GMS owns the HBM pages. There is intentionally
no engine-owned ``torch.empty`` fallback in GMS mode; if the persistent path is
unavailable the engine should fail startup rather than silently losing restart
reuse semantics.

These tests run without tensorrt_llm installed. The connector falls back to stub
bases, and the persistent allocator client is monkeypatched so no real daemon
or CUDA device is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from gpu_memory_service.integrations.trtllm.gms_connector import GMSTrtllmKvCacheWorker

torch = pytest.importorskip("torch")


def _make_llm_args(block_size=4):
    return SimpleNamespace(
        kv_cache_config=SimpleNamespace(tokens_per_block=block_size),
    )


@pytest.fixture
def fake_device():
    """A torch.device-like value with no CUDA dependency for unit tests."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def _clear_vmm_env(monkeypatch):
    for name in (
        "GMS_TRTLLM_VMM_IPC",
        "GMS_TRTLLM_VMM_IPC_KV",
        "GMS_TRTLLM_VMM_IPC_SOCKET",
        "GMS_TRTLLM_VMM_IPC_ENGINE_ID",
        "GMS_TRTLLM_ENGINE_ID",
        "DYN_GMS_ENGINE_ID",
        "DYN_ENGINE_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def test_allocate_routes_through_persistent_pool_by_default(
    monkeypatch,
    fake_device,
):
    monkeypatch.setenv("GMS_TRTLLM_VMM_IPC_SOCKET", "/tmp/gms-fake.sock")
    monkeypatch.setenv("GMS_TRTLLM_VMM_IPC_ENGINE_ID", "stable-trt-engine")

    captured: dict = {}

    def fake_get_or_create(socket, dev_idx, engine_id, *, tag, shared=False):
        captured["register"] = (socket, dev_idx, engine_id, tag, shared)

    from contextlib import contextmanager

    @contextmanager
    def fake_use_pool(tag, dev_idx):
        captured["scope"] = (tag, dev_idx)
        yield

    import gpu_memory_service.client.torch.allocator as alloc_mod

    monkeypatch.setattr(
        alloc_mod, "get_or_create_persistent_allocator", fake_get_or_create
    )
    monkeypatch.setattr(alloc_mod, "gms_use_persistent_pool", fake_use_pool)

    worker = GMSTrtllmKvCacheWorker(_make_llm_args())
    out = worker.allocate_kv_caches(
        (8, 4, 2, 64),
        torch.float16,
        fake_device,
    )

    assert tuple(out.shape) == (8, 4, 2, 64)
    assert out.dtype == torch.float16
    assert worker._vmm_ipc_enabled is True
    assert captured["register"] == (
        "/tmp/gms-fake.sock",
        0,
        "stable-trt-engine",
        "kv_pool",
        False,
    )
    assert captured["scope"] == ("kv_pool", 0)


def test_allocate_raises_on_register_failure(monkeypatch, fake_device):
    monkeypatch.setenv("GMS_TRTLLM_VMM_IPC_SOCKET", "/tmp/gms-fake.sock")

    def fake_get_or_create(*a, **kw):
        raise ConnectionError("simulated daemon unreachable")

    import gpu_memory_service.client.torch.allocator as alloc_mod

    monkeypatch.setattr(
        alloc_mod, "get_or_create_persistent_allocator", fake_get_or_create
    )

    worker = GMSTrtllmKvCacheWorker(_make_llm_args())
    with pytest.raises(RuntimeError, match="persistent KV allocation failed"):
        worker.allocate_kv_caches((4, 2, 2, 16), torch.float16, fake_device)


def test_allocate_raises_on_pool_scope_failure(monkeypatch, fake_device):
    monkeypatch.setenv("GMS_TRTLLM_VMM_IPC_SOCKET", "/tmp/gms-fake.sock")

    def fake_get_or_create(*a, **kw):
        return None

    from contextlib import contextmanager

    @contextmanager
    def fake_use_pool_raising(tag, dev_idx):
        raise RuntimeError("simulated daemon-side error")
        yield  # pragma: no cover

    import gpu_memory_service.client.torch.allocator as alloc_mod

    monkeypatch.setattr(
        alloc_mod, "get_or_create_persistent_allocator", fake_get_or_create
    )
    monkeypatch.setattr(alloc_mod, "gms_use_persistent_pool", fake_use_pool_raising)

    worker = GMSTrtllmKvCacheWorker(_make_llm_args())
    with pytest.raises(RuntimeError, match="persistent KV allocation failed"):
        worker.allocate_kv_caches((4, 2, 2, 16), torch.float16, fake_device)


def test_trtllm_install_fails_without_engine_hook(monkeypatch):
    import gpu_memory_service.integrations.trtllm.install_vmm_ipc_kv as installer

    monkeypatch.setattr(installer, "_patched_trtllm_has_allocate_hook", lambda: False)
    with pytest.raises(RuntimeError, match="allocate_kv_caches engine patch"):
        installer.install()
