# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for GMS vLLM integration patches."""

from __future__ import annotations

import gc
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.integrations.vllm import patches

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_workspace_manager_retains_native_pool_across_resizes(monkeypatch):
    events = []
    ubatch_id = 0

    class Tensor:
        def __init__(self, size):
            self.size = size

    class WorkspaceManager:
        def __init__(self):
            self._device = "cuda:0"
            self._current_workspaces = [None]

        @staticmethod
        def _workspace_size_bytes(workspace):
            return 0 if workspace is None else workspace.size

        def _ensure_workspace_size(self, required_bytes):
            current = self._current_workspaces[ubatch_id]
            if self._workspace_size_bytes(current) < required_bytes:
                events.append(("allocate", required_bytes, active_pools[-1]))
                self._current_workspaces[ubatch_id] = Tensor(required_bytes)
            return self._current_workspaces[ubatch_id]

    workspace_module = ModuleType("vllm.v1.worker.workspace")
    workspace_module.WorkspaceManager = WorkspaceManager
    workspace_module.dbo_current_ubatch_id = lambda: ubatch_id
    worker_module = ModuleType("vllm.v1.worker")
    worker_module.workspace = workspace_module
    v1_module = ModuleType("vllm.v1")
    v1_module.worker = worker_module
    vllm_module = ModuleType("vllm")
    vllm_module.v1 = v1_module
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", worker_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.workspace", workspace_module)

    class Pool:
        next_id = 0

        def __init__(self):
            self.id = Pool.next_id
            Pool.next_id += 1
            events.append(("create", self.id))

        def __del__(self):
            events.append(("destroy", self.id))

    active_pools = []

    @contextmanager
    def use_mem_pool(pool, *, device):
        assert device == "cuda:0"
        active_pools.append(pool.id)
        try:
            yield
        finally:
            active_pools.pop()

    @contextmanager
    def use_device(device):
        assert device == "cuda:0"
        yield

    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(
        MemPool=Pool,
        device=use_device,
        use_mem_pool=use_mem_pool,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setattr(patches, "_workspace_manager_patched", False)

    patches.patch_workspace_manager()
    manager = WorkspaceManager()

    assert manager._ensure_workspace_size(1024).size == 1024
    assert ("allocate", 1024, 0) in events
    assert ("destroy", 0) not in events

    assert manager._ensure_workspace_size(512).size == 1024
    assert [event for event in events if event[0] == "create"] == [("create", 0)]

    assert manager._ensure_workspace_size(2048).size == 2048
    gc.collect()

    assert ("allocate", 2048, 0) in events
    assert [event for event in events if event[0] == "create"] == [("create", 0)]
    assert ("destroy", 0) not in events

    del manager
    gc.collect()
    assert ("destroy", 0) in events


def test_workspace_manager_patch_is_idempotent(monkeypatch):
    class WorkspaceManager:
        def _ensure_workspace_size(self, required_bytes):
            return required_bytes

    workspace_module = ModuleType("vllm.v1.worker.workspace")
    workspace_module.WorkspaceManager = WorkspaceManager
    worker_module = ModuleType("vllm.v1.worker")
    worker_module.workspace = workspace_module
    v1_module = ModuleType("vllm.v1")
    v1_module.worker = worker_module
    vllm_module = ModuleType("vllm")
    vllm_module.v1 = v1_module
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", worker_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.workspace", workspace_module)
    monkeypatch.setattr(patches, "_workspace_manager_patched", False)

    patches.patch_workspace_manager()
    patched = WorkspaceManager._ensure_workspace_size
    patches.patch_workspace_manager()

    assert WorkspaceManager._ensure_workspace_size is patched
