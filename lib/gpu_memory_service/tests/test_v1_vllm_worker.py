# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]


@pytest.fixture(scope="module")
def vllm_modules():
    pytest.importorskip("vllm.device_allocator.sleep_mode_backend")
    pytest.importorskip("vllm.model_executor.model_loader.base_loader")
    pytest.importorskip("vllm.v1.worker.gpu_worker")
    pytest.importorskip("vllm.v1.worker.workspace")
    backend = importlib.import_module("gpu_memory_service.v1.integrations.vllm.backend")
    patches = importlib.import_module("gpu_memory_service.v1.integrations.vllm.patches")
    worker = importlib.import_module("gpu_memory_service.v1.integrations.vllm.worker")
    return backend, patches, worker


def test_worker_installs_loader_patch_and_leaves_vllm_scopes_default(
    vllm_modules,
    monkeypatch,
) -> None:
    backend, _patches, worker_module = vllm_modules
    events = []
    workspace = object()
    client = SimpleNamespace(close=lambda: events.append("client_close"))
    manager = object()
    pool = object()
    runtime = SimpleNamespace(manager=manager, pool=pool)

    def upstream_init(instance) -> None:
        events.append("upstream_init")
        instance.device = SimpleNamespace(index=3)

    monkeypatch.setattr(worker_module.Worker, "init_device", upstream_init)
    monkeypatch.setattr(worker_module, "current_workspace_manager", lambda: workspace)
    monkeypatch.setattr(
        worker_module, "get_socket_path", lambda device, tag: f"/{device}/{tag}"
    )
    monkeypatch.setattr(
        worker_module,
        "AllocationClient",
        lambda path: events.append(("client", path)) or client,
    )
    monkeypatch.setattr(worker_module, "get_vmm", lambda: "vmm")
    monkeypatch.setattr(
        worker_module,
        "SnapshotMemoryManager",
        lambda received_client, vmm, device: (
            events.append(("manager", received_client, vmm, device)) or manager
        ),
    )
    monkeypatch.setattr(
        worker_module,
        "SnapshotTorchPool",
        lambda received: events.append(("pool", received)) or pool,
    )
    monkeypatch.setattr(
        worker_module,
        "install_vllm_integration",
        lambda received_workspace, received_pool: events.append(
            ("install", received_workspace, received_pool)
        ),
    )
    monkeypatch.setattr(
        worker_module,
        "install_runtime",
        lambda received_manager, received_pool: (
            events.append(("runtime", received_manager, received_pool)) or runtime
        ),
    )

    worker = object.__new__(worker_module.GMSV1Worker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_sleep_mode=True,
            sleep_mode_backend="cumem",
        )
    )
    worker.init_device()

    assert worker.vllm_config.model_config.sleep_mode_backend == backend.BACKEND_NAME
    assert worker._maybe_get_memory_pool_context("weights").__enter__() is None
    assert worker._maybe_get_memory_pool_context("kv_cache").__enter__() is None
    assert events == [
        "upstream_init",
        ("client", "/3/snapshot-v1"),
        ("manager", client, "vmm", 3),
        ("pool", manager),
        ("install", workspace, pool),
        ("runtime", manager, pool),
    ]


def test_loader_only_uses_gms_and_workspace_growth_uses_native_pool(
    vllm_modules,
    monkeypatch,
) -> None:
    _backend, patches, _worker = vllm_modules
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    events = []
    model = torch.nn.Module()

    @contextmanager
    def scope(name):
        events.append(f"{name}_enter")
        yield
        events.append(f"{name}_exit")

    pool = SimpleNamespace(
        model_load_pool=lambda: scope("gms"),
        native_workspace_pool=lambda: scope("native"),
        finalize_model_load=lambda received: events.append(
            ("finalize", received, list(events))
        ),
        abort_model_load=lambda cause: events.append(("abort", cause)),
    )
    workspace = SimpleNamespace(
        _current_workspaces=[None],
        _workspace_size_bytes=lambda current: 0 if current is None else current,
    )

    def workspace_growth(required_bytes):
        events.append(("workspace_growth", required_bytes))
        workspace._current_workspaces[0] = required_bytes
        return required_bytes

    workspace._ensure_workspace_size = workspace_growth
    original = BaseModelLoader.load_model

    def normal_loader(_loader, *args, **kwargs):
        events.append("base_loader")
        return model

    monkeypatch.setattr(BaseModelLoader, "load_model", normal_loader)
    monkeypatch.setattr("vllm.v1.worker.workspace.dbo_current_ubatch_id", lambda: 0)
    try:
        patches.install_vllm_integration(workspace, pool)
        assert BaseModelLoader.load_model(object()) is model
        events.append("v2_post_loader_runtime")
        assert workspace._ensure_workspace_size(4096) == 4096
    finally:
        BaseModelLoader.load_model = original

    assert events[:5] == [
        "gms_enter",
        "base_loader",
        "gms_exit",
        ("finalize", model, ["gms_enter", "base_loader", "gms_exit"]),
        "v2_post_loader_runtime",
    ]
    assert events[5:] == [
        "native_enter",
        ("workspace_growth", 4096),
        "native_exit",
    ]


def test_backend_orders_pool_sleep_before_reconnect_wake(
    vllm_modules,
    monkeypatch,
) -> None:
    backend, _patches, _worker_module = vllm_modules
    events = []
    runtime = SimpleNamespace(
        pool=SimpleNamespace(prepare_snapshot=lambda: events.append("sleep")),
        manager=SimpleNamespace(wake=lambda: events.append("reconnect_wake")),
    )
    monkeypatch.setattr(backend, "current_runtime", lambda: runtime)

    instance = backend.GMSV1SleepModeBackend()
    instance.suspend()
    instance.resume()

    assert events == ["sleep", "reconnect_wake"]
    assert instance.state() == "RUNNING"


def test_worker_rejects_partial_lifecycle_and_exits_on_transition_failure(
    vllm_modules,
    monkeypatch,
) -> None:
    _backend, _patches, worker_module = vllm_modules
    events = []

    def fail_sleep(_instance, level=1):
        events.append(("sleep", level))
        raise RuntimeError("partial suspend")

    def fail_wake(_instance, tags=None):
        events.append(("wake_up", tags))
        raise RuntimeError("partial resume")

    monkeypatch.setattr(worker_module.Worker, "sleep", fail_sleep)
    monkeypatch.setattr(worker_module.Worker, "wake_up", fail_wake)
    worker = object.__new__(worker_module.GMSV1Worker)

    with pytest.raises(ValueError, match="whole-engine"):
        worker.sleep(2)
    with pytest.raises(ValueError, match="partial-tag"):
        worker.wake_up(["weights"])
    assert events == []

    with pytest.raises(SystemExit, match="1"):
        worker.sleep()
    with pytest.raises(SystemExit, match="1"):
        worker.wake_up()
    assert events == [("sleep", 1), ("wake_up", None)]
