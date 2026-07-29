# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def test_worker_routes_managed_scopes_and_backend_orders_lifecycle(
    monkeypatch,
) -> None:
    pytest.importorskip("vllm.device_allocator.sleep_mode_backend")
    pytest.importorskip("vllm.v1.worker.gpu_worker")
    backend_module = importlib.import_module(
        "gpu_memory_service.v1.integrations.vllm.backend"
    )
    worker_module = importlib.import_module(
        "gpu_memory_service.v1.integrations.vllm.worker"
    )
    events = []
    final_model = object()

    class RoutedBackend:
        @contextmanager
        def capture_weights(self, model):
            events.append("weights_enter")
            yield
            events.append(("weights_exit", model()))

        @contextmanager
        def capture_kv_cache(self):
            events.append("kv_enter")
            yield
            events.append("kv_exit")

    routed_backend = RoutedBackend()

    def init_device(instance):
        events.append("vllm_init")
        instance.model_runner = SimpleNamespace(
            model=None,
            get_model=lambda: instance.model_runner.model,
        )

    monkeypatch.setattr(worker_module.Worker, "init_device", init_device)
    monkeypatch.setattr(
        worker_module.Worker,
        "_get_sleep_mode_backend",
        lambda _instance: routed_backend,
    )
    monkeypatch.setattr(
        worker_module.Worker,
        "_maybe_get_memory_pool_context",
        lambda _instance, tag: events.append(("super", tag)) or nullcontext(),
    )

    worker = object.__new__(worker_module.GMSV1Worker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_sleep_mode=True,
            sleep_mode_backend="cumem",
        )
    )
    worker.init_device()
    with worker._maybe_get_memory_pool_context("weights"):
        worker.model_runner.model = final_model
    with worker._maybe_get_memory_pool_context("kv_cache"):
        pass
    with worker._maybe_get_memory_pool_context("activation"):
        pass

    backend = object.__new__(backend_module.GMSV1SleepModeBackend)
    backend_module.SleepModeBackend.__init__(backend)
    backend._device = 0
    info_messages = []
    monkeypatch.setattr(
        backend_module.logger,
        "info",
        lambda message, *_args: info_messages.append(message),
    )
    monkeypatch.setattr(backend_module.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(
        backend_module.torch.cuda,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    backend._raise_if_allocator_failed = lambda: events.append("allocator_ok")
    backend._weights = SimpleNamespace(
        unmap_all_vas=lambda: events.append("weights_unmap"),
        disconnect=lambda: events.append("weights_disconnect"),
        connect=lambda mode: events.append(("weights_connect", mode.value)),
        remap_all_vas=lambda: events.append("weights_remap"),
    )
    backend._kv_cache = SimpleNamespace(
        unmap_all_vas=lambda: events.append("kv_unmap"),
        disconnect=lambda: events.append("kv_disconnect"),
        connect=lambda mode: events.append(("kv_connect", mode.value)),
        reallocate_all_handles=lambda: events.append("kv_reallocate"),
        remap_all_vas=lambda: events.append("kv_remap"),
    )
    backend.suspend()
    backend.resume()

    assert worker.vllm_config.model_config.sleep_mode_backend == (
        backend_module.BACKEND_NAME
    )
    assert events == [
        "vllm_init",
        "weights_enter",
        ("weights_exit", final_model),
        "kv_enter",
        "kv_exit",
        ("super", "activation"),
        "gc",
        "allocator_ok",
        "weights_unmap",
        "weights_disconnect",
        "kv_unmap",
        "kv_disconnect",
        "empty_cache",
        ("kv_connect", "rw"),
        "kv_reallocate",
        "kv_remap",
        ("weights_connect", "ro"),
        "weights_remap",
    ]
    assert backend.state() == "RUNNING"
    assert info_messages == [
        "GMS V1 KV wake device=%d connect_elapsed=%.3fs "
        "reallocate_elapsed=%.3fs remap_elapsed=%.3fs total_elapsed=%.3fs",
        "GMS V1 weights wake device=%d connect_elapsed=%.3fs "
        "remap_elapsed=%.3fs total_elapsed=%.3fs",
        "GMS V1 wake complete device=%d total_elapsed=%.3fs",
    ]
