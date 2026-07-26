# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.fixture(scope="module")
def vllm_modules():
    pytest.importorskip("vllm.device_allocator.sleep_mode_backend")
    pytest.importorskip("vllm.v1.worker.gpu_worker")
    backend = importlib.import_module("gpu_memory_service.v1.integrations.vllm.backend")
    worker = importlib.import_module("gpu_memory_service.v1.integrations.vllm.worker")
    return backend, worker


def test_worker_routes_weights_and_kv_cache_scopes_to_gms(
    vllm_modules,
    monkeypatch,
) -> None:
    backend_module, worker_module = vllm_modules
    events = []
    final_model = object()

    class ModelRunner:
        model = None

        def get_model(self):
            events.append(("get_model", self.model))
            return self.model

    class Backend:
        @contextmanager
        def capture_weights(self, model):
            events.append(("gms_enter", "weights"))
            yield
            events.append(("gms_exit", "weights", model()))

        @contextmanager
        def capture_kv_cache(self):
            events.append(("gms_enter", "kv_cache"))
            yield
            events.append(("gms_exit", "kv_cache"))

    selected_backend = Backend()

    def init_device(instance):
        events.append("vllm_init")
        instance.model_runner = ModelRunner()

    monkeypatch.setattr(worker_module.Worker, "init_device", init_device)
    monkeypatch.setattr(
        worker_module.Worker,
        "_get_sleep_mode_backend",
        lambda _instance: events.append("get_backend") or selected_backend,
    )
    monkeypatch.setattr(
        worker_module.Worker,
        "_maybe_get_memory_pool_context",
        lambda _instance, tag: pytest.fail(f"native vLLM pool used for {tag}"),
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
        events.append("load_model")
        worker.model_runner.model = final_model
    with worker._maybe_get_memory_pool_context("kv_cache"):
        events.append("allocate_kv")

    assert worker.vllm_config.model_config.sleep_mode_backend == (
        backend_module.BACKEND_NAME
    )
    assert events == [
        "vllm_init",
        "get_backend",
        "get_backend",
        ("gms_enter", "weights"),
        "load_model",
        ("get_model", final_model),
        ("gms_exit", "weights", final_model),
        "get_backend",
        ("gms_enter", "kv_cache"),
        "allocate_kv",
        ("gms_exit", "kv_cache"),
    ]


def test_backend_orders_gms_lifecycle_without_native_cumem(
    vllm_modules,
    monkeypatch,
) -> None:
    backend_module, _worker_module = vllm_modules
    events = []

    def native_allocator():
        pytest.fail("GMS V1 invoked vLLM's native CuMem allocator")

    monkeypatch.setattr(
        "vllm.device_allocator.get_mem_allocator_instance",
        native_allocator,
    )

    backend = object.__new__(backend_module.GMSV1SleepModeBackend)
    backend_module.SleepModeBackend.__init__(backend)
    backend._pools = SimpleNamespace(
        raise_if_allocator_failed=lambda: events.append("allocator_ok"),
        weights=SimpleNamespace(
            sleep=lambda: events.append("weights_sleep"),
            wake=lambda: events.append("weights_wake"),
        ),
        kv_cache=SimpleNamespace(
            sleep=lambda: events.append("kv_sleep"),
            wake=lambda: events.append("kv_wake"),
        ),
    )

    backend.suspend()
    backend.resume()

    assert events == [
        "allocator_ok",
        "weights_sleep",
        "kv_sleep",
        "kv_wake",
        "weights_wake",
    ]
    assert backend.state() == "RUNNING"
