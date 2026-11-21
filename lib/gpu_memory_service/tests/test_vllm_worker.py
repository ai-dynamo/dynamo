# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip("gpu_memory_service is required", allow_module_level=True)
if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)
pytest.importorskip("vllm")

from gpu_memory_service.integrations.vllm import worker as gms_worker  # noqa: E402
from gpu_memory_service.v1.integrations.vllm import (  # noqa: E402
    worker as gms_v1_worker,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def test_init_device_forwards_ro_connect_timeout(monkeypatch, tmp_path):
    manager_factory = Mock()
    monkeypatch.setattr(
        gms_worker, "get_or_create_gms_client_memory_manager", manager_factory
    )
    monkeypatch.setattr(gms_worker, "_get_dp_adjusted_local_rank", lambda *_: 0)
    monkeypatch.setattr(
        gms_worker, "get_vmm_device_type", lambda: SimpleNamespace(value="cuda")
    )
    monkeypatch.setattr(
        gms_worker, "get_socket_path", lambda *_: str(tmp_path / "weights.sock")
    )
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(set_device=lambda _: None),
    )
    monkeypatch.setattr(gms_worker._BaseWorker, "init_device", lambda _: None)

    worker = gms_worker.GMSWorker.__new__(gms_worker.GMSWorker)
    worker.local_rank = 0
    worker.parallel_config = None
    worker.vllm_config = SimpleNamespace(
        load_config=SimpleNamespace(
            model_loader_extra_config={
                "gms_read_only": True,
                "gms_ro_connect_timeout_ms": 4200,
            }
        )
    )

    worker.init_device()

    assert manager_factory.call_args.kwargs["timeout_ms"] == 4200


def test_v1_worker_uses_upstream_sleep_backend_accessor(monkeypatch):
    monkeypatch.setattr(gms_v1_worker.Worker, "init_device", lambda _: None)
    backend = Mock()
    worker = gms_v1_worker.GMSV1Worker.__new__(gms_v1_worker.GMSV1Worker)
    worker._sleep_mode_backend = backend
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=True)
    )
    worker.model_runner = SimpleNamespace(get_model=Mock())

    worker.init_device()

    assert (
        worker.vllm_config.model_config.sleep_mode_backend == gms_v1_worker.BACKEND_NAME
    )
    assert (
        worker._maybe_get_memory_pool_context("weights")
        is backend.capture_weights.return_value
    )
    backend.capture_weights.assert_called_once_with(worker.model_runner.get_model)
    assert (
        worker._maybe_get_memory_pool_context("kv_cache")
        is backend.capture_kv_cache.return_value
    )


_GIB = 1 << 30
# vLLM's budget for an engine whose GMS weights it never charged.
_VLLM_AVAILABLE = 144 * _GIB
_IMPORTED_WEIGHTS = 51 * _GIB


def _default_kv_path_worker(
    monkeypatch,
    *,
    pending_write: bool,
    imported_weights: int = _IMPORTED_WEIGHTS,
    kv_cache_memory_bytes=None,
    vllm_available: int = _VLLM_AVAILABLE,
):
    """A GMSWorker on the default (non-scratch) KV path with vLLM faked out."""
    pending = {"write": pending_write}
    publish = Mock(side_effect=lambda: pending.update(write=False))
    abort = Mock(side_effect=lambda: pending.update(write=False))

    def fake_vllm_determine_available_memory(self):
        self.available_kv_cache_memory_bytes = vllm_available
        return vllm_available

    monkeypatch.setattr(gms_worker, "is_scratch_kv_enabled", lambda: False)
    monkeypatch.setattr(gms_worker, "has_pending_gms_write", lambda: pending["write"])
    monkeypatch.setattr(
        gms_worker, "get_imported_weights_bytes", lambda: imported_weights
    )
    monkeypatch.setattr(gms_worker, "publish_pending_gms_write", publish)
    monkeypatch.setattr(gms_worker, "abort_pending_gms_write", abort)
    monkeypatch.setattr(
        gms_worker._BaseWorker,
        "determine_available_memory",
        fake_vllm_determine_available_memory,
    )

    worker = gms_worker.GMSWorker.__new__(gms_worker.GMSWorker)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=kv_cache_memory_bytes)
    worker.requested_memory = 148 * _GIB
    return worker, publish, abort


def test_default_kv_path_charges_read_only_imported_weights(monkeypatch):
    worker, publish, _ = _default_kv_path_worker(monkeypatch, pending_write=False)

    available = worker.determine_available_memory()

    assert available == _VLLM_AVAILABLE - _IMPORTED_WEIGHTS
    assert worker.available_kv_cache_memory_bytes == available
    publish.assert_called_once_with()


def test_default_kv_path_does_not_double_charge_pending_writer(monkeypatch):
    # vLLM already charged the writer's weights: they were allocated after its
    # initial memory snapshot. Publication only happens after sizing.
    worker, publish, _ = _default_kv_path_worker(monkeypatch, pending_write=True)

    available = worker.determine_available_memory()

    assert available == _VLLM_AVAILABLE
    assert worker.available_kv_cache_memory_bytes == _VLLM_AVAILABLE
    publish.assert_called_once_with()


def test_default_kv_path_keeps_explicit_kv_cache_memory_bytes(monkeypatch):
    explicit = 64 * _GIB
    worker, _, _ = _default_kv_path_worker(
        monkeypatch,
        pending_write=False,
        kv_cache_memory_bytes=explicit,
        vllm_available=explicit,
    )

    assert worker.determine_available_memory() == explicit


def test_default_kv_path_unchanged_without_imported_weights(monkeypatch):
    worker, _, _ = _default_kv_path_worker(
        monkeypatch, pending_write=False, imported_weights=0
    )

    assert worker.determine_available_memory() == _VLLM_AVAILABLE
    assert worker.available_kv_cache_memory_bytes == _VLLM_AVAILABLE


def test_default_kv_path_raises_when_imported_weights_leave_no_budget(monkeypatch):
    worker, publish, abort = _default_kv_path_worker(
        monkeypatch, pending_write=False, vllm_available=_IMPORTED_WEIGHTS
    )

    with pytest.raises(ValueError, match="leave no KV cache memory"):
        worker.determine_available_memory()

    abort.assert_called_once_with()
    publish.assert_not_called()
