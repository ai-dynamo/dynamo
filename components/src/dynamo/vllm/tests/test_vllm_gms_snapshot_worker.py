# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

pytest.importorskip("vllm.v1.worker.gpu_worker")

from gpu_memory_service.integrations.vllm.worker import GMSWorker  # noqa: E402
from vllm.v1.worker.gpu_worker import Worker  # noqa: E402

from dynamo.vllm.handlers import (  # noqa: E402
    BaseWorkerHandler,
    VllmEnginePauseController,
)
from dynamo.vllm.snapshot_backend import (  # noqa: E402
    DynamoGMSSnapshotBackend,
    GMSKVRestoreError,
    GMSWeightRestoreError,
    SnapshotTerminalError,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _snapshot_worker() -> GMSWorker:
    worker = object.__new__(GMSWorker)
    backend = MagicMock()
    backend.state.return_value = "RUNNING"
    worker._sleep_mode_backend = backend
    worker._get_sleep_mode_backend = MagicMock(return_value=backend)
    worker.model_runner = MagicMock()
    return worker


def test_snapshot_suspend_failure_terminates_worker(monkeypatch):
    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", "/run/dynamo/snapshot")
    worker = _snapshot_worker()
    worker.gms_ro_connect_timeout_ms = 4321

    with (
        patch.object(
            Worker,
            "sleep",
            autospec=True,
            side_effect=RuntimeError("partial unmap"),
        ),
        pytest.raises(SystemExit, match="1"),
    ):
        worker.sleep()

    worker._sleep_mode_backend.set_ro_connect_timeout_ms.assert_called_once_with(4321)


def test_nixl_registration_retries_after_post_wake_failure(monkeypatch):
    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", "/run/dynamo/snapshot")
    worker = _snapshot_worker()
    kv_cache = MagicMock()
    worker._register_kv_caches_with_nixl = MagicMock()

    with (
        patch(
            "gpu_memory_service.integrations.vllm.worker.get_gms_client_memory_manager",
            return_value=kv_cache,
        ),
        patch(
            "gpu_memory_service.integrations.vllm.worker.is_scratch",
            side_effect=[True, False],
        ),
        patch.object(
            Worker,
            "wake_up",
            autospec=True,
            side_effect=[RuntimeError("post wake failed"), None],
        ),
    ):
        with pytest.raises(RuntimeError, match="post wake failed"):
            worker.wake_up(["kv_cache"])

        worker.wake_up(["kv_cache"])

    worker._register_kv_caches_with_nixl.assert_called_once_with()
    assert worker._snapshot_nixl_registration_pending is False


@pytest.mark.asyncio
async def test_pending_nixl_registration_blocks_non_kv_wake(monkeypatch):
    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", "/run/dynamo/snapshot")
    worker = _snapshot_worker()
    kv_cache = MagicMock()
    worker._register_kv_caches_with_nixl = MagicMock(
        side_effect=[RuntimeError("NIXL registration failed"), None]
    )
    parent_wake = MagicMock()

    async def wake_up(tags=None):
        worker.wake_up(tags)

    engine_client = SimpleNamespace(
        wake_up=AsyncMock(side_effect=wake_up),
        resume_generation=AsyncMock(),
    )
    controller = VllmEnginePauseController(engine_client)
    controller._is_paused = True
    controller._generation_paused = True
    endpoint = SimpleNamespace(register_endpoint_instance=AsyncMock())
    handler = SimpleNamespace(
        _pause_controller=controller,
        _pause_lock=asyncio.Lock(),
        generate_endpoint=endpoint,
    )

    with (
        patch(
            "gpu_memory_service.integrations.vllm.worker.get_gms_client_memory_manager",
            return_value=kv_cache,
        ),
        patch(
            "gpu_memory_service.integrations.vllm.worker.is_scratch",
            side_effect=[True, False],
        ),
        patch.object(Worker, "wake_up", autospec=True, side_effect=parent_wake),
    ):
        initial_result = await BaseWorkerHandler.wake_up(
            handler, {"tags": ["kv_cache"]}
        )
        assert initial_result == {
            "status": "error",
            "message": "NIXL registration failed",
        }

        rejected_result = await BaseWorkerHandler.wake_up(
            handler, {"tags": ["weights"]}
        )
        assert rejected_result["status"] == "error"
        assert "pending NIXL KV-cache registration" in rejected_result["message"]

        parent_wake.assert_called_once_with(worker, ["kv_cache"])
        assert worker._register_kv_caches_with_nixl.call_count == 1
        assert worker._snapshot_nixl_registration_pending is True
        engine_client.resume_generation.assert_not_awaited()
        endpoint.register_endpoint_instance.assert_not_awaited()

        retry_result = await BaseWorkerHandler.wake_up(handler, {})

    assert retry_result == {"status": "ok", "message": "Engine woke"}
    assert parent_wake.call_args_list == [
        call(worker, ["kv_cache"]),
        call(worker, None),
    ]
    assert worker._register_kv_caches_with_nixl.call_count == 2
    assert worker._snapshot_nixl_registration_pending is False
    engine_client.resume_generation.assert_awaited_once_with()
    endpoint.register_endpoint_instance.assert_awaited_once_with()


def test_nixl_registration_retries_after_publish_failure(monkeypatch):
    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", "/run/dynamo/snapshot")
    worker = _snapshot_worker()
    kv_cache = MagicMock()
    worker._register_kv_caches_with_nixl = MagicMock(
        side_effect=[RuntimeError("NIXL registration failed"), None]
    )

    with (
        patch(
            "gpu_memory_service.integrations.vllm.worker.get_gms_client_memory_manager",
            return_value=kv_cache,
        ),
        patch(
            "gpu_memory_service.integrations.vllm.worker.is_scratch",
            side_effect=[True, False],
        ),
        patch.object(Worker, "wake_up", autospec=True),
    ):
        with pytest.raises(RuntimeError, match="NIXL registration failed"):
            worker.wake_up(["kv_cache"])

        worker.wake_up(["kv_cache"])

    assert worker._register_kv_caches_with_nixl.call_count == 2
    assert worker._snapshot_nixl_registration_pending is False


def test_partial_kv_reallocation_is_fatal_and_never_published(monkeypatch):
    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", "/run/dynamo/snapshot")
    worker = _snapshot_worker()
    backend = DynamoGMSSnapshotBackend()
    backend._state = "SUSPENDED"
    backend._restored_tags.clear()
    backend._communicators_restored = False
    worker._sleep_mode_backend = backend
    worker._get_sleep_mode_backend = MagicMock(return_value=backend)
    worker._register_kv_caches_with_nixl = MagicMock()

    class PartiallyReallocatedManager:
        is_unmapped = True
        is_connected = False

        def __init__(self):
            self.mapping_allocation_ids = ["old-0", "old-1"]
            self.aborted = False
            self.remapped = False

        def connect(self, lock_type):
            self.is_connected = True

        def prepare_scratch_for_reallocation(self):
            pass

        def reallocate_all_handles(self, *, tag):
            self.mapping_allocation_ids[0] = f"{tag}-new-0"
            raise RuntimeError("second allocation failed")

        def remap_all_vas(self):
            self.remapped = True

        def abort(self):
            self.aborted = True
            self.is_connected = False

    kv_cache = PartiallyReallocatedManager()

    def wake_backend(_worker, tags):
        backend.resume(tags)

    with (
        patch(
            "gpu_memory_service.integrations.vllm.worker.get_gms_client_memory_manager",
            return_value=kv_cache,
        ),
        patch(
            "gpu_memory_service.client.torch.allocator.get_gms_client_memory_manager",
            return_value=kv_cache,
        ),
        patch(
            "gpu_memory_service.integrations.vllm.worker.is_scratch",
            return_value=True,
        ),
        patch(
            "gpu_memory_service.client.torch.allocator.is_scratch",
            return_value=True,
        ),
        patch.object(Worker, "wake_up", autospec=True, side_effect=wake_backend),
        pytest.raises(SystemExit, match="1"),
    ):
        worker.wake_up(["kv_cache"])

    assert kv_cache.mapping_allocation_ids == ["kv_cache-new-0", "old-1"]
    assert kv_cache.aborted is True
    assert kv_cache.remapped is False
    assert isinstance(backend._fatal_error, GMSKVRestoreError)
    worker._register_kv_caches_with_nixl.assert_not_called()

    with pytest.raises(SnapshotTerminalError, match="unavailable"):
        backend.resume(["kv_cache"])


def test_weight_timeout_cleans_connection_and_exits(monkeypatch):
    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", "/run/dynamo/snapshot")
    worker = _snapshot_worker()
    weights = MagicMock(is_connected=True)
    error = GMSWeightRestoreError("RO lock timeout")
    error.__cause__ = TimeoutError("RO lock timeout")

    with (
        patch(
            "gpu_memory_service.integrations.vllm.worker.get_gms_client_memory_manager",
            return_value=weights,
        ),
        patch.object(
            Worker,
            "wake_up",
            autospec=True,
            side_effect=error,
        ),
        pytest.raises(SystemExit, match="1"),
    ):
        worker.wake_up(["weights"])

    weights.abort.assert_called_once_with()


def test_flashinfer_restore_failure_terminates_worker(monkeypatch):
    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", "/run/dynamo/snapshot")
    worker = _snapshot_worker()

    with (
        patch(
            "gpu_memory_service.integrations.vllm.worker.get_gms_client_memory_manager",
            return_value=None,
        ),
        patch.object(
            Worker,
            "wake_up",
            autospec=True,
            side_effect=SnapshotTerminalError("FlashInfer restore failed"),
        ),
        pytest.raises(SystemExit, match="1"),
    ):
        worker.wake_up()
