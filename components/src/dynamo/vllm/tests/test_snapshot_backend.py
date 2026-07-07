# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from gpu_memory_service.common.locks import RequestedLockType

pytest.importorskip("vllm.device_allocator.sleep_mode_backend")

import dynamo.vllm.snapshot_backend as snapshot_backend  # noqa: E402
from dynamo.vllm.snapshot_backend import (  # noqa: E402
    GMS_BACKEND_NAME,
    DynamoGMSSnapshotBackend,
    SnapshotRollbackError,
    register_dynamo_gms_snapshot_backend,
    select_dynamo_gms_snapshot_backend,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_registration_is_idempotent():
    factory = snapshot_backend.SleepModeBackendFactory
    assert factory is not None
    factory._registry.pop(GMS_BACKEND_NAME, None)

    register_dynamo_gms_snapshot_backend()
    register_dynamo_gms_snapshot_backend()

    assert factory.get_backend_class(GMS_BACKEND_NAME) is DynamoGMSSnapshotBackend


def test_selection_requires_gms_load_format():
    config = SimpleNamespace(
        model_config=SimpleNamespace(sleep_mode_backend="cumem"),
        load_config=SimpleNamespace(load_format="auto"),
    )

    select_dynamo_gms_snapshot_backend(config)
    assert config.model_config.sleep_mode_backend == "cumem"

    config.load_config.load_format = "gms"
    select_dynamo_gms_snapshot_backend(config)
    assert config.model_config.sleep_mode_backend == GMS_BACKEND_NAME


def test_missing_vllm_integration_only_rejects_gms(monkeypatch):
    compatibility_error = ImportError("missing sleep backend")
    monkeypatch.setattr(
        snapshot_backend, "_SLEEP_MODE_IMPORT_ERROR", compatibility_error
    )
    monkeypatch.setattr(snapshot_backend, "SleepModeBackendFactory", None)
    config = SimpleNamespace(
        model_config=SimpleNamespace(sleep_mode_backend="cumem"),
        load_config=SimpleNamespace(load_format="auto"),
    )

    select_dynamo_gms_snapshot_backend(config)

    config.load_config.load_format = "gms"
    with pytest.raises(RuntimeError, match="exact integration overlay"):
        select_dynamo_gms_snapshot_backend(config)


def test_missing_checkpoint_hooks_has_actionable_error(monkeypatch):
    monkeypatch.delattr(
        "vllm.distributed.parallel_state.checkpoint_prepare_distributed_state",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="matching custom FlashInfer build"):
        snapshot_backend._checkpoint_hooks()


def test_suspend_orders_checkpoint_before_unmap():
    backend = DynamoGMSSnapshotBackend()
    calls = []
    backend._unmap_gms_tags = lambda touched: calls.append("gms_unmap")

    with (
        patch.object(
            snapshot_backend,
            "_checkpoint_hooks",
            return_value=(
                lambda: calls.append("checkpoint_prepare"),
                lambda: calls.append("checkpoint_restore"),
            ),
        ),
        patch("torch.cuda.empty_cache"),
    ):
        backend.suspend()

    assert calls == ["checkpoint_prepare", "gms_unmap"]
    assert backend.state() == "SUSPENDED"


def test_suspend_unmaps_all_canonical_gms_tags():
    backend = DynamoGMSSnapshotBackend()
    weights = MagicMock(is_unmapped=False)
    kv_cache = MagicMock(is_unmapped=False)
    managers = {"weights": weights, "kv_cache": kv_cache}
    touched = []

    with patch(
        "gpu_memory_service.client.torch.allocator.get_gms_client_memory_manager",
        side_effect=managers.__getitem__,
    ):
        backend._unmap_gms_tags(touched)

    assert touched == [("weights", weights), ("kv_cache", kv_cache)]
    for manager in managers.values():
        manager.unmap_all_vas.assert_called_once_with()
        manager.abort.assert_called_once_with()


def test_failed_partial_unmap_remaps_every_touched_manager_fail_closed():
    backend = DynamoGMSSnapshotBackend()
    weights = MagicMock(is_unmapped=False)
    kv_cache = MagicMock(is_unmapped=False)
    unmap_error = RuntimeError("mid-manager unmap")

    def unmap_weights():
        weights.is_unmapped = True

    def partially_unmap_kv_cache():
        kv_cache.is_unmapped = True
        raise unmap_error

    weights.unmap_all_vas.side_effect = unmap_weights
    kv_cache.unmap_all_vas.side_effect = partially_unmap_kv_cache
    managers = {"weights": weights, "kv_cache": kv_cache}

    def force_remap(_tag, *, manager, force_remap):
        assert force_remap is True
        manager.is_unmapped = False

    backend._resume_gms_tag = MagicMock(side_effect=force_remap)
    checkpoint_restore = MagicMock()

    with (
        patch(
            "gpu_memory_service.client.torch.allocator.get_gms_client_memory_manager",
            side_effect=managers.__getitem__,
        ),
        patch.object(
            snapshot_backend,
            "_checkpoint_hooks",
            return_value=(MagicMock(), checkpoint_restore),
        ),
        pytest.raises(RuntimeError, match="mid-manager unmap"),
    ):
        backend.suspend()

    assert backend._resume_gms_tag.call_args_list == [
        call("weights", manager=weights, force_remap=True),
        call("kv_cache", manager=kv_cache, force_remap=True),
    ]
    weights.unmap_all_vas.assert_called_once_with()
    weights.abort.assert_called_once_with()
    kv_cache.unmap_all_vas.assert_called_once_with()
    kv_cache.abort.assert_not_called()
    assert weights.is_unmapped is False
    assert kv_cache.is_unmapped is False
    checkpoint_restore.assert_called_once_with()
    assert backend.state() == "SUSPENDED"
    assert backend._communicators_restored is True
    assert backend._fatal_error is unmap_error

    with pytest.raises(RuntimeError, match="unavailable after a failed suspend"):
        backend.resume()


def test_failed_unmap_and_rollback_raise_transaction_error():
    backend = DynamoGMSSnapshotBackend()
    manager = MagicMock(is_unmapped=False)
    manager.unmap_all_vas.side_effect = RuntimeError("unmap failed")
    backend._resume_gms_tag = MagicMock(side_effect=RuntimeError("remap failed"))
    checkpoint_restore = MagicMock()

    with (
        patch(
            "gpu_memory_service.client.torch.allocator.get_gms_client_memory_manager",
            return_value=manager,
        ),
        patch.object(
            snapshot_backend,
            "_checkpoint_hooks",
            return_value=(MagicMock(), checkpoint_restore),
        ),
        pytest.raises(SnapshotRollbackError, match="rollback error: remap failed"),
    ):
        backend.suspend()

    checkpoint_restore.assert_not_called()
    assert backend.state() == "SUSPENDED"


def test_failed_checkpoint_prepare_rollback_is_fail_closed():
    backend = DynamoGMSSnapshotBackend()

    with (
        patch.object(
            snapshot_backend,
            "_checkpoint_hooks",
            return_value=(
                MagicMock(side_effect=RuntimeError("prepare failed")),
                MagicMock(side_effect=RuntimeError("restore failed")),
            ),
        ),
        pytest.raises(SnapshotRollbackError, match="prepare failed"),
    ):
        backend.suspend()

    assert backend.state() == "SUSPENDED"


def test_partial_wakes_restore_communicators_after_all_tags():
    backend = DynamoGMSSnapshotBackend()
    backend._state = "SUSPENDED"
    backend._restored_tags.clear()
    backend._communicators_restored = False
    calls = []
    backend._resume_gms_tag = lambda tag: calls.append(("gms_resume", tag))

    with patch.object(
        snapshot_backend,
        "_checkpoint_hooks",
        return_value=(
            MagicMock(),
            lambda: calls.append(("checkpoint_restore", None)),
        ),
    ):
        backend.resume(["weights"])
        assert backend.state() == "RESUMING"
        assert calls == [("gms_resume", "weights")]

        backend.resume(["kv_cache"])

    assert calls == [
        ("gms_resume", "weights"),
        ("gms_resume", "kv_cache"),
        ("checkpoint_restore", None),
    ]
    assert backend.state() == "RUNNING"


def test_flashinfer_restore_failure_is_fail_closed():
    backend = DynamoGMSSnapshotBackend()
    backend._state = "SUSPENDED"
    backend._restored_tags.clear()
    backend._communicators_restored = False
    backend._resume_gms_tag = MagicMock()
    checkpoint_restore = MagicMock(side_effect=RuntimeError("reattach failed"))

    with patch.object(
        snapshot_backend,
        "_checkpoint_hooks",
        return_value=(MagicMock(), checkpoint_restore),
    ):
        with pytest.raises(RuntimeError, match="reattach failed"):
            backend.resume()
        assert backend.state() == "SUSPENDED"

        with pytest.raises(RuntimeError, match="unavailable after a failed suspend"):
            backend.resume()

    assert backend._resume_gms_tag.call_args_list == [
        call("weights"),
        call("kv_cache"),
    ]
    checkpoint_restore.assert_called_once_with()
    assert backend.state() == "SUSPENDED"


def test_weight_restore_honors_configured_timeout():
    backend = DynamoGMSSnapshotBackend()
    backend.set_ro_connect_timeout_ms(12345)
    weights = MagicMock(is_unmapped=True, is_connected=False)

    with patch(
        "gpu_memory_service.client.torch.allocator.get_gms_client_memory_manager",
        return_value=weights,
    ):
        backend._resume_gms_tag("weights")

    weights.connect.assert_called_once_with(
        RequestedLockType.RO,
        timeout_ms=12345,
    )
    weights.remap_all_vas.assert_called_once_with()


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("RO lock timeout"),
        ConnectionError("server unavailable"),
        RuntimeError("remap failed"),
    ],
)
def test_weight_restore_failure_cleans_partial_connection(error):
    backend = DynamoGMSSnapshotBackend()
    weights = MagicMock(is_unmapped=True, is_connected=True)
    weights.remap_all_vas.side_effect = error

    with (
        patch(
            "gpu_memory_service.client.torch.allocator.get_gms_client_memory_manager",
            return_value=weights,
        ),
        pytest.raises(snapshot_backend.GMSWeightRestoreError),
    ):
        backend._resume_gms_tag("weights")

    weights.abort.assert_called_once_with()


def test_restore_prepares_scratch_kv_before_reallocation():
    backend = DynamoGMSSnapshotBackend()
    kv_cache = MagicMock(is_unmapped=True, is_connected=False)

    with (
        patch(
            "gpu_memory_service.client.torch.allocator.get_gms_client_memory_manager",
            return_value=kv_cache,
        ),
        patch(
            "gpu_memory_service.client.torch.allocator.is_scratch",
            return_value=True,
        ),
    ):
        backend._resume_gms_tag("kv_cache")

    assert kv_cache.method_calls == [
        call.connect(RequestedLockType.RW),
        call.prepare_scratch_for_reallocation(),
        call.reallocate_all_handles(tag="kv_cache"),
        call.remap_all_vas(),
    ]


def test_unknown_partial_wake_tag_is_rejected():
    backend = DynamoGMSSnapshotBackend()
    backend._state = "SUSPENDED"

    with pytest.raises(ValueError, match="Unknown GMS snapshot tags"):
        backend.resume(["unknown"])
