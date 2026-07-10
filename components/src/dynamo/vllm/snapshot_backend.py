# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
from collections.abc import Iterable
from typing import Any

import torch

try:
    from vllm.device_allocator.sleep_mode_backend import (
        SleepModeBackend,
        SleepModeBackendFactory,
    )
except (ImportError, AttributeError) as exc:
    _SLEEP_MODE_IMPORT_ERROR: Exception | None = exc

    class SleepModeBackend:  # type: ignore[no-redef]
        """Import-only fallback for vLLM releases without pluggable backends."""

        def __init__(self) -> None:
            self._state = "RUNNING"

        def state(self) -> str:
            return self._state

    SleepModeBackendFactory = None  # type: ignore[assignment,misc]
else:
    _SLEEP_MODE_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

GMS_BACKEND_NAME = "dynamo_gms_snapshot"


class DynamoGMSSnapshotBackend(SleepModeBackend):
    """GMS unmap/remap with FlashInfer checkpoint lifecycle ordering."""

    def __init__(self) -> None:
        super().__init__()
        self._restored_tags = set(_get_gms_tags())
        self._communicators_restored = True
        self._ro_connect_timeout_ms: int | None = None
        self._fatal_error: Exception | None = None

    def set_ro_connect_timeout_ms(self, timeout_ms: int | None) -> None:
        self._ro_connect_timeout_ms = timeout_ms

    def suspend(self, level: int = 1) -> None:
        del level
        if self._state != "RUNNING":
            raise RuntimeError(
                f"Cannot suspend GMS snapshot backend from {self._state}"
            )

        checkpoint_prepare, _ = _checkpoint_hooks()
        self._communicators_restored = False
        try:
            checkpoint_prepare()
        except Exception as exc:
            self._state = "SUSPENDED"
            self._fatal_error = SnapshotTerminalError(
                f"FlashInfer checkpoint preparation failed: {exc}"
            )
            raise self._fatal_error from exc

        try:
            self._unmap_gms_tags()
        except Exception as exc:
            self._state = "SUSPENDED"
            self._fatal_error = SnapshotTerminalError(
                f"GMS unmap failed after FlashInfer checkpoint preparation: {exc}"
            )
            raise self._fatal_error from exc

        self._restored_tags.clear()
        gc.collect()
        torch.cuda.empty_cache()
        self._state = "SUSPENDED"

    def resume(self, tags: list[str] | None = None) -> None:
        if self._fatal_error is not None:
            raise SnapshotTerminalError(
                "GMS snapshot backend is unavailable after a failed suspend"
            ) from self._fatal_error
        restore_tags = _validated_restore_tags(tags)
        if self._state == "RUNNING":
            return

        self._state = "RESUMING"
        try:
            for tag in restore_tags:
                if tag in self._restored_tags:
                    continue
                self._resume_gms_tag(tag)
                self._restored_tags.add(tag)

            if (
                self._restored_tags == set(_get_gms_tags())
                and not self._communicators_restored
            ):
                try:
                    _, checkpoint_restore = _checkpoint_hooks()
                    checkpoint_restore()
                except Exception as exc:
                    raise SnapshotTerminalError(
                        f"FlashInfer checkpoint restore failed: {exc}"
                    ) from exc
                self._communicators_restored = True
        except Exception as exc:
            if isinstance(exc, SnapshotTerminalError):
                self._fatal_error = exc
            self._state = "SUSPENDED"
            raise

        if self._restored_tags == set(_get_gms_tags()) and self._communicators_restored:
            self._state = "RUNNING"

    def is_tag_restored(self, tag: str) -> bool:
        _validated_restore_tags((tag,))
        return tag in self._restored_tags

    def _unmap_gms_tags(self) -> None:
        from gpu_memory_service.client.torch.allocator import (
            get_gms_client_memory_manager,
        )

        for tag in _get_gms_tags():
            manager = get_gms_client_memory_manager(tag)
            assert manager is not None, f"GMS {tag} client is not initialized"
            assert not manager.is_unmapped, f"GMS {tag} is already unmapped"
            manager.unmap_all_vas()
            manager.abort()

    def _resume_gms_tags(self, tags: list[str] | None = None) -> None:
        for tag in _validated_restore_tags(tags):
            self._resume_gms_tag(tag)

    def _resume_gms_tag(
        self,
        tag: str,
        *,
        manager: Any | None = None,
    ) -> None:
        from gpu_memory_service.client.torch.allocator import (
            get_gms_client_memory_manager,
            is_scratch,
        )
        from gpu_memory_service.common.locks import RequestedLockType

        if manager is None:
            manager = get_gms_client_memory_manager(tag)
        assert manager is not None, f"GMS {tag} client is not initialized"
        if not manager.is_unmapped:
            return

        reconnected = not manager.is_connected
        if tag == "weights":
            try:
                if reconnected:
                    manager.connect(
                        RequestedLockType.RO,
                        timeout_ms=self._ro_connect_timeout_ms,
                    )
                manager.remap_all_vas()
            except Exception as exc:
                if manager.is_connected:
                    try:
                        manager.abort()
                    except Exception:
                        logger.exception(
                            "Failed to clean partially connected GMS weights client"
                        )
                raise GMSWeightRestoreError(str(exc)) from exc
            return

        if reconnected:
            manager.connect(RequestedLockType.RW)

        scratch = is_scratch(manager)
        try:
            if scratch:
                manager.prepare_scratch_for_reallocation()
            if reconnected:
                manager.reallocate_all_handles(tag=tag)
            manager.remap_all_vas()
        except Exception as exc:
            if manager.is_connected:
                try:
                    manager.abort()
                except Exception:
                    logger.exception(
                        "Failed to abort GMS %s after destructive restore failure",
                        tag,
                    )
            raise GMSKVRestoreError(
                f"GMS {tag} restore left an uncertain memory layout: {exc}"
            ) from exc

    @classmethod
    def preserves_communicators(cls) -> bool:
        return True

    @classmethod
    def preserves_graphs_with_communicators(cls) -> bool:
        return True


class SnapshotTerminalError(RuntimeError):
    """A snapshot lifecycle failure that requires worker termination."""


class GMSWeightRestoreError(SnapshotTerminalError):
    """A fatal weight reconnect/remap failure."""


class GMSKVRestoreError(SnapshotTerminalError):
    """A fatal destructive KV restoration failure."""


def _get_gms_tags() -> tuple[str, ...]:
    from gpu_memory_service.common.utils import GMS_TAGS

    return tuple(GMS_TAGS)


def _validated_restore_tags(tags: Iterable[str] | None) -> tuple[str, ...]:
    canonical_tags = _get_gms_tags()
    restore_tags = canonical_tags if tags is None else tuple(tags)
    unknown = set(restore_tags).difference(canonical_tags)
    if unknown:
        raise ValueError(f"Unknown GMS snapshot tags: {sorted(unknown)}")
    return restore_tags


def _checkpoint_hooks() -> tuple[Any, Any]:
    try:
        from vllm.distributed.parallel_state import (
            checkpoint_prepare_distributed_state,
            checkpoint_restore_distributed_state,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "GMS snapshot requires vLLM checkpoint hooks from the exact "
            "integration overlay and the matching custom FlashInfer build"
        ) from exc
    return (
        checkpoint_prepare_distributed_state,
        checkpoint_restore_distributed_state,
    )


def _require_sleep_mode_backend() -> None:
    if _SLEEP_MODE_IMPORT_ERROR is not None or SleepModeBackendFactory is None:
        raise RuntimeError(
            "GMS snapshot requires vLLM's pluggable sleep backend from the exact "
            "integration overlay; the declared vLLM 0.24.0 dependency does not "
            "provide this API"
        ) from _SLEEP_MODE_IMPORT_ERROR


def register_dynamo_gms_snapshot_backend() -> None:
    _require_sleep_mode_backend()
    assert SleepModeBackendFactory is not None
    if GMS_BACKEND_NAME in SleepModeBackendFactory._registry:
        return
    SleepModeBackendFactory.register_backend(
        GMS_BACKEND_NAME,
        "dynamo.vllm.snapshot_backend",
        "DynamoGMSSnapshotBackend",
    )


def select_dynamo_gms_snapshot_backend(vllm_config: object) -> None:
    load_format = getattr(
        getattr(vllm_config, "load_config", None), "load_format", None
    )
    if load_format != "gms":
        return
    register_dynamo_gms_snapshot_backend()
    vllm_config.model_config.sleep_mode_backend = GMS_BACKEND_NAME
