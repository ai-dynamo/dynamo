# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned host-buffer helpers shared by snapshot transfer backends."""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from typing import Any, List, Sequence, Tuple

from gpu_memory_service.common import cuda_utils
from gpu_memory_service.common.snapshot_profile import SnapshotProfile

PINNED_COPY_CHUNK_SIZE = 64 * 1024 * 1024

_LOGGER = logging.getLogger(__name__)
_ALIGNMENT = 4096
_LIBC = ctypes.CDLL(None)
_LIBC.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_LIBC.posix_memalign.restype = ctypes.c_int
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None


def _allocate_aligned_buffer(size: int) -> Tuple[memoryview, Any, int]:
    ptr = ctypes.c_void_p()
    rc = _LIBC.posix_memalign(ctypes.byref(ptr), _ALIGNMENT, size)
    if rc != 0:
        raise OSError(rc, os.strerror(rc))
    array = (ctypes.c_ubyte * size).from_address(ptr.value)
    return memoryview(array), array, int(ptr.value)


def _free_aligned_buffer(view: memoryview, ptr: int) -> None:
    view.release()
    _LIBC.free(ctypes.c_void_p(ptr))


class PinnedCopySlot:
    """One reusable pinned host buffer and CUDA stream."""

    def __init__(
        self,
        size: int = PINNED_COPY_CHUNK_SIZE,
        *,
        profile: SnapshotProfile | None = None,
        cuda_operations: Any = None,
        arena: "PinnedCopyArena | None" = None,
        arena_slot: int | None = None,
        **profile_fields: Any,
    ) -> None:
        size = int(size)
        self._profile = profile or SnapshotProfile("loader", enabled=False)
        self._profile_fields = profile_fields
        self._cuda = (
            cuda_utils.RUNTIME_CUDA_TRANSFER_OPERATIONS
            if cuda_operations is None
            else cuda_operations
        )
        self._cuda_profile_fields = (
            {"cuda_api": self._cuda.api} if self._cuda.api == "driver" else {}
        )
        self._resource_profile_fields = {
            name: value
            for name, value in self._profile_fields.items()
            if name != "worker"
        }
        self._arena = arena
        if arena is None:
            if arena_slot is not None:
                raise ValueError("arena_slot requires an arena")
            with self._profile.aggregate(
                "pinned_host_memory_create",
                byte_count=size,
                **self._resource_profile_fields,
            ):
                self.view, self._raw, self.ptr = _allocate_aligned_buffer(size)
        else:
            if arena_slot is None:
                raise ValueError("arena_slot is required with an arena")
            self.view, self._raw, self.ptr = arena.claim_slot(arena_slot, size)
        self.stream = None
        self.busy = False
        self._copy_wall_start_ns = 0
        self._copy_bytes = 0
        self._copy_direction = ""
        self._start_event = None
        self._end_event = None
        self._registered = False
        self._closed = False
        try:
            with self._profile.aggregate(
                "cuda_stream_create",
                **self._cuda_profile_fields,
                **self._resource_profile_fields,
            ):
                self.stream = self._cuda.stream_create_nonblocking()
            if self._profile.enabled:
                with self._profile.aggregate(
                    "cuda_event_create",
                    count=2,
                    **self._cuda_profile_fields,
                    **self._resource_profile_fields,
                ):
                    self._start_event = self._cuda.event_create()
                    self._end_event = self._cuda.event_create()
            if arena is None:
                with self._profile.aggregate(
                    "cuda_host_register",
                    byte_count=size,
                    api=(
                        "cuMemHostRegister"
                        if self._cuda.api == "driver"
                        else "cudaHostRegister"
                    ),
                    **self._cuda_profile_fields,
                    **self._resource_profile_fields,
                ):
                    self._cuda.host_register(self.ptr, size)
                self._registered = True
        except Exception:
            try:
                self._destroy_events_and_stream(suppress_errors=True)
            finally:
                if self._arena is None:
                    _free_aligned_buffer(self.view, self.ptr)
                else:
                    self._arena.release_slot(self.ptr, self.view)
            raise

    def copy_to_device_async(self, dst_ptr: int, size: int) -> None:
        if not self._profile.enabled:
            self._cuda.memcpy_h2d_async(dst_ptr, self.ptr, size, self.stream)
            self.busy = True
            return
        self._enqueue_copy(
            "h2d",
            size,
            lambda: self._cuda.memcpy_h2d_async(dst_ptr, self.ptr, size, self.stream),
        )
        self.busy = True

    def copy_from_device_async(self, src_ptr: int, size: int) -> None:
        if not self._profile.enabled:
            self._cuda.memcpy_d2h_async(self.ptr, src_ptr, size, self.stream)
            self.busy = True
            return
        self._enqueue_copy(
            "d2h",
            size,
            lambda: self._cuda.memcpy_d2h_async(self.ptr, src_ptr, size, self.stream),
        )
        self.busy = True

    def _enqueue_copy(
        self,
        direction: str,
        size: int,
        copy: Any,
    ) -> None:
        self._copy_wall_start_ns = time.time_ns() if self._profile.enabled else 0
        self._copy_bytes = size
        self._copy_direction = direction
        with self._profile.aggregate(
            f"{direction}_enqueue_cpu",
            byte_count=size,
            **self._profile_fields,
        ):
            if self._start_event is not None:
                self._cuda.event_record(self._start_event, self.stream)
            copy()
            self.busy = True
            if self._end_event is not None:
                self._cuda.event_record(self._end_event, self.stream)

    def wait(self) -> None:
        if not self.busy:
            return
        if not self._profile.enabled:
            self._cuda.stream_synchronize(self.stream)
            self.busy = False
            return
        with self._profile.aggregate(
            "cuda_stream_wait_cpu",
            **self._profile_fields,
        ):
            self._cuda.stream_synchronize(self.stream)
        if self._end_event is not None:
            wall_end_ns = time.time_ns()
            duration_ns = self._cuda.event_elapsed_ns(
                self._start_event,
                self._end_event,
            )
            self._profile.add_aggregate(
                f"{self._copy_direction}_device",
                wall_start_ns=self._copy_wall_start_ns,
                wall_end_ns=wall_end_ns,
                duration_ns=duration_ns,
                count=1,
                byte_count=self._copy_bytes,
                observable_wall_bounds="enqueue_to_existing_stream_wait",
                **self._profile_fields,
            )
        self.busy = False

    def _destroy_events_and_stream(self, *, suppress_errors: bool) -> None:
        first_error = None
        for event_name in ("_start_event", "_end_event"):
            event = getattr(self, event_name)
            if event is None:
                continue
            try:
                with self._profile.aggregate(
                    "cuda_event_destroy",
                    **self._cuda_profile_fields,
                    **self._resource_profile_fields,
                ):
                    self._cuda.event_destroy(event)
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = exc
            finally:
                setattr(self, event_name, None)
        if self.stream is not None:
            try:
                with self._profile.aggregate(
                    "cuda_stream_destroy",
                    **self._cuda_profile_fields,
                    **self._resource_profile_fields,
                ):
                    self._cuda.stream_destroy(self.stream)
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = exc
            finally:
                self.stream = None
        if first_error is not None and not suppress_errors:
            raise first_error

    def close(self) -> None:
        if self._closed:
            return
        error = None
        try:
            self.wait()
        except Exception as exc:  # noqa: BLE001
            error = exc
        try:
            if self._registered:
                with self._profile.aggregate(
                    "cuda_host_unregister",
                    byte_count=len(self.view),
                    api=(
                        "cuMemHostUnregister"
                        if self._cuda.api == "driver"
                        else "cudaHostUnregister"
                    ),
                    **self._cuda_profile_fields,
                    **self._resource_profile_fields,
                ):
                    self._cuda.host_unregister(self.ptr)
                self._registered = False
        except Exception as exc:  # noqa: BLE001
            if error is None:
                error = exc
            else:
                _LOGGER.warning(
                    "failed to unregister pinned host buffer", exc_info=True
                )
        try:
            self._destroy_events_and_stream(suppress_errors=False)
        except Exception as exc:  # noqa: BLE001
            if error is None:
                error = exc
            else:
                _LOGGER.warning("failed to destroy CUDA copy stream", exc_info=True)
        try:
            if self._arena is None:
                with self._profile.aggregate(
                    "pinned_host_memory_free",
                    **self._resource_profile_fields,
                ):
                    _free_aligned_buffer(self.view, self.ptr)
            else:
                self._arena.release_slot(self.ptr, self.view)
            self._closed = True
        except Exception as exc:  # noqa: BLE001
            if error is None:
                error = exc
            else:
                _LOGGER.warning("failed to free aligned host buffer", exc_info=True)
        if error is not None:
            raise error


class PinnedCopyArena:
    """One contiguous registered host allocation backing logical copy slots."""

    def __init__(
        self,
        slot_count: int,
        registration_groups: int,
        *,
        slot_size: int = PINNED_COPY_CHUNK_SIZE,
        profile: SnapshotProfile | None = None,
        cuda_operations: Any = None,
    ) -> None:
        self.slot_count = int(slot_count)
        self.slot_size = int(slot_size)
        self.registration_groups = int(registration_groups)
        if self.slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if self.slot_size <= 0 or self.slot_size % _ALIGNMENT:
            raise ValueError(f"slot_size must be {_ALIGNMENT}-byte aligned")
        if not 1 <= self.registration_groups <= self.slot_count:
            raise ValueError(
                "registration_groups must be between 1 and slot_count inclusive"
            )
        self._profile = profile or SnapshotProfile("loader", enabled=False)
        self._cuda = (
            cuda_utils.RUNTIME_CUDA_TRANSFER_OPERATIONS
            if cuda_operations is None
            else cuda_operations
        )
        self._cuda_profile_fields = (
            {"cuda_api": self._cuda.api} if self._cuda.api == "driver" else {}
        )
        self._lock = threading.Lock()
        self._claimed_slots: dict[int, int] = {}
        self._registered_ranges: List[Tuple[int, int]] = []
        self._closed = False
        self.size = self.slot_count * self.slot_size
        with self._profile.aggregate(
            "pinned_host_memory_create",
            byte_count=self.size,
            allocation="arena",
        ):
            self.view, self._raw, self.ptr = _allocate_aligned_buffer(self.size)
        try:
            for first_slot, slots_in_group in self._registration_ranges():
                group_ptr = self.ptr + first_slot * self.slot_size
                group_size = slots_in_group * self.slot_size
                with self._profile.aggregate(
                    "cuda_host_register",
                    byte_count=group_size,
                    api=(
                        "cuMemHostRegister"
                        if self._cuda.api == "driver"
                        else "cudaHostRegister"
                    ),
                    **self._cuda_profile_fields,
                ):
                    self._cuda.host_register(group_ptr, group_size)
                self._registered_ranges.append((group_ptr, group_size))
        except Exception:
            self._cleanup_registration(suppress_errors=True)
            _free_aligned_buffer(self.view, self.ptr)
            raise

    def _registration_ranges(self) -> List[Tuple[int, int]]:
        base, extra = divmod(self.slot_count, self.registration_groups)
        ranges = []
        first_slot = 0
        for group in range(self.registration_groups):
            slots_in_group = base + (1 if group < extra else 0)
            ranges.append((first_slot, slots_in_group))
            first_slot += slots_in_group
        return ranges

    def claim_slot(self, slot: int, size: int) -> Tuple[memoryview, Any, int]:
        slot = int(slot)
        if size != self.slot_size:
            raise ValueError(
                f"arena slot size mismatch: requested={size} arena={self.slot_size}"
            )
        if not 0 <= slot < self.slot_count:
            raise IndexError(f"arena slot {slot} is outside [0, {self.slot_count})")
        ptr = self.ptr + slot * self.slot_size
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot claim a closed pinned arena")
            if ptr in self._claimed_slots:
                raise RuntimeError(f"pinned arena slot {slot} is already claimed")
            self._claimed_slots[ptr] = slot
        start = slot * self.slot_size
        return self.view[start : start + self.slot_size], self._raw, ptr

    def release_slot(self, ptr: int | None, view: memoryview) -> None:
        if ptr is None:
            raise ValueError("arena slot pointer is required")
        with self._lock:
            if ptr not in self._claimed_slots:
                raise RuntimeError(f"pinned arena pointer {ptr} is not claimed")
            self._claimed_slots.pop(ptr)
        view.release()

    def _cleanup_registration(self, *, suppress_errors: bool) -> None:
        first_error = None
        for ptr, size in reversed(self._registered_ranges):
            try:
                with self._profile.aggregate(
                    "cuda_host_unregister",
                    byte_count=size,
                    api=(
                        "cuMemHostUnregister"
                        if self._cuda.api == "driver"
                        else "cudaHostUnregister"
                    ),
                    **self._cuda_profile_fields,
                ):
                    self._cuda.host_unregister(ptr)
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = exc
        self._registered_ranges.clear()
        if first_error is not None and not suppress_errors:
            raise first_error

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._claimed_slots:
                raise RuntimeError(
                    "cannot close pinned arena while logical slots remain active: "
                    f"{sorted(self._claimed_slots.values())}"
                )
            self._closed = True
        error = None
        try:
            self._cleanup_registration(suppress_errors=False)
        except Exception as exc:  # noqa: BLE001
            error = exc
        try:
            with self._profile.aggregate(
                "pinned_host_memory_free",
                allocation="arena",
            ):
                _free_aligned_buffer(self.view, self.ptr)
        except Exception as exc:  # noqa: BLE001
            if error is None:
                error = exc
        if error is not None:
            raise error


def make_pinned_copy_slots(
    count: int,
    *,
    profile: SnapshotProfile | None = None,
    cuda_operations: Any = None,
    arena: PinnedCopyArena | None = None,
    first_arena_slot: int = 0,
    **profile_fields: Any,
) -> List[PinnedCopySlot]:
    slots: List[PinnedCopySlot] = []
    try:
        for _ in range(count):
            slots.append(
                PinnedCopySlot(
                    profile=profile,
                    cuda_operations=cuda_operations,
                    arena=arena,
                    arena_slot=(first_arena_slot + len(slots) if arena else None),
                    **profile_fields,
                )
            )
    except Exception:
        for slot in slots:
            try:
                slot.close()
            except Exception:
                _LOGGER.warning(
                    "failed to close partially created pinned copy slot",
                    exc_info=True,
                )
        raise
    return slots


def close_pinned_copy_slots(
    slots: Sequence[PinnedCopySlot],
    logger: logging.Logger,
    warning: str,
    *args: Any,
) -> None:
    for slot in slots:
        try:
            slot.close()
        except Exception:
            logger.warning(warning, *args, exc_info=True)
