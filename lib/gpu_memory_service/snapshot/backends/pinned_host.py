# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned host-buffer helpers shared by snapshot transfer backends."""

from __future__ import annotations

import ctypes
import logging
import os
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
        **profile_fields: Any,
    ) -> None:
        size = int(size)
        self._profile = profile or SnapshotProfile("loader", enabled=False)
        self._profile_fields = profile_fields
        with self._profile.aggregate(
            "pinned_slot_allocation",
            byte_count=size,
            **self._profile_fields,
        ):
            self.view, self._raw, self.ptr = _allocate_aligned_buffer(size)
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
                **self._profile_fields,
            ):
                self.stream = cuda_utils.cuda_stream_create_nonblocking()
            if self._profile.enabled:
                with self._profile.aggregate(
                    "cuda_event_create",
                    count=2,
                    **self._profile_fields,
                ):
                    self._start_event = cuda_utils.cuda_event_create()
                    self._end_event = cuda_utils.cuda_event_create()
            with self._profile.aggregate(
                "cuda_host_register",
                byte_count=size,
                **self._profile_fields,
            ):
                cuda_utils.cuda_host_register(self.ptr, size)
            self._registered = True
        except Exception:
            try:
                self._destroy_events_and_stream(suppress_errors=True)
            finally:
                _free_aligned_buffer(self.view, self.ptr)
            raise

    def copy_to_device_async(self, dst_ptr: int, size: int) -> None:
        if not self._profile.enabled:
            cuda_utils.cuda_memcpy_h2d_async(dst_ptr, self.ptr, size, self.stream)
            self.busy = True
            return
        self._enqueue_copy(
            "h2d",
            size,
            lambda: cuda_utils.cuda_memcpy_h2d_async(
                dst_ptr, self.ptr, size, self.stream
            ),
        )
        self.busy = True

    def copy_from_device_async(self, src_ptr: int, size: int) -> None:
        if not self._profile.enabled:
            cuda_utils.cuda_memcpy_d2h_async(self.ptr, src_ptr, size, self.stream)
            self.busy = True
            return
        self._enqueue_copy(
            "d2h",
            size,
            lambda: cuda_utils.cuda_memcpy_d2h_async(
                self.ptr, src_ptr, size, self.stream
            ),
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
                cuda_utils.cuda_event_record(self._start_event, self.stream)
            copy()
            self.busy = True
            if self._end_event is not None:
                cuda_utils.cuda_event_record(self._end_event, self.stream)

    def wait(self) -> None:
        if not self.busy:
            return
        if not self._profile.enabled:
            cuda_utils.cuda_stream_synchronize(self.stream)
            self.busy = False
            return
        with self._profile.aggregate(
            "cuda_stream_wait_cpu",
            **self._profile_fields,
        ):
            cuda_utils.cuda_stream_synchronize(self.stream)
        if self._end_event is not None:
            wall_end_ns = time.time_ns()
            duration_ns = cuda_utils.cuda_event_elapsed_ns(
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
                    **self._profile_fields,
                ):
                    cuda_utils.cuda_event_destroy(event)
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = exc
            finally:
                setattr(self, event_name, None)
        if self.stream is not None:
            try:
                with self._profile.aggregate(
                    "cuda_stream_destroy",
                    **self._profile_fields,
                ):
                    cuda_utils.cuda_stream_destroy(self.stream)
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
                    **self._profile_fields,
                ):
                    cuda_utils.cuda_host_unregister(self.ptr)
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
            with self._profile.aggregate(
                "pinned_slot_free",
                **self._profile_fields,
            ):
                _free_aligned_buffer(self.view, self.ptr)
            self._closed = True
        except Exception as exc:  # noqa: BLE001
            if error is None:
                error = exc
            else:
                _LOGGER.warning("failed to free aligned host buffer", exc_info=True)
        if error is not None:
            raise error


def make_pinned_copy_slots(
    count: int,
    *,
    profile: SnapshotProfile | None = None,
    **profile_fields: Any,
) -> List[PinnedCopySlot]:
    slots: List[PinnedCopySlot] = []
    try:
        for _ in range(count):
            slots.append(
                PinnedCopySlot(
                    profile=profile,
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
