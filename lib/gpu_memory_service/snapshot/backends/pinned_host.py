# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned host-buffer helpers shared by snapshot transfer backends."""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Any, List, Sequence, Tuple

from gpu_memory_service.common import cuda_utils

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

    def __init__(self, size: int = PINNED_COPY_CHUNK_SIZE) -> None:
        size = int(size)
        self.view, self._raw, self.ptr = _allocate_aligned_buffer(size)
        self.stream = None
        self.busy = False
        self._registered = False
        self._closed = False
        try:
            self.stream = cuda_utils.cuda_stream_create_nonblocking()
            cuda_utils.cuda_host_register(self.ptr, size)
            self._registered = True
        except Exception:
            try:
                if self.stream is not None:
                    cuda_utils.cuda_stream_destroy(self.stream)
            finally:
                _free_aligned_buffer(self.view, self.ptr)
            raise

    def copy_to_device_async(self, dst_ptr: int, size: int) -> None:
        cuda_utils.cuda_memcpy_h2d_async(dst_ptr, self.ptr, size, self.stream)
        self.busy = True

    def copy_from_device_async(self, src_ptr: int, size: int) -> None:
        cuda_utils.cuda_memcpy_d2h_async(self.ptr, src_ptr, size, self.stream)
        self.busy = True

    def wait(self) -> None:
        if not self.busy:
            return
        cuda_utils.cuda_stream_synchronize(self.stream)
        self.busy = False

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
            if self.stream is not None:
                cuda_utils.cuda_stream_destroy(self.stream)
                self.stream = None
        except Exception as exc:  # noqa: BLE001
            if error is None:
                error = exc
            else:
                _LOGGER.warning("failed to destroy CUDA copy stream", exc_info=True)
        try:
            _free_aligned_buffer(self.view, self.ptr)
            self._closed = True
        except Exception as exc:  # noqa: BLE001
            if error is None:
                error = exc
            else:
                _LOGGER.warning("failed to free aligned host buffer", exc_info=True)
        if error is not None:
            raise error


def make_pinned_copy_slots(count: int) -> List[PinnedCopySlot]:
    slots: List[PinnedCopySlot] = []
    try:
        for _ in range(count):
            slots.append(PinnedCopySlot())
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
