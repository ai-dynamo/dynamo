# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-process shared counter array for cuStreamWaitValue32-based
restore-completion signalling.

Mechanism (per-engine_id):
  1. Engine creates a tmpfs file of size N*4 bytes, mmaps it
     MAP_SHARED, and `cuMemHostRegister`s it with
     CU_MEMHOSTREGISTER_DEVICEMAP. Calls `cuMemHostGetDevicePointer`
     to obtain a CUdeviceptr usable in `cuStreamWaitValue32`.
  2. Daemon attaches the same tmpfs file via mmap MAP_SHARED.
     The daemon does NOT need CUDA mapping — it writes the counter
     via host atomic stores. The engine's GPU sees the writes via
     PCIe-coherent host memory (zero-copy mapped page).
  3. Each chunk-restore reserves a counter slot (round-robin). The
     daemon atomic-stores `counter_target` after the chunk's H2Ds
     are issued. The engine's stream waits via cuStreamWaitValue32
     for `counter_target` (using CU_STREAM_WAIT_VALUE_GEQ).

Per-engine state:
  - Counters monotonically increase (engine never reuses a slot
    until the previous wait has completed).
  - Slot allocation is round-robin; max in-flight restores = N
    counters per engine.
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def _file_size(num_counters: int) -> int:
    # Each counter is u32 = 4 bytes. Pad to 4 KiB page for cleanliness.
    raw = num_counters * 4
    return (raw + 4095) & ~4095


# ---------------------------------------------------------------------------
# Engine-side: create + CUDA-map for cuStreamWaitValue32
# ---------------------------------------------------------------------------


class EngineCounterArray:
    """Engine-side handle. Owns the tmpfs file + CUDA host-registration.

    Counters are written by the daemon (not by us). We *read* them
    indirectly via `cuStreamWaitValue32` on `self.device_ptr` (offset
    in 4-byte slots).
    """

    def __init__(
        self,
        path: str,
        num_counters: int,
        *,
        ipc_compat: bool = True,
    ) -> None:
        self.path = path
        self.num_counters = num_counters
        self._size = _file_size(num_counters)
        self._fd = -1
        self.buf: Optional[mmap.mmap] = None
        self._host_addr: int = 0
        self.device_ptr: int = 0
        self._registered = False
        self._slot_seq: int = 0
        self._slot_target: list = [0] * num_counters
        self._slot_lock = threading.Lock()
        self._ipc_compat = ipc_compat

    @classmethod
    def create(
        cls,
        path: str,
        num_counters: int = 512,
        *,
        ipc_compat: bool = True,
    ) -> "EngineCounterArray":
        inst = cls(path, num_counters, ipc_compat=ipc_compat)
        inst._open(create=True)
        inst._register_with_cuda()
        return inst

    def _open(self, *, create: bool) -> None:
        flags = os.O_CREAT | os.O_RDWR if create else os.O_RDWR
        self._fd = os.open(self.path, flags, 0o600)
        if create:
            os.ftruncate(self._fd, self._size)
        self.buf = mmap.mmap(
            self._fd,
            self._size,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )
        # Zero-initialize.
        if create:
            self.buf[:] = b"\x00" * self._size
        self._host_addr = self._buf_address()

    def _buf_address(self) -> int:
        # ctypes hack to get the raw pointer of the mmap'd region.
        import ctypes

        if self.buf is None:
            raise RuntimeError("mmap is not open")
        return ctypes.addressof(ctypes.c_char.from_buffer(self.buf))

    def _register_with_cuda(self) -> None:
        from cuda.bindings import driver as drv

        flags = drv.CU_MEMHOSTREGISTER_DEVICEMAP
        if self._ipc_compat:
            # PORTABLE allows the device pointer to be valid across
            # contexts on the same device.
            flags |= drv.CU_MEMHOSTREGISTER_PORTABLE
        err = drv.cuMemHostRegister(
            self._host_addr,
            self._size,
            int(flags),
        )[0]
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"cuMemHostRegister failed: {msg.decode() if msg else err}"
            )
        self._registered = True
        err, devptr = drv.cuMemHostGetDevicePointer(
            self._host_addr,
            0,
        )
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"cuMemHostGetDevicePointer failed: " f"{msg.decode() if msg else err}"
            )
        self.device_ptr = int(devptr)
        logger.info(
            "EngineCounterArray: registered %d counters at host=0x%x "
            "→ device=0x%x (path=%s)",
            self.num_counters,
            self._host_addr,
            self.device_ptr,
            self.path,
        )

    def reserve_slot(self) -> tuple[int, int]:
        """Reserve a counter slot for one chunk restore. Returns
        `(slot_idx, target_value)`. The target is monotonically
        increasing per slot; the daemon must atomic-store `target` to
        slot[slot_idx] when the restore is complete.

        Round-robin allocator with a per-slot monotonic counter.
        Caller must wait on the slot (via cuStreamWaitValue32) before
        this slot's target wraps around.
        """
        with self._slot_lock:
            slot = self._slot_seq % self.num_counters
            self._slot_seq += 1
            self._slot_target[slot] += 1
            target = self._slot_target[slot]
        return slot, target

    def slot_device_ptr(self, slot: int) -> int:
        if slot < 0 or slot >= self.num_counters:
            raise ValueError(f"slot {slot} out of range")
        return self.device_ptr + slot * 4

    def wait_value32(
        self,
        stream_handle: int,
        slot: int,
        target: int,
    ) -> None:
        """Make `stream_handle` wait until slot[slot] reaches >= target.

        Uses cuStreamWaitValue32 with CU_STREAM_WAIT_VALUE_GEQ —
        entirely GPU-side, no host sync.
        """
        from cuda.bindings import driver as drv

        addr = self.slot_device_ptr(slot)
        flags = drv.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ
        err = drv.cuStreamWaitValue32(
            stream_handle,
            addr,
            int(target) & 0xFFFFFFFF,
            int(flags),
        )[0]
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"cuStreamWaitValue32 failed: {msg.decode() if msg else err}"
            )

    def read_slot(self, slot: int) -> int:
        """Read current value of a counter slot (host side, for debug)."""
        return struct.unpack_from("<I", self.buf, slot * 4)[0]

    def close(self) -> None:
        if self._registered:
            try:
                from cuda.bindings import driver as drv

                drv.cuMemHostUnregister(self._host_addr)
            except Exception:
                pass
            self._registered = False
        if self.buf is not None:
            try:
                self.buf.close()
            except Exception:
                pass
            self.buf = None
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = -1


# ---------------------------------------------------------------------------
# Daemon-side: attach + atomic store
# ---------------------------------------------------------------------------


class DaemonCounterArray:
    """Daemon-side handle. mmap's the same tmpfs file MAP_SHARED.

    Two write modes:
      - `store(slot, value)`: host atomic store. Only safe when the
        caller knows all GPU work that should precede the counter
        bump has already drained. NOT safe to use after issuing
        async cuMemcpyAsync's — the engine could observe the counter
        bump before the H2D completes.
      - `write_on_stream(stream, slot, value)`: enqueues
        cuStreamWriteValue32 on the daemon's CUDA stream. Ordered
        AFTER any prior async work on that stream. This is the
        correct write to pair with the engine's cuStreamWaitValue32.

    To use `write_on_stream`, attach with `attach_with_cuda(path)`
    which performs the cuMemHostRegister so the daemon has its own
    device pointer.
    """

    def __init__(self, path: str, num_counters: int) -> None:
        self.path = path
        self.num_counters = num_counters
        self._size = _file_size(num_counters)
        self._fd = -1
        self.buf: Optional[mmap.mmap] = None
        self._host_addr: int = 0
        self.device_ptr: int = 0
        self._registered: bool = False

    @classmethod
    def attach(cls, path: str, num_counters: int = 512) -> "DaemonCounterArray":
        inst = cls(path, num_counters)
        inst._open()
        return inst

    @classmethod
    def attach_with_cuda(
        cls,
        path: str,
        num_counters: int = 512,
    ) -> "DaemonCounterArray":
        """Attach AND register with CUDA so `write_on_stream` works.

        Requires a CUDA context on the calling thread (daemon process
        will have one because it already does cuMemcpyAsync's)."""
        inst = cls(path, num_counters)
        inst._open()
        inst._register_with_cuda()
        return inst

    def _open(self) -> None:
        self._fd = os.open(self.path, os.O_RDWR)
        actual = os.fstat(self._fd).st_size
        if actual < self.num_counters * 4:
            os.close(self._fd)
            raise ValueError(
                f"counter file too small: {actual} < {self.num_counters * 4}",
            )
        self.buf = mmap.mmap(
            self._fd,
            self._size,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )
        import ctypes

        self._host_addr = ctypes.addressof(
            ctypes.c_char.from_buffer(self.buf),
        )

    def _register_with_cuda(self) -> None:
        from cuda.bindings import driver as drv

        flags = drv.CU_MEMHOSTREGISTER_DEVICEMAP | drv.CU_MEMHOSTREGISTER_PORTABLE
        err = drv.cuMemHostRegister(
            self._host_addr,
            self._size,
            int(flags),
        )[0]
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"DaemonCounterArray cuMemHostRegister failed: "
                f"{msg.decode() if msg else err}"
            )
        self._registered = True
        err, devptr = drv.cuMemHostGetDevicePointer(self._host_addr, 0)
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"DaemonCounterArray cuMemHostGetDevicePointer failed: "
                f"{msg.decode() if msg else err}"
            )
        self.device_ptr = int(devptr)

    def store(self, slot: int, value: int) -> None:
        """Host atomic store. Use only when no in-flight async GPU
        work needs to drain first."""
        if slot < 0 or slot >= self.num_counters:
            raise ValueError(f"slot {slot} out of range")
        struct.pack_into("<I", self.buf, slot * 4, int(value) & 0xFFFFFFFF)

    def write_on_stream(
        self,
        stream_handle: int,
        slot: int,
        value: int,
    ) -> None:
        """Enqueue cuStreamWriteValue32 on the daemon stream. The write
        is ordered AFTER any prior async work on `stream_handle` —
        which is exactly what we want when the prior work is the
        chunk's cuMemcpyAsync calls."""
        if not self._registered:
            raise RuntimeError(
                "DaemonCounterArray was not CUDA-registered; "
                "use attach_with_cuda() instead of attach()."
            )
        if slot < 0 or slot >= self.num_counters:
            raise ValueError(f"slot {slot} out of range")
        from cuda.bindings import driver as drv

        addr = self.device_ptr + slot * 4
        # WRITE_VALUE_DEFAULT: just a store, no memory fence — fine
        # because the prior cuMemcpyAsync's on this stream already
        # ordered the writes.
        err = drv.cuStreamWriteValue32(
            stream_handle,
            addr,
            int(value) & 0xFFFFFFFF,
            0,
        )[0]
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"cuStreamWriteValue32 failed: {msg.decode() if msg else err}"
            )

    def read(self, slot: int) -> int:
        if slot < 0 or slot >= self.num_counters:
            raise ValueError(f"slot {slot} out of range")
        return struct.unpack_from("<I", self.buf, slot * 4)[0]

    def close(self) -> None:
        if self._registered:
            try:
                from cuda.bindings import driver as drv

                drv.cuMemHostUnregister(self._host_addr)
            except Exception:
                pass
            self._registered = False
        if self.buf is not None:
            try:
                self.buf.close()
            except Exception:
                pass
            self.buf = None
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = -1
