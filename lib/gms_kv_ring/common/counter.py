# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared semaphore counter array for cuStreamWaitValue32 sync.

How it works:

  1. Engine creates a /dev/shm file of size N*4 bytes; mmaps it
     MAP_SHARED; `cuMemHostRegister`s with DEVICEMAP; obtains a
     CUdeviceptr via `cuMemHostGetDevicePointer`.
  2. Daemon attaches the same /dev/shm file, also registers with
     CUDA so it can use `cuStreamWriteValue32` on its CUDA stream.
  3. Per restore, engine reserves a slot K and target value N.
  4. Engine queues `cuStreamWaitValue32(compute_stream, counter[K],
     N, GEQ)` BEFORE any kernel that needs the restored bytes.
  5. Daemon does its H2D's on the daemon CUDA stream; AFTER them,
     enqueues `cuStreamWriteValue32(daemon_stream, counter[K], N)`.
  6. The write is stream-ordered AFTER the H2D's. When the write
     fires, the engine's stream wait releases.

Why this is better than cudaEvent IPC:
  • No IPC event handle exchange (saves a socket round-trip).
  • No `cudaIpcGetEventHandle` / `cudaIpcOpenEventHandle` calls.
  • Pure GPU-side serialization once the counter array is set up.

Constraints:
  • The counter file MUST live on tmpfs (/dev/shm).
    `cuMemHostRegister` rejects regular-fs mmaps.
  • One counter array per engine. Engines don't share slots.
  • Round-robin slot reuse with per-slot monotonic targets — caller
    must wait on slot K's prior target before reusing K.
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
    raw = num_counters * 4
    return (raw + 4095) & ~4095  # page-aligned


class EngineCounterArray:
    """Engine-side: owns the tmpfs file + CUDA host-registration.

    Engine *waits* on counter slots via `wait_value32`. The daemon
    *writes* slot values to signal completion.
    """

    def __init__(self, path: str, num_counters: int) -> None:
        self.path = path
        self.num_counters = num_counters
        self._size = _file_size(num_counters)
        self._fd = -1
        self.buf: Optional[mmap.mmap] = None
        self._host_addr: int = 0
        self.device_ptr: int = 0
        self._registered = False
        self._slot_seq = 0
        self._slot_target = [0] * num_counters
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        path: str,
        num_counters: int = 512,
    ) -> "EngineCounterArray":
        inst = cls(path, num_counters)
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
        if create:
            self.buf[:] = b"\x00" * self._size
        import ctypes

        self._host_addr = ctypes.addressof(
            ctypes.c_char.from_buffer(self.buf),
        )

    def _register_with_cuda(self) -> None:
        from cuda.bindings import driver as drv

        flags = drv.CU_MEMHOSTREGISTER_DEVICEMAP | drv.CU_MEMHOSTREGISTER_PORTABLE
        err = drv.cuMemHostRegister(self._host_addr, self._size, int(flags))[0]
        if err == drv.CUresult.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            # Same process already pinned these tmpfs pages via another
            # mmap. That's fine — just skip re-register, get the devptr.
            self._registered = False
        elif err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"cuMemHostRegister failed: " f"{msg.decode() if msg else err}",
            )
        else:
            self._registered = True
        err, devptr = drv.cuMemHostGetDevicePointer(self._host_addr, 0)
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"cuMemHostGetDevicePointer failed: " f"{msg.decode() if msg else err}",
            )
        self.device_ptr = int(devptr)

    def reserve_slot(self) -> tuple[int, int]:
        """Reserve a (slot_idx, success_target) for one restore op.

        Each reservation consumes TWO consecutive target values per
        slot: success_target = N and failure_target = N+1. The daemon
        writes N on success, N+1 on failure. Engine's
        `cuStreamWaitValue32(GEQ N)` satisfies on either (so the
        engine's compute stream doesn't hang on a daemon-side error),
        and the engine then reads the counter host-side after sync to
        learn the actual outcome via `restore_succeeded` on the handle.

        Slots round-robin; per-slot target monotonically increases."""
        with self._lock:
            slot = self._slot_seq % self.num_counters
            self._slot_seq += 1
            return self._reserve_slot_locked(slot)

    def try_reserve_completed_slot(self) -> Optional[tuple[int, int]]:
        """Reserve a counter slot only if its prior op completed.

        Returns ``None`` when every slot still has an in-flight target.
        This prevents wrapping the fixed counter array under high
        concurrency and reusing a slot whose previous daemon write has
        not become visible yet.
        """
        with self._lock:
            for _ in range(self.num_counters):
                slot = self._slot_seq % self.num_counters
                self._slot_seq += 1
                previous_success_target = self._slot_target[slot] - 1
                if previous_success_target > 0:
                    if self.read_slot(slot) < previous_success_target:
                        continue
                return self._reserve_slot_locked(slot)
            return None

    def _reserve_slot_locked(self, slot: int) -> tuple[int, int]:
        # Consume two values: N (success) and N+1 (failure).
        success_target = self._slot_target[slot] + 1
        self._slot_target[slot] += 2
        return slot, success_target

    def mark_slot_failed(self, slot: int, target: int) -> None:
        """Host-side failure mark for a reserved slot with no daemon op.

        Used when the producer reserved a slot but could not enqueue the
        ring record. It makes the slot reusable and wakes any accidental
        GEQ wait as a failure.
        """
        if not 0 <= slot < self.num_counters:
            raise ValueError(f"slot {slot} out of range")
        struct.pack_into("<I", self.buf, slot * 4, int(target) + 1)

    def read_slot(self, slot: int) -> int:
        """Host-side read of counter[slot]. Used post-sync by the
        engine to learn whether a restore succeeded — daemon writes
        target on success, target+1 on failure."""
        if not 0 <= slot < self.num_counters:
            raise ValueError(f"slot {slot} out of range")
        import struct

        return struct.unpack_from("<I", self.buf, slot * 4)[0]

    def slot_device_ptr(self, slot: int) -> int:
        if not 0 <= slot < self.num_counters:
            raise ValueError(f"slot {slot} out of range")
        return self.device_ptr + slot * 4

    def wait_value32(self, stream: int, slot: int, target: int) -> None:
        """Make `stream` wait until slot's value reaches `target` (GEQ).
        Pure GPU sync — no host syscall."""
        from cuda.bindings import driver as drv

        addr = self.slot_device_ptr(slot)
        err = drv.cuStreamWaitValue32(
            stream,
            addr,
            int(target) & 0xFFFFFFFF,
            int(drv.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
        )[0]
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"cuStreamWaitValue32 failed: " f"{msg.decode() if msg else err}",
            )

    def close(self) -> None:
        # Always try to unregister, even if our own register() was
        # skipped because the pages were already pinned via a different
        # VA in this process. If we leave a phantom pinning entry behind
        # for the VA we're about to unmap, a later mmap of a different
        # tmpfs file that the kernel happens to place at the same VA
        # will see cuMemHostGetDevicePointer return a stale device
        # pointer — and `cuStreamWaitValue32` ends up polling memory
        # that nothing writes to. Quietly tolerate the unregister
        # error if no pinning is recorded for this exact VA.
        if self._host_addr:
            try:
                from cuda.bindings import driver as drv

                drv.cuMemHostUnregister(self._host_addr)
            except Exception:
                pass
            self._registered = False
            self._host_addr = 0
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


class DaemonCounterArray:
    """Daemon-side: attaches the same file + uses cuStreamWriteValue32
    to signal completion on the daemon's CUDA stream."""

    def __init__(self, path: str, num_counters: int) -> None:
        self.path = path
        self.num_counters = num_counters
        self._size = _file_size(num_counters)
        self._fd = -1
        self.buf: Optional[mmap.mmap] = None
        self._host_addr = 0
        self.device_ptr = 0
        self._registered = False
        self._in_process = False

    @classmethod
    def attach(
        cls,
        path: str,
        num_counters: int = 512,
    ) -> "DaemonCounterArray":
        inst = cls(path, num_counters)
        inst._open()
        inst._register_with_cuda()
        return inst

    @classmethod
    def attach_in_process(
        cls,
        host_addr: int,
        num_counters: int = 512,
    ) -> "DaemonCounterArray":
        """Same-process attach: reuse the engine's already-pinned VA.

        The engine's `EngineCounterArray` mmap'd the counter file and
        called `cuMemHostRegister` with DEVICEMAP. In a same-process
        daemon, we don't need our own mmap or register — we already
        share the address space, and under UVA the device pointer
        equals the host address. Just remember the host_addr and use
        it as the device pointer.

        Cross-process daemons must use `attach(path, num_counters)`
        instead, which does its own mmap + register.

        Skipping the second register also avoids the
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED path entirely, which
        is what caused phantom pinning entries to leak across tests."""
        inst = cls("<in-process>", num_counters)
        inst._host_addr = int(host_addr)
        inst.device_ptr = int(host_addr)
        inst._registered = False
        inst._in_process = True
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
        err = drv.cuMemHostRegister(self._host_addr, self._size, int(flags))[0]
        if err == drv.CUresult.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            # Same process already pinned this tmpfs region via another
            # mmap (e.g. EngineCounterArray on the same path). Skip
            # re-register; cuMemHostGetDevicePointer still works.
            self._registered = False
        elif err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"DaemonCounterArray cuMemHostRegister failed: "
                f"{msg.decode() if msg else err}",
            )
        else:
            self._registered = True
        err, devptr = drv.cuMemHostGetDevicePointer(self._host_addr, 0)
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"DaemonCounterArray cuMemHostGetDevicePointer failed: "
                f"{msg.decode() if msg else err}",
            )
        self.device_ptr = int(devptr)

    def close(self) -> None:
        if self._in_process:
            # Engine owns the registration + mmap. Nothing to release.
            self._host_addr = 0
            self.device_ptr = 0
            return
        if self._host_addr:
            try:
                from cuda.bindings import driver as drv

                drv.cuMemHostUnregister(self._host_addr)
            except Exception:
                pass
            self._registered = False
            self._host_addr = 0
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

    def write_host(self, slot: int, value: int) -> None:
        """Write `value` to counter[slot] directly via CPU. Caller is
        responsible for ensuring prior GPU work (e.g. cuMemcpyBatchAsync
        on the daemon stream) has been drained — cuStreamSynchronize
        first if you care about ordering.

        Pinned host memory mapped with DEVICEMAP is coherent: a CPU
        store is visible to GPU reads via cuStreamWaitValue32. This
        is the same write that cuStreamWriteValue32 would have done,
        just from the CPU side — no devptr round-trip, no risk of
        aliasing through a stale registration."""
        if not 0 <= slot < self.num_counters:
            raise ValueError(f"slot {slot} out of range")
        import ctypes

        addr = self._host_addr + slot * 4
        if not addr:
            raise RuntimeError("counter not attached")
        ctypes.c_uint32.from_address(addr).value = int(value) & 0xFFFFFFFF

    def write_on_stream(
        self,
        stream: int,
        slot: int,
        value: int,
    ) -> None:
        """Enqueue `value → counter[slot]` on the daemon stream.
        Ordered AFTER prior async work on the same stream."""
        if not 0 <= slot < self.num_counters:
            raise ValueError(f"slot {slot} out of range")
        from cuda.bindings import driver as drv

        addr = self.device_ptr + slot * 4
        err = drv.cuStreamWriteValue32(
            stream,
            addr,
            int(value) & 0xFFFFFFFF,
            0,
        )[0]
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(
                f"cuStreamWriteValue32 failed: " f"{msg.decode() if msg else err}",
            )

    def read(self, slot: int) -> int:
        if not 0 <= slot < self.num_counters:
            raise ValueError(f"slot {slot} out of range")
        if self._in_process:
            import ctypes

            return ctypes.c_uint32.from_address(self._host_addr + slot * 4).value
        return struct.unpack_from("<I", self.buf, slot * 4)[0]
