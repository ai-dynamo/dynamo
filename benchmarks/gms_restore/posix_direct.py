# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental O_DIRECT -> pinned buffers -> CUDA backend, without NIXL startup.

Benchmark-only backend injection. Every lane owns two buffers and one device
context; completion and cleanup finish before the V1 loader publishes its lease.
"""

import fcntl
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from threading import Event

from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.snapshot.backends.pinned_host import PinnedCopySlot
from gpu_memory_service.snapshot.transfer import validate_transfer_targets

CHUNK = 16 * 1024**2


class PosixDirect:
    def __init__(self, config, emit=None):
        self.config = config
        self.emit = emit or (lambda *_args, **_kwargs: None)
        self.instrumented = emit is not None

    def start_restore(self, sources):
        self.sources = list(sources)
        self.emit(
            "allocation_import_complete",
            allocations=len(self.sources),
            bytes=sum(source.byte_count for source in self.sources),
        )
        return self

    def restore(self, targets):
        validate_transfer_targets(self.sources, targets, device=self.config.device)
        work = []
        for source in self.sources:
            for offset in range(0, source.byte_count, CHUNK):
                work.append(
                    (
                        source.file_path,
                        source.file_offset + offset,
                        targets[source.allocation_id].va + offset,
                        min(CHUNK, source.byte_count - offset),
                    )
                )
        lanes = min(self.config.max_workers, len(work))
        stop = Event()

        def lane(index, items):
            vmm = get_vmm()
            self.emit("lane_context_start", lane=index)
            vmm.runtime_set_device(self.config.device)
            self.emit("lane_context_ready", lane=index)
            slots = []
            fds = {}
            try:
                self.emit("lane_buffers_start", lane=index)
                slots = [PinnedCopySlot(vmm, CHUNK)]
                slots.append(PinnedCopySlot(vmm, CHUNK))
                self.emit("lane_ready", lane=index, pinned_bytes=2 * CHUNK)
                for i, (path, offset, dst, length) in enumerate(items):
                    if stop.is_set():
                        return
                    if path not in fds:
                        fds[path] = os.open(
                            path,
                            os.O_RDONLY
                            | (
                                0
                                if os.environ.get("GMS_PROTOTYPE_BUFFERED_READS") == "1"
                                else os.O_DIRECT
                            ),
                        )
                        flags = fcntl.fcntl(fds[path], fcntl.F_GETFL)
                        direct = bool(flags & os.O_DIRECT)
                        if os.environ.get("GMS_PROTOTYPE_BUFFERED_READS") != "1":
                            assert direct, "artifact FD must use O_DIRECT"
                        record = {"path": path, "flags": flags, "o_direct": direct}
                        if self.instrumented:
                            self.emit("artifact_open", lane=index, **record)
                        else:
                            print(
                                json.dumps({"event": "artifact_open", **record}),
                                flush=True,
                            )
                    slot = slots[i % 2]
                    slot.wait()
                    view = slot.view.cast("B")[:length]
                    try:
                        if i == 0:
                            self.emit("first_read_start", lane=index, bytes=length)
                        read = os.preadv(fds[path], [view], offset)
                    finally:
                        view.release()
                    if read != length:
                        raise OSError(f"short direct read: {read}/{length}")
                    if i == 0:
                        self.emit("first_read_complete", lane=index, bytes=read)
                        self.emit("first_copy_start", lane=index, bytes=length)
                    slot.copy_to_device_async(dst, length)
            except BaseException:
                stop.set()
                raise
            finally:
                try:
                    for slot in slots:
                        slot.close()
                finally:
                    for fd in fds.values():
                        os.close(fd)
            self.emit("lane_transfer_complete", lane=index)

        with ThreadPoolExecutor(max_workers=lanes) as pool:
            futures = [pool.submit(lane, i, work[i::lanes]) for i in range(lanes)]
            for future in futures:
                future.result()
        self.emit("transfer_complete", bytes=sum(x.byte_count for x in self.sources))

    def close(self):
        pass


class ResidentLanePool:
    """Warm copy lanes without opening or reading any weight payload.

    Each dedicated thread retains its current device, streams and two pinned
    buffers. CUDA's primary context is shared within the process; these are not
    separate driver contexts per lane. This experimental pool handles one load.
    """

    def __init__(self, device, workers, chunk, emit):
        if workers < 1 or chunk < 1:
            raise ValueError("workers and chunk must be positive")
        self.device, self.workers, self.chunk = device, workers, chunk
        self.emit = emit
        self.stop = Event()
        self.closed = False
        self.used = False
        self.queues = [Queue() for _ in range(workers)]
        self.ready = [Future() for _ in range(workers)]
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.futures = [self.pool.submit(self._lane, i) for i in range(workers)]
        try:
            for ready in self.ready:
                ready.result()
        except BaseException:
            self.close()
            raise

    def _lane(self, index):
        slots = []
        try:
            vmm = get_vmm()
            self.emit("lane_context_start", lane=index)
            vmm.runtime_set_device(self.device)
            self.emit("lane_context_ready", lane=index)
            self.emit("lane_buffers_start", lane=index)
            for _ in range(2):
                slots.append(PinnedCopySlot(vmm, self.chunk))
            self.emit("lane_ready", lane=index, pinned_bytes=2 * self.chunk)
            self.ready[index].set_result(None)
            request = self.queues[index].get()
            if request is None:
                return
            items, result = request
            try:
                self._copy(index, slots, items)
            except BaseException as error:  # noqa: BLE001 - delivered to caller below
                self.stop.set()
                result.set_exception(error)
            else:
                result.set_result(None)
            # Keep warmed resources resident through publication and inference.
            if self.queues[index].get() is not None:
                raise RuntimeError("resident lanes support exactly one load")
        except BaseException as error:
            if not self.ready[index].done():
                self.ready[index].set_exception(error)
            raise
        finally:
            for slot in slots:
                slot.close()

    def _copy(self, index, slots, items):
        fds = {}
        copied = 0
        try:
            for i, (path, offset, dst, length) in enumerate(items):
                if self.stop.is_set():
                    raise RuntimeError("another resident transfer lane failed")
                if path not in fds:
                    fds[path] = os.open(path, os.O_RDONLY | os.O_DIRECT)
                    flags = fcntl.fcntl(fds[path], fcntl.F_GETFL)
                    if not flags & os.O_DIRECT:
                        raise RuntimeError("artifact FD must use O_DIRECT")
                    self.emit(
                        "artifact_open",
                        lane=index,
                        path=path,
                        flags=flags,
                        o_direct=True,
                    )
                slot = slots[i % 2]
                slot.wait()
                view = slot.view.cast("B")[:length]
                try:
                    if i == 0:
                        self.emit("first_read_start", lane=index, bytes=length)
                    read = os.preadv(fds[path], [view], offset)
                finally:
                    view.release()
                if read != length:
                    raise OSError(f"short direct read: {read}/{length}")
                if i == 0:
                    self.emit("first_read_complete", lane=index, bytes=read)
                    self.emit("first_copy_start", lane=index, bytes=length)
                slot.copy_to_device_async(dst, length)
                copied += length
            for slot in slots:
                slot.wait()
            self.emit("lane_transfer_complete", lane=index, bytes=copied)
        finally:
            # Even a failed read must drain earlier asynchronous copies before
            # V1's error cleanup can unmap their destinations.
            try:
                for slot in slots:
                    slot.wait()
            finally:
                for fd in fds.values():
                    os.close(fd)

    def restore(self, sources, targets):
        if self.closed or self.used:
            raise RuntimeError("resident transfer pool must be fresh and open")
        self.used = True
        work = []
        for source in sources:
            for offset in range(0, source.byte_count, self.chunk):
                work.append(
                    (
                        source.file_path,
                        source.file_offset + offset,
                        targets[source.allocation_id].va + offset,
                        min(self.chunk, source.byte_count - offset),
                    )
                )
        results = [Future() for _ in self.queues]
        for i, result in enumerate(results):
            self.queues[i].put((work[i :: self.workers], result))
        # Drain every lane even when another has failed before mappings are freed.
        failure = None
        for result in results:
            try:
                result.result()
            except BaseException as error:  # noqa: BLE001 - drain every lane, then raise
                failure = failure or error
        if failure is not None:
            raise failure
        self.emit("transfer_complete", bytes=sum(x.byte_count for x in sources))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for queue in self.queues:
            queue.put(None)
        self.pool.shutdown(wait=True)
        for future in self.futures:
            future.result()


class ResidentPosixDirect:
    """V1 backend facade over a prestarted, single-load lane pool."""

    def __init__(self, config, pool):
        if config.device != pool.device or config.max_workers != pool.workers:
            raise ValueError("resident lane configuration differs from loader")
        self.config, self.pool = config, pool

    def start_restore(self, sources):
        self.sources = list(sources)
        self.pool.emit(
            "allocation_import_complete",
            allocations=len(self.sources),
            bytes=sum(source.byte_count for source in self.sources),
        )
        return self

    def restore(self, targets):
        validate_transfer_targets(self.sources, targets, device=self.config.device)
        self.pool.restore(self.sources, targets)

    def close(self):
        # The resident worker owns the pool; copies were drained in restore().
        pass


def install(resident_pool=None, emit=None):
    from gpu_memory_service.v1.snapshot import weight_artifact

    original = weight_artifact.create_transfer_backend

    def create(name, config):
        if name != "posix-direct":
            return original(name, config)
        if resident_pool is not None:
            resident_pool.emit("allocation_import_start")
            return ResidentPosixDirect(config, resident_pool)
        if emit is not None:
            emit("allocation_import_start")
        return PosixDirect(config, emit=emit)

    weight_artifact.create_transfer_backend = create


if __name__ == "__main__":
    import argparse

    from gpu_memory_service.common.vmm import VMMDeviceType, init_vmm
    from gpu_memory_service.v1.device import get_socket_path
    from gpu_memory_service.v1.snapshot.weight_artifact import load_weights

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--transfer-backend", default="posix-direct")
    p.add_argument("--max-workers", type=int, default=4)
    a = p.parse_args()
    if a.max_workers < 1:
        p.error("max-workers must be positive")
    init_vmm(VMMDeviceType.CUDA)
    install()
    load_weights(
        a.checkpoint_dir + "/device-0",
        get_socket_path(0),
        0,
        transfer_backend="posix-direct",
        max_workers=a.max_workers,
    )
