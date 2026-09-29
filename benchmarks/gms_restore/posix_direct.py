# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental O_DIRECT -> pinned buffers -> CUDA backend, without NIXL startup.

Benchmark-only backend injection. Every lane owns two buffers and one device
context; completion and cleanup finish before the V1 loader publishes its lease.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from threading import Event

from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.snapshot.backends.pinned_host import PinnedCopySlot
from gpu_memory_service.snapshot.transfer import validate_transfer_targets

CHUNK = 16 * 1024**2


class PosixDirect:
    def __init__(self, config):
        self.config = config

    def start_restore(self, sources):
        self.sources = list(sources)
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

        def lane(items):
            vmm = get_vmm()
            vmm.runtime_set_device(self.config.device)
            slots = []
            fds = {}
            try:
                slots = [PinnedCopySlot(vmm, CHUNK)]
                slots.append(PinnedCopySlot(vmm, CHUNK))
                for i, (path, offset, dst, length) in enumerate(items):
                    if stop.is_set():
                        return
                    if path not in fds:
                        fds[path] = os.open(path, os.O_RDONLY | os.O_DIRECT)
                    slot = slots[i % 2]
                    slot.wait()
                    view = slot.view.cast("B")[:length]
                    try:
                        read = os.preadv(fds[path], [view], offset)
                    finally:
                        view.release()
                    if read != length:
                        raise OSError(f"short direct read: {read}/{length}")
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

        with ThreadPoolExecutor(max_workers=lanes) as pool:
            futures = [pool.submit(lane, work[i::lanes]) for i in range(lanes)]
            for future in futures:
                future.result()

    def close(self):
        pass


def install():
    from gpu_memory_service.v1.snapshot import weight_artifact

    original = weight_artifact.create_transfer_backend

    def create(name, config):
        return PosixDirect(config) if name == "posix-direct" else original(name, config)

    weight_artifact.create_transfer_backend = create


if __name__ == "__main__":
    import argparse
    from gpu_memory_service.common.vmm import init_vmm, VMMDeviceType
    from gpu_memory_service.v1.snapshot.weight_artifact import load_weights
    from gpu_memory_service.v1.device import get_socket_path

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
