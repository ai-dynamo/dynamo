# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare unchanged GMS/PB transfer code on identical PVC bytes, without restore.

This isolates the transfer layer. It does not exercise GMS RPC/publication or
PageBroker CUDA checkpoint reconstruction. One process sees one DRA GPU.
"""

import argparse
import ctypes
import fcntl
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument("--rank", type=int, required=True)
p.add_argument("--backend", choices=["gms", "pagebroker"], required=True)
p.add_argument("--workers", type=int, default=16)
p.add_argument("--chunk-mib", type=int, default=16)
p.add_argument("--slots", type=int, default=32)
p.add_argument("--case", required=True)
p.add_argument("--mount", default="/checkpoints")
p.add_argument("--prewarm", action="store_true")
a = p.parse_args()
started = time.time()
from cuda.bindings import driver as cu
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
from gpu_memory_service.snapshot.transfer import FileTransferSource as TransferSource
from gpu_memory_service.snapshot.transfer import GMSTransferTarget as TransferTarget
from gpu_memory_service.v1.device import get_device_uuid


def check(result):
    assert int(result[0]) == 0, result
    return result[1] if len(result) == 2 else result[1:]


init_vmm(VMMDeviceType.CUDA)
vmm = get_vmm()
vmm.ensure_initialized()
vmm.runtime_set_device(0)
# Bind CPU and pinned-buffer first touch to the same GPU's host NUMA node.
device = check(cu.cuDeviceGet(0))
node = check(
    cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device
    )
)
cpus = set()
for part in (
    Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(",")
):
    ends = [int(x) for x in part.split("-")]
    cpus.update(range(ends[0], ends[-1] + 1))
os.sched_setaffinity(0, cpus & os.sched_getaffinity(0))
root = Path(a.mount) / f"gms-restore-0929/default-capture-2/device-{a.rank}"
manifest = json.loads((root / "manifest.json").read_text())
allocs = manifest["allocations"]
size = sum(x["aligned_size"] for x in allocs)
va = int(check(cu.cuMemAlloc(size)))
sources = []
targets = {}
offset = 0
files = {}
for x in allocs:
    sources.append(
        TransferSource(
            allocation_id=x["allocation_id"],
            file_path=str(root / x["shard"]),
            file_offset=x["offset"],
            byte_count=x["aligned_size"],
        )
    )
    targets[x["allocation_id"]] = TransferTarget(
        allocation_id=x["allocation_id"],
        va=va + offset,
        byte_count=x["aligned_size"],
        device=0,
    )
    if x["shard"] not in files:
        assert x["offset"] == 0
        files[x["shard"]] = [va + offset, 0]
    assert files[x["shard"]][0] + x["offset"] == va + offset
    files[x["shard"]][1] += x["aligned_size"]
    offset += x["aligned_size"]
setup = {}
handle = None
warm_slots = []
if a.backend == "pagebroker":
    lib = ctypes.CDLL("/gms/libpbprobe.so")
    lib.probe_create.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    lib.probe_create.restype = ctypes.c_void_p
    lib.probe_transfer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_ulonglong,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.probe_destroy.argtypes = [ctypes.c_void_p]
    s = time.perf_counter()
    handle = lib.probe_create(a.slots, a.chunk_mib * 1024**2)
    assert handle
    setup["ring_init_s"] = time.perf_counter() - s
else:
    import posix_direct

    posix_direct.CHUNK = a.chunk_mib * 1024**2
    if a.prewarm:
        from queue import Queue

        class WarmSlot:
            def __init__(self, slot):
                self.slot = slot
                self.view = slot.view

            def wait(self):
                self.slot.wait()

            def copy_to_device_async(self, dst, length):
                self.slot.copy_to_device_async(dst, length)

            def close(self):
                # Preserve the backend's drain guarantee while retaining buffers.
                self.slot.wait()

        ring_start = time.perf_counter()
        pool = Queue()
        for _ in range(a.workers * 2):
            slot = posix_direct.PinnedCopySlot(vmm, posix_direct.CHUNK)
            warm_slots.append(slot)
            pool.put(WarmSlot(slot))
        setup["ring_init_s"] = time.perf_counter() - ring_start
        posix_direct.PinnedCopySlot = lambda *_: pool.get_nowait()
    backend = posix_direct.PosixDirect(SimpleNamespace(device=0, max_workers=a.workers))
    backend.start_restore(sources)
# Cross-container barrier excludes pod/python/context/destination setup from transfer.
run = Path("/gms") / a.case
run.mkdir(exist_ok=True)
(run / f"ready-{a.rank}").write_text(
    json.dumps(
        {
            "uuid": get_device_uuid(0),
            "size": size,
            "setup": setup,
            "startup_s": time.time() - started,
        }
    )
)
deadline = time.monotonic() + 120
while not (run / "go").exists():
    if time.monotonic() > deadline:
        raise TimeoutError("barrier")
    time.sleep(0.002)


def cpu_stats():
    values = {}
    for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
        key, value = line.split()
        values[key] = int(value)
    return values


profile = []
cpu_before = cpu_stats()
began = time.time()
if a.backend == "pagebroker":
    for shard, (dst, n) in files.items():
        fd = os.open(root / shard, os.O_RDONLY | os.O_DIRECT)
        try:
            assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT
            metrics = (ctypes.c_double * 5)()
            assert lib.probe_transfer(handle, fd, dst, n, metrics) == 0
            profile.append(
                {"shard": shard, "bytes": n, "metrics": list(metrics), "o_direct": True}
            )
        finally:
            os.close(fd)
else:
    backend.restore(targets)
check(cu.cuCtxSynchronize())
ended = time.time()
cpu_after = cpu_stats()
# Verify first/middle/last page of every 2GiB allocation after timed transfer.
buf = (ctypes.c_ubyte * 4096)()
samples = 0
for source in sources:
    with open(source.file_path, "rb") as f:
        for off in [0, source.byte_count // 2, source.byte_count - 4096]:
            check(
                cu.cuMemcpyDtoH(
                    ctypes.addressof(buf), targets[source.allocation_id].va + off, 4096
                )
            )
            f.seek(source.file_offset + off)
            assert bytes(buf) == f.read(4096)
            samples += 1
record = {
    "rank": a.rank,
    "backend": a.backend,
    "bytes": size,
    "start_epoch": began,
    "end_epoch": ended,
    "transfer_s": ended - began,
    "setup": setup,
    "profile": profile,
    "verified_samples": samples,
    "workers": a.workers,
    "slots": a.slots,
    "chunk_mib": a.chunk_mib,
    "uuid": get_device_uuid(0),
    "numa_node": node,
    "prewarm": a.prewarm,
    "cpu_delta": {k: cpu_after[k] - v for k, v in cpu_before.items()},
    "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
    "cpu_weight": Path("/sys/fs/cgroup/cpu.weight").read_text().strip(),
}
(run / f"result-{a.rank}.json").write_text(json.dumps(record))
print(json.dumps(record), flush=True)
if handle:
    lib.probe_destroy(handle)
for slot in warm_slots:
    slot.close()
check(cu.cuMemFree(va))
