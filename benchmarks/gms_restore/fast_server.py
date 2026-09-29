# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental fused V1 server/loader: one CUDA init and no NIXL startup."""

import argparse
import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Event, Lock

started = time.time()
p = argparse.ArgumentParser()
p.add_argument("--rank", type=int, required=True)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--chunk-mib", type=int, default=16)
p.add_argument("--numa", action="store_true")
p.add_argument("--artifact-root", required=True)
p.add_argument("--checkpoint-target", action="store_true")
p.add_argument("--retain-anchors", action="store_true")
a = p.parse_args()
if a.chunk_mib < 1 or a.workers < 1:
    p.error("chunk size and workers must be positive")
output_lock = Lock()


def emit(event, **values):
    with output_lock:
        epoch = time.time()
        print(
            json.dumps(
                {
                    "event": event,
                    "rank": a.rank,
                    "epoch": epoch,
                    "elapsed_s": epoch - started,
                    **values,
                }
            ),
            flush=True,
        )


def cpu_snapshot():
    """Read this rank container's counters without sampling another workload."""
    root = Path("/sys/fs/cgroup")
    values = {}
    for line in (root / "cpu.stat").read_text().splitlines():
        key, value = line.split()
        values[key] = int(value)
    return {
        "epoch": time.time(),
        "stat": values,
        "pressure": (root / "cpu.pressure").read_text(),
        "max": (root / "cpu.max").read_text().strip(),
        "weight": (root / "cpu.weight").read_text().strip(),
    }


cpu_before = cpu_snapshot()
emit("process_start", started_epoch=started)
emit("imports_start")
import posix_direct
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
from gpu_memory_service.v1.checkpoint import GMSCheckpointClient, GMSCheckpointLifecycle
from gpu_memory_service.v1.cli import run_servers
from gpu_memory_service.v1.device import get_device_uuid, get_socket_path
from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager
from gpu_memory_service.v1.snapshot.weight_artifact import load_weights
from posix_direct import install

emit("imports_complete")
posix_direct.CHUNK = a.chunk_mib * 1024**2

if os.environ.get("GMS_PROTOTYPE_INTERPOSE_CUDA_PYTHON") == "1":
    from ctypes_interpose import install as install_interpose

    install_interpose()
emit("vmm_create_start")
init_vmm(VMMDeviceType.CUDA)
vmm = get_vmm()
emit("vmm_create_complete")
emit("cuinit_start")
vmm.ensure_initialized()
emit("cuinit_complete")
uuid = get_device_uuid(0)
if a.numa:
    emit("numa_start")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"],
        text=True,
    )
    pci = next(
        line.split(",")[1].strip()
        for line in rows.splitlines()
        if line.split(",")[0].strip() == uuid
    )
    domain, rest = pci.split(":", 1)
    pci = f"{int(domain, 16):04x}:{rest}".lower()
    node = int(Path(f"/sys/bus/pci/devices/{pci}/numa_node").read_text())
    if node < 0:
        raise RuntimeError("GPU NUMA node unknown")
    cpus = set()
    for part in (
        Path(f"/sys/devices/system/node/node{node}/cpulist")
        .read_text()
        .strip()
        .split(",")
    ):
        limits = [int(x) for x in part.split("-")]
        cpus.update(range(limits[0], limits[-1] + 1))
    os.sched_setaffinity(0, cpus & os.sched_getaffinity(0))
    emit("numa_affinity", node=node, cpus=sorted(os.sched_getaffinity(0)))
root = Path(os.environ["GMS_SOCKET_DIR"])
root.mkdir(exist_ok=True)
lifecycle = GMSCheckpointLifecycle()
with ExitStack() as stack:
    emit("server_init_start")
    managers = {
        d: GMSServerMemoryManager(uuid, vmm, 0, checkpoint_lifecycle=lifecycle)
        for d in ("weights", "kv_cache")
    }
    lifecycle.bind_domains(managers)
    servers = [
        stack.enter_context(GMSRPCServer(get_socket_path(a.rank, d), managers[d]))
        for d in managers
    ]
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    emit("sockets", uuid=uuid)
    with ThreadPoolExecutor(max_workers=1) as pool:
        serving = pool.submit(run_servers, servers, stop)
        try:
            install(emit=emit)
            # Attribute the existing loader's primary-context setup separately;
            # load_weights repeats cudaSetDevice on the already-current device.
            emit("loader_context_start")
            vmm.runtime_set_device(0)
            emit("loader_context_ready")
            emit("load_weights_start")
            load_weights(
                f"{a.artifact_root}/device-{a.rank}",
                get_socket_path(a.rank),
                0,
                transfer_backend="posix-direct",
                max_workers=a.workers,
            )
            published = time.time()
            record = {
                "rank": a.rank,
                "uuid": uuid,
                "event": "published",
                "chunk_mib": a.chunk_mib,
                "workers": a.workers,
                "cpu_before": cpu_before,
                "cpu_after": cpu_snapshot(),
                "started_epoch": started,
                "published_epoch": published,
                "elapsed_s": published - started,
            }
            (root / f"rank-{a.rank}.json").write_text(json.dumps(record))
            (root / f"published-{a.rank}").write_text(uuid)
            print(json.dumps(record), flush=True)
            if a.checkpoint_target:
                response = GMSCheckpointClient(get_socket_path(a.rank)).prepare()
                anchors = []
                if a.retain_anchors:
                    from gpu_memory_service.common.locks import GrantedLockType

                    for manager in managers.values():
                        for allocation_id, size in manager.allocation_snapshot():
                            handle = manager._allocations._allocations[allocation_id]
                            va = vmm.address_reserve(
                                size, int(vmm.get_allocation_granularity(0))
                            )
                            vmm.map(va, size, handle)
                            vmm.set_access(va, size, 0, GrantedLockType.RO)
                            anchors.append((manager, allocation_id, va, size))
                Path("/snapshot-control/checkpoint-token").write_text(response.token)
                Path("/snapshot-control/ready-for-snapshot").touch()
            restored = False
            while not stop.wait(0.01):
                if (
                    a.checkpoint_target
                    and not restored
                    and Path("/snapshot-control/restore-complete").exists()
                ):
                    # Prototype restore hook: CUDA has remapped the process, so
                    # refresh identity before admitting any engine connection.
                    from gpu_memory_service.v1.device import (
                        invalidate_device_uuid_cache,
                    )

                    if a.retain_anchors:
                        from cuda.bindings import driver as cuda

                        for manager, allocation_id, va, size in anchors:
                            result, handle = cuda.cuMemRetainAllocationHandle(va)
                            if int(result) != 0:
                                raise RuntimeError(
                                    f"retain restored allocation: {result}"
                                )
                            manager._allocations._allocations[allocation_id] = int(
                                handle
                            )
                    invalidate_device_uuid_cache()
                    restored_uuid = get_device_uuid(0)
                    with lifecycle.condition:
                        for manager in managers.values():
                            manager._identity = (manager.identity[0], restored_uuid)
                    GMSCheckpointClient(get_socket_path(a.rank)).complete(
                        response.token
                    )
                    Path("/snapshot-control/server-restore-ready").write_text(
                        restored_uuid
                    )
                    restored = True
                if serving.done():
                    serving.result()
                    raise RuntimeError("server stopped")
        finally:
            stop.set()
            serving.result()
