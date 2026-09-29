# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental, single-load resident GMS V1 server and PVC/O_DIRECT loader.

One container exposes one GPU as cuda:0; --rank determines captured socket names.
Readiness means responsive V1 sockets plus warm loader contexts/streams/pinned
buffers. Payload files and GPU weight allocations are untouched until an exact
capture/generation trigger arrives. Restart the worker for another load.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Event, Lock


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def validate_trigger(trigger, capture_id, generation):
    if trigger.get("capture_id") != capture_id:
        raise ValueError("load trigger capture_id does not match resident capture")
    if trigger.get("generation") != generation:
        raise ValueError("load trigger generation does not match resident worker")
    epoch = trigger.get("trigger_written_epoch")
    if not isinstance(epoch, (int, float)) or not math.isfinite(epoch) or epoch <= 0:
        raise ValueError("load trigger must include a finite positive epoch")
    return trigger


def cpu_snapshot():
    root = Path("/sys/fs/cgroup")
    return {
        "epoch": time.time(),
        "stat": {
            key: int(value)
            for key, value in (
                line.split() for line in (root / "cpu.stat").read_text().splitlines()
            )
        },
        "pressure": (root / "cpu.pressure").read_text(),
        "max": (root / "cpu.max").read_text().strip(),
        "weight": (root / "cpu.weight").read_text().strip(),
    }


def bind_numa(uuid, emit):
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


def main():
    daemon_started = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--numa", action="store_true")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--expected-uuid")
    parser.add_argument("--trigger", default="/gms/load-trigger.json")
    args = parser.parse_args()
    if args.rank < 0 or args.workers < 1 or args.chunk_mib < 1:
        parser.error("rank must be nonnegative; workers/chunk size must be positive")
    if os.environ.get("GMS_PROTOTYPE_BUFFERED_READS") == "1":
        parser.error("resident experiment requires PVC/O_DIRECT payload reads")
    output_lock = Lock()

    def emit(event, **values):
        with output_lock:
            epoch = time.time()
            print(
                json.dumps(
                    {
                        "event": event,
                        "rank": args.rank,
                        "epoch": epoch,
                        "daemon_elapsed_s": epoch - daemon_started,
                        **values,
                    }
                ),
                flush=True,
            )

    emit("daemon_start", daemon_started_epoch=daemon_started)
    prewarm_cpu_before = cpu_snapshot()
    emit("imports_start")
    # Imports intentionally occur here so the benchmark measures their latency.
    import posix_direct
    from gpu_memory_service.common.locks import RequestedLockType
    from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
    from gpu_memory_service.v1.checkpoint import GMSCheckpointLifecycle
    from gpu_memory_service.v1.cli import run_servers
    from gpu_memory_service.v1.client.session import _GMSClientSession
    from gpu_memory_service.v1.device import get_device_uuid, get_socket_path
    from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager
    from gpu_memory_service.v1.snapshot.weight_artifact import load_weights

    emit("imports_complete")
    emit("vmm_create_start")
    init_vmm(VMMDeviceType.CUDA)
    vmm = get_vmm()
    emit("vmm_create_complete")
    emit("cuinit_start")
    vmm.ensure_initialized()
    emit("cuinit_complete")
    uuid = get_device_uuid(0)
    if args.expected_uuid and args.expected_uuid != uuid:
        raise RuntimeError(
            f"rank {args.rank}: expected {args.expected_uuid}, got {uuid}"
        )
    if args.numa:
        bind_numa(uuid, emit)
    root = Path(os.environ["GMS_SOCKET_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    online_path = root / f"online-{args.rank}.json"
    for path in (
        online_path,
        root / f"rank-{args.rank}.json",
        root / f"published-{args.rank}",
    ):
        path.unlink(missing_ok=True)
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    lifecycle = GMSCheckpointLifecycle()
    with ExitStack() as stack:
        emit("server_init_start")
        managers = {
            domain: GMSServerMemoryManager(uuid, vmm, 0, checkpoint_lifecycle=lifecycle)
            for domain in ("weights", "kv_cache")
        }
        lifecycle.bind_domains(managers)
        servers = [
            stack.enter_context(
                GMSRPCServer(get_socket_path(args.rank, domain), manager)
            )
            for domain, manager in managers.items()
        ]
        emit("sockets", uuid=uuid, elapsed_s=time.time() - daemon_started)
        with ThreadPoolExecutor(max_workers=1) as executor:
            serving = executor.submit(run_servers, servers, stop)
            lanes = None
            try:
                # A completed V1 handshake proves both serving threads respond;
                # abort the empty RW lease without publishing an empty artifact.
                for domain in managers:
                    session = _GMSClientSession(
                        get_socket_path(args.rank, domain),
                        RequestedLockType.RW,
                        connect_timeout=10,
                        admission_timeout=10,
                    )
                    try:
                        if session.identity[1] != uuid:
                            raise RuntimeError("resident server GPU identity mismatch")
                    finally:
                        session.close()
                emit("server_responsive", uuid=uuid)
                emit("loader_context_start")
                vmm.runtime_set_device(0)
                emit("loader_context_ready")
                lanes = posix_direct.ResidentLanePool(
                    0,
                    args.workers,
                    args.chunk_mib * 1024**2,
                    emit,
                )
                posix_direct.install(resident_pool=lanes)
                online = {
                    "event": "online",
                    "rank": args.rank,
                    "uuid": uuid,
                    "capture_id": args.capture_id,
                    "generation": args.generation,
                    "daemon_started_epoch": daemon_started,
                    "online_epoch": time.time(),
                    "workers": args.workers,
                    "chunk_mib": args.chunk_mib,
                    "pinned_bytes": args.workers * 2 * args.chunk_mib * 1024**2,
                    "server_domains_responsive": list(managers),
                    "loader_lanes_ready": args.workers,
                    "payload_bytes_read": 0,
                    "weight_allocations": 0,
                    "prewarm_cpu_before": prewarm_cpu_before,
                    "prewarm_cpu_after": cpu_snapshot(),
                }
                if any(manager.allocation_snapshot() for manager in managers.values()):
                    raise RuntimeError("weights unexpectedly allocated before trigger")
                atomic_json(online_path, online)
                emit("online", **{k: v for k, v in online.items() if k != "event"})
                trigger_path = Path(args.trigger)
                rejected = None
                while not stop.is_set():
                    if serving.done():
                        serving.result()
                        raise RuntimeError("resident GMS server stopped")
                    if trigger_path.exists():
                        trigger = json.loads(trigger_path.read_text())
                        try:
                            validate_trigger(trigger, args.capture_id, args.generation)
                        except ValueError as error:
                            if str(error) != rejected:
                                emit("trigger_rejected", reason=str(error))
                                rejected = str(error)
                        else:
                            break
                    stop.wait(0.005)
                else:
                    return
                started = time.time()
                emit("trigger_observed", trigger=trigger, started_epoch=started)
                cpu_before = cpu_snapshot()
                emit("load_weights_start")
                load_weights(
                    f"{args.artifact_root}/device-{args.rank}",
                    get_socket_path(args.rank),
                    0,
                    transfer_backend="posix-direct",
                    max_workers=args.workers,
                )
                published = time.time()
                record = {
                    "rank": args.rank,
                    "uuid": uuid,
                    "event": "published",
                    "capture_id": args.capture_id,
                    "generation": args.generation,
                    "chunk_mib": args.chunk_mib,
                    "workers": args.workers,
                    "cpu_before": cpu_before,
                    "cpu_after": cpu_snapshot(),
                    "daemon_started_epoch": daemon_started,
                    "online_epoch": online["online_epoch"],
                    "trigger_written_epoch": trigger["trigger_written_epoch"],
                    "started_epoch": started,
                    "published_epoch": published,
                    "elapsed_s": published - started,
                }
                atomic_json(root / f"rank-{args.rank}.json", record)
                (root / f"published-{args.rank}").write_text(uuid)
                emit("published", **{k: v for k, v in record.items() if k != "event"})
                while not stop.wait(0.01):
                    if serving.done():
                        serving.result()
                        raise RuntimeError("resident GMS server stopped")
            finally:
                online_path.unlink(missing_ok=True)
                stop.set()
                try:
                    if lanes is not None:
                        lanes.close()
                finally:
                    serving.result()


if __name__ == "__main__":
    main()
