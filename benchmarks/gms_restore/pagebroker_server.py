# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-GPU GMS V1 allocation owner for the native PageBroker restore experiment.

This process serves weights and KV-cache sockets. It initializes the CUDA driver
without creating a context, importing a loader, opening payload files, or owning
transfer streams/buffers. PageBroker acquires the weights RW lease and performs
all allocation imports and PVC transfers in its native GPU engine.

The service starts without a capture or model binding. Its service generation
and server nonces identify this lifetime independently of later restore requests.
"""

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


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


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


def context_state(driver, device=0):
    """Observe without creating/retaining a context; reject unexpected activation."""

    def checked(result, operation):
        if int(result[0]) != 0:
            raise RuntimeError(f"{operation} failed: CUDA result {int(result[0])}")
        return result[1:]

    (current,) = checked(driver.cuCtxGetCurrent(), "cuCtxGetCurrent")
    (cuda_device,) = checked(driver.cuDeviceGet(device), "cuDeviceGet")
    flags, active = checked(
        driver.cuDevicePrimaryCtxGetState(cuda_device), "cuDevicePrimaryCtxGetState"
    )
    record = {
        "epoch": time.time(),
        "current_context": 0 if current is None else int(current),
        "primary_context_active": bool(active),
        "primary_context_flags": int(flags),
    }
    if record["current_context"] != 0 or record["primary_context_active"]:
        raise RuntimeError("server-only GMS unexpectedly acquired a CUDA context")
    return record


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--service-generation", required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--numa", action="store_true")
    args = parser.parse_args(argv)
    if args.rank < 0 or not args.service_generation:
        parser.error("rank must be nonnegative; service generation is required")
    return args


def main():
    started = time.time()
    args = parse_args()
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
                        "daemon_elapsed_s": epoch - started,
                        **values,
                    }
                ),
                flush=True,
            )

    emit(
        "daemon_start",
        mode="pagebroker",
        daemon_started_epoch=started,
        service_generation=args.service_generation,
    )
    before = cpu_snapshot()
    emit("imports_start")
    # Keep imports inside this benchmark entrypoint to measure startup cost;
    # CPU-only tests can import the service identity and context checks.
    from cuda.bindings import driver
    from gpu_memory_service.common.locks import RequestedLockType
    from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
    from gpu_memory_service.v1.checkpoint import GMSCheckpointLifecycle
    from gpu_memory_service.v1.cli import run_servers
    from gpu_memory_service.v1.client.session import _GMSClientSession
    from gpu_memory_service.v1.device import get_device_uuid, get_socket_path
    from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager

    emit("imports_complete")
    init_vmm(VMMDeviceType.CUDA)
    vmm = get_vmm()
    emit("cuinit_start")
    vmm.ensure_initialized()
    emit("cuinit_complete")
    result, count = driver.cuDeviceGetCount()
    if int(result) != 0 or count != 1:
        raise RuntimeError("server-only rank container must expose exactly one CUDA GPU")
    uuid = get_device_uuid(0)
    if uuid != args.expected_uuid:
        raise RuntimeError(f"rank {args.rank}: expected {args.expected_uuid}, got {uuid}")
    initial_context = context_state(driver)
    emit("context_state", stage="driver_initialized", **initial_context)
    if args.numa:
        bind_numa(uuid, emit)
    root = Path(os.environ["GMS_SOCKET_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    online_path = root / f"online-{args.rank}.json"
    if online_path.exists():
        raise RuntimeError(f"GMS service lifetime already has online state: {online_path}")
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
        emit("sockets", uuid=uuid)
        with ThreadPoolExecutor(max_workers=1) as executor:
            serving = executor.submit(run_servers, servers, stop)
            try:
                for domain, manager in managers.items():
                    session = _GMSClientSession(
                        get_socket_path(args.rank, domain),
                        RequestedLockType.RW,
                        expected_identity=manager.identity,
                        connect_timeout=10,
                        admission_timeout=10,
                    )
                    try:
                        if session.identity != manager.identity:
                            raise RuntimeError("server-only GMS readiness identity mismatch")
                    finally:
                        session.close()  # Abort the empty lease; publish nothing.
                if any(manager.allocation_snapshot() for manager in managers.values()):
                    raise RuntimeError("server-only readiness allocated GPU weights")
                ready_context = context_state(driver)
                online = {
                    "event": "online",
                    "mode": "pagebroker",
                    "rank": args.rank,
                    "uuid": uuid,
                    "service_generation": args.service_generation,
                    "daemon_started_epoch": started,
                    "online_epoch": time.time(),
                    "local_device": 0,
                    "visible_device_count": count,
                    "socket_device": args.rank,
                    "socket_dir": str(root),
                    "weights_server_nonce": managers["weights"].identity[0],
                    "kv_cache_server_nonce": managers["kv_cache"].identity[0],
                    "server_domains_responsive": list(managers),
                    "payload_bytes_read": 0,
                    "weight_allocations": 0,
                    "pinned_bytes": 0,
                    "loader_lanes_ready": 0,
                    "cuda_context_current": False,
                    "cuda_primary_context_active": False,
                    "context_state": ready_context,
                    "prewarm_cpu_before": before,
                    "prewarm_cpu_after": cpu_snapshot(),
                }
                atomic_json(online_path, online)
                emit("online", **{k: v for k, v in online.items() if k != "event"})
                allocated_seen = committed_seen = False
                while not stop.wait(0.05):
                    if serving.done():
                        serving.result()
                        raise RuntimeError("server-only GMS serving thread stopped")
                    weights = managers["weights"]
                    allocations = weights.allocation_snapshot()
                    if allocations and not allocated_seen:
                        emit(
                            "allocation_context_state",
                            allocation_count=len(allocations),
                            context_state=context_state(driver),
                        )
                        allocated_seen = True
                    if weights.session_snapshot().committed and not committed_seen:
                        emit(
                            "weights_committed",
                            allocation_count=len(allocations),
                            allocated_bytes=sum(size for _, size in allocations),
                            context_state=context_state(driver),
                            weights_server_nonce=weights.identity[0],
                        )
                        committed_seen = True
            except BaseException as error:
                emit("server_error", error=str(error))
                raise
            finally:
                online_path.unlink(missing_ok=True)
                stop.set()
                serving.result(timeout=10)


if __name__ == "__main__":
    main()
