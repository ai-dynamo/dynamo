# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure fresh single-rank V1 server + loader and verify published bytes.

Run inside an isolated one-GPU container with this checkout on PYTHONPATH.
Synthetic artifacts test plumbing and startup; they are NOT model checkpoints.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--mib", type=int, default=256)
p.add_argument("--backend", default="nixl")
p.add_argument("--iterations", type=int, default=3)
a = p.parse_args()
root = Path(a.root)
root.mkdir(parents=True, exist_ok=True)
artifact = root / f"synthetic-{a.mib}" / "device-0"
artifact.mkdir(parents=True, exist_ok=True)
size = a.mib * 1024**2
assert size > 0 and size % (2 * 1024**2) == 0
shard = artifact / "weights.bin"
if not shard.exists():
    with shard.open("wb") as f:
        for _ in range(a.mib):
            f.write(b"\x5a" * 1024**2)
(artifact / "manifest.json").write_text(
    json.dumps(
        {
            "version": 1,
            "allocations": [
                {
                    "allocation_id": "synthetic-rank-0",
                    "aligned_size": size,
                    "shard": "weights.bin",
                    "offset": 0,
                }
            ],
        }
    )
)

# Imports happen outside timed fresh server/loader subprocesses, only for validation.
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.device import get_device_uuid
from gpu_memory_service.v1.protocol import (
    ListAllocationsRequest,
    ListAllocationsResponse,
)
from gpu_memory_service.v1.snapshot.weight_artifact import _map_export, _release_mapping

for i in range(a.iterations):
    sockets = root / f"sockets-{os.getpid()}-{i}"
    sockets.mkdir()
    env = {**os.environ, "GMS_SOCKET_DIR": str(sockets), "DYN_GMS_USE_V1": "true"}
    socket = str(sockets / "gms_0_weights.sock")
    started = time.monotonic()
    with (root / f"server-{i}.log").open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "gpu_memory_service.v1.cli"],
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            until = started + 60
            while not Path(socket).exists():
                if server.poll() is not None:
                    raise RuntimeError("server exited; see log")
                if time.monotonic() > until:
                    raise TimeoutError("server startup")
                time.sleep(0.005)
            admitted = time.monotonic()
            with (root / f"loader-{i}.log").open("w") as loadlog:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        (
                            "posix_direct"
                            if a.backend == "posix-direct"
                            else "gpu_memory_service.v1.snapshot.loader"
                        ),
                        "--checkpoint-dir",
                        str(artifact.parent),
                        "--transfer-backend",
                        a.backend,
                    ],
                    env=env,
                    stdout=loadlog,
                    stderr=loadlog,
                    check=True,
                    timeout=180,
                )
            published = time.monotonic()
            session = _GMSClientSession(
                socket, RequestedLockType.RO, connect_timeout=2, admission_timeout=2
            )
            try:
                records = session._call(
                    ListAllocationsRequest(), ListAllocationsResponse
                ).allocations
                assert [(r.allocation_id, r.aligned_size) for r in records] == [
                    ("synthetic-rank-0", size)
                ]
                init_vmm(VMMDeviceType.CUDA)
                vmm = get_vmm()
                vmm.ensure_initialized()
                vmm.runtime_set_device(0)
                assert session.identity[1] == get_device_uuid(0)
                mapping = _map_export(
                    session,
                    records[0],
                    vmm,
                    0,
                    int(vmm.get_allocation_granularity(0)),
                    GrantedLockType.RO,
                )
                try:
                    buf = (ctypes.c_ubyte * 4096)()
                    from cuda.bindings import driver as cuda

                    for offset in (0, size // 2, size - 4096):
                        (rc,) = cuda.cuMemcpyDtoH(
                            ctypes.addressof(buf), mapping[0].base + offset, 4096
                        )
                        assert int(rc) == 0
                        assert bytes(buf) == b"\x5a" * 4096
                finally:
                    _release_mapping(vmm, mapping)
            finally:
                session.close()
            print(
                json.dumps(
                    {
                        "iteration": i,
                        "mib": a.mib,
                        "backend": a.backend,
                        "server_socket_s": admitted - started,
                        "loader_process_s": published - admitted,
                        "total_to_published_s": published - started,
                        "uuid": get_device_uuid(0),
                        "exact_ids_sizes": True,
                        "sampled_bytes_verified": True,
                    }
                ),
                flush=True,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
