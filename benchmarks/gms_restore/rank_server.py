# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One isolated GPU, stable engine socket ordinal, optional exact artifact save."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

started = time.time()
p = argparse.ArgumentParser()
p.add_argument("--rank", type=int, required=True)
p.add_argument("--artifact-root", required=True)
p.add_argument("--mode", choices=["capture", "load"], default="capture")
a = p.parse_args()
root = Path(os.environ["GMS_SOCKET_DIR"])
server = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "gpu_memory_service.v1.cli",
        "--device",
        "0",
        "--socket-device",
        str(a.rank),
    ]
)
try:
    from gpu_memory_service.common.vmm import VMMDeviceType, init_vmm
    from gpu_memory_service.v1.device import get_device_uuid, get_socket_path

    init_vmm(VMMDeviceType.CUDA)
    uuid = get_device_uuid(0)
    (root / f"rank-{a.rank}.json").write_text(
        json.dumps({"rank": a.rank, "uuid": uuid})
    )
    for domain in ("weights", "kv_cache"):
        while not Path(get_socket_path(a.rank, domain)).exists():
            if server.poll() is not None:
                raise RuntimeError("server exited during initialization")
            time.sleep(0.01)
    sockets = time.time()
    print(json.dumps({"event": "sockets", "elapsed_s": sockets - started}), flush=True)
    if a.mode == "capture":
        while not (root / "save").exists():
            if server.poll() is not None:
                raise RuntimeError("server exited")
            time.sleep(0.05)
        from gpu_memory_service.v1.snapshot.weight_artifact import save_weights

        manifest = save_weights(
            f"{a.artifact_root}/device-{a.rank}", get_socket_path(a.rank), 0
        )
    else:
        from gpu_memory_service.snapshot.backends import nixl_staging
        from gpu_memory_service.v1.snapshot.weight_artifact import load_weights

        original_open = nixl_staging.open_direct_read_fd

        def verified_open(path, **kwargs):
            fd = original_open(path, **kwargs)
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            assert flags & os.O_DIRECT, "artifact FD must use O_DIRECT"
            print(
                json.dumps(
                    {
                        "event": "artifact_open",
                        "path": path,
                        "flags": flags,
                        "o_direct": True,
                    }
                ),
                flush=True,
            )
            return fd

        nixl_staging.open_direct_read_fd = verified_open
        load_weights(f"{a.artifact_root}/device-{a.rank}", get_socket_path(a.rank), 0)
    published = time.time()
    record = {
        "rank": a.rank,
        "uuid": uuid,
        "started_epoch": started,
        "published_epoch": published,
        "elapsed_s": published - started,
    }
    (root / f"rank-{a.rank}.json").write_text(json.dumps(record))
    print(json.dumps(record), flush=True)
    (root / f"published-{a.rank}").write_text(uuid)
    server.wait()
    raise RuntimeError(f"server exited {server.returncode}")
finally:
    server.terminate()
    server.wait(timeout=10)
