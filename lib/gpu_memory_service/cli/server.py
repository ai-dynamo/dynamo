# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS server entry point.

Launches two GMS server processes per GPU (one for weights, one for kv_cache).
Writes a ready file once all expected UDS sockets are present. Runs until
SIGTERM (pod termination kills it).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from gpu_memory_service.common.cuda_utils import list_devices
from gpu_memory_service.common.utils import get_socket_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_TAGS = ("weights", "kv_cache")
_READY_FILE = "gms-ready"


def main() -> None:
    ready_file = Path(os.environ.get("GMS_SOCKET_DIR", "/tmp")) / _READY_FILE
    ready_file.unlink(missing_ok=True)

    devices = list_devices()
    processes = []
    for device in devices:
        for tag in _TAGS:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gpu_memory_service",
                    "--device",
                    str(device),
                    "--tag",
                    tag,
                ]
            )
            logger.info("Started GMS device=%d tag=%s pid=%d", device, tag, proc.pid)
            processes.append(proc)

    def shutdown() -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    def terminate(*_args) -> None:
        shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    ready_written = False
    while True:
        if not ready_written:
            sockets_ready = all(
                os.path.exists(get_socket_path(device, tag))
                for device in devices
                for tag in _TAGS
            )
            if sockets_ready:
                ready_file.write_text("ready", encoding="utf-8")
                ready_written = True

        running = False
        for process in processes:
            exit_code = process.poll()
            if exit_code is None:
                running = True
                continue
            shutdown()
            raise SystemExit(exit_code)

        if not running:
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
