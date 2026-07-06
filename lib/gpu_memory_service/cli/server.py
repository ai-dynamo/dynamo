# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS server entry point.

Launches one GMS server process per GPU serving both the weights and kv_cache
tags, then supervises them: terminates the rest if any child exits, and
propagates the first non-zero exit code. Runs until SIGTERM (pod termination
kills it) or until a child exits.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import time

from gpu_memory_service.common.cuda_utils import list_devices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_TAGS = ("weights", "kv_cache")


def _start_processes(devices: list[int]) -> list[subprocess.Popen]:
    processes = []
    for device in devices:
        command = [sys.executable, "-m", "gpu_memory_service", "--device", str(device)]
        for tag in _TAGS:
            command += ["--tag", tag]
        proc = subprocess.Popen(command)
        logger.info("Started GMS device=%d pid=%d", device, proc.pid)
        processes.append(proc)
    return processes


def main() -> None:
    processes = _start_processes(list_devices())

    def shutdown() -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    def terminate(*_args) -> None:
        shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    while True:
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
