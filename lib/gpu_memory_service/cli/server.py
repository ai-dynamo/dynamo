# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS server entry point.

Launches one GMS server process per configured tag and GPU, then supervises
them: terminates the rest if any child exits, and propagates the first non-zero
exit code. Runs until SIGTERM (pod termination kills it)
or until a child exits.
"""

from __future__ import annotations

import logging
import os
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


_DEFAULT_TAGS = ("weights", "kv_cache")


def _tags_from_env() -> tuple[str, ...]:
    raw = os.environ.get("GMS_SERVER_TAGS")
    if not raw:
        return _DEFAULT_TAGS
    tags = tuple(tag.strip() for tag in raw.split(",") if tag.strip())
    if not tags:
        raise RuntimeError("GMS_SERVER_TAGS must contain at least one tag")
    return tags


def _child_command(device: int, tag: str | None = None) -> list[str]:
    command = [sys.executable, "-m", "gpu_memory_service", "--device", str(device)]
    if tag is not None:
        command.extend(("--tag", tag))
    return command


def _terminate_all(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()


def _supervise(processes: list[subprocess.Popen]) -> int:
    """Block until any child exits, terminate the rest, and return its exit code."""
    while processes:
        for process in processes:
            exit_code = process.poll()
            if exit_code is not None:
                _terminate_all(processes)
                return exit_code
        time.sleep(1)
    return 0


def main() -> None:
    processes = []
    for device in list_devices():
        for tag in _tags_from_env():
            proc = subprocess.Popen(_child_command(device, tag))
            logger.info("Started GMS device=%d tag=%s pid=%d", device, tag, proc.pid)
            processes.append(proc)

    def terminate(*_args) -> None:
        _terminate_all(processes)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    raise SystemExit(_supervise(processes))


if __name__ == "__main__":
    main()
