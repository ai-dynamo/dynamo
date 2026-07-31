# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS server entry point.

Launches one GMS server process per GPU serving every production GMS tag,
then supervises them: terminates the rest if any child exits, and propagates
the first non-zero exit code. Runs until SIGTERM (pod termination kills it)
or until a child exits.
"""

from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import time

from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _child_command(device: int, device_type: str) -> list[str]:
    """Command for one child process serving every production tag on one GPU."""
    return [
        sys.executable,
        "-m",
        "gpu_memory_service",
        "--device",
        str(device),
        "--device-type",
        device_type,
    ]


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
    parser = argparse.ArgumentParser(
        description="GPU Memory Service supervisor (one server per (device, tag))."
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default=VMMDeviceType.CUDA.value,
        choices=[d.value for d in VMMDeviceType],
        help="VMM device type forwarded to server (default: cuda).",
    )
    args = parser.parse_args()

    init_vmm(VMMDeviceType.from_str(args.device_type))
    vmm = get_vmm()
    vmm.ensure_initialized()
    devices = vmm.list_devices()
    processes = []
    for device in devices:
        proc = subprocess.Popen(_child_command(device, args.device_type))
        logger.info(
            "Started GMS device=%d device_type=%s pid=%d",
            device,
            args.device_type,
            proc.pid,
        )
        processes.append(proc)

    def terminate(*_args) -> None:
        _terminate_all(processes)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    raise SystemExit(_supervise(processes))


if __name__ == "__main__":
    main()
