# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import subprocess
import sys
import time

from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm

logger = logging.getLogger(__name__)


def requests_help(argv: list[str]) -> bool:
    # argparse also accepts unambiguous abbreviations of --help, such as --hel.
    return any(
        argument == "-h" or (len(argument) > 2 and "--help".startswith(argument))
        for argument in argv
    )


def _is_scoped(argv: list[str]) -> bool:
    return requests_help(argv) or any(
        argument == "--device" or argument.startswith("--device=") for argument in argv
    )


def start_per_device(
    module: str, argv: list[str], devices: list[int]
) -> list[subprocess.Popen]:
    scoped = _is_scoped(argv)
    processes = []
    for device in [None] if scoped else devices:
        extra = [] if device is None else ["--device", str(device)]
        process = subprocess.Popen([sys.executable, "-m", module, *argv, *extra])
        logger.info("Started %s device=%s pid=%d", module, device, process.pid)
        processes.append(process)
    return processes


def run_per_device(module: str, argv: list[str]) -> None:
    scoped = _is_scoped(argv)
    if not scoped:
        init_vmm(VMMDeviceType.CUDA)
    processes = start_per_device(
        module, argv, [] if scoped else get_vmm().list_devices()
    )
    try:
        pending = list(processes)
        while pending:
            for process in list(pending):
                exit_code = process.poll()
                if exit_code is None:
                    continue
                if exit_code:
                    raise SystemExit(exit_code)
                pending.remove(process)
            if pending:
                time.sleep(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
