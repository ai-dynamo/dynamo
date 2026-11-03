# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import time

from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm

logger = logging.getLogger(__name__)


def start_per_device(
    module: str, argv: list[str], devices: list[int]
) -> list[subprocess.Popen]:
    scoped = any(
        argument in {"-h", "--help"}
        or argument == "--device"
        or argument.startswith("--device=")
        for argument in argv
    )
    processes = []
    try:
        for device in [None] if scoped else devices:
            extra = [] if device is None else ["--device", str(device)]
            process = subprocess.Popen([sys.executable, "-m", module, *argv, *extra])
            logger.info("Started %s device=%s pid=%d", module, device, process.pid)
            processes.append(process)
    except BaseException:
        # The caller never receives these processes, so stop them here.
        _stop_all(processes)
        raise
    return processes


def _stop_all(processes: list[subprocess.Popen]) -> None:
    """Terminate the processes and wait for their own cleanup."""
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        process.wait()


def _exit_on_sigterm(signum: int, _frame) -> None:
    # Ignore a repeated SIGTERM so it cannot cut short the children's cleanup.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def run_per_device(
    module: str, argv: list[str], devices: list[int] | None = None
) -> None:
    """Run ``module`` once per device and wait. ``devices`` defaults to every
    CUDA device, which starts CUDA in this process."""
    scoped = any(
        argument in {"-h", "--help"}
        or argument == "--device"
        or argument.startswith("--device=")
        for argument in argv
    )
    if not scoped and devices is None:
        init_vmm(VMMDeviceType.CUDA)
        devices = get_vmm().list_devices()
    # Without a handler SIGTERM would kill this process without running the
    # finally below, or, as a container's PID 1, be ignored until SIGKILL.
    # Either way the children would never get to clean up.
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    processes = start_per_device(module, argv, [] if scoped else devices)
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
        _stop_all(processes)
