# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import time

from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm

logger = logging.getLogger(__name__)


def _device_scoped(argv: list[str]) -> bool:
    return any(
        a in {"-h", "--help"} or a == "--device" or a.startswith("--device=")
        for a in argv
    )


def start_per_device(
    module: str, argv: list[str], devices: list[int]
) -> list[subprocess.Popen]:
    targets: list[int | None] = [None] if _device_scoped(argv) else list(devices)
    processes = []
    for device in targets:
        extra = [] if device is None else ["--device", str(device)]
        process = subprocess.Popen([sys.executable, "-m", module, *argv, *extra])
        logger.info("Started %s device=%s pid=%d", module, device, process.pid)
        processes.append(process)
    return processes


def run_per_device(module: str, argv: list[str]) -> None:
    if _device_scoped(argv):
        importlib.import_module(module).main(argv)
        return
    init_vmm(VMMDeviceType.CUDA)
    processes = start_per_device(module, argv, get_vmm().list_devices())
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
