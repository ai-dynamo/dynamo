# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot saver/loader CLI helpers."""

from __future__ import annotations

import logging
import subprocess
import sys
import time

from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm

logger = logging.getLogger(__name__)


def run_v1_per_device(module: str, argv: list[str], label: str) -> None:
    init_vmm(VMMDeviceType.CUDA)
    processes: list[subprocess.Popen] = []
    try:
        for device in get_vmm().list_devices():
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    module,
                    *argv,
                    "--device",
                    str(device),
                ]
            )
            logger.info(
                "Started GMS V1 %s device=%d pid=%d", label, device, process.pid
            )
            processes.append(process)

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
