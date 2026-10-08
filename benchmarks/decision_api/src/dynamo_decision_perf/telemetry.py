# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only process telemetry without recording environment or command secrets."""

import json
import time
from pathlib import Path

import psutil


def snapshot(pids: list[int]) -> dict:
    processes = {}
    for pid in pids:
        try:
            parent = psutil.Process(pid)
            for process in (parent, *parent.children(recursive=True)):
                with process.oneshot():
                    cpu = process.cpu_times()
                    processes[process.pid] = {
                        "pid": process.pid,
                        "name": process.name(),
                        "cpu_user_seconds": cpu.user,
                        "cpu_system_seconds": cpu.system,
                        "rss_bytes": process.memory_info().rss,
                        "cpu_affinity": process.cpu_affinity(),
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "unix_ns": time.time_ns(),
        "host_cpu_percent": psutil.cpu_percent(),
        "host_memory_available_bytes": psutil.virtual_memory().available,
        "processes": list(processes.values()),
    }


def record_sample(path: Path, pids: list[int]) -> None:
    with path.open("a") as output:
        output.write(json.dumps(snapshot(pids)) + "\n")
