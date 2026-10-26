# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start one GMS V1 server process per GPU of the container.

GMS device N is the container's N-th GPU in PCI bus order, the order engines
see with CUDA_DEVICE_ORDER=PCI_BUS_ID. Its process is pinned to that GPU with
CUDA_VISIBLE_DEVICES and serves device N's sockets.
"""

from __future__ import annotations

import functools
import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gpu_memory_service.v1 import landlock

logger = logging.getLogger(__name__)

_PROC_GPUS = Path("/proc/driver/nvidia/gpus")
_DEV = Path("/dev")


@dataclass(frozen=True)
class Gpu:
    bus_id: str
    minor: int
    uuid: str


def node_gpus(proc_gpus: Path = _PROC_GPUS, dev: Path = _DEV) -> list[Gpu]:
    """GPUs this container can open, in PCI bus order, without starting CUDA."""
    gpus = []
    for entry in proc_gpus.iterdir():
        info = dict(
            line.split(":", 1)
            for line in (entry / "information").read_text().splitlines()
            if ":" in line
        )
        minor = int(info["Device Minor"])
        if os.access(dev / f"nvidia{minor}", os.R_OK | os.W_OK):
            gpus.append(Gpu(entry.name.lower(), minor, info["GPU UUID"].strip()))
    return sorted(gpus, key=lambda gpu: gpu.bus_id)


def start_ranks(
    gpus: list[Gpu], command: Callable[[int], list[str]], *, confine: bool
) -> list[subprocess.Popen]:
    """Start ``command(n)`` for GMS device n on each GPU.

    With ``confine``, each process can open no GPU node but its own (see
    ``landlock``), so CUDA starts on that GPU alone.
    """
    processes = []
    for device, gpu in enumerate(gpus):
        process = subprocess.Popen(
            command(device),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu.uuid},
            preexec_fn=(
                functools.partial(landlock.confine_to_gpu, gpu.minor)
                if confine
                else None
            ),
        )
        logger.info(
            "Started GMS V1 device=%d gpu=%s bus=%s confined=%s pid=%d",
            device,
            gpu.uuid,
            gpu.bus_id,
            confine,
            process.pid,
        )
        processes.append(process)
    return processes
