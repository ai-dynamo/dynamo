# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start one GMS V1 server process per GPU of the container.

GMS device N's process is pinned to the container's N-th GPU in PCI bus order
(``device.node_gpus``) with CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import functools
import logging
import os
import subprocess
from collections.abc import Callable

from gpu_memory_service.v1 import landlock
from gpu_memory_service.v1.device import Gpu

logger = logging.getLogger(__name__)


def start_ranks(
    gpus: list[Gpu], command: Callable[[int, Gpu], list[str]], *, confine: bool
) -> list[subprocess.Popen]:
    """Start ``command(n, gpu)`` for GMS device n on each GPU.

    With ``confine``, each process can open no GPU node but its own (see
    ``landlock``), so CUDA starts on that GPU alone. Such a process cannot
    list the other GPUs, so ``command`` must tell it which GPU is its own.
    """
    processes = []
    for device, gpu in enumerate(gpus):
        process = subprocess.Popen(
            command(device, gpu),
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
