#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Warn when the frontend can run across multiple NUMA nodes."""

import logging
import os
from collections.abc import Callable
from pathlib import Path

_CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")


def warn_if_frontend_cpu_affinity_spans_numa_nodes(
    logger: logging.Logger,
    *,
    cpu_root: Path = _CPU_SYSFS_ROOT,
    affinity_getter: Callable[[int], set[int]] | None = None,
) -> None:
    """Log a warning if the current CPU affinity covers multiple NUMA nodes."""

    affinity_getter = affinity_getter or getattr(os, "sched_getaffinity", None)
    if affinity_getter is None:
        return

    try:
        cpus = sorted(affinity_getter(0))
        nodes = sorted(
            {
                int(node_path.name[4:])
                for cpu in cpus
                for node_path in (cpu_root / f"cpu{cpu}").glob("node[0-9]*")
                if node_path.name[4:].isdigit()
            }
        )
    except Exception:
        return

    if len(nodes) < 2:
        return

    border = "=" * 80
    logger.warning(
        "%s\n"
        "WARNING: Frontend CPU affinity spans multiple NUMA nodes!\n"
        "Tokio worker threads may execute across NUMA domains, which can "
        "degrade performance.\n"
        "Available CPUs: %s\n"
        "NUMA nodes: %s\n"
        "Pin the frontend to CPUs from a single NUMA node.\n"
        "%s",
        border,
        cpus,
        nodes,
        border,
    )
