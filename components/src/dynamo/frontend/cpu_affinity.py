#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Warn when the frontend can run across multiple NUMA nodes."""

import logging
import os
from pathlib import Path

_CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")


def _format_cpu_ranges(cpus: list[int]) -> str:
    ranges: list[tuple[int, int]] = []
    for cpu in cpus:
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], cpu)
        else:
            ranges.append((cpu, cpu))
    return ",".join(
        str(start) if start == end else f"{start}-{end}" for start, end in ranges
    )


def warn_if_frontend_cpu_affinity_spans_numa_nodes(
    logger: logging.Logger,
) -> None:
    """Log a warning if the current CPU affinity covers multiple NUMA nodes."""

    try:
        cpus = sorted(os.sched_getaffinity(0))
        nodes = sorted(
            {
                node_path.name
                for cpu in cpus
                for node_path in (_CPU_SYSFS_ROOT / f"cpu{cpu}").glob("node[0-9]*")
            }
        )
    except (AttributeError, OSError) as error:
        logger.debug("NUMA affinity check skipped: %s", error, exc_info=True)
        return

    if not nodes:
        logger.debug(
            "NUMA affinity check skipped: no NUMA node links found for CPUs %s",
            _format_cpu_ranges(cpus),
        )
        return

    if len(nodes) == 1:
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
        _format_cpu_ranges(cpus),
        ", ".join(nodes),
        border,
    )
