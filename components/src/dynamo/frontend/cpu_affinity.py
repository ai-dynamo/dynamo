#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Detect the CPU and NUMA affinity inherited by the frontend runtime."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_NUMA_NODE_ROOT = Path("/sys/devices/system/node")

AffinityGetter = Callable[[int], Iterable[int]]


class NumaAffinityStatus(str, Enum):
    """Classification of the frontend's effective CPU affinity."""

    SINGLE_NODE = "single_node"
    MULTIPLE_NODES = "multiple_nodes"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CpuNumaAffinity:
    """Effective CPU affinity and its NUMA-node coverage."""

    available_cpus: tuple[int, ...]
    numa_nodes: tuple[int, ...]
    unmapped_cpus: tuple[int, ...]
    status: NumaAffinityStatus
    reason: str | None = None


def parse_cpu_list(cpu_list: str) -> set[int]:
    """Parse Linux cpulist syntax such as ``0-3,8,10-11``."""

    value = cpu_list.strip()
    if not value:
        return set()

    cpus: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"invalid empty CPU-list element in {cpu_list!r}")

        if "-" not in part:
            cpu = _parse_cpu_id(part, cpu_list)
            cpus.add(cpu)
            continue

        bounds = part.split("-")
        if len(bounds) != 2:
            raise ValueError(f"invalid CPU range {part!r} in {cpu_list!r}")
        start = _parse_cpu_id(bounds[0], cpu_list)
        end = _parse_cpu_id(bounds[1], cpu_list)
        if start > end:
            raise ValueError(f"descending CPU range {part!r} in {cpu_list!r}")
        cpus.update(range(start, end + 1))

    return cpus


def _parse_cpu_id(value: str, cpu_list: str) -> int:
    if not value.isdigit():
        raise ValueError(f"invalid CPU ID {value!r} in {cpu_list!r}")
    return int(value)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Format CPU or NUMA-node IDs using compact Linux cpulist syntax."""

    values = sorted(set(cpus))
    if not values:
        return "none"
    if values[0] < 0:
        raise ValueError("CPU-list values must be non-negative")

    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(_format_range(start, previous))
        start = previous = value
    ranges.append(_format_range(start, previous))
    return ",".join(ranges)


def _format_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def detect_cpu_numa_affinity(
    *,
    node_root: Path = _NUMA_NODE_ROOT,
    affinity_getter: AffinityGetter | None = None,
) -> CpuNumaAffinity:
    """Detect NUMA coverage of the current thread's effective CPU affinity.

    The frontend calls this before constructing ``DistributedRuntime``, so the
    resulting mask is inherited by the Tokio worker threads.
    """

    if affinity_getter is None:
        affinity_getter = getattr(os, "sched_getaffinity", None)
        if affinity_getter is None:
            return _unknown(reason="os.sched_getaffinity is unavailable")

    try:
        available_cpus = tuple(sorted(set(affinity_getter(0))))
    except (OSError, NotImplementedError) as error:
        return _unknown(reason=f"sched_getaffinity failed: {error}")

    if not available_cpus:
        return _unknown(reason="sched_getaffinity returned an empty CPU set")

    try:
        cpu_to_node = _read_cpu_to_node_map(node_root)
    except (OSError, ValueError) as error:
        return _unknown(
            available_cpus=available_cpus,
            unmapped_cpus=available_cpus,
            reason=f"failed to read NUMA topology: {error}",
        )

    mapped_nodes = {cpu_to_node[cpu] for cpu in available_cpus if cpu in cpu_to_node}
    unmapped_cpus = tuple(cpu for cpu in available_cpus if cpu not in cpu_to_node)
    numa_nodes = tuple(sorted(mapped_nodes))

    if len(numa_nodes) > 1:
        status = NumaAffinityStatus.MULTIPLE_NODES
        reason = None
    elif len(numa_nodes) == 1 and not unmapped_cpus:
        status = NumaAffinityStatus.SINGLE_NODE
        reason = None
    else:
        status = NumaAffinityStatus.UNKNOWN
        reason = (
            "NUMA topology did not map all available CPUs "
            f"({format_cpu_list(unmapped_cpus)})"
        )

    return CpuNumaAffinity(
        available_cpus=available_cpus,
        numa_nodes=numa_nodes,
        unmapped_cpus=unmapped_cpus,
        status=status,
        reason=reason,
    )


def _unknown(
    *,
    available_cpus: tuple[int, ...] = (),
    unmapped_cpus: tuple[int, ...] = (),
    reason: str,
) -> CpuNumaAffinity:
    return CpuNumaAffinity(
        available_cpus=available_cpus,
        numa_nodes=(),
        unmapped_cpus=unmapped_cpus,
        status=NumaAffinityStatus.UNKNOWN,
        reason=reason,
    )


def _read_cpu_to_node_map(node_root: Path) -> dict[int, int]:
    if not node_root.is_dir():
        raise OSError(f"NUMA node directory does not exist: {node_root}")

    cpu_to_node: dict[int, int] = {}
    node_paths = sorted(
        (
            path
            for path in node_root.iterdir()
            if path.is_dir()
            and path.name.startswith("node")
            and path.name[4:].isdigit()
        ),
        key=lambda path: int(path.name[4:]),
    )
    if not node_paths:
        raise OSError(f"no NUMA nodes found under {node_root}")

    for node_path in node_paths:
        node = int(node_path.name[4:])
        node_cpus = parse_cpu_list((node_path / "cpulist").read_text())
        for cpu in node_cpus:
            previous = cpu_to_node.setdefault(cpu, node)
            if previous != node:
                raise ValueError(
                    f"CPU {cpu} belongs to both NUMA node {previous} and {node}"
                )

    if not cpu_to_node:
        raise ValueError("NUMA topology contains no CPUs")
    return cpu_to_node


def warn_if_frontend_cpu_affinity_spans_numa_nodes(
    logger: logging.Logger,
    *,
    node_root: Path = _NUMA_NODE_ROOT,
    affinity_getter: AffinityGetter | None = None,
) -> CpuNumaAffinity:
    """Warn if the frontend can run across NUMA nodes without blocking startup."""

    try:
        result = detect_cpu_numa_affinity(
            node_root=node_root,
            affinity_getter=affinity_getter,
        )
    except Exception as error:
        result = _unknown(reason=f"unexpected affinity detection error: {error}")
    if result.status == NumaAffinityStatus.MULTIPLE_NODES:
        border = "=" * 80
        logger.warning(
            "%s\n"
            "WARNING: Frontend CPU affinity spans multiple NUMA nodes!\n"
            "Tokio worker threads may execute across NUMA domains, which can "
            "degrade performance.\n"
            "Available CPUs: %s\n"
            "NUMA nodes: %s\n"
            "Unmapped CPUs: %s\n"
            "Pin the frontend to CPUs from a single NUMA node.\n"
            "%s",
            border,
            format_cpu_list(result.available_cpus),
            format_cpu_list(result.numa_nodes),
            format_cpu_list(result.unmapped_cpus),
            border,
        )

    return result
