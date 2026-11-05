#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a symmetric two-arm CPU/GPU layout from the SLURM allocation."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _thread_group(cpu: int, allowed: set[int]) -> tuple[int, ...]:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    siblings: set[int] = set()
    for part in path.read_text(encoding="utf-8").strip().split(","):
        bounds = [int(value) for value in part.split("-", 1)]
        siblings.update(range(bounds[0], bounds[-1] + 1))
    return tuple(sorted(siblings & allowed))


def _numa_node(cpu: int) -> int:
    matches = sorted(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
    return int(matches[0].name.removeprefix("node")) if matches else -1


def _format_cpu_list(cpus: list[int]) -> str:
    if not cpus:
        raise ValueError("empty CPU assignment")
    ranges: list[str] = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def main() -> int:
    visible = [
        value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    ]
    if len(visible) != 2 or not all(value.isdigit() for value in visible):
        raise ValueError(
            "same-node demo requires exactly two integer CUDA_VISIBLE_DEVICES entries"
        )

    allowed = set(os.sched_getaffinity(0))
    groups = sorted({_thread_group(cpu, allowed) for cpu in allowed})
    by_numa: dict[int, list[tuple[int, ...]]] = {}
    for group in groups:
        by_numa.setdefault(_numa_node(group[0]), []).append(group)

    assignments: list[list[int]] = [[], []]
    for numa_groups in by_numa.values():
        for index, group in enumerate(sorted(numa_groups)):
            assignments[index % 2].extend(group)
    if abs(len(assignments[0]) - len(assignments[1])) > 1:
        raise ValueError("unable to derive balanced CPU assignments")

    output = {
        "control": {
            "gpu_index": int(visible[0]),
            "cpuset": _format_cpu_list(sorted(assignments[0])),
        },
        "dynamo-vllm": {
            "gpu_index": int(visible[1]),
            "cpuset": _format_cpu_list(sorted(assignments[1])),
        },
        "allowed_cpu_count": len(allowed),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
