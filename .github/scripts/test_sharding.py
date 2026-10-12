# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Duration-balanced GPU shards, applied after pytest's normal selection hooks.

Profiled tests already run in independent subprocesses, so each can be assigned
separately. Other tests stay with their module to preserve fixture sharing.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import pytest


def junit_identity(nodeid: str) -> str:
    parts = nodeid.split("::")
    classname = parts[0].removesuffix(".py").replace("/", ".")
    if len(parts) > 2:
        classname += "." + ".".join(parts[1:-1])
    return classname + "::" + parts[-1]


def assign_groups(
    groups: dict[str, float], count: int
) -> tuple[dict[str, int], list[float]]:
    """Longest processing time first, with stable ties for identical collections."""
    if count < 1:
        raise ValueError("shard count must be positive")
    loads = [0.0] * count
    assignment = {}
    for group, duration in sorted(groups.items(), key=lambda pair: (-pair[1], pair[0])):
        shard = min(range(count), key=lambda index: (loads[index], index))
        assignment[group] = shard
        loads[shard] += duration
    return assignment, loads


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # Execute after marker/VRAM deselection. The GPU run loop reads
    # session.items, which contains this shard; extra metadata is harmless.
    yield
    count = int(os.environ.get("DYNAMO_GPU_SHARD_COUNT", "1"))
    if count == 1:
        return
    index = int(os.environ["DYNAMO_GPU_SHARD_INDEX"])
    if count < 1 or not 0 <= index < count:
        raise pytest.UsageError("invalid GPU shard index/count")
    pool = os.environ["DYNAMO_GPU_SHARD_POOL"]
    data = json.loads(Path(os.environ["DYNAMO_GPU_SHARD_DURATIONS"]).read_text())
    if pool not in data:
        raise pytest.UsageError(f"GPU shard timing pool is missing: {pool}")
    weights = data[pool]
    fallback = max(
        60.0, statistics.median(value for value in weights.values() if value > 0)
    )
    groups: dict[str, float] = {}
    item_groups = {}
    unknown = 0
    for item in items:
        identity = junit_identity(item.nodeid)
        # The subprocess orchestrator already isolates each profiled test.
        group = (
            item.nodeid
            if item.get_closest_marker("profiled_vram_gib")
            else item.nodeid.split("::")[0]
        )
        item_groups[item.nodeid] = group
        duration = weights.get(identity, fallback)
        unknown += identity not in weights
        groups[group] = groups.get(group, 0.0) + duration
    assignment, loads = assign_groups(groups, count)
    selected, deselected = [], []
    for item in items:
        (
            selected if assignment[item_groups[item.nodeid]] == index else deselected
        ).append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_line(
            f"GPU shard {index + 1}/{count}: {len(selected)} tests; "
            f"estimated serial work {[round(load, 1) for load in loads]}s; "
            f"{unknown} tests use the {fallback:.1f}s fallback"
        )
    # An empty shard is legitimate when a marker pool has fewer groups than
    # shards. Exit 5 would otherwise turn that case into a failed CI job.
    config._dynamo_empty_gpu_shard = not selected


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED and getattr(
        session.config, "_dynamo_empty_gpu_shard", False
    ):
        session.exitstatus = pytest.ExitCode.OK
