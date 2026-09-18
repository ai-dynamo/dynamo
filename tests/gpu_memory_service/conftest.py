# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module-scoped guard: GMS tests must not leak GPU memory.

GMSServer hosts the RPC server in a subprocess so CUDA state dies with it.
The guard measures only pytest's process tree: unrelated workloads may share
these GPUs and must not make an otherwise isolated test fail.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# Per-GPU threshold: absorbs small driver baseline residue, catches any real
# leak (the bug that motivated the subprocess refactor was ~2.4 GiB).
_LEAK_THRESHOLD_MIB = int(os.environ.get("GMS_TEST_LEAK_THRESHOLD_MIB", "100"))
_REQUIRE_GPU_MEMORY_CHECK = os.environ.get(
    "GMS_TEST_REQUIRE_GPU_MEMORY_CHECK", "0"
).lower() in ("1", "true", "yes", "on")


@pytest.fixture(scope="module", autouse=True)
def _assert_no_gpu_memory_leak():
    root_pid = os.getpid()
    before = _gpu_memory_usage(_process_tree(root_pid))
    yield
    # An externally owned GMS server intentionally remains alive until its
    # CUDA/CRIU controller completes validation and cleanup.
    if os.environ.get("DYN_GMS_EXTERNAL_SERVER") == "1":
        return
    after = _gpu_memory_usage(_process_tree(root_pid))
    if before is None or after is None:
        return

    keys = before.keys() | after.keys()
    leaked_mib = {
        key: after.get(key, 0) - before.get(key, 0)
        for key in keys
        if after.get(key, 0) - before.get(key, 0) >= _LEAK_THRESHOLD_MIB
    }
    logger.info("GPU process memory before/after (MiB): %s / %s", before, after)
    assert not leaked_mib, (
        f"GMS tests leaked GPU memory in pytest process(es): {leaked_mib} "
        f"(threshold {_LEAK_THRESHOLD_MIB} MiB per process/device)."
    )


def _process_tree(root_pid: int) -> set[int]:
    """Return live descendants without depending on psutil in minimal CI images."""
    parents: dict[int, int] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            # comm is parenthesized and may contain spaces or ')' characters;
            # the fields after the final ')' start with state and ppid.
            pid = int(stat_path.parent.name)
            fields = stat_path.read_text().rsplit(")", 1)[1].split()
            parents[pid] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue

    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def _gpu_memory_usage(pids: set[int]) -> dict[tuple[int, str], int] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
        usage: dict[tuple[int, str], int] = {}
        for line in out.strip().splitlines():
            pid_text, gpu_uuid, memory_text = (part.strip() for part in line.split(","))
            pid = int(pid_text)
            if pid in pids:
                key = (pid, gpu_uuid)
                usage[key] = usage.get(key, 0) + int(memory_text)
        return usage
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        if _REQUIRE_GPU_MEMORY_CHECK:
            detail = getattr(exc, "output", None) or str(exc)
            raise AssertionError(
                f"required GPU leak measurement failed: {detail}"
            ) from exc
        logger.warning("Skipping unavailable GPU leak measurement: %s", exc)
        return None
