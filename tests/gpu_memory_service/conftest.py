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
import threading
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
    initial = _process_snapshot()
    seen = _descendant_identities(root_pid, initial)
    before = _gpu_memory_usage(set(seen))
    stop = threading.Event()

    def sample_descendants() -> None:
        while not stop.wait(0.02):
            snapshot = _process_snapshot()
            seen.update(_descendant_identities(root_pid, snapshot))

    monitor = threading.Thread(
        target=sample_descendants, name="gms-test-child-monitor", daemon=True
    )
    monitor.start()
    yield
    stop.set()
    monitor.join(timeout=1.0)
    # An externally owned GMS server intentionally remains alive until its
    # CUDA/CRIU controller completes validation and cleanup.
    if os.environ.get("DYN_GMS_EXTERNAL_SERVER") == "1":
        return
    final = _process_snapshot()
    seen.update(_descendant_identities(root_pid, final))
    live_seen = _live_seen_identities(seen, final)
    after = _gpu_memory_usage(live_seen)
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


def _process_snapshot() -> dict[int, tuple[int, int]]:
    """Return pid -> (ppid, birth tick), robust to names and PID reuse."""
    result = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            pid = int(stat_path.parent.name)
            fields = stat_path.read_text().rsplit(")", 1)[1].split()
            result[pid] = (int(fields[1]), int(fields[19]))
        except (OSError, ValueError, IndexError):
            continue
    return result


def _descendant_identities(
    root_pid: int, snapshot: dict[int, tuple[int, int]]
) -> dict[int, int]:
    """Capture descendants by birth identity before they can be reparented."""
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _start_time) in snapshot.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return {pid: snapshot[pid][1] for pid in result if pid in snapshot}


def _live_seen_identities(
    seen: dict[int, int], final: dict[int, tuple[int, int]]
) -> set[int]:
    """Return captured birth identities still alive, even if reparented."""
    return {
        pid
        for pid, start_time in seen.items()
        if final.get(pid, (None, None))[1] == start_time
    }


def _process_tree(root_pid: int) -> set[int]:
    """Compatibility helper returning currently live descendants."""
    return set(_descendant_identities(root_pid, _process_snapshot()))


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
