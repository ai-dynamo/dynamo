# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module-scoped guard: GMS tests must not leak GPU memory on any device.

GMSServer hosts the RPC server in a subprocess so CUDA state dies with it.
If that ever regresses, this fixture catches it at the module boundary with
a clear message instead of leaving it as a mystery OOM downstream.
"""

from __future__ import annotations

import logging
import os
import subprocess

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
    before = _gpu_memory_usage()
    yield
    # An externally owned GMS server intentionally remains alive until its
    # CUDA/CRIU controller completes validation and cleanup.
    if os.environ.get("DYN_GMS_EXTERNAL_SERVER") == "1":
        return
    after = _gpu_memory_usage()
    if before is None or after is None:
        return

    leaked_mib = [end - start for start, end in zip(before, after)]
    logger.info(
        "GPU memory.used before/after/delta (MiB): %s",
        list(zip(before, after, leaked_mib)),
    )
    leakers = [
        (gpu_id, mib)
        for gpu_id, mib in enumerate(leaked_mib)
        if mib >= _LEAK_THRESHOLD_MIB
    ]
    assert not leakers, (
        f"GMS tests leaked GPU memory on device(s): {leakers} "
        f"(threshold {_LEAK_THRESHOLD_MIB} MiB per device)."
    )


def _gpu_memory_usage() -> list[int] | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
        usage = [int(line) for line in out.strip().splitlines()]
        if not usage:
            raise ValueError("nvidia-smi returned no GPU rows")
        return usage
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        if _REQUIRE_GPU_MEMORY_CHECK:
            detail = getattr(exc, "output", None) or str(exc)
            raise AssertionError(
                f"required GPU leak measurement failed: {detail}"
            ) from exc
        logger.warning("Skipping unavailable GPU leak measurement: %s", exc)
        return None
