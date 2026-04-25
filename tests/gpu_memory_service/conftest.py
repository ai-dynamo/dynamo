# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module-scoped guard: GMS tests must not leak GPU memory on any device.

GMSServer hosts the RPC server in a subprocess so CUDA state dies with it.
If that ever regresses, this fixture catches it at the module boundary with
a clear message instead of leaving it as a mystery OOM downstream.
"""

from __future__ import annotations

import logging
import subprocess

import pytest

logger = logging.getLogger(__name__)

# Per-GPU threshold: absorbs small driver baseline residue, catches any real
# leak (the bug that motivated the subprocess refactor was ~2.4 GiB).
_LEAK_THRESHOLD_MIB = 100


@pytest.fixture(scope="module", autouse=True)
def _assert_no_gpu_memory_leak():
    yield
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        usage_mib = [int(line) for line in out.strip().splitlines()]
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return  # No GPU or unparseable output — skip the check.

    # Surfaced live in CI via pyproject.toml's `log_cli_level = "INFO"` so we
    # can confirm the guard is firing with expected values. Safe to leave in —
    # one line per module, useful forensics if a later test ever OOMs.
    logger.info("post-module memory.used per GPU (MiB): %s", usage_mib)

    leakers = [
        (gpu_id, mib)
        for gpu_id, mib in enumerate(usage_mib)
        if mib >= _LEAK_THRESHOLD_MIB
    ]
    assert not leakers, (
        f"GMS tests left memory pinned on GPU(s): {leakers} "
        f"(threshold {_LEAK_THRESHOLD_MIB} MiB per device)."
    )
