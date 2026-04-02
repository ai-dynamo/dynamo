# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU VRAM utilities for parallel test execution.

Functions:
    detect_gpus()                  Enumerate GPUs via pynvml
    auto_worker_count(gpus, limit) Calculate slot count for -n auto
    write_test_meta(items)         Serialize profiled/requested vram + timeout
    load_test_meta()               Read the serialized test metadata
    print_gpu_plan(gpus, limit, would_run)  Dry-run GPU plan summary

Usage:
    # Sequential (filter only)
    pytest --max-vram-gib=10 -m "gpu_1 and vllm" tests/serve/

    # Parallel (VRAM-aware scheduling)
    pytest --max-vram-gib=10 -n auto -m "gpu_1 and vllm" tests/serve/
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import pynvml

_logger = logging.getLogger(__name__)

# When 2+ tests run concurrently, reserve 15% of GPU VRAM for CUDA context
# overhead across processes.  A single test gets the full GPU (0% margin).
VRAM_MULTI_PROC_MARGIN = 0.15

_TEST_META_FILENAME = "pytest_gpu_parallel_test_meta.json"


def detect_gpus() -> list[dict]:
    """Return list of dicts with 'index', 'name', 'total_mib' per GPU.

    Uses pynvml (already a dependency via profile_pytest.py).
    Returns empty list if no GPUs or pynvml is unavailable.
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return []
    try:
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "total_mib": mem.total // (1024 * 1024),
                }
            )
        return gpus
    finally:
        pynvml.nvmlShutdown()


def auto_worker_count(
    gpus: list[dict],
    vram_limit: float,
    test_profiled_gibs: list[float] | None = None,
) -> int:
    """Calculate slot count for -n auto.

    Uses the smallest profiled test size (if provided) to maximize parallelism.
    Falls back to vram_limit when no test sizes are available.
    """
    if not gpus or vram_limit <= 0:
        return len(gpus) or 1
    min_gpu_gib = min(g["total_mib"] for g in gpus) / 1024.0
    budget_gib = min_gpu_gib * (1.0 - VRAM_MULTI_PROC_MARGIN)
    divisor = vram_limit
    if test_profiled_gibs:
        nonzero = [g for g in test_profiled_gibs if g > 0]
        if nonzero:
            divisor = min(nonzero)
    workers_per_gpu = max(1, int(budget_gib / divisor)) if divisor > 0 else 1
    return len(gpus) * workers_per_gpu


def write_test_meta(items, dest_dir: str | None = None) -> None:
    """Serialize profiled_vram_gib, timeout, and KV cache markers to JSON.

    Called from pytest_collection_modifyitems so the GPU orchestrator can
    read test metadata without re-collecting.
    """
    test_meta: dict[str, dict] = {}
    for item in items:
        meta: dict = {}
        profiled_mark = item.get_closest_marker("profiled_vram_gib")
        if profiled_mark and profiled_mark.args:
            meta["profiled_vram_gib"] = profiled_mark.args[0]
        kv_bytes_mark = item.get_closest_marker("requested_vllm_kv_cache_bytes")
        if kv_bytes_mark and kv_bytes_mark.args:
            meta["requested_vllm_kv_cache_bytes"] = kv_bytes_mark.args[0]
        timeout_mark = item.get_closest_marker("timeout")
        if timeout_mark and timeout_mark.args:
            meta["timeout"] = timeout_mark.args[0]
        kv_tokens_mark = item.get_closest_marker("requested_sglang_kv_tokens")
        if kv_tokens_mark and kv_tokens_mark.args:
            meta["requested_sglang_kv_tokens"] = kv_tokens_mark.args[0]
        skip_mark = item.get_closest_marker("skip")
        if skip_mark:
            reason = skip_mark.kwargs.get("reason", "")
            if not reason and skip_mark.args:
                reason = skip_mark.args[0]
            meta["skip_reason"] = reason or "skipped"
        if meta:
            test_meta[item.nodeid] = meta
    if test_meta:
        path = os.path.join(dest_dir or tempfile.gettempdir(), _TEST_META_FILENAME)
        with open(path, "w") as f:
            json.dump(test_meta, f)


def load_test_meta() -> dict[str, dict]:
    """Load the nodeid -> {profiled_vram_gib, timeout, ...} map."""
    path = os.path.join(tempfile.gettempdir(), _TEST_META_FILENAME)
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def print_gpu_plan(
    gpus: list[dict], vram_limit: float, would_run: list[tuple[str, float]]
) -> None:
    """Print the GPU-parallel plan section for --dry-run output."""
    min_gpu_gib = min(g["total_mib"] for g in gpus) / 1024.0
    budget_gib = min_gpu_gib * (1.0 - VRAM_MULTI_PROC_MARGIN)
    profiled_gibs = [gib for _, gib in would_run if gib is not None and gib > 0]
    min_test_gib = min(profiled_gibs) if profiled_gibs else vram_limit
    auto_slots = max(1, int(budget_gib / min_test_gib)) if min_test_gib > 0 else 1

    print(f"\n{'=' * 60}")
    print("GPU-Parallel Plan")
    print(f"{'=' * 60}")
    for gpu in gpus:
        gib = gpu["total_mib"] / 1024
        print(f"  GPU {gpu['index']}: {gpu['name']} ({gib:.1f} GiB)")
    print(f"\n  Usable VRAM: {budget_gib:.0f} GiB")
    print("\n  Run options:")
    print("    (no -n)  : sequential, 1 test at a time")
    print(
        f"    -n auto  : up to {auto_slots} slots per GPU "
        f"({budget_gib:.0f} / {min_test_gib:.0f} GiB smallest test)"
    )
    print(f"    -n N     : N concurrent slots across {len(gpus)} GPU(s)")
    print("\n  Usage:")
    print(
        f"    pytest --max-vram-gib={vram_limit:.0f} -n {auto_slots} "
        f'-m "gpu_1 and vllm" tests/serve/'
    )
