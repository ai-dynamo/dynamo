# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock GPU allocation test for profiler validation."""

import logging
import os
import time

import pytest

torch = pytest.importorskip("torch", reason="torch required for GPU allocation test")

logger = logging.getLogger(__name__)

ALLOC_MIB = 4096  # 4 GiB


@pytest.mark.gpu_1
@pytest.mark.timeout(30)
def test_mock_4gb_gpu_alloc():
    """Allocate 4 GiB of GPU VRAM, hold 5s, release. Honors DYN_GPU_MEMORY_FRACTION_OVERRIDE."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = 0
    total_mib = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)

    gpu_util = os.environ.get("DYN_GPU_MEMORY_FRACTION_OVERRIDE")
    if gpu_util is not None:
        cap_mib = total_mib * float(gpu_util)
        logger.info(
            "DYN_GPU_MEMORY_FRACTION_OVERRIDE=%.2f -> cap %.0f MiB (%.1f GiB) of %.0f MiB total",
            float(gpu_util),
            cap_mib,
            cap_mib / 1024,
            total_mib,
        )
        if ALLOC_MIB > cap_mib:
            raise RuntimeError(
                f"Requested {ALLOC_MIB} MiB exceeds DYN_GPU_MEMORY_FRACTION_OVERRIDE "
                f"cap of {cap_mib:.0f} MiB ({gpu_util})"
            )

    num_elements = (ALLOC_MIB * 1024 * 1024) // 4
    logger.info(
        "Allocating %d MiB (%.1f GiB) on cuda:%d ...",
        ALLOC_MIB,
        ALLOC_MIB / 1024,
        device,
    )

    tensor = torch.empty(num_elements, dtype=torch.float32, device=f"cuda:{device}")
    logger.info(
        "Allocated. torch reports %.0f MiB in use.",
        torch.cuda.memory_allocated(device) / (1024 * 1024),
    )

    time.sleep(2.0)

    del tensor
    torch.cuda.empty_cache()
    logger.info(
        "Released. torch reports %.0f MiB in use.",
        torch.cuda.memory_allocated(device) / (1024 * 1024),
    )
