# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock GPU allocation test for profiler validation.

Local-only: this test is skipped in CI (GitHub Actions / GitLab CI).
Do NOT mark it as pre_merge, post_merge, nightly, or e2e -- it exists
solely to validate profile_pytest.py's binary search locally.
"""

import logging
import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") is not None
    or os.environ.get("GITHUB_ACTIONS") is not None
    or os.environ.get("GITLAB_CI") is not None,
    reason="Mock GPU allocation test is for local profiling only, not CI",
)

torch = pytest.importorskip("torch", reason="torch required for GPU allocation test")

logger = logging.getLogger(__name__)

ALLOC_MIB = 4096  # 4 GiB


# This cannot be pre_merge, post_merge, nightly, or e2e. It's a mock test for local testing.
@pytest.mark.gpu_1
@pytest.mark.timeout(30)
def test_mock_4gb_gpu_alloc():
    """Allocate 4 GiB of GPU VRAM, hold 2s, release. Honors _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = 0
    total_mib = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)

    kv_bytes_str = os.environ.get("_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES")
    if kv_bytes_str is not None:
        cap_mib = int(kv_bytes_str) / (1024 * 1024)
        logger.info(
            "_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES=%s -> cap %.0f MiB (%.1f GiB) of %.0f MiB total",
            kv_bytes_str,
            cap_mib,
            cap_mib / 1024,
            total_mib,
        )
        if ALLOC_MIB > cap_mib:
            raise RuntimeError(
                f"Requested {ALLOC_MIB} MiB exceeds KV cache cap "
                f"of {cap_mib:.0f} MiB ({kv_bytes_str} bytes)"
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
