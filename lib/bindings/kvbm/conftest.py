# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for kvbm bindings.

Skips v1 integration modules (vllm_integration, trtllm_integration) which
are deprecated and not actively maintained. Only v2 code is tested.

Also conditionally skips v2/vllm/ when vllm is not installed.
"""

from pathlib import Path

import pytest

# Check if vllm is available (for v2 vllm integration)
try:
    import vllm  # noqa: F401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# V1 integration patterns - always skip these (deprecated)
V1_SKIP_PATTERNS = [
    "/vllm_integration/",
    "\\vllm_integration\\",
    "/trtllm_integration/",
    "\\trtllm_integration\\",
    "test_kvbm_vllm_integration.py",
]

# V2 vllm patterns - skip only when vllm is not installed
V2_VLLM_PATTERNS = ["/v2/vllm/", "\\v2\\vllm\\"]


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """
    Skip collection of deprecated v1 integration files and v2 vllm files
    when vllm is not installed.
    """
    path_str = str(collection_path)

    # Always skip v1 integration files (deprecated)
    for pattern in V1_SKIP_PATTERNS:
        if pattern in path_str:
            return True

    # Skip v2 vllm files only when vllm is not installed
    if not VLLM_AVAILABLE:
        for pattern in V2_VLLM_PATTERNS:
            if pattern in path_str:
                return True

    return False


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_vllm: mark test as requiring vllm to be installed",
    )
