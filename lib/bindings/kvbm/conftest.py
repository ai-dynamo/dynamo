# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for kvbm bindings.

Conditionally skips v1 and v2 vllm integration when vllm is not installed,
and v1 trtllm integration when tensorrt_llm is not installed. Both v1 and
v2 are first-class — no blanket skips.
"""

import importlib.util
from pathlib import Path

import pytest

VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
TRTLLM_AVAILABLE = importlib.util.find_spec("tensorrt_llm") is not None

# vllm-gated patterns: skip when vllm is not installed
VLLM_PATTERNS = [
    "/v1/vllm_integration/",
    "\\v1\\vllm_integration\\",
    "/v2/vllm/",
    "\\v2\\vllm\\",
    "test_kvbm_vllm_integration.py",
]

# trtllm-gated patterns: skip when tensorrt_llm is not installed
TRTLLM_PATTERNS = [
    "/v1/trtllm_integration/",
    "\\v1\\trtllm_integration\\",
]


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """Skip vllm/trtllm-dependent tests when those packages are missing."""
    path_str = str(collection_path)

    if not VLLM_AVAILABLE:
        for pattern in VLLM_PATTERNS:
            if pattern in path_str:
                return True

    if not TRTLLM_AVAILABLE:
        for pattern in TRTLLM_PATTERNS:
            if pattern in path_str:
                return True

    return False


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_vllm: mark test as requiring vllm to be installed",
    )
