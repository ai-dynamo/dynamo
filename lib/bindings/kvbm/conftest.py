# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for kvbm bindings.

Conditionally skips v2 vllm integration when vllm is not installed.
"""

import importlib.util
from pathlib import Path

import pytest

VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

# vllm-gated patterns: skip when vllm is not installed
VLLM_PATTERNS = [
    "/v2/vllm/",
    "\\v2\\vllm\\",
    "test_kvbm_vllm_integration.py",
]


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """Skip vllm-dependent tests when vllm is missing."""
    path_str = str(collection_path)

    if not VLLM_AVAILABLE:
        for pattern in VLLM_PATTERNS:
            if pattern in path_str:
                return True

    return False


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_vllm: mark test as requiring vllm to be installed",
    )
