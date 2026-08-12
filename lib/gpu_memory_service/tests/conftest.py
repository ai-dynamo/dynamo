# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest bootstrap for gpu_memory_service unit tests.

This keeps local source-tree imports stable and avoids collection-time import
errors on environments where optional dependencies are not installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# Ensure `import gpu_memory_service` resolves from source checkout when tests are
# run from the repository root without an editable install.
_REPO_LIB_DIR = Path(__file__).resolve().parents[2]
if str(_REPO_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_LIB_DIR))


def _has_spec(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


_HAS_GMS = _has_spec("gpu_memory_service")
_HAS_MSGSPEC = _has_spec("msgspec")

# The v1 test modules import gpu_memory_service/msgspec at module import time.
# Skip collecting them when dependencies are unavailable to avoid hard errors.
if not (_HAS_GMS and _HAS_MSGSPEC):
    collect_ignore_glob = ["test_v1_*.py"]


def pytest_ignore_collect(collection_path, config):
    """Skip v1 test modules before import when optional deps are missing."""
    if _HAS_GMS and _HAS_MSGSPEC:
        return False
    return collection_path.name.startswith("test_v1_")
