# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test setup for the docs/fern/scripts generator tests.

The top-level pyproject.toml applies ``--ignore-glob=docs/*`` so these tests
never run under the repo's default pytest invocation. Run them explicitly
with an isolated config::

    python3 -m pytest docs/fern/scripts/tests -c /dev/null

The conftest adds ``docs/fern/scripts`` to ``sys.path`` so the generator can
be imported by name without a package install step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def pytest_configure(config: pytest.Config) -> None:
    """Register the repository's required scheduling markers in isolation."""
    config.addinivalue_line("markers", "pre_merge: runs before merging")
    config.addinivalue_line("markers", "gpu_0: requires no GPU")
    config.addinivalue_line("markers", "unit: isolated unit test")
