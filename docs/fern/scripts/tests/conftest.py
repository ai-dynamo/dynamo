# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test setup for the docs/fern/scripts generator tests.

The top-level pyproject.toml applies ``--ignore-glob=docs/*`` to skip docs
site content during ordinary test collection, but pytest treats explicit
path arguments as opt-in overrides, so running::

    python3 -m pytest docs/fern/scripts/tests

collects these tests under the repository's shared pytest configuration
(strict markers, warning filters, and the registered marker list). The
conftest adds ``docs/fern/scripts`` to ``sys.path`` so the generator can
be imported by name without a package install step.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
