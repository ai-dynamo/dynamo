# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect AI Simulate tests only where its standalone package is available.

PR CI runs the pre-merge subset in the Planner CPU jobs on amd64 and arm64.
Post-merge Planner jobs also select the integration-marked subset.
"""

from importlib.util import find_spec
from pathlib import Path

# Most Dynamo runtime images copy the repository into /workspace but intentionally
# do not install AI Simulate. Only the planner image installs the standalone wheel.
# Ignore this optional suite before importing its test modules in those images.
try:
    _core_available = find_spec("aisimulate.runner") is not None
except ModuleNotFoundError:
    _core_available = False

try:
    _sweeper_available = find_spec("aisimulate.sweeper.config") is not None
except ModuleNotFoundError:
    _sweeper_available = False

collect_ignore = []
if not _core_available:
    collect_ignore.extend(
        str(Path(__file__).parent / test_file)
        for test_file in (
            "test_runner.py",
            "test_traffic.py",
        )
    )
if not _sweeper_available:
    collect_ignore.append(str(Path(__file__).parent / "sweeper"))
