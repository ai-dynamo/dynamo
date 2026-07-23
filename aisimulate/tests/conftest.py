# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect AI Simulate tests only where its standalone package is available."""

from importlib.util import find_spec
from pathlib import Path

# Most Dynamo runtime images copy the repository into /workspace but intentionally
# do not install AI Simulate. Only the planner image installs the standalone wheel.
# Ignore this optional suite before importing its test modules in those images.
try:
    _spica_available = find_spec("spica.config") is not None
except ModuleNotFoundError:
    _spica_available = False

collect_ignore = []
if not _spica_available:
    collect_ignore.append(str(Path(__file__).parent / "spica"))
