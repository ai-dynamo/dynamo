# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensure the repository root is importable when pytest roots at benchmarks/."""

import sys
from pathlib import Path

# benchmarks/pyproject.toml defines [tool.pytest.ini_options], so running a bare
# `pytest benchmarks/multimodal/jsonl/tests/...` sets rootdir to benchmarks/ and
# omits the repository root from sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
