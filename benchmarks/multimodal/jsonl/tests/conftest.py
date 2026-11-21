# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# benchmarks/pyproject.toml carries its own [tool.pytest.ini_options], so
# collecting a path under benchmarks/ makes that directory the rootdir and the
# repository root is never added to sys.path. These tests import
# benchmarks.multimodal.jsonl, which then fails to resolve under a bare
# `pytest benchmarks/...` invocation. Add the repository root explicitly, the
# same shim benchmarks/router/tests/conftest.py uses for its own imports.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
