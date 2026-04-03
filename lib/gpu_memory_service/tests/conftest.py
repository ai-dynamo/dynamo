# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_PATHS = [
    REPO_ROOT,
    REPO_ROOT / "components" / "src",
    REPO_ROOT / "lib" / "bindings" / "python" / "src",
    REPO_ROOT / "lib",
]

for path in reversed(PROJECT_PATHS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
