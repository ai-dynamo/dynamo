# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test bootstrap — put the repo root on ``sys.path`` so tests can import
``container.compliance.lockfile.*`` without requiring an installable package.

The surrounding directories (``container/``, ``container/compliance/``) have no
``__init__.py`` — they are a script layout, not a Python package — and this
sub-package is intentionally self-contained. Adding an __init__.py to those
dirs would alter the existing script-invocation contract, so instead we just
make the namespace importable for tests via sys.path manipulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# tests -> lockfile -> compliance -> container -> <repo_root>
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
