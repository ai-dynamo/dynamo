# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make ``dynamo_test`` importable when the harness has not been installed.

The repository root has no ``testpaths``, so a bare ``pytest`` from the root
collects this directory. Without this file every module here fails to import
with ``ModuleNotFoundError: No module named 'dynamo_test'`` and takes the whole
repository's collection down with it — a new package must not be able to break
the existing suite just by existing.

An installed copy always wins: the path is only added when the import genuinely
fails, so ``pip install -e ./harness`` behaves exactly as it would without this
file.
"""

import sys
from pathlib import Path

try:  # pragma: no cover - the installed case has nothing to do
    import dynamo_test  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
