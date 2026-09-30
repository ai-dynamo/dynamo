# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make the ``compliance`` package importable for these tests.

The compliance Dockerfile stage runs the package with ``PYTHONPATH=container``.
The repo-wide pytest collection (e.g. the dynamo-runtime suite) doesn't set that
path, so ``from compliance...`` would fail at collection time. Prepend the
``container/`` dir here so collection succeeds wherever these tests are gathered.
"""

import sys
from pathlib import Path

_CONTAINER_DIR = str(Path(__file__).resolve().parents[2])  # -> container/
if _CONTAINER_DIR not in sys.path:
    sys.path.insert(0, _CONTAINER_DIR)
