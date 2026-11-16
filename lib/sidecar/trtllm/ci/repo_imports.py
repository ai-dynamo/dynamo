# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make checkout test utilities available to standalone CI entry points."""

import sys
from pathlib import Path


def add_repo_root():
    root = str(Path(__file__).resolve().parents[4])
    if root not in sys.path:
        sys.path.insert(0, root)
