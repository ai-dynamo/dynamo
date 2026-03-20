#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility wrapper for the built-in ``dynamo.fastvideo`` backend."""

from __future__ import annotations

import sys
from pathlib import Path

from dynamo.fastvideo.main import main

REPO_ROOT = Path(__file__).resolve().parents[2]
components_src = REPO_ROOT / "components/src"
if components_src.exists():
    sys.path.insert(0, str(components_src))

if __name__ == "__main__":
    main()
