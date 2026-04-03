# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for GPU Memory Service tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Respect an already-resolvable GMS package first. This lets containerized runs
# point pytest at an explicit source overlay without this hook overriding it.
if importlib.util.find_spec("gpu_memory_service") is None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "lib" / "gpu_memory_service").is_dir():
            lib_dir = parent / "lib"
            if str(lib_dir) not in sys.path:
                sys.path.insert(0, str(lib_dir))
            break
    else:
        collect_ignore_glob = ["*"]
