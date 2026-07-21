# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency availability flags for gpu_memory_service tests."""

from __future__ import annotations

import importlib.util

HAS_PYNVML = importlib.util.find_spec("pynvml") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _check_gms_usable() -> bool:
    """Check if gpu_memory_service is fully importable (including submodules)."""
    try:
        if importlib.util.find_spec("gpu_memory_service") is None:
            return False
        if importlib.util.find_spec("gpu_memory_service.client.rpc") is None:
            return False
        if importlib.util.find_spec("gpu_memory_service.server.rpc") is None:
            return False
        if importlib.util.find_spec("msgspec") is None:
            return False
        return True
    except ModuleNotFoundError:
        return False


HAS_GMS = _check_gms_usable()

# CUDA availability requires a full torch import
HAS_CUDA = False
if HAS_TORCH:
    import torch

    try:
        HAS_CUDA = torch.cuda.is_available()
    except Exception:
        HAS_CUDA = False
