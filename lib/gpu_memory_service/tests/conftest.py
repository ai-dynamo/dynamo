# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for gpu_memory_service tests."""

import importlib
import importlib.util


def check_module_available(module_name: str) -> bool:
    """Check if a Python module is available and importable.

    Args:
        module_name: Name of the module to check (e.g., "pynvml", "torch")

    Returns:
        True if the module is available and importable, False otherwise
    """
    if importlib.util.find_spec(module_name) is None:
        return False
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


HAS_PYNVML = check_module_available("pynvml")
HAS_TORCH = check_module_available("torch")


def _check_gms_usable() -> bool:
    """Check if gpu_memory_service is fully importable (including dependencies)."""
    try:
        import gpu_memory_service  # noqa: F401
        import msgspec  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


HAS_GMS = _check_gms_usable()

# CUDA availability requires torch to be importable first
HAS_CUDA = False
if HAS_TORCH:
    import torch

    HAS_CUDA = torch.cuda.is_available()
