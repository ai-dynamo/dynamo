# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for dynamo.sglang unit tests only.
Handles conditional test collection to prevent import errors when the sglang
framework is not installed in the current container.
"""

import importlib.util
import sys

import pytest


def _sgl_kernel_functional() -> bool:
    """Check if sgl_kernel's CUDA ops can be loaded.

    sglang-kernel 0.4.1+ ships precompiled CUDA .so files that fail to load
    on platforms without a compatible GPU (e.g., arm64 CPU-only CI nodes).
    Cache the result so we only probe once per session.
    """
    if not hasattr(_sgl_kernel_functional, "_result"):
        try:
            import sgl_kernel  # noqa: F401

            _sgl_kernel_functional._result = True
        except (ImportError, OSError):
            _sgl_kernel_functional._result = False
    return _sgl_kernel_functional._result


# Test files that import from request_handlers (which transitively pulls in
# sgl_kernel via the sglang quantization/engine import chain).
_NEEDS_SGL_KERNEL = {
    "test_sglang_decode_handler.py",
    "test_sglang_image_diffusion_handler.py",
    "test_sglang_memory_occupation_handlers.py",
    "test_sglang_multimodal_embedding_cache.py",
    "test_sglang_unit.py",
}


def pytest_ignore_collect(collection_path, config):
    """Skip collecting sglang test files when required dependencies are missing.

    - All test_sglang_*.py files: skip if sglang is not installed.
    - Files that import request_handlers: also skip if sgl_kernel CUDA ops
      can't load (arm64 CPU-only CI).
    """
    filename = collection_path.name
    if filename.startswith("test_sglang_"):
        if importlib.util.find_spec("sglang") is None:
            return True  # sglang not available, skip this file
        if filename in _NEEDS_SGL_KERNEL and not _sgl_kernel_functional():
            return True  # CUDA ops unavailable, skip to avoid collection error
    return None


def make_cli_args_fixture(module_name: str):
    """Create a pytest fixture for mocking CLI arguments for sglang backend."""

    @pytest.fixture
    def mock_cli_args(monkeypatch):
        def set_args(*args, **kwargs):
            if args:
                argv = [module_name, *args]
            else:
                argv = [module_name]
                for param_name, param_value in kwargs.items():
                    cli_flag = f"--{param_name.replace('_', '-')}"
                    argv.extend([cli_flag, str(param_value)])
            monkeypatch.setattr(sys, "argv", argv)

        return set_args

    return mock_cli_args
