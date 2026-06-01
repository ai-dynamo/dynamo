# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for dynamo.vllm unit tests only.
Handles conditional test collection to prevent import errors when the vllm
framework is not installed in the current container.
"""

import importlib
import importlib.util
import sys

import pytest

# Cached result of attempting to import the omni handler module.
# `None` = not yet attempted, `True` = succeeded, `False` = raised.
_omni_importable: bool | None = None


def _can_import_omni() -> bool:
    """Try to import dynamo.vllm.omni.base_handler once and cache the result.

    Catches any exception, not just ImportError — vllm_omni's import chain
    can raise NotImplementedError (and other types) when vllm._C / libcuda
    aren't available on a CPU-only runner. importlib.util.find_spec is
    insufficient because it only resolves the top-level package, not the
    transitive imports that actually fail.
    """
    global _omni_importable
    if _omni_importable is None:
        try:
            importlib.import_module("dynamo.vllm.omni.base_handler")
            _omni_importable = True
        except Exception:
            _omni_importable = False
    return _omni_importable


def pytest_ignore_collect(collection_path, config):
    """Skip collecting vllm test files if vllm module isn't installed.
    Checks test file naming pattern: test_vllm_*.py
    """
    filename = collection_path.name
    if filename.startswith("test_vllm_"):
        if importlib.util.find_spec("vllm") is None:
            return True  # vllm not available, skip this file
    # Omni tests import dynamo.vllm.omni.* which transitively imports
    # vllm_omni at module load. On CPU-only sample-runtime runners the
    # import chain reaches vllm._C and raises (NotImplementedError when
    # libcuda.so.1 is missing). Each file's local try/except ImportError
    # doesn't catch this, so skip collection up-front if the canonical
    # omni module isn't importable.
    parts = collection_path.parts
    if "omni" in parts and filename.startswith("test_"):
        if not _can_import_omni():
            return True
    return None


def make_cli_args_fixture(module_name: str):
    """Create a pytest fixture for mocking CLI arguments for vllm backend."""

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
