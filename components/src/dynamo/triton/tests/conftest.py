# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for the Triton worker unit tests.

Skips collecting triton tests when ``tritonclient`` or ``tritonserver`` is
missing, mirroring the per-backend skip-guards used elsewhere in Dynamo.
"""

import importlib
import sys

import pytest

# Cached result of probing tritonclient / tritonserver.
# `None` = not attempted, `True` = importable, `False` = raised.
_tritonclient_importable: bool | None = None
_tritonserver_importable: bool | None = None


def _can_import_tritonclient() -> bool:
    """Import the worker's tritonclient deps once and cache the result."""
    global _tritonclient_importable
    if _tritonclient_importable is None:
        try:
            importlib.import_module("tritonclient.grpc.model_config_pb2")
            importlib.import_module("tritonclient.utils")
            _tritonclient_importable = True
        except Exception:
            _tritonclient_importable = False

    return _tritonclient_importable


def _can_import_tritonserver() -> bool:
    """Import the tritonserver modules the tests use once and cache the result."""
    global _tritonserver_importable
    if _tritonserver_importable is None:
        try:
            importlib.import_module("tritonserver")
            # test_triton_logging.py imports the compiled bindings directly, so
            # probe the deepest module to reject a stub namespace lacking them.
            importlib.import_module("tritonserver._c.triton_bindings")
            _tritonserver_importable = True
        except Exception:
            _tritonserver_importable = False

    return _tritonserver_importable


def pytest_ignore_collect(collection_path, config):
    """Skip collecting triton test files if triton deps aren't installed."""
    filename = collection_path.name
    if filename.startswith("test_triton_"):
        if not _can_import_tritonclient() or not _can_import_tritonserver():
            return True

    return None


def make_cli_args_fixture(module_name: str):
    """Create a pytest fixture for mocking CLI arguments for triton backend."""

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
