# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for dynamo.sglang unit tests only.
Handles conditional test collection to prevent import errors when the sglang
framework is not installed in the current container.
"""

import importlib.util
import sys

import pytest


def pytest_ignore_collect(collection_path, config):
    """Skip collecting sglang test files if sglang module isn't installed.
    Checks test file naming pattern: test_sglang_*.py
    """
    filename = collection_path.name
    if filename.startswith("test_sglang_"):
        if importlib.util.find_spec("sglang") is None:
            return True  # sglang not available, skip this file
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
