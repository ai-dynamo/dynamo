# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for dynamo backend unit tests.

Handles conditional test collection to prevent import errors when backend
frameworks are not installed in the current container.
"""

import importlib.util
import sys

import pytest


def pytest_ignore_collect(collection_path, config):
    """Skip collecting backend test files if their framework isn't installed.

    Checks test file naming pattern: test_<backend>_*.py
    """
    filename = collection_path.name

    # Map test file prefixes to required modules
    backend_requirements = {
        "test_vllm_": "vllm",
        "test_sglang_": "sglang",
        "test_trtllm_": "tensorrt_llm",
    }

    for prefix, required_module in backend_requirements.items():
        if filename.startswith(prefix):
            if importlib.util.find_spec(required_module) is None:
                return True  # Module not available, skip this file

    return None  # Not a backend test or module available


def make_cli_args_fixture(module_name: str):
    """Create a pytest fixture for mocking CLI arguments for a backend.

    The returned fixture supports two call styles:

    1. Explicit:
        mock_vllm_cli("--model", "gpt-2", "--custom-jinja-template", "path.jinja")

    2. Kwargs (auto-converts underscores to hyphens):
        mock_vllm_cli(model="gpt-2", custom_jinja_template="path.jinja")

    Both produce: ["dynamo.vllm", "--model", "gpt-2", "--custom-jinja-template", "path.jinja"]

    Args:
        module_name: Module identifier for argv[0] (e.g., "dynamo.vllm")

    Returns:
        Pytest fixture that mocks sys.argv via monkeypatch
    """

    @pytest.fixture
    def mock_cli_args(monkeypatch):
        """Mock sys.argv with CLI arguments (explicit or kwargs style)."""

        def set_args(*args, **kwargs):
            if args:
                # Explicit style: pass argv elements directly
                argv = [module_name, *args]
            else:
                # Kwargs style: convert python arg to CLI arg
                argv = [module_name]
                for param_name, param_value in kwargs.items():
                    cli_flag = f"--{param_name.replace('_', '-')}"
                    argv.extend([cli_flag, str(param_value)])

            monkeypatch.setattr(sys, "argv", argv)

        return set_args

    return mock_cli_args
