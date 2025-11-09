# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for dynamo.trtllm unit tests only.
Handles conditional test collection to prevent import errors when the tensorrt_llm
framework is not installed in the current container.
"""

import importlib.util
import sys

import pytest


def pytest_ignore_collect(collection_path, config):
    """Skip collecting trtllm test files if tensorrt_llm module isn't installed.
    Checks test file naming pattern: test_trtllm_*.py
    """
    filename = collection_path.name
    if filename.startswith("test_trtllm_"):
        if importlib.util.find_spec("tensorrt_llm") is None:
            return True  # tensorrt_llm not available, skip this file
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """
    Auto-apply nightly marker to tests with certain markers.
    """
    auto_nightly_markers = {"pre_merge", "post_merge"}

    for item in items:
        # Auto-apply nightly marker FIRST before skip check
        item_marker_names = {m.name for m in item.own_markers}
        if (
            item_marker_names & auto_nightly_markers
            and "nightly" not in item_marker_names
        ):
            item.add_marker(pytest.mark.nightly)


def make_cli_args_fixture(module_name: str):
    """Create a pytest fixture for mocking CLI arguments for trtllm backend."""

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
