# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for dynamo.fastvideo unit tests."""

import importlib.util
import sys
from pathlib import Path

import pytest


def pytest_ignore_collect(collection_path, config):
    """Skip FastVideo tests in broad collection if FastVideo is unavailable."""
    path = Path(str(collection_path)).resolve()
    if not path.name.startswith("test_fastvideo"):
        return None
    if importlib.util.find_spec("fastvideo") is not None:
        return None
    if _is_direct_fastvideo_test_collection(path, config):
        return None
    return True


def _is_direct_fastvideo_test_collection(path: Path, config) -> bool:
    requested_paths = []
    for arg in getattr(config, "args", ()):
        try:
            requested_paths.append(Path(arg).resolve())
        except OSError:
            continue
    return path in requested_paths or path.parent in requested_paths


def make_cli_args_fixture(module_name: str):
    """Create a pytest fixture for mocking CLI arguments for FastVideo."""

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
