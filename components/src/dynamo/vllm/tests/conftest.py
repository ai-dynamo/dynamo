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
_multimodal_utils_importable: bool | None = None
_vllm_internals_importable: bool | None = None


def _can_import_vllm_internals() -> bool:
    """Try to import a vllm internal submodule once and cache the result.

    ``find_spec("vllm")`` returns non-None even when vllm is only partially
    installed (e.g. the top-level package exists but vllm.v1, vllm.config,
    vllm.inputs etc. are absent).  An actual import attempt is the only
    reliable way to detect this — used to gate all test_vllm_* files and
    any other tests that import vllm internals.
    """
    global _vllm_internals_importable
    if _vllm_internals_importable is None:
        try:
            importlib.import_module("vllm.v1.engine.async_llm")
            _vllm_internals_importable = True
        except Exception:
            _vllm_internals_importable = False
    return _vllm_internals_importable


def _can_import_multimodal_utils() -> bool:
    """Try to import dynamo.vllm.multimodal_utils once and cache the result.

    The multimodal_utils sub-package transitively imports vllm internals
    (e.g. chat_message_utils → vllm) that are not available in the
    dynamo-runtime image.  ``find_spec("vllm")`` returns non-None even
    when vllm is partially installed, so the spec check is insufficient —
    only a real import attempt reveals the failure.
    """
    global _multimodal_utils_importable
    if _multimodal_utils_importable is None:
        try:
            importlib.import_module("dynamo.vllm.multimodal_utils")
            _multimodal_utils_importable = True
        except Exception:
            _multimodal_utils_importable = False
    return _multimodal_utils_importable


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
    """Skip collecting test files that need vllm internals or optional deps.

    Uses real import attempts (not find_spec) because vllm can be partially
    installed: the top-level package resolves under find_spec but submodules
    such as vllm.v1, vllm.config, and vllm.inputs are absent in the
    dynamo-runtime image, causing collection errors at import time.
    """
    filename = collection_path.name
    parts = collection_path.parts

    # test_vllm_*.py files import vllm internals (vllm.v1, vllm.config, …).
    # Replace the find_spec guard with a real import check.
    if filename.startswith("test_vllm_") and not _can_import_vllm_internals():
        return True

    # tests/frontend/test_prepost*.py import vllm.entrypoints.openai which
    # also requires full vllm internals absent in dynamo-runtime.
    if filename.startswith("test_prepost") and "frontend" in parts:
        if not _can_import_vllm_internals():
            return True

    # examples/backends/sglang/test_sglang_expert_info.py requires pybase64
    # which is not installed in all CI images.
    if filename == "test_sglang_expert_info.py":
        if importlib.util.find_spec("pybase64") is None:
            return True

    # multimodal_utils tests import dynamo.vllm.multimodal_utils which
    # transitively imports vllm internals not present in dynamo-runtime.
    if "multimodal_utils" in parts and filename.startswith("test_"):
        if not _can_import_multimodal_utils():
            return True

    # Omni tests import dynamo.vllm.omni.* which transitively imports
    # vllm_omni at module load. On CPU-only sample-runtime runners the
    # import chain reaches vllm._C and raises (NotImplementedError when
    # libcuda.so.1 is missing).
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
