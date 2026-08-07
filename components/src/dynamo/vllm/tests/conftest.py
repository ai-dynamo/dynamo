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
_vllm_importable: bool | None = None


def _can_import_vllm() -> bool:
    """Try to import a canonical vLLM submodule once and cache the result.

    Some CI images carry a top-level ``vllm`` namespace without the full vLLM
    package. ``find_spec("vllm")`` is true there, but backend tests still fail
    during collection when they import ``vllm.config``/``vllm.inputs``.
    """
    global _vllm_importable
    if _vllm_importable is None:
        try:
            importlib.import_module("vllm.config")
            _vllm_importable = True
        except Exception:
            _vllm_importable = False
    return _vllm_importable


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
        if not _can_import_vllm():
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


@pytest.fixture
def vllm_cpu_platform_when_no_accelerator():
    """Pin vLLM's CpuPlatform on hosts with no accelerator.

    Tests that build the vLLM engine argument parser hit
    ``DeviceConfig.__post_init__``, which resolves ``current_platform`` when
    ``device`` is left at ``"auto"``. On a host with no accelerator every
    builtin platform plugin declines, ``resolve_current_platform_cls_qualname``
    falls back to ``UnspecifiedPlatform`` (``device_type == ""``) and the
    dataclass raises ``RuntimeError: Failed to infer device type``. The parser
    instantiates each config dataclass to compute its argparse default, so this
    fires while *building* the parser -- passing ``--device cpu`` cannot avoid
    it.

    ``vllm.platforms`` defines a module-level ``__setattr__`` whose whole job is
    installing a platform object, so assigning through it is the supported way
    in. Pin it only when torch reports no CUDA device, so GPU runners keep
    exercising the real detection path. Save and restore the private
    ``_current_platform`` rather than reading the public attribute, because
    reading it would itself force lazy resolution and cache
    ``UnspecifiedPlatform``; teardown then keeps a worker that ran these tests
    from leaking a pinned platform into another module's tests.
    """
    import torch

    if torch.cuda.is_available():
        yield
        return

    import vllm.platforms as vllm_platforms
    from vllm.platforms.cpu import CpuPlatform

    previous = vllm_platforms._current_platform
    vllm_platforms.current_platform = CpuPlatform()
    yield
    vllm_platforms.current_platform = previous


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
