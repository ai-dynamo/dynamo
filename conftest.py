# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block dynamo.vllm/sglang from shadowing the installed vllm/sglang.

Pytest collection puts components/src/dynamo on sys.path, which makes
`import vllm` resolve to dynamo.vllm. Spawned subprocesses (EngineCore,
sglang scheduler) inherit that and crash on `from vllm.v1 ...`.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

_vllm_internals_importable: bool | None = None


def _can_import_vllm_internals() -> bool:
    """Try to import a vllm internal submodule once and cache the result."""
    global _vllm_internals_importable
    if _vllm_internals_importable is None:
        try:
            importlib.import_module("vllm.v1.engine.async_llm")
            _vllm_internals_importable = True
        except Exception:
            _vllm_internals_importable = False
    return _vllm_internals_importable


def pytest_ignore_collect(collection_path, config):
    """Skip test files that need optional deps not available in all CI images."""
    filename = collection_path.name
    parts = collection_path.parts

    # tests/frontend/test_prepost*.py import vllm.entrypoints.openai which
    # requires full vllm internals absent in dynamo-runtime.
    if filename.startswith("test_prepost") and "frontend" in parts:
        if not _can_import_vllm_internals():
            return True

    # examples/backends/sglang/test_sglang_expert_info.py requires pybase64
    # which is not installed in all CI images.
    if filename == "test_sglang_expert_info.py":
        if importlib.util.find_spec("pybase64") is None:
            return True

    return None


# Seed sys.modules with the venv copies before pytest collection runs.
for _name in ("vllm", "sglang"):
    try:
        importlib.import_module(_name)
    except ImportError:
        pass

# Suppress ImportPathMismatchError when pytest later loads dynamo.vllm
# under the bare name "vllm".
os.environ.setdefault("PY_IGNORE_IMPORTMISMATCH", "1")

_BAD_DYNAMO_PATH = str(
    Path(__file__).resolve().parent / "components" / "src" / "dynamo"
)


def _strip_bad_path() -> None:
    while _BAD_DYNAMO_PATH in sys.path:
        sys.path.remove(_BAD_DYNAMO_PATH)


# Strip the bad path before multiprocessing.spawn freezes sys.path for the
# child — catches re-insertions that happen during fixture/test execution.
try:
    import multiprocessing.spawn as _mps

    _orig_get_preparation_data = _mps.get_preparation_data

    def _patched_get_preparation_data(name):
        _strip_bad_path()
        return _orig_get_preparation_data(name)

    _mps.get_preparation_data = _patched_get_preparation_data
except Exception:
    pass


def pytest_runtest_setup(item):
    _strip_bad_path()
