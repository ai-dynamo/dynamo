# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block dynamo.vllm/sglang from shadowing the installed vllm/sglang.

Pytest collection puts components/src/dynamo on sys.path, which makes
`import vllm` resolve to dynamo.vllm. Spawned subprocesses (EngineCore,
sglang scheduler) inherit that and crash on `from vllm.v1 ...`.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

from tests.marker_categories import REQUIRED_CATEGORIES

_NO_DEFAULT_MARKERS_ENV = "DYNAMO_PYTEST_NO_DEFAULT_MARKERS"
_SUITE_MARKERS = REQUIRED_CATEGORIES["Lifecycle"]
_MACHINE_MARKERS = REQUIRED_CATEGORIES["Hardware"]

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


def pytest_itemcollected(item):
    """Apply CI defaults to tests missing lifecycle or hardware markers.

    This hook lives in the repository-root conftest so it applies to every
    collected test tree, including ``tests/``, ``components/src``, and
    ``aisimulate/tests``. It runs before pytest's marker filter.

    ``DYNAMO_PYTEST_NO_DEFAULT_MARKERS=1`` disables the defaults so the marker
    report can inspect authored markers only.
    """
    if os.environ.get(_NO_DEFAULT_MARKERS_ENV) == "1":
        return
    if not any(item.get_closest_marker(marker) for marker in _SUITE_MARKERS):
        item.add_marker(pytest.mark.pre_merge)
    if not any(item.get_closest_marker(marker) for marker in _MACHINE_MARKERS):
        item.add_marker(pytest.mark.gpu_0)
