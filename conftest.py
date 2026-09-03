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

# Trees that ship runnable demos and agent scripts rather than CI suites.
# Their files are still collected, so a developer can run one explicitly, but
# they must not be defaulted into a lifecycle: an unmarked demo helper such as
# examples/backends/sglang/test_sglang_profile.py assumes a server the harness
# never starts, and defaulting it into pre_merge makes the CPU job run it.
# Anything under these roots that carries its own markers is unaffected.
# ``skills`` is a symlink to ``.agents/skills``, so resolve() lands on
# ``.agents``; both names are listed because either can reach collection.
_UNMANAGED_ROOTS = frozenset({"examples", "skills", ".agents"})
_REPO_ROOT = Path(__file__).resolve().parent

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


def _is_unmanaged(item) -> bool:
    """True when the item lives in a demo or script tree, not a CI suite."""
    try:
        path = Path(str(item.path)).resolve()
    except (AttributeError, OSError, ValueError):
        return False
    try:
        parts = path.relative_to(_REPO_ROOT).parts
    except ValueError:
        return False
    return bool(parts) and parts[0] in _UNMANAGED_ROOTS


def pytest_itemcollected(item):
    """Apply CI defaults to tests missing lifecycle or hardware markers.

    This hook lives in the repository-root conftest so it applies to every
    collected test tree, including ``tests/``, ``components/src``, and
    ``aisimulate/tests``. It runs before pytest's marker filter.

    Anything defaulted also gets ``defaulted``, which routes it to the CPU
    parallel job. Without that marker every defaulted test matches
    ``not parallel`` and lands in the sequential job, which already carries the
    fault-tolerance suite in a single process -- the imports alone cost ~90 MB
    resident for the rest of the run, and the job was OOM-killed (exit 137).

    Trees in ``_UNMANAGED_ROOTS`` are skipped entirely; they keep the
    pre-default behaviour of being deselected unless they mark themselves.

    ``DYNAMO_PYTEST_NO_DEFAULT_MARKERS=1`` disables the defaults so the marker
    report can inspect authored markers only.
    """
    if os.environ.get(_NO_DEFAULT_MARKERS_ENV) == "1":
        return
    if _is_unmanaged(item):
        return
    defaulted = False
    if not any(item.get_closest_marker(marker) for marker in _SUITE_MARKERS):
        item.add_marker(pytest.mark.pre_merge)
        defaulted = True
    if not any(item.get_closest_marker(marker) for marker in _MACHINE_MARKERS):
        item.add_marker(pytest.mark.gpu_0)
        defaulted = True
    if defaulted:
        item.add_marker(pytest.mark.defaulted)
