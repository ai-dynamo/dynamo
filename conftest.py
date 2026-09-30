# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block dynamo.vllm/sglang/triton from shadowing the installed packages.

Pytest collection puts components/src/dynamo on sys.path, which makes
`import vllm` resolve to dynamo.vllm. Spawned subprocesses (EngineCore,
sglang scheduler) inherit that and crash on `from vllm.v1 ...`. The same
applies to dynamo.triton vs. the triton compiler package torch imports.
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
# Best-effort: an engine that is present but not importable (an editable install
# with an unreadable source tree raises PermissionError) must not abort collection.
for _name in ("vllm", "sglang", "triton"):
    try:
        importlib.import_module(_name)
    # ImportError is a missing engine; OSError is one whose sources are present
    # but unreadable. A failure of any other kind comes from inside an engine
    # that did start importing, and swallowing it here would run the suite
    # against a half-initialized engine, so it propagates.
    except (ImportError, OSError) as _exc:
        if isinstance(_exc, ModuleNotFoundError) and _exc.name == _name:
            continue  # the engine is simply not installed; nothing to report
        # Anything else means the install is broken, and one line here names
        # the cause rather than leaving a bare collection error downstream.
        print(
            f"conftest: {_name} is installed but could not be imported, so "
            f"tests that import it will fail: {type(_exc).__name__}: {_exc}",
            file=sys.stderr,
        )

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


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_protocol(item, nextitem):
    """Tear down the parent's setup stack after a forked item.

    pytest-forked runs setup and teardown only in the child process
    (https://github.com/pytest-dev/pytest-forked, ``forked_run_report``), so
    the parent never calls ``teardown_exact(nextitem)`` for a forked item.
    Collectors the parent set up for earlier non-forked items, such as the
    ``Package`` of a test directory whose last tests are forked, then stay on
    the stack, and the first item of the next package fails setup with
    "previous item was not torn down properly". Doing the teardown pytest
    would have done keeps the parent's stack in step with ``nextitem``.

    A parent-side finalizer failure is reported as a teardown error for the
    item, as pytest's own teardown phase would, rather than escaping the hook
    as an INTERNALERROR. The child already reported a passing teardown, so
    nothing more is reported when the parent teardown succeeds.
    """
    result = yield
    if item.config.pluginmanager.has_plugin("pytest_forked") and (
        item.config.getoption("forked", default=False)
        or item.get_closest_marker("forked")
    ):
        # Match pytest's teardown when the session will not run the next item.
        if item.session.shouldfail or item.session.shouldstop:
            nextitem = None
        reraise: tuple[type[BaseException], ...] = (pytest.exit.Exception,)
        if not item.config.getoption("usepdb", False):
            reraise += (KeyboardInterrupt,)
        call = pytest.CallInfo.from_call(
            lambda: item.session._setupstate.teardown_exact(nextitem),
            when="teardown",
            reraise=reraise,
        )
        if call.excinfo is not None:
            # Built directly: makereport hook wrappers such as
            # pytest-rerunfailures expect per-item state that their protocol
            # hook sets, and pytest-forked's protocol hook skips it.
            report = pytest.TestReport.from_item_and_call(item, call)
            item.ihook.pytest_runtest_logreport(report=report)
            if not call.excinfo.errisinstance(pytest.skip.Exception):
                item.ihook.pytest_exception_interact(
                    node=item, call=call, report=report
                )
    return result


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
