# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend entry-point wiring for the jemalloc preload.

lib/bindings/python/tests/test_jemalloc_preload.py covers the preload itself.
"""

import ctypes.util
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

ENTRYPOINT = Path(__file__).resolve().parents[1] / "__main__.py"


@pytest.fixture
def startup(monkeypatch):
    for name in ("DYN_JEMALLOC", "DYN_FRONTEND_JEMALLOC", "LD_PRELOAD"):
        monkeypatch.delenv(name, raising=False)
    runtime = ModuleType("dynamo.frontend.main")
    runtime.main = Mock()
    monkeypatch.setitem(sys.modules, "dynamo.frontend.main", runtime)
    execve = Mock(side_effect=SystemExit)
    monkeypatch.setattr(
        ctypes.util, "find_library", Mock(return_value="libjemalloc.so.2")
    )
    monkeypatch.setattr(os, "execve", execve)
    return execve, runtime.main


def test_disabled(startup):
    execve, main = startup
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    execve.assert_not_called()
    main.assert_called_once_with()


@pytest.mark.parametrize("env_var", ["DYN_JEMALLOC", "DYN_FRONTEND_JEMALLOC"])
def test_preloads_before_runtime_import(startup, monkeypatch, env_var):
    monkeypatch.setenv(env_var, "1")
    # Importing the runtime before the re-exec raises ImportError, not SystemExit.
    monkeypatch.setitem(sys.modules, "dynamo.frontend.main", None)
    execve, _ = startup
    with pytest.raises(SystemExit):
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    _, _, env = execve.call_args.args
    assert env["LD_PRELOAD"] == "libjemalloc.so.2"


def test_import_does_not_restart(startup, monkeypatch):
    monkeypatch.setenv("DYN_JEMALLOC", "1")
    execve, main = startup
    runpy.run_path(str(ENTRYPOINT), run_name="dynamo.frontend.__main__")
    execve.assert_not_called()
    main.assert_not_called()
