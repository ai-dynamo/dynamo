# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router entry-point wiring for the jemalloc preload.

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
def execve(monkeypatch):
    for name in ("DYN_JEMALLOC", "DYN_FRONTEND_JEMALLOC", "LD_PRELOAD"):
        monkeypatch.delenv(name, raising=False)
    execve = Mock(side_effect=SystemExit)
    monkeypatch.setattr(
        ctypes.util, "find_library", Mock(return_value="libjemalloc.so.2")
    )
    monkeypatch.setattr(os, "execve", execve)
    return execve


def test_preloads_before_runtime_import(execve, monkeypatch):
    monkeypatch.setenv("DYN_JEMALLOC", "1")
    # Importing the runtime before the re-exec raises ImportError, not SystemExit.
    monkeypatch.setitem(sys.modules, "dynamo.router.main", None)
    with pytest.raises(SystemExit):
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    _, _, env = execve.call_args.args
    assert env["LD_PRELOAD"] == "libjemalloc.so.2"


@pytest.mark.parametrize("env_var", [None, "DYN_FRONTEND_JEMALLOC"])
def test_runs_without_preload(execve, monkeypatch, env_var):
    if env_var is not None:
        monkeypatch.setenv(env_var, "1")
    runtime = ModuleType("dynamo.router.main")
    runtime.main = Mock()
    monkeypatch.setitem(sys.modules, "dynamo.router.main", runtime)
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    execve.assert_not_called()
    runtime.main.assert_called_once_with()
