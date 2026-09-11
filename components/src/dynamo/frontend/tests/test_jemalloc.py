# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    monkeypatch.delenv("DYN_FRONTEND_JEMALLOC", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    runtime = ModuleType("dynamo.frontend.main")
    runtime.main = Mock()
    monkeypatch.setitem(sys.modules, "dynamo.frontend.main", runtime)
    find_library = Mock(return_value="libjemalloc.so.2")
    execv = Mock(side_effect=SystemExit)
    monkeypatch.setattr(ctypes.util, "find_library", find_library)
    monkeypatch.setattr(os, "execv", execv)
    return find_library, execv, runtime.main


@pytest.mark.parametrize("enabled", [None, "0", "true"])
def test_disabled(startup, monkeypatch, enabled):
    if enabled is not None:
        monkeypatch.setenv("DYN_FRONTEND_JEMALLOC", enabled)
    find_library, execv, main = startup
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    find_library.assert_not_called()
    execv.assert_not_called()
    main.assert_called_once_with()
    assert "LD_PRELOAD" not in os.environ


def test_already_preloaded(startup, monkeypatch):
    monkeypatch.setenv("DYN_FRONTEND_JEMALLOC", "1")
    preload = "libother.so:/usr/lib/libjemalloc.so.2"
    monkeypatch.setenv("LD_PRELOAD", preload)
    find_library, execv, main = startup
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    find_library.assert_not_called()
    execv.assert_not_called()
    main.assert_called_once_with()
    assert os.environ["LD_PRELOAD"] == preload


def test_missing_library(startup, monkeypatch, capsys):
    monkeypatch.setenv("DYN_FRONTEND_JEMALLOC", "1")
    find_library, execv, main = startup
    find_library.return_value = None
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    find_library.assert_called_once_with("jemalloc")
    execv.assert_not_called()
    main.assert_called_once_with()
    assert "libjemalloc was not found" in capsys.readouterr().err
    assert "LD_PRELOAD" not in os.environ


@pytest.mark.parametrize("existing", ["", "libother.so:libanother.so"])
def test_restart(startup, monkeypatch, existing):
    monkeypatch.setenv("DYN_FRONTEND_JEMALLOC", "1")
    monkeypatch.setenv("LD_PRELOAD", existing)
    monkeypatch.setattr(sys, "argv", ["frontend", "--http-port", "8123"])
    find_library, execv, main = startup
    with pytest.raises(SystemExit):
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    find_library.assert_called_once_with("jemalloc")
    execv.assert_called_once_with(
        sys.executable,
        [sys.executable, "-m", "dynamo.frontend", "--http-port", "8123"],
    )
    assert os.environ["LD_PRELOAD"] == "libjemalloc.so.2" + (
        f":{existing}" if existing else ""
    )
    main.assert_not_called()
