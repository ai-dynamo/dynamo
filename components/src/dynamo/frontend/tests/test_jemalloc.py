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
    execve = Mock(side_effect=SystemExit)
    monkeypatch.setattr(ctypes.util, "find_library", find_library)
    monkeypatch.setattr(os, "execve", execve)
    return find_library, execve, runtime.main


@pytest.mark.parametrize("enabled", [None, "0", "false"])
def test_disabled(startup, monkeypatch, enabled):
    if enabled is not None:
        monkeypatch.setenv("DYN_FRONTEND_JEMALLOC", enabled)
    find_library, execve, main = startup
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    find_library.assert_not_called()
    execve.assert_not_called()
    main.assert_called_once_with()
    assert "LD_PRELOAD" not in os.environ


@pytest.mark.parametrize("separator", [":", " "])
def test_already_preloaded(startup, monkeypatch, separator):
    monkeypatch.setenv("DYN_FRONTEND_JEMALLOC", "1")
    preload = f"libother.so{separator}/usr/lib/libjemalloc.so.2"
    monkeypatch.setenv("LD_PRELOAD", preload)
    find_library, execve, main = startup
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    find_library.assert_not_called()
    execve.assert_not_called()
    main.assert_called_once_with()
    assert os.environ["LD_PRELOAD"] == preload


def test_missing_library(startup, monkeypatch, capsys):
    monkeypatch.setenv("DYN_FRONTEND_JEMALLOC", "1")
    find_library, execve, main = startup
    find_library.return_value = None
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    find_library.assert_called_once_with("jemalloc")
    execve.assert_not_called()
    main.assert_called_once_with()
    assert "libjemalloc was not found" in capsys.readouterr().err
    assert "LD_PRELOAD" not in os.environ


@pytest.mark.parametrize(
    "enabled,existing",
    [
        ("1", ""),
        ("true", "libother.so:libanother.so"),
        ("YES", "/opt/jemalloc-tools/libheaptrace.so"),
    ],
)
def test_restart(startup, monkeypatch, enabled, existing):
    monkeypatch.setenv("DYN_FRONTEND_JEMALLOC", enabled)
    monkeypatch.setenv("LD_PRELOAD", existing)
    argv = [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        "-m",
        "dynamo.frontend",
        "--http-port",
        "8123",
    ]
    monkeypatch.setattr(sys, "orig_argv", argv)
    find_library, execve, main = startup
    stdout, stderr = Mock(), Mock()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    def replace_process(*args):
        stdout.flush.assert_called_once_with()
        stderr.flush.assert_called_once_with()
        raise SystemExit

    execve.side_effect = replace_process
    expected_env = dict(
        os.environ, LD_PRELOAD="libjemalloc.so.2" + (f":{existing}" if existing else "")
    )
    with pytest.raises(SystemExit):
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    find_library.assert_called_once_with("jemalloc")
    execve.assert_called_once_with(sys.executable, argv, expected_env)
    assert os.environ["LD_PRELOAD"] == existing
    main.assert_not_called()


@pytest.mark.parametrize("existing", [None, "", "libother.so"])
def test_exec_failure(startup, monkeypatch, capsys, existing):
    monkeypatch.setenv("DYN_FRONTEND_JEMALLOC", "1")
    if existing is not None:
        monkeypatch.setenv("LD_PRELOAD", existing)
    _, execve, main = startup
    execve.side_effect = OSError("exec blocked")
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    main.assert_called_once_with()
    assert os.environ.get("LD_PRELOAD") == existing
    assert "exec blocked" in capsys.readouterr().err


def test_import_does_not_restart(startup, monkeypatch):
    monkeypatch.setenv("DYN_FRONTEND_JEMALLOC", "1")
    find_library, execve, main = startup
    runpy.run_path(str(ENTRYPOINT), run_name="dynamo.frontend.__main__")
    find_library.assert_not_called()
    execve.assert_not_called()
    main.assert_not_called()
