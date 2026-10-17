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

from dynamo._jemalloc import maybe_preload_jemalloc

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

ALIAS = "DYN_FRONTEND_JEMALLOC"
INDEXER_ENTRYPOINT = (
    Path(__file__).resolve().parents[1] / "src" / "dynamo" / "indexer" / "__main__.py"
)

# Environment, alias passed by the entry point, and the variable warnings name.
ENABLED = [
    pytest.param({"DYN_JEMALLOC": "1"}, None, "DYN_JEMALLOC", id="generic"),
    pytest.param({"DYN_JEMALLOC": "true"}, ALIAS, "DYN_JEMALLOC", id="generic-alias"),
    pytest.param({ALIAS: "YES"}, ALIAS, ALIAS, id="alias"),
    pytest.param({"DYN_JEMALLOC": "0", ALIAS: "1"}, ALIAS, ALIAS, id="alias-only"),
]


@pytest.fixture
def startup(monkeypatch):
    for name in ("DYN_JEMALLOC", ALIAS, "LD_PRELOAD"):
        monkeypatch.delenv(name, raising=False)
    find_library = Mock(return_value="libjemalloc.so.2")
    execve = Mock(side_effect=SystemExit)
    monkeypatch.setattr(ctypes.util, "find_library", find_library)
    monkeypatch.setattr(os, "execve", execve)
    return find_library, execve


def set_env(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize("alias", [None, ALIAS])
@pytest.mark.parametrize("value", [None, "", "0", "false", "no"])
def test_disabled(startup, monkeypatch, alias, value):
    if value is not None:
        set_env(monkeypatch, {"DYN_JEMALLOC": value, ALIAS: value})
    find_library, execve = startup
    maybe_preload_jemalloc(alias=alias)
    find_library.assert_not_called()
    execve.assert_not_called()
    assert "LD_PRELOAD" not in os.environ


def test_alias_ignored_unless_passed(startup, monkeypatch):
    monkeypatch.setenv(ALIAS, "1")
    find_library, execve = startup
    maybe_preload_jemalloc()
    find_library.assert_not_called()
    execve.assert_not_called()


@pytest.mark.parametrize("separator", [":", " "])
def test_already_preloaded(startup, monkeypatch, separator):
    monkeypatch.setenv("DYN_JEMALLOC", "1")
    preload = f"libother.so{separator}/usr/lib/libjemalloc.so.2"
    monkeypatch.setenv("LD_PRELOAD", preload)
    find_library, execve = startup
    maybe_preload_jemalloc()
    find_library.assert_not_called()
    execve.assert_not_called()
    assert os.environ["LD_PRELOAD"] == preload


@pytest.mark.parametrize("env,alias,enabled_by", ENABLED)
def test_missing_library(startup, monkeypatch, capsys, env, alias, enabled_by):
    set_env(monkeypatch, env)
    find_library, execve = startup
    find_library.return_value = None
    maybe_preload_jemalloc(alias=alias)
    find_library.assert_called_once_with("jemalloc")
    execve.assert_not_called()
    err = capsys.readouterr().err
    assert f"{enabled_by} is enabled but libjemalloc was not found" in err
    assert "LD_PRELOAD" not in os.environ


@pytest.mark.parametrize(
    "existing", ["", "libother.so:libanother.so", "/opt/jemalloc-tools/libheaptrace.so"]
)
def test_restart(startup, monkeypatch, existing):
    monkeypatch.setenv("DYN_JEMALLOC", "1")
    monkeypatch.setenv("LD_PRELOAD", existing)
    argv = [sys.executable, "-u", "-X", "faulthandler", "-m", "dynamo.router"]
    argv += ["--endpoint", "dynamo.prefill.generate"]
    monkeypatch.setattr(sys, "orig_argv", argv)
    find_library, execve = startup
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
        maybe_preload_jemalloc()
    find_library.assert_called_once_with("jemalloc")
    execve.assert_called_once_with(sys.executable, argv, expected_env)
    assert os.environ["LD_PRELOAD"] == existing


@pytest.mark.parametrize("env,alias,enabled_by", ENABLED)
@pytest.mark.parametrize("existing", [None, "", "libother.so"])
def test_exec_failure(startup, monkeypatch, capsys, env, alias, enabled_by, existing):
    set_env(monkeypatch, env)
    if existing is not None:
        monkeypatch.setenv("LD_PRELOAD", existing)
    _, execve = startup
    execve.side_effect = OSError("exec blocked")
    maybe_preload_jemalloc(alias=alias)
    assert os.environ.get("LD_PRELOAD") == existing
    err = capsys.readouterr().err
    assert f"{enabled_by} is enabled but re-exec with jemalloc failed" in err
    assert "exec blocked" in err


def test_indexer_preloads_before_runtime_import(startup, monkeypatch):
    monkeypatch.setenv("DYN_JEMALLOC", "1")
    # Importing the runtime before the re-exec raises ImportError, not SystemExit.
    monkeypatch.setitem(sys.modules, "dynamo.indexer.main", None)
    _, execve = startup
    with pytest.raises(SystemExit):
        runpy.run_path(str(INDEXER_ENTRYPOINT), run_name="__main__")
    _, _, env = execve.call_args.args
    assert env["LD_PRELOAD"] == "libjemalloc.so.2"


@pytest.mark.parametrize("env", [{}, {ALIAS: "1"}])
def test_indexer_runs_without_preload(startup, monkeypatch, env):
    set_env(monkeypatch, env)
    runtime = ModuleType("dynamo.indexer.main")
    runtime.main = Mock(return_value=0)
    monkeypatch.setitem(sys.modules, "dynamo.indexer.main", runtime)
    _, execve = startup
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(INDEXER_ENTRYPOINT), run_name="__main__")
    assert exit_info.value.code == 0
    execve.assert_not_called()
    runtime.main.assert_called_once_with()
