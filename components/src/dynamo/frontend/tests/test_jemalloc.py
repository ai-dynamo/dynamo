# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from dynamo import _jemalloc

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

ENTRYPOINT = Path(__file__).resolve().parents[1] / "__main__.py"


@pytest.fixture
def startup(monkeypatch, tmp_path):
    monkeypatch.delenv(_jemalloc.DISABLE_ENV, raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    library = tmp_path / "libjemalloc_dynamo.so.2"
    library.touch()
    monkeypatch.setattr(_jemalloc, "LIBRARY", library)
    runtime = ModuleType("dynamo.frontend.main")
    runtime.main = Mock()
    monkeypatch.setitem(sys.modules, "dynamo.frontend.main", runtime)
    execve = Mock(side_effect=SystemExit)
    monkeypatch.setattr(os, "execve", execve)
    return library, execve, runtime.main


@pytest.mark.parametrize(
    "existing",
    ["", "libother.so:libanother.so", "/opt/jemalloc-tools/libheaptrace.so"],
)
def test_restart(startup, monkeypatch, existing):
    monkeypatch.setenv("LD_PRELOAD", existing)
    argv = [sys.executable, "-u", "-m", "dynamo.frontend", "--http-port", "8123"]
    monkeypatch.setattr(sys, "orig_argv", argv)
    library, execve, main = startup
    stdout, stderr = Mock(), Mock()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    def replace_process(*args):
        stdout.flush.assert_called_once_with()
        stderr.flush.assert_called_once_with()
        raise SystemExit

    execve.side_effect = replace_process
    preload = f"{library}:{existing}" if existing else str(library)
    with pytest.raises(SystemExit):
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    execve.assert_called_once_with(
        sys.executable, argv, dict(os.environ, LD_PRELOAD=preload)
    )
    assert os.environ["LD_PRELOAD"] == existing
    main.assert_not_called()


@pytest.mark.parametrize("disabled", ["1", "true", "YES"])
def test_disabled(startup, monkeypatch, disabled):
    monkeypatch.setenv(_jemalloc.DISABLE_ENV, disabled)
    _, execve, main = startup
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    execve.assert_not_called()
    main.assert_called_once_with()
    assert "LD_PRELOAD" not in os.environ


@pytest.mark.parametrize(
    "preload",
    [
        "libother.so:/usr/lib/libjemalloc.so.2",
        "libother.so /usr/lib/libjemalloc.so.2",
        "/site-packages/dynamo/libjemalloc_dynamo.so.2",
    ],
)
def test_already_preloaded(startup, monkeypatch, preload):
    monkeypatch.setenv("LD_PRELOAD", preload)
    _, execve, main = startup
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    execve.assert_not_called()
    main.assert_called_once_with()
    assert os.environ["LD_PRELOAD"] == preload


def test_library_not_bundled(startup, monkeypatch, capsys):
    library, execve, main = startup
    library.unlink()
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    execve.assert_not_called()
    main.assert_called_once_with()
    assert "is not bundled" in capsys.readouterr().err
    assert "LD_PRELOAD" not in os.environ


@pytest.mark.parametrize("existing", [None, "libother.so"])
def test_exec_failure(startup, monkeypatch, capsys, existing):
    if existing is not None:
        monkeypatch.setenv("LD_PRELOAD", existing)
    _, execve, main = startup
    execve.side_effect = OSError("exec blocked")
    runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    main.assert_called_once_with()
    assert os.environ.get("LD_PRELOAD") == existing
    assert "exec blocked" in capsys.readouterr().err


def test_import_does_not_restart(startup):
    _, execve, main = startup
    runpy.run_path(str(ENTRYPOINT), run_name="dynamo.frontend.__main__")
    execve.assert_not_called()
    main.assert_not_called()
