# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolving ``-n auto`` from GPU VRAM ahead of pytest-xdist.

pytest-xdist converts ``-n auto`` to a CPU count in its own tryfirst
``pytest_cmdline_main``. ``tests/conftest.py`` must observe the "auto" request
before that conversion so ``--max-vram-gib`` runs get the VRAM-derived slot
count. These tests drive a real pytest session (pytester, in-process) with the
real xdist plugin loaded and read ``numprocesses`` after both hooks have run.
No GPU or ``pynvml`` required -- GPU detection is stubbed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.utils import vram_utils

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]
pytest_plugins = ["pytester"]

_FAKE_GPUS = [
    {"index": 0, "name": "fake-80g", "total_mib": 80 * 1024},
    {"index": 1, "name": "fake-80g", "total_mib": 80 * 1024},
]
_VRAM_LIMIT = 10.0
# 2 GPUs x int(80 GiB * 0.85 / 10 GiB) = 2 x 6
_EXPECTED_AUTO_SLOTS = vram_utils.auto_worker_count(_FAKE_GPUS, _VRAM_LIMIT)
# Sentinel returned in place of xdist's CPU-count probe, so a value resolved by
# xdist (rather than by the conftest) is unambiguous on any host.
_XDIST_CPU_SENTINEL = 777

# pytest_cmdline_main here is plain priority: pluggy calls it after every
# tryfirst impl (xdist's conversion and the conftest wrapper) and, because it
# is registered after _pytest.main, before pytest's own session runner.
# Returning an exit code short-circuits the firstresult hook so no session runs.
_RECORDER_PLUGIN = f"""
import pytest


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_xdist_auto_num_workers(config):
    return {_XDIST_CPU_SENTINEL}


def pytest_cmdline_main(config):
    print(f"RECORDED_NUMPROCESSES={{config.option.numprocesses!r}}")
    return 0
"""


@pytest.fixture
def fake_gpus(monkeypatch):
    monkeypatch.setattr(vram_utils, "detect_gpus", lambda: list(_FAKE_GPUS))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.fixture
def run_cmdline(pytester, monkeypatch):
    """Run the real tests/conftest.py + xdist through pytest_cmdline_main."""
    conftest_path = Path(__file__).parents[1] / "conftest.py"
    pytester.makeconftest(conftest_path.read_text())
    pytester.makepyfile(xdist_recorder=_RECORDER_PLUGIN)
    pytester.syspathinsert()
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    def _run(*args: str):
        result = pytester.runpytest(
            "-o", "addopts=", "-p", "xdist.plugin", "-p", "xdist_recorder", *args
        )
        assert result.ret == 0, result.stderr.str()
        match = re.search(r"RECORDED_NUMPROCESSES=(.+)", result.stdout.str())
        assert match, result.stdout.str()
        return ast.literal_eval(match.group(1))

    return _run


def test_auto_resolves_to_vram_slot_count_before_xdist(run_cmdline, fake_gpus):
    numproc = run_cmdline("-n", "auto", f"--max-vram-gib={_VRAM_LIMIT}")
    assert numproc == _EXPECTED_AUTO_SLOTS


def test_logical_resolves_like_auto(run_cmdline, fake_gpus):
    numproc = run_cmdline("-n", "logical", f"--max-vram-gib={_VRAM_LIMIT}")
    assert numproc == _EXPECTED_AUTO_SLOTS


def test_auto_honours_cuda_visible_devices(run_cmdline, fake_gpus, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    numproc = run_cmdline("-n", "auto", f"--max-vram-gib={_VRAM_LIMIT}")
    assert numproc == _EXPECTED_AUTO_SLOTS // 2


def test_explicit_numeric_n_is_untouched(run_cmdline, fake_gpus):
    numproc = run_cmdline("-n", "3", f"--max-vram-gib={_VRAM_LIMIT}")
    assert numproc == 3


def test_auto_without_max_vram_gib_is_left_to_xdist(run_cmdline, fake_gpus):
    numproc = run_cmdline("-n", "auto")
    assert numproc == _XDIST_CPU_SENTINEL


def test_auto_with_collect_only_is_left_to_xdist(run_cmdline, fake_gpus):
    numproc = run_cmdline(
        "--collect-only", "-n", "auto", f"--max-vram-gib={_VRAM_LIMIT}"
    )
    assert numproc == _XDIST_CPU_SENTINEL


def test_auto_without_gpus_falls_back_to_one_slot(run_cmdline, monkeypatch):
    monkeypatch.setattr(vram_utils, "detect_gpus", lambda: [])
    numproc = run_cmdline("-n", "auto", f"--max-vram-gib={_VRAM_LIMIT}")
    assert numproc == 1


def test_conftest_hook_is_a_tryfirst_wrapper():
    """Guard the ordering guarantee relative to xdist's tryfirst hookimpl."""
    import xdist.plugin

    from tests import conftest

    ours = conftest.pytest_cmdline_main.pytest_impl  # type: ignore[attr-defined]
    theirs = xdist.plugin.pytest_cmdline_main.pytest_impl  # type: ignore[attr-defined]
    assert theirs.get("tryfirst") and not theirs.get("wrapper")
    assert ours.get("tryfirst") and ours.get("wrapper")
