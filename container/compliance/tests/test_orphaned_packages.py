# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A reviewed orphan removal must never cascade into serving dependencies."""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "deps/vllm/remove_orphaned_package.py"
_SPEC = importlib.util.spec_from_file_location("remove_orphaned_package", _SCRIPT)
removal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(removal)
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]


def commands(monkeypatch, plan, installed=True):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = (
            "libplacebo338\tinstalled\nlibc6\tinstalled\n"
            if installed
            else "libc6\tinstalled\n"
        )
        if command[0] == "dpkg":
            output = "amd64\n"
        if command[0] == "apt-get":
            output = plan
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(removal.subprocess, "run", run)
    return calls


def test_removes_only_the_reviewed_orphan(monkeypatch):
    calls = commands(monkeypatch, "Purg libplacebo338 [6.338.2-2build1]\n")
    removal.remove_orphans(["libplacebo338"])
    assert len(calls) == 4
    assert "--simulate" in calls[2]
    assert "--yes" in calls[3]
    assert "APT::Get::AutomaticRemove=false" in calls[3]
    assert calls[3][-2:] == ["purge", "libplacebo338:amd64"]


@pytest.mark.parametrize(
    "plan",
    [
        "Purg libplacebo338 [6.338.2-2build1]\nRemv serving-dependency [1]\n",
        "Purg libplacebo338:amd64 [6.338.2-2build1]\nPurg libplacebo338:i386 [6.338.2-2build1]\n",
        "Purg libplacebo338 [6.338.2-2build1]\nInst unrelated [2]\n",
        "Purg libplacebo338 [6.338.2-2build1]\nConf unrelated [2]\n",
        "",
    ],
)
def test_dependency_or_upgrade_changes_abort_before_mutation(monkeypatch, plan):
    calls = commands(monkeypatch, plan)
    with pytest.raises(RuntimeError, match="unrelated changes"):
        removal.remove_orphans(["libplacebo338"])
    assert len(calls) == 3
    assert not any("--yes" in command for command in calls)


def test_absent_package_is_a_noop(monkeypatch):
    calls = commands(monkeypatch, "", installed=False)
    removal.remove_orphans(["libplacebo338"])
    assert len(calls) == 2


@pytest.mark.parametrize(
    "info,packages",
    [
        ("GUI: QT5\nOpenGL support: YES\n", ["opencv-python-headless"]),
        ("GUI: NONE\nOpenGL support: YES\n", ["opencv-python-headless"]),
        ("GUI: NONE\n", ["opencv-python-headless", "PyOpenGL"]),
        ("GUI: NONE\n", ["opencv-python"]),
    ],
)
def test_gui_or_new_gl_bindings_block_cleanup(info, packages):
    with pytest.raises(RuntimeError, match="Refusing GL library removal"):
        removal.validate_headless(info, packages)


def test_headless_opencv_without_gl_is_accepted():
    removal.validate_headless(
        "GUI: NONE\nOpenGL support: NO\n", ["opencv-python-headless"]
    )


def test_partial_installed_set_removes_only_present_reviewed_packages(monkeypatch):
    calls = commands(monkeypatch, "Purg libplacebo338 [6.338.2-2build1]\n")
    removal.remove_orphans(["libplacebo338", "libgl1", "mesa-libgallium"])
    assert calls[-1][-2:] == ["purge", "libplacebo338:amd64"]


def test_post_cleanup_guard_uses_a_fresh_interpreter(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "sys.argv",
        [str(_SCRIPT), "--package", "libplacebo338", "--require-headless-opencv"],
    )
    monkeypatch.setattr(
        removal.importlib,
        "import_module",
        lambda name: SimpleNamespace(getBuildInformation=lambda: "GUI: NONE"),
    )
    monkeypatch.setattr(removal.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(
        removal, "validate_headless", lambda *args: calls.append("preflight")
    )
    monkeypatch.setattr(
        removal, "remove_orphans", lambda packages: calls.append("purge")
    )
    monkeypatch.setattr(
        removal.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )
    removal.main()
    assert calls[:2] == ["preflight", "purge"]
    assert calls[2][-2:] == [str(_SCRIPT), "--check-headless-only"]
