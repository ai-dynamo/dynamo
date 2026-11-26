# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared Cargo license harvest helper.

The wheel builder and EPP images call it with ``--policy best-effort``; the
sidecar image calls it with ``--policy strict``. These tests run the real
script under ``sh`` and, when installed, ``bash``.

Run from the repo root:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_harvest_cargo_licenses.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

_SCRIPT = Path(__file__).resolve().parents[1] / "harvest_cargo_licenses.sh"
_SHELLS = [s for s in ("sh", "bash") if shutil.which(s)]
_POLICIES = ["best-effort", "strict"]


def _harvest(
    shell: str, policy: str, out: Path, *roots: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [shell, str(_SCRIPT), "--policy", policy, "--out", str(out), *map(str, roots)],
        capture_output=True,
        text=True,
        check=False,
    )


def _crate(root: Path, index: str, name: str, *files: str) -> Path:
    crate = root / index / name
    crate.mkdir(parents=True)
    for f in files:
        (crate / f).parent.mkdir(parents=True, exist_ok=True)
        (crate / f).write_text(f"{name}:{f}\n")
    return crate


def _tree(out: Path) -> list[str]:
    return sorted(str(p.relative_to(out)) for p in out.rglob("*"))


@pytest.mark.parametrize("shell", _SHELLS)
@pytest.mark.parametrize("policy", _POLICIES)
def test_layout_and_selection(tmp_path: Path, shell: str, policy: str) -> None:
    root_a = tmp_path / "a/registry/src"
    root_b = tmp_path / "b/registry/src"
    _crate(root_a, "index.crates.io-1", "serde-1.0.0", "LICENSE-MIT", "README.md")
    _crate(root_a, "index.crates.io-1", "tokio-1.52.4", "LICENCE", "sub/LICENSE")
    _crate(root_a, "index.crates.io-2", "ring-0.17.0", "COPYING", "NOTICE")
    _crate(root_b, "index.crates.io-1", "unl-0.1.0", "UNLICENSE.txt")
    _crate(root_b, "index.crates.io-1", "nolicense-1.0.0", "README.md")
    out = tmp_path / "out"

    result = _harvest(shell, policy, out, root_a, root_b)

    assert result.returncode == 0, result.stderr
    assert _tree(out) == [
        "ring-0.17.0",
        "ring-0.17.0/COPYING",
        "ring-0.17.0/NOTICE",
        "serde-1.0.0",
        "serde-1.0.0/LICENSE-MIT",
        "tokio-1.52.4",
        "tokio-1.52.4/LICENCE",
        "unl-0.1.0",
        "unl-0.1.0/UNLICENSE.txt",
    ]
    assert (out / "serde-1.0.0/LICENSE-MIT").read_text() == "serde-1.0.0:LICENSE-MIT\n"
    assert f"4 crates with license files in {out}" in result.stdout


@pytest.mark.parametrize("shell", _SHELLS)
@pytest.mark.parametrize("policy", _POLICIES)
def test_missing_root_is_skipped(tmp_path: Path, shell: str, policy: str) -> None:
    out = tmp_path / "out"

    result = _harvest(shell, policy, out, tmp_path / "missing/registry/src")

    assert result.returncode == 0, result.stderr
    assert _tree(out) == []


@pytest.mark.parametrize("shell", _SHELLS)
@pytest.mark.parametrize("policy", _POLICIES)
def test_matching_directory_entry(tmp_path: Path, shell: str, policy: str) -> None:
    # best-effort selects any path (-e), so its failed cp leaves an empty crate
    # directory; strict selects regular files only (-f).
    root = tmp_path / "registry/src"
    crate = _crate(root, "index", "dual-1.0.0")
    (crate / "LICENSES").mkdir()
    out = tmp_path / "out"

    result = _harvest(shell, policy, out, root)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    if policy == "best-effort":
        assert _tree(out) == ["dual-1.0.0"]
    else:
        assert _tree(out) == []


@pytest.mark.parametrize("shell", _SHELLS)
@pytest.mark.parametrize("policy", _POLICIES)
def test_copy_failure(tmp_path: Path, shell: str, policy: str) -> None:
    # A directory at the destination path makes cp fail, even as root.
    root = tmp_path / "registry/src"
    _crate(root, "index", "aaa-1.0.0", "LICENSE")
    _crate(root, "index", "zzz-1.0.0", "LICENSE")
    out = tmp_path / "out"
    (out / "aaa-1.0.0/LICENSE").mkdir(parents=True)
    (out / "zzz-1.0.0/LICENSE").mkdir(parents=True)
    _crate(root, "index", "ok-1.0.0", "LICENSE")

    result = _harvest(shell, policy, out, root)

    if policy == "best-effort":
        assert result.returncode == 0
        assert result.stderr == ""
        assert (out / "ok-1.0.0/LICENSE").is_file()
    else:
        assert result.returncode != 0
        assert "LICENSE" in result.stderr
