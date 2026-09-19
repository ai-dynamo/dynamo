# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dpkg inventory scan.

Run from the repo root with the compliance package on the path:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_dpkg_generator.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from compliance.generators import dpkg

# CPU-only unit tests; markers are required by .ai/pytest-guidelines.md
# (lifecycle / test-type / hardware categories).
pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _fake_dpkg_query(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    """Make dpkg.collect_components read `stdout` instead of the real tool."""

    def _run(cmd, **kwargs):  # noqa: ANN001 - mirrors subprocess.run's signature
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(dpkg.subprocess, "run", _run)


def _names(components) -> set[str]:  # noqa: ANN001 - list[Component]
    return {c.name for c in components}


def test_skips_packages_whose_files_were_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `config-files` package is a dpkg entry with no files, not a dependency.

    The TensorRT-LLM base images install DOCA packages and remove them again,
    which leaves `dpdk-community` and `doca-sdk-common` in this state. They have
    no /usr/share/doc/<pkg>/copyright, so reporting them costs a license
    override for software the image does not carry.
    """
    _fake_dpkg_query(
        monkeypatch,
        "bash\t5.2.21-2ubuntu4\tinstalled\n"
        "dpdk-community\t26.03.0.5-1\tconfig-files\n"
        "doca-sdk-common\t3.5.0095-1\tconfig-files\n"
        "somepkg\t1.0-1\tnot-installed\n",
    )

    names = _names(dpkg.collect_components(tmp_path))

    assert names == {"bash"}


def test_keeps_unpacked_and_half_configured_packages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every state other than the two fileless ones leaves the files in place."""
    _fake_dpkg_query(
        monkeypatch,
        "a\t1\tinstalled\n"
        "b\t2\tunpacked\n"
        "c\t3\thalf-configured\n"
        "d\t4\thalf-installed\n"
        "e\t5\ttriggers-pending\n"
        "f\t6\ttriggers-awaited\n",
    )

    names = _names(dpkg.collect_components(tmp_path))

    assert names == {"a", "b", "c", "d", "e", "f"}


def test_keeps_everything_when_the_status_field_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """dpkg-query prints an empty value for a field it does not know, and exits 0.

    An unrecognized field must leave the inventory complete rather than empty:
    an over-complete inventory is noise, an empty one passes the policy gate
    while declaring nothing.
    """
    _fake_dpkg_query(monkeypatch, "bash\t5.2.21-2ubuntu4\t\nzlib1g\t1:1.3.dfsg-3.1\t\n")

    names = _names(dpkg.collect_components(tmp_path))

    assert names == {"bash", "zlib1g"}


@pytest.mark.skipif(
    shutil.which("dpkg-query") is None, reason="needs a Debian/Ubuntu environment"
)
def test_real_dpkg_query_reports_the_status_field() -> None:
    """Pin the field name against the real tool.

    `${db:Status-Status}` is what the filter reads. dpkg-query prints an empty
    string for an unknown field and exits 0, so a typo here would disable the
    filter silently on every image.
    """
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${Package}\\t${db:Status-Status}\\n"],
        capture_output=True,
        text=True,
        check=True,
    )
    statuses = {
        line.split("\t", 1)[1] for line in result.stdout.splitlines() if "\t" in line
    }

    assert statuses, "no packages reported; cannot tell whether the field resolved"
    assert "" not in statuses, f"dpkg-query did not resolve the field: {statuses}"
    assert statuses <= {
        "not-installed",
        "config-files",
        "half-installed",
        "unpacked",
        "half-configured",
        "triggers-awaited",
        "triggers-pending",
        "installed",
    }, f"unexpected dpkg state: {statuses}"
