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


def _fake_dpkg_query(
    monkeypatch: pytest.MonkeyPatch,
    listing: str,
    conffiles: dict[str, str] | None = None,
    conffiles_returncode: int = 0,
) -> None:
    """Make dpkg.collect_components read `listing` instead of the real tool.

    The per-package `${Conffiles}` query answers from `conffiles`, keyed by the
    package name the generator asks about.
    """
    conffiles = conffiles or {}

    def _run(cmd, **kwargs):  # noqa: ANN001 - mirrors subprocess.run's signature
        if "-f=${Conffiles}\\n" in cmd:
            stdout = conffiles.get(cmd[-1], "")
            return subprocess.CompletedProcess(
                cmd, conffiles_returncode, stdout=stdout, stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout=listing, stderr="")

    monkeypatch.setattr(dpkg.subprocess, "run", _run)


def _names(components) -> set[str]:  # noqa: ANN001 - list[Component]
    return {c.name for c in components}


def _touch(root: Path, path: str) -> None:
    target = root / path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x\n")


# The conffile paths the TensorRT-LLM 1.3.0rc27 base image lists for the two
# packages it removes, in dpkg's Conffiles field shape (placeholder md5 sums).
_DPDK_CONFFILES = (
    " /etc/ld.so.conf.d/dpdk-x86_64-linux-gnu.conf 0123456789abcdef0123456789abcdef\n"
)
_DOCA_CONFFILES = (
    " /etc/ld.so.conf.d/doca-runtime-x86_64-linux-gnu.conf 0123456789abcdef0123456789abcdef\n"
    " /etc/profile.d/doca-runtime.sh fedcba9876543210fedcba9876543210\n"
)
_REMOVED_LISTING = (
    "bash\t5.2.21-2ubuntu4\tinstalled\n"
    "dpdk-community\t26.03.0.5-1\tconfig-files\n"
    "doca-sdk-common\t3.5.0095-1\tconfig-files\n"
)


def test_keeps_removed_packages_whose_conffiles_remain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `config-files` package still ships the conffiles it left, so it stays.

    The TensorRT-LLM base images install DOCA packages and remove them again:
    dpdk-community keeps an ld.so.conf.d entry, and doca-sdk-common keeps a
    script under /etc/profile.d. One surviving conffile is enough.
    """
    _touch(tmp_path, "/etc/ld.so.conf.d/dpdk-x86_64-linux-gnu.conf")
    _touch(tmp_path, "/etc/profile.d/doca-runtime.sh")
    _fake_dpkg_query(
        monkeypatch,
        _REMOVED_LISTING,
        conffiles={
            "dpdk-community": _DPDK_CONFFILES,
            "doca-sdk-common": _DOCA_CONFFILES,
        },
    )

    names = _names(dpkg.collect_components(tmp_path))

    assert names == {"bash", "dpdk-community", "doca-sdk-common"}


def test_skips_removed_packages_with_no_conffile_left(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Listed conffiles that are all gone prove the removal left nothing behind."""
    _fake_dpkg_query(
        monkeypatch,
        _REMOVED_LISTING + "somepkg\t1.0-1\tnot-installed\n",
        conffiles={
            "dpdk-community": _DPDK_CONFFILES,
            "doca-sdk-common": _DOCA_CONFFILES,
        },
    )

    names = _names(dpkg.collect_components(tmp_path))

    assert names == {"bash"}


@pytest.mark.parametrize(
    ("conffiles", "returncode"),
    [({}, 0), ({"dpdk-community": _DPDK_CONFFILES}, 1)],
    ids=["nothing-listed", "query-failed"],
)
def test_keeps_a_removed_package_whose_conffiles_are_not_listed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    conffiles: dict[str, str],
    returncode: int,
) -> None:
    """Without a conffile list there is no proof that nothing was left, so keep it."""
    _fake_dpkg_query(
        monkeypatch,
        "dpdk-community\t26.03.0.5-1\tconfig-files\n",
        conffiles=conffiles,
        conffiles_returncode=returncode,
    )

    names = _names(dpkg.collect_components(tmp_path))

    assert names == {"dpdk-community"}


def test_keeps_unpacked_and_half_configured_packages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every state other than not-installed and config-files leaves the files in place."""
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


@pytest.mark.skipif(
    shutil.which("dpkg-query") is None, reason="needs a Debian/Ubuntu environment"
)
def test_real_dpkg_query_reports_conffiles_in_the_parsed_shape() -> None:
    """Pin `${Conffiles}` and its line shape against the real tool.

    An unknown field prints nothing, and a line the pattern does not match is
    dropped. Either one keeps every removed package, which fails safe but hides
    the drift, so check that real conffile lines all parse.
    """
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${Conffiles}\\n"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]

    assert lines, "no conffiles reported; cannot tell whether the field resolved"
    unparsed = [line for line in lines if not dpkg._CONFFILE_LINE.match(line)]
    assert not unparsed, f"conffile lines the pattern does not match: {unparsed[:5]}"
