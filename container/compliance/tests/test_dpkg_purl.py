# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for self-describing dpkg Package URLs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from compliance.generators import dpkg

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_ubuntu_component_purl_includes_distro_and_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An Ubuntu dpkg component should carry matching distro and arch qualifiers."""
    etc = tmp_path / "etc"
    etc.mkdir()
    (etc / "os-release").write_text(
        'ID="ubuntu"\nVERSION_ID="24.04"\n',
        encoding="utf-8",
    )

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Return one deterministic dpkg-query row."""
        assert "--admindir=" in command[1]
        assert "${Architecture}" in command[-1]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="libssl3t64\t3.0.13-0ubuntu3.5\tamd64\n",
            stderr="",
        )

    monkeypatch.setattr(dpkg.subprocess, "run", fake_run)
    monkeypatch.setattr(dpkg, "_resolve_license", lambda *args: "Apache-2.0")

    components = dpkg.collect_components(tmp_path)

    assert len(components) == 1
    assert components[0].purl == (
        "pkg:deb/ubuntu/libssl3t64@3.0.13-0ubuntu3.5" "?arch=amd64&distro=ubuntu-24.04"
    )


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("arm64", "arch=arm64"),
        ("all", "arch=all"),
    ],
)
def test_dpkg_purl_preserves_architecture(architecture: str, expected: str) -> None:
    """Architecture qualifiers should remain explicit for every generated artifact."""
    purl = dpkg._dpkg_purl(
        "package",
        "1:2.0+build1",
        architecture,
        "ubuntu",
        "24.04",
    )
    assert expected in purl
    assert "@1%3A2.0%2Bbuild1?" in purl
