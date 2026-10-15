# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise package inventory against an isolated, real dpkg status database."""

from __future__ import annotations

import shutil

import pytest
from compliance.generators.dpkg import collect_components

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.integration]


def test_inventory_excludes_removed_packages_but_retains_payloads(tmp_path):
    if shutil.which("dpkg-query") is None:
        pytest.skip("dpkg-query is required to read the synthetic package database")

    packages = {
        "installed": "install ok installed",
        "held": "hold ok installed",
        "pending-removal": "deinstall ok installed",
        "unpacked": "install ok unpacked",
        "half-configured": "install ok half-configured",
        "half-installed": "install reinstreq half-installed",
        "removed": "deinstall ok config-files",
        "purged": "purge ok not-installed",
    }
    admindir = tmp_path / "var/lib/dpkg"
    admindir.mkdir(parents=True)
    (admindir / "status").write_text(
        "\n".join(
            f"Package: {name}\nStatus: {status}\nVersion: 1.0-1\n"
            "Architecture: amd64\nMaintainer: Test <test@example.com>\n"
            "Description: Synthetic inventory fixture\n"
            # Prove that the removed package's listed conffile is gone. An
            # absent conffile list is intentionally retained conservatively.
            + (
                "Conffiles:\n /etc/removed.conf " + "0" * 32 + "\n"
                if name == "removed"
                else ""
            )
            for name, status in packages.items()
        )
    )

    components = collect_components(root=tmp_path)

    assert {c.name for c in components} == {
        "installed",
        "held",
        "pending-removal",
        "unpacked",
        "half-configured",
        "half-installed",
    }
    # Missing license evidence must still reach the policy gate as UNKNOWN.
    assert all(c.version == "1.0-1" and c.spdx == "UNKNOWN" for c in components)
