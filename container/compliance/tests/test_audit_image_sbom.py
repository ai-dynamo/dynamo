# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current-build inventory selection and verified dpkg ownership regressions."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from compliance.audit_image_sbom import _covered_by_owner, _ownership_coverage, main

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _component(*paths):
    return {
        "name": "dbus-python",
        "version": "1.3.2",
        "purl": "pkg:pypi/dbus-python@1.3.2",
        "properties": [
            {"name": f"syft:location:{i}:path", "value": path}
            for i, path in enumerate(paths)
        ],
    }


def _native(path, name="python3-dbus", kind="deb"):
    return {
        "artifacts": [
            {"name": name, "type": kind, "metadata": {"files": [{"path": path}]}}
        ]
    }


@pytest.mark.parametrize("component_prefix", ["", "/", "./"])
@pytest.mark.parametrize("owner_prefix", ["", "/", "./"])
def test_dpkg_ownership_accepts_root_relative_and_absolute_paths(
    component_prefix, owner_prefix
):
    path = "usr/lib/python3/dist-packages/dbus_python-1.3.2.egg-info/PKG-INFO"
    owners = _ownership_coverage(_native(owner_prefix + path), {"python3-dbus"})
    assert _covered_by_owner(_component(component_prefix + path), owners)


def test_ownership_requires_attributed_package_and_every_metadata_file():
    path = "usr/lib/python3/dist-packages/dbus_python-1.3.2.egg-info/PKG-INFO"
    native = _native("/" + path)
    assert not _covered_by_owner(_component(path), _ownership_coverage(native, set()))
    owners = _ownership_coverage(native, {"python3-dbus"})
    assert not _covered_by_owner(_component(path, path + ".unowned"), owners)
    assert not _covered_by_owner(_component(), owners)
    # An installed wheel cannot stand in for a separately licensed vendored dist.
    wheel = _native(path, name="parent-wheel", kind="python")
    assert not _covered_by_owner(
        _component(path), _ownership_coverage(wheel, {"parent-wheel"})
    )
    outside = "/usr/local/lib/python3.12/dist-packages/vendor.dist-info/METADATA"
    assert not _covered_by_owner(
        _component(outside),
        _ownership_coverage(_native(outside), {"python3-dbus"}),
    )


def test_audit_cli_requires_verified_ownership(tmp_path):
    path = "usr/lib/python3/dist-packages/dbus_python-1.3.2.egg-info/PKG-INFO"
    sbom = tmp_path / "syft.cdx.json"
    native = tmp_path / "syft.json"
    inventory = tmp_path / "inventory.csv"
    report = tmp_path / "report.txt"
    sbom.write_text(json.dumps({"components": [_component(path)]}))
    native.write_text(json.dumps(_native("/" + path)))
    inventory.write_text("name\npython3-dbus\n")
    args = [
        "--syft-sbom",
        str(sbom),
        "--our-csv",
        str(inventory),
        "--report",
        str(report),
        "--fail-on-findings",
    ]
    assert main(args) == 1
    assert main(args + ["--syft-json", str(native)]) == 0
    assert not report.read_text().strip()
    inventory.write_text("name\n")
    assert main(args + ["--syft-json", str(native)]) == 1


@pytest.mark.timeout(10)
@pytest.mark.parametrize("platform", ["amd64", "arm64"])
@pytest.mark.parametrize("current_count", [0, 1, 2])
def test_workflow_selects_unique_current_inventory(tmp_path, platform, current_count):
    workflow = (
        Path(__file__).resolve().parents[3]
        / ".github/workflows/shared-compliance-audit.yml"
    )
    source = workflow.read_text()
    if "matrix.platform" not in source and platform == "arm64":
        pytest.skip("This workflow audits amd64 only")
    # Execute the actual selector embedded in the workflow, with an isolated root.
    start = source.index("          shopt -s nullglob")
    end = source.index('          echo "auditing against', start)
    script = "set -euo pipefail\n" + source[start:end]
    script = script.replace("/tmp/compliance", str(tmp_path))
    script = script.replace("${{ matrix.platform }}", platform)
    script += '\nprintf "%s\\n" "$OUR_CSV"\n'
    arch = tmp_path / f"linux_{platform}"
    baseline = arch / "baseline"
    baseline.mkdir(parents=True)
    for path in [
        baseline / "osrb-old-12345678.csv",
        arch / "osrb-image-12345678.diff.csv",
        arch / "osrb-image-deadbeef.csv",
        tmp_path / "linux_other/osrb-image-12345678.csv",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for i in range(current_count):
        (arch / f"osrb-image{i}-12345678.csv").touch()
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "GITHUB_SHA": "12345678" + "0" * 32},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == (0 if current_count == 1 else 1), (
        result.stdout + result.stderr
    )
    if current_count == 1:
        assert result.stdout.strip() == str(arch / "osrb-image0-12345678.csv")
