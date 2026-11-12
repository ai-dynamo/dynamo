# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current-build inventory selection and verified dpkg ownership regressions."""

import json

import pytest
from compliance.audit_image_sbom import _covered_by_owner, _ownership_coverage, main

pytestmark = [
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


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
            {
                "name": name,
                "version": "1.3.2-1",
                "type": kind,
                "metadata": {"files": [{"path": path}]},
            }
        ]
    }


@pytest.mark.parametrize("component_prefix", ["", "/", "./"])
@pytest.mark.parametrize("owner_prefix", ["", "/", "./"])
def test_dpkg_ownership_accepts_root_relative_and_absolute_paths(
    component_prefix, owner_prefix
):
    path = "usr/lib/python3/dist-packages/dbus_python-1.3.2.egg-info/PKG-INFO"
    owners = _ownership_coverage(
        _native(owner_prefix + path), {("dpkg", "python3-dbus", "1.3.2-1")}
    )
    assert _covered_by_owner(_component(component_prefix + path), owners)


def test_ownership_requires_attributed_package_and_every_metadata_file():
    path = "usr/lib/python3/dist-packages/dbus_python-1.3.2.egg-info/PKG-INFO"
    native = _native("/" + path)
    assert not _covered_by_owner(_component(path), _ownership_coverage(native, set()))
    owners = _ownership_coverage(native, {("dpkg", "python3-dbus", "1.3.2-1")})
    assert not _covered_by_owner(_component(path, path + ".unowned"), owners)
    assert not _covered_by_owner(_component(), owners)
    # An installed wheel cannot stand in for a separately licensed vendored dist.
    wheel = _native(path, name="parent-wheel", kind="python")
    assert not _covered_by_owner(
        _component(path),
        _ownership_coverage(wheel, {("python", "parent-wheel", "1.3.2-1")}),
    )
    outside = "/usr/local/lib/python3.12/dist-packages/vendor.dist-info/METADATA"
    assert not _covered_by_owner(
        _component(outside),
        _ownership_coverage(_native(outside), {("dpkg", "python3-dbus", "1.3.2-1")}),
    )


def test_audit_cli_requires_verified_ownership(tmp_path):
    path = "usr/lib/python3/dist-packages/dbus_python-1.3.2.egg-info/PKG-INFO"
    sbom = tmp_path / "syft.cdx.json"
    native = tmp_path / "syft.json"
    inventory = tmp_path / "inventory.csv"
    report = tmp_path / "report.txt"
    sbom.write_text(json.dumps({"components": [_component(path)]}))
    native.write_text(json.dumps(_native("/" + path)))
    inventory.write_text("ecosystem,name,version\ndpkg,python3-dbus,1.3.2-1\n")
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
    inventory.write_text("ecosystem,name,version\n")
    assert main(args + ["--syft-json", str(native)]) == 1


@pytest.mark.parametrize(
    "ecosystem,version",
    [("dpkg", "1.3.2-0"), ("python", "1.3.2-1")],
)
def test_owner_from_another_version_or_ecosystem_is_not_attribution(ecosystem, version):
    path = "/usr/lib/python3/dist-packages/dbus_python-1.3.2.egg-info/PKG-INFO"
    owners = _ownership_coverage(_native(path), {(ecosystem, "python3-dbus", version)})
    assert not _covered_by_owner(_component(path), owners)


def test_external_root_ownership_uses_the_same_root_relative_paths():
    path = "/usr/lib/python3/dist-packages/dbus_python-1.3.2.egg-info/PKG-INFO"
    owners = _ownership_coverage(
        _native(path), {("dpkg", "python3-dbus", "1.3.2-1")}, "/rootfs"
    )
    assert _covered_by_owner(_component("/rootfs" + path), owners, "/rootfs")
    assert not _covered_by_owner(_component("/rootfs-other" + path), owners, "/rootfs")


def _audit_fixture(tmp_path, components, rows):
    sbom = tmp_path / "image.json"
    sbom.write_text(json.dumps({"components": components}))
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("ecosystem,name,version,spdx\n" + rows)
    return [
        "--syft-sbom",
        str(sbom),
        "--our-csv",
        str(inventory),
        "--fail-on-findings",
    ]


@pytest.mark.parametrize(
    "ecosystem,purl_type,recorded,scanned,expected",
    [
        ("python", "pypi", "1.0", "2.0", 1),
        ("python", "pypi", "1.0", "1.0", 0),
        ("rust", "cargo", "1.0", "v1.0", 1),
        ("go", "golang", "1.0", "v1.0", 0),
        ("go", "golang", "v1.0", "1.0", 0),
        ("go", "golang", "v1.0", "v2.0", 1),
    ],
)
def test_audit_requires_exact_version(
    tmp_path, ecosystem, purl_type, recorded, scanned, expected
):
    component = {
        "name": "demo",
        "version": scanned,
        "purl": f"pkg:{purl_type}/demo@{scanned}",
        "licenses": [{"license": {"id": "AGPL-3.0-only" if expected else "MIT"}}],
    }
    args = _audit_fixture(tmp_path, [component], f"{ecosystem},demo,{recorded},MIT\n")
    assert main(args) == expected


def test_baseline_cannot_hide_a_different_ecosystem(tmp_path):
    component = {"name": "demo", "version": "1", "purl": "pkg:pypi/demo@1"}
    args = _audit_fixture(tmp_path, [component], "")
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps({"components": [{**component, "purl": "pkg:cargo/demo@1"}]})
    )
    assert main(args + ["--base-sbom", str(base)]) == 1
    base.write_text(json.dumps({"components": [component]}))
    assert main(args + ["--base-sbom", str(base)]) == 0


def test_missing_requested_baseline_fails(tmp_path):
    args = _audit_fixture(tmp_path, [], "")
    with pytest.raises(SystemExit, match="2"):
        main(args + ["--base-sbom", str(tmp_path / "absent.json")])


def test_audit_validates_resolved_inventory_policy(tmp_path):
    args = _audit_fixture(tmp_path, [], "python,demo,1,AGPL-3.0-only\n")
    policy = tmp_path / "licenses.toml"
    policy.write_text('[licenses]\nallow = ["MIT"]\ndeny = ["AGPL-3.0-only"]\n')
    assert main(args + ["--policy", str(policy)]) == 1
    (tmp_path / "inventory.csv").write_text(
        "ecosystem,name,version,spdx\npython,demo,1,MIT\n"
    )
    assert main(args + ["--policy", str(policy)]) == 0


@pytest.mark.parametrize(
    "contents",
    [
        "change,name,new_version\nadded,demo,1\n",
        "ecosystem,name,version\npython,demo,\n",
        "ecosystem,name,version,spdx\npython,demo,1,MIT\npython,demo,1,AGPL-3.0-only\n",
    ],
)
def test_audit_rejects_incomplete_or_conflicting_inventory(tmp_path, contents):
    args = _audit_fixture(tmp_path, [], "")
    (tmp_path / "inventory.csv").write_text(contents)
    with pytest.raises(ValueError):
        main(args)


def test_report_preserves_each_missing_version(tmp_path):
    components = [
        {"name": "demo", "version": version, "purl": f"pkg:pypi/demo@{version}"}
        for version in ("1", "2")
    ]
    args = _audit_fixture(tmp_path, components, "")
    report = tmp_path / "report.txt"
    assert main(args + ["--report", str(report)]) == 1
    assert report.read_text().splitlines() == ["pypi\tdemo\t1", "pypi\tdemo\t2"]


@pytest.mark.parametrize(
    "kind,expected", [("file", 0), ("operating-system", 0), ("library", 1)]
)
def test_cyclonedx_evidence_is_not_a_package_but_unknown_package_versions_are(
    tmp_path, kind, expected
):
    args = _audit_fixture(
        tmp_path, [{"type": kind, "name": "example", "version": ""}], ""
    )
    assert main(args) == expected
