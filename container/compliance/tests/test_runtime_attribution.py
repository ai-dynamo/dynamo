# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression cases from the Super VL runtime filesystem audit."""

from __future__ import annotations

import io
import tarfile

import pytest
from compliance.audit_image_sbom import (
    _covered_by_owner,
    _normalize,
    _ownership_coverage,
)
from compliance.binary_dependencies import go_license_expression, notices, tar_files
from compliance.generators.dpkg import _parse_dep5
from compliance.generators.python import collect_components, installed_search_paths

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _distribution(site, name, version, license_id):
    directory = site / f"{name}-{version}.dist-info"
    directory.mkdir(parents=True)
    (directory / "METADATA").write_text(
        f"Name: {name}\nVersion: {version}\nLicense-Expression: {license_id}\n"
    )
    return directory


def test_system_and_vendored_python_have_independent_attribution(tmp_path):
    site = tmp_path / "usr/local/lib/python3.12/dist-packages"
    _distribution(site, "setuptools", "80.0", "MIT")
    vendor = _distribution(
        site / "setuptools/_vendor", "autocommand", "2.2.2", "LGPL-3.0-only"
    )
    (vendor / "METADATA").write_text(
        "Name: autocommand\nVersion: 2.2.2\nLicense: LGPLv3\n"
        "Classifier: License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)\n"
    )
    (vendor / "LICENSE").write_text("Vendored LGPL copyright and license")
    _distribution(site / "pkg_resources/tests/data", "my-test-package", "1.0", "MIT")
    _distribution(
        tmp_path / "opt/dynamo/venv/lib/python3.12/site-packages",
        "urllib3",
        "2.8.0",
        "MIT",
    )
    components = collect_components(installed_search_paths(tmp_path))
    by_name = {c.name: c for c in components}
    assert set(by_name) == {"setuptools", "autocommand", "urllib3"}
    assert by_name["autocommand"].spdx == "LGPL-3.0-only"
    assert "Vendored LGPL copyright" in by_name["autocommand"].license_text


def test_python_retains_all_bundled_notices(tmp_path):
    directory = _distribution(tmp_path, "example", "1.0", "MIT OR Apache-2.0")
    for name, text in [
        ("LICENSE.MIT", "MIT copyright"),
        ("LICENSE.APACHE", "Apache copyright"),
        ("NOTICE", "Additional acknowledgment"),
    ]:
        (directory / name).write_text(text)
    text = collect_components([tmp_path])[0].license_text
    assert all(
        s in text
        for s in ["MIT copyright", "Apache copyright", "Additional acknowledgment"]
    )


def test_nrl_dep5_keeps_custom_terms():
    text = """Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/

Files: *
License: NRL-2-clause

Files: src/unix/*
License: NRL-3-clause

Files: sim/*
License: BSD-4-clause-UC
"""
    expression = _parse_dep5(text)
    assert "LicenseRef-NRL-2-clause" in expression
    assert "LicenseRef-NRL-3-clause" in expression
    assert "BSD-4-Clause-UC" in expression


def _cdx(name, path, cataloger="python-installed-package-cataloger", purl=None):
    return {
        "name": name,
        "purl": purl or f"pkg:pypi/{name}@1.0",
        "properties": [
            {"name": "syft:location:0:path", "value": path},
            {"name": "syft:package:foundBy", "value": cataloger},
        ],
    }


def test_only_verified_owner_records_cover_duplicate_detections():
    deb_path = "/usr/lib/python3/dist-packages/blinker-1.7.dist-info/METADATA"
    pe_path = "/usr/local/lib/python3.12/dist-packages/setuptools/cli.exe"
    vendor_path = (
        "/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/"
        "autocommand-2.2.2.dist-info/METADATA"
    )
    native = {
        "artifacts": [
            {
                "name": "python3-blinker",
                "type": "deb",
                "metadata": {"files": [{"path": deb_path}]},
            },
            {
                "name": "setuptools",
                "type": "python",
                "metadata": {
                    "sitePackagesRootPath": "/usr/local/lib/python3.12/dist-packages",
                    "files": [{"path": "setuptools/cli.exe"}, {"path": vendor_path}],
                },
            },
        ]
    }
    owners = _ownership_coverage(native, {"python3-blinker", "setuptools"})
    assert _covered_by_owner(_cdx("blinker", deb_path), owners)
    assert not _covered_by_owner(_cdx("blinker", deb_path), {})
    assert _covered_by_owner(
        _cdx("cli", pe_path, "pe-binary-package-cataloger"), owners
    )
    assert not _covered_by_owner(_cdx("autocommand", vendor_path), owners)
    assert not _covered_by_owner(
        _cdx("unknown", "/opt/unknown.exe", "pe-binary-package-cataloger"), owners
    )


def test_archive_preserves_license_ids_and_never_extracts_paths(tmp_path):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in ["pkg/MPL-2.0.txt", "pkg/LICENSE-MIT", "pkg/NOTICE"]:
            data = (name + " license body").encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    files = tar_files(stream.getvalue())
    text = notices(files)
    assert "MPL-2.0.txt license body" in text
    assert "LICENSE-MIT license body" in text
    assert "NOTICE license body" in text
    assert not list(tmp_path.iterdir())


def test_pep503_names_match_vendored_python_distributions():
    assert _normalize("jaraco.context") == _normalize("jaraco-context")
    assert _normalize("backports.tarfile") == _normalize("backports_tarfile")


def test_go_license_templates_do_not_relicense_the_module():
    artifact = {
        "licenses": [
            {
                "spdxExpression": "Apache-2.0",
                "urls": ["https://proxy.golang.org/cobra.zip#cobra@v1/LICENSE.txt"],
            },
            {
                "spdxExpression": "AGPL-3.0-only",
                "urls": [
                    "https://proxy.golang.org/cobra.zip#cobra@v1/cobra/cmd/license_agpl.go"
                ],
            },
        ]
    }
    assert go_license_expression(artifact) == "Apache-2.0"
    assert go_license_expression({"licenses": []}) is None


def test_distlib_launcher_requires_pip_owner_and_retained_license():
    launcher = "/usr/local/lib/python3.12/dist-packages/pip/_vendor/distlib/w64.exe"
    component = _cdx("Simple Launcher", launcher, "pe-binary-package-cataloger")
    owners = {launcher: {"pip"}}
    assert not _covered_by_owner(component, owners)
    owners[launcher.rsplit("/", 1)[0] + "/LICENSE.txt"] = {"pip"}
    assert _covered_by_owner(component, owners)


def test_standalone_distlib_launcher_requires_record_owned_license():
    root = "/opt/sglang/lib/python3.12/site-packages"
    launcher = root + "/distlib/w64.exe"
    component = _cdx("Simple Launcher", launcher, "pe-binary-package-cataloger")
    owners = {launcher: {"distlib"}}
    assert not _covered_by_owner(component, owners)
    owners[root + "/distlib-0.4.0.dist-info/licenses/LICENSE"] = {"distlib"}
    assert _covered_by_owner(component, owners)
    owners[launcher] = {"other-package"}
    assert not _covered_by_owner(component, owners)


def test_pre_commit_template_requires_exact_path_and_owner():
    path = "/opt/venv/lib/python3.12/site-packages/pre_commit/resources/empty_template_pre-commit-package-dev-1.rockspec"
    component = _cdx("pre-commit-package", path, "rocksdb-cataloger")
    assert not _covered_by_owner(component, {})
    assert _covered_by_owner(component, {path: {"pre-commit"}})
    component["name"] = "independent-vendored-distribution"
    assert not _covered_by_owner(component, {path: {"pre-commit"}})


def test_source_declarations_are_retained_without_hiding_binary_gaps(
    tmp_path, monkeypatch
):
    import json
    import sys

    from compliance.audit_image_sbom import main

    source = _cdx(
        "source-only",
        "/opt/project/Cargo.lock",
        "rust-cargo-lock-cataloger",
        "pkg:cargo/source-only@1.0",
    )
    source["version"] = "1.0"
    binary = _cdx(
        "compiled-dependency",
        "/opt/bin/tool",
        "cargo-auditable-binary-cataloger",
        "pkg:cargo/compiled-dependency@2.0",
    )
    binary["version"] = "2.0"
    sbom = tmp_path / "image.json"
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("ecosystem,name,version,spdx\n")
    declarations = tmp_path / "declarations.txt"
    report = tmp_path / "report.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--syft-sbom",
            str(sbom),
            "--our-csv",
            str(inventory),
            "--installed-only",
            "--fail-on-findings",
            "--report",
            str(report),
            "--declarations-report",
            str(declarations),
        ],
    )
    sbom.write_text(json.dumps({"components": [source, binary]}))
    assert main() == 1
    assert "compiled-dependency" in report.read_text()
    assert "source-only" in declarations.read_text()
    sbom.write_text(json.dumps({"components": [source]}))
    assert main() == 0
