# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions found when auditing the independently audited runtime images."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from compliance import license_materials
from compliance.audit_image_sbom import main as audit
from compliance.binary_dependencies import collect as collect_binary
from compliance.binary_dependencies import debian_git_lfs_main, etcd_binary_version
from compliance.generators.common import (
    Component,
    dedupe_by_name_version,
    write_cyclonedx,
)
from compliance.generators.go import collect_components as collect_go
from compliance.generators.native import (
    collect_bundled_python_binaries,
    collect_python_interpreters,
)
from compliance.generators.python import collect_components, installed_search_paths
from compliance.policy.validate import load_policy, validate_row

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize("failure", [None, "unowned", "changed-license"])
@pytest.mark.parametrize("version", ["3.2.2", "3.3.2"])
def test_bundled_cbc_keeps_its_own_license_and_record_evidence(
    tmp_path, failure, version
):
    site = tmp_path / "site-packages"
    info = site / f"pulp-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Name: PuLP\nVersion: {version}\nLicense: MIT\n")
    binary = site / "pulp/solverdir/cbc/linux/i64/cbc"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"standalone solver fixture")
    notice = binary.parent / "coin-license.txt"
    text = "Eclipse Public License - v 1.0\nSolver's own upstream terms\n"
    notice.write_text("MIT" if failure == "changed-license" else text)
    owned = [binary] if failure == "unowned" else [binary, notice]
    (info / "RECORD").write_text(
        "".join(f"{path.relative_to(site)},,\n" for path in owned)
    )
    legal = tmp_path / "legal"
    if failure:
        with pytest.raises(ValueError, match="ownership|license"):
            collect_bundled_python_binaries([site], legal)
        return
    (component,) = collect_bundled_python_binaries([site], legal)
    assert component.ecosystem == "native" and component.spdx == "EPL-1.0"
    assert component.license_text == text
    provenance = json.loads((legal / "bundled-native-provenance.json").read_text())
    assert provenance[0]["binary"] == str(binary)
    assert provenance[0]["owner_version"] == version
    policy = load_policy(Path(__file__).resolve().parents[1] / "policy/licenses.toml")
    assert validate_row(policy, "native", "cbc", "UNKNOWN", "EPL-1.0") is None
    assert validate_row(policy, "rust", "cbc", "0.1.2", "EPL-1.0") is not None
    assert validate_row(policy, "native", "cbc", "UNKNOWN", "EPL-2.0") is not None


@pytest.mark.parametrize(
    "location",
    ["/usr/bin/git-lfs", "usr/bin/git-lfs", "./usr/bin/git-lfs", "/bin/git-lfs"],
)
def test_debian_git_lfs_main_requires_exact_installed_owner(tmp_path, location):
    info = tmp_path / "var/lib/dpkg/info"
    info.mkdir(parents=True)
    (info / "git-lfs.list").write_text("/usr/bin/git-lfs\n")
    (info.parent / "status").write_text(
        "Package: git-lfs\nStatus: install ok installed\nVersion: 3.4.1-1\n"
    )
    binary = tmp_path / "usr/bin/git-lfs"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"installed Git LFS fixture")
    (tmp_path / "bin").symlink_to("usr/bin", target_is_directory=True)
    copyright = tmp_path / "usr/share/doc/git-lfs/copyright"
    copyright.parent.mkdir(parents=True)
    copyright.write_text(
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n\n"
        "Files: *\nCopyright: Git LFS contributors\nLicense: Expat\n"
    )
    artifact = {
        "name": "github.com/git-lfs/git-lfs",
        "version": "UNKNOWN",
        "metadata": {"mainModule": "github.com/git-lfs/git-lfs"},
        "locations": [{"path": location}],
    }
    component = debian_git_lfs_main(artifact, tmp_path)
    assert component.spdx == "MIT"
    assert component.source_url == "pkg:deb/ubuntu/git-lfs@3.4.1-1"
    assert component.license_text == copyright.read_text()
    # The collector output is consumed by the Go notices generator. Debian
    # provenance must survive without turning the Go module into a deb entry.
    output = tmp_path / "go.cdx.json"
    write_cyclonedx([component], output)
    assert json.loads(output.read_text())["components"][0]["purl"].startswith(
        "pkg:golang/"
    )
    licenses = tmp_path / "licenses"
    harvested = licenses / "github.com/git-lfs/git-lfs@UNKNOWN"
    harvested.mkdir(parents=True)
    (harvested / "LICENSE-NOTICES.txt").write_text(component.license_text)
    (generated,) = collect_go([output], licenses)
    assert (
        generated.ecosystem,
        generated.name,
        generated.version,
        generated.spdx,
        generated.source_url,
    ) == (
        component.ecosystem,
        component.name,
        component.version,
        component.spdx,
        component.source_url,
    )
    assert component.license_text.strip() in generated.license_text
    copied = tmp_path / "other-git-lfs"
    copied.write_bytes(binary.read_bytes())
    artifact["locations"] = [{"path": "/other-git-lfs"}]
    with pytest.raises(ValueError, match="executable locations"):
        debian_git_lfs_main(artifact, tmp_path)
    artifact["locations"] = [{"path": location}]
    (info / "git-lfs.list").write_text("/usr/share/doc/git-lfs/copyright\n")
    with pytest.raises(ValueError, match="file ownership"):
        debian_git_lfs_main(artifact, tmp_path)


def test_debian_git_lfs_main_does_not_cover_other_go_modules(tmp_path):
    artifact = {"name": "github.com/other/dependency", "version": "UNKNOWN"}
    assert debian_git_lfs_main(artifact, tmp_path) is None


def test_etcd_release_identity_is_read_from_exact_executable(tmp_path, monkeypatch):
    def version(command, **kwargs):
        assert command == [str(tmp_path / "usr/local/bin/etcd/etcd"), "--version"]
        assert kwargs["timeout"] == 10
        return type("Result", (), {"stdout": "etcd Version: 3.5.33\nGit SHA: abc\n"})()

    monkeypatch.setattr("compliance.binary_dependencies.subprocess.run", version)
    assert etcd_binary_version("/usr/local/bin/etcd/etcd", tmp_path) == "v3.5.33"
    with pytest.raises(ValueError, match="unrecognized"):
        etcd_binary_version("/some/other/etcd", tmp_path)


@pytest.mark.parametrize(
    "artifacts",
    [[], [{"name": "dynamo-llm", "foundBy": "cargo-auditable-binary-cataloger"}]],
)
def test_frontend_without_uv_or_etcd_needs_no_unrelated_binary_source(
    tmp_path, monkeypatch, artifacts
):
    output = tmp_path / "output"
    output.mkdir()

    def scan(command, **kwargs):
        Path(command[-1].split("=", 1)[1]).write_text('{"artifacts": []}')

    monkeypatch.setattr("compliance.binary_dependencies.subprocess.run", scan)
    assert collect_binary({"artifacts": artifacts}, output, tmp_path, "syft") == []


def test_duplicate_package_copies_preserve_all_bundled_attribution():
    first = Component(
        "python", "example", "1.0", "MIT", license_text="Copyright holder A"
    )
    second = Component(
        "python", "example", "1.0", "MIT", license_text="Copyright holder B"
    )
    (component,) = dedupe_by_name_version([first, second, first])
    assert "Copyright holder A" in component.license_text
    assert "Copyright holder B" in component.license_text


def test_upstream_license_wins_over_canonical_fallback():
    fallback = Component(
        "python",
        "example",
        "1.0",
        "MIT",
        license_text="Canonical MIT text",
        license_text_is_canonical=True,
    )
    upstream = Component(
        "python", "example", "1.0", "MIT", license_text="Copyright upstream holder"
    )
    (component,) = dedupe_by_name_version([fallback, upstream])
    assert component.license_text == upstream.license_text
    assert not component.license_text_is_canonical


def test_hidden_versioned_environment_and_nested_cutlass(tmp_path):
    site = tmp_path / "opt/musespark-trtllm/.venv-3.12/lib/python3.12/site-packages"
    cutlass = site / "tensorrt_llm/include/trtllm_gen_kernels/fmha/cutlass"
    egg = cutlass / "python/cutlass_library.egg-info"
    egg.mkdir(parents=True)
    (egg / "PKG-INFO").write_text("Name: cutlass_library\nVersion: 4.4.2\n")
    (cutlass / "LICENSE.txt").write_text(
        "SPDX-License-Identifier: BSD-3-Clause\nCopyright upstream holder\n"
    )
    components = collect_components(installed_search_paths(tmp_path))
    assert [(c.name, c.version, c.spdx) for c in components] == [
        ("cutlass_library", "4.4.2", "BSD-3-Clause")
    ]
    assert "Copyright upstream holder" in components[0].license_text


@pytest.mark.parametrize(
    "ownership",
    ["exact", "unowned", "unknown-license", "ambiguous", "ambiguous-unknown"],
)
def test_system_python_requires_unambiguous_attributed_dpkg_owner(tmp_path, ownership):
    site = tmp_path / "usr/lib/python3/dist-packages"
    info = site / "os_package-1.0.egg-info"
    info.mkdir(parents=True)
    metadata = info / "PKG-INFO"
    metadata.write_text("Name: os-package\nVersion: 1.0\n")
    records = tmp_path / "var/lib/dpkg/info"
    records.mkdir(parents=True)
    filename = "/" + str(metadata.relative_to(tmp_path))
    (records / "python3-os-package.list").write_text(
        filename if ownership != "unowned" else "/some/other/file"
    )
    packages = [
        Component(
            "dpkg",
            "python3-os-package",
            "1.0-1",
            "UNKNOWN" if ownership == "unknown-license" else "MIT",
        )
    ]
    if ownership.startswith("ambiguous"):
        packages.append(
            Component(
                "dpkg",
                "another-package",
                "1.0",
                "UNKNOWN" if ownership == "ambiguous-unknown" else "MIT",
            )
        )
        (records / "another-package.list").write_text(filename)
    components = collect_components([site], packages, tmp_path)
    if ownership == "exact":
        assert components == []
    else:
        assert [(c.name, c.spdx) for c in components] == [("os-package", "UNKNOWN")]


def test_copied_interpreter_has_individual_full_notice(tmp_path):
    prefix = tmp_path / "opt/python/cpython-3.12.15-linux-aarch64-gnu"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python3").write_bytes(b"interpreter")
    license_file = prefix / "lib/python3.12/LICENSE.txt"
    license_file.parent.mkdir(parents=True)
    license_file.write_text(
        "Python license\nCopyright upstream Python holder\nZERO-CLAUSE BSD LICENSE\n"
    )
    (component,) = collect_python_interpreters(tmp_path)
    assert (component.ecosystem, component.name, component.version, component.spdx) == (
        "native",
        "python",
        "3.12.15",
        "Python-2.0 AND 0BSD",
    )
    assert component.license_text == license_file.read_text()


def test_binary_version_mismatch_is_not_covered_by_other_version(tmp_path):
    sbom = tmp_path / "syft.json"
    sbom.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "name": "sample-crate",
                        "version": "2.0",
                        "purl": "pkg:cargo/sample-crate@2.0",
                        "properties": [
                            {
                                "name": "syft:package:foundBy",
                                "value": "cargo-auditable-binary-cataloger",
                            }
                        ],
                    }
                ]
            }
        )
    )
    inventory = tmp_path / "osrb.csv"
    with inventory.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ecosystem", "name", "version", "spdx"])
        writer.writerow(["rust", "sample-crate", "1.0", "MIT"])
    assert (
        audit(
            [
                "--syft-sbom",
                str(sbom),
                "--our-csv",
                str(inventory),
                "--fail-on-findings",
            ]
        )
        == 1
    )


def test_soxr_source_and_relink_materials_are_shipped(tmp_path, monkeypatch):
    info = (
        tmp_path / "opt/dynamo/venv/lib/python3.12/site-packages/soxr-1.1.0.dist-info"
    )
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Name: soxr\nVersion: 1.1.0\n")
    data = b"source archive fixture"
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(license_materials, "SOXR_SHA256", sha)
    release = {
        "urls": [
            {
                "packagetype": "sdist",
                "digests": {"sha256": sha},
                "url": "https://sources.invalid/soxr.tar.gz",
            }
        ]
    }
    monkeypatch.setattr(
        license_materials.urllib.request,
        "urlopen",
        lambda url, **kwargs: io.BytesIO(
            json.dumps(release).encode() if url.endswith("/json") else data
        ),
    )
    legal = tmp_path / "legal"
    license_materials.collect(tmp_path, legal)
    assert (legal / "sources/soxr-1.1.0/soxr-1.1.0.tar.gz").read_bytes() == data
    assert "Patch and relink" in (legal / "sources/soxr-1.1.0/README.md").read_text()


def test_soxr_source_checksum_failure_prevents_distribution(tmp_path, monkeypatch):
    info = (
        tmp_path / "opt/dynamo/venv/lib/python3.12/site-packages/soxr-1.1.0.dist-info"
    )
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Name: soxr\nVersion: 1.1.0\n")
    release = {
        "urls": [
            {
                "packagetype": "sdist",
                "digests": {"sha256": license_materials.SOXR_SHA256},
                "url": "https://sources.invalid/soxr.tar.gz",
            }
        ]
    }
    monkeypatch.setattr(
        license_materials.urllib.request,
        "urlopen",
        lambda url, **kwargs: io.BytesIO(
            json.dumps(release).encode() if url.endswith("/json") else b"wrong bytes"
        ),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        license_materials.collect(tmp_path, tmp_path / "legal")
    assert not (tmp_path / "legal/sources").exists()


def test_relative_ci_paths_match_exact_dpkg_file_ownership(tmp_path):
    metadata = "usr/lib/python3/dist-packages/dbus_python-1.3.2.egg-info/PKG-INFO"
    component = {
        "name": "dbus-python",
        "version": "1.3.2",
        "purl": "pkg:pypi/dbus-python@1.3.2",
        "properties": [{"name": "syft:location:0:path", "value": metadata}],
    }
    sbom = tmp_path / "image.json"
    sbom.write_text(json.dumps({"components": [component]}))
    native = tmp_path / "native.json"
    native.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "name": "python3-dbus",
                        "type": "deb",
                        "metadata": {"files": [{"path": "/" + metadata}]},
                    }
                ]
            }
        )
    )
    inventory = tmp_path / "osrb.csv"
    inventory.write_text("ecosystem,name,version,spdx\ndpkg,python3-dbus,1.3.2,MIT\n")
    arguments = [
        "--syft-sbom",
        str(sbom),
        "--syft-json",
        str(native),
        "--our-csv",
        str(inventory),
        "--fail-on-findings",
    ]
    assert audit(arguments) == 0
    native.write_text(json.dumps({"artifacts": []}))
    assert audit(arguments) == 1
