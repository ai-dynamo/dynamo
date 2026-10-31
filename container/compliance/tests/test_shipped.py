# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shipped files need attribution even when they are nested, source, or resources."""

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest
from compliance import shipped
from compliance.audit_image_sbom import main as audit
from compliance.generators.__main__ import main as generate
from compliance.generators.common import Component
from compliance.generators.python import collect_components
from compliance.identity import component_identity
from compliance.policy.validate import load_policy, validate_row
from jinja2 import Environment, StrictUndefined

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(url):
        raise AssertionError(f"unexpected network request: {url}")

    monkeypatch.setattr(shipped, "_read_url", fail)


def artifact(name, kind, path, version="1.0", **kwargs):
    return {
        "id": name + path,
        "name": name,
        "version": version,
        "type": kind,
        "foundBy": "test",
        "locations": [{"path": path}],
        **kwargs,
    }


def owner(path, name="setuptools", license_id="MIT"):
    return artifact(
        name,
        "python",
        "site/METADATA",
        licenses=[{"spdxExpression": license_id}],
        metadata={"sitePackagesRootPath": "site", "files": [{"path": path}]},
    )


def test_nested_distributions_are_attributed_under_their_own_license(tmp_path):
    for prefix, name, license_id in [
        ("", "parent", "MIT"),
        ("parent/_vendor/", "nested", "EPL-1.0"),
    ]:
        directory = tmp_path / (prefix + name + "-1.0.dist-info")
        directory.mkdir(parents=True)
        (directory / "METADATA").write_text(
            f"Name: {name}\nVersion: 1.0\nLicense-Expression: {license_id}\n"
        )
        (directory / "LICENSE").write_text(license_id + " upstream notice")
    components = {c.name: c for c in collect_components([tmp_path])}
    assert components["nested"].spdx == "EPL-1.0"
    assert components["nested"].license_text == "EPL-1.0 upstream notice"


def test_crate_uses_exact_version_checksum_and_real_notice(tmp_path, monkeypatch):
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w:gz") as archive:
        entry = tarfile.TarInfo("demo-1.2/LICENSE")
        text = b"Copyright upstream. MPL 2.0."
        entry.size = len(text)
        archive.addfile(entry, io.BytesIO(text))
        manifest = b'[package]\nname = "demo"\nversion = "1.2"\nlicense = "MPL-2.0"\n'
        entry = tarfile.TarInfo("demo-1.2/Cargo.toml")
        entry.size = len(manifest)
        archive.addfile(entry, io.BytesIO(manifest))
    package = body.getvalue()
    metadata = json.dumps(
        {"vers": "1.2", "cksum": hashlib.sha256(package).hexdigest()}
    ).encode()
    calls = []

    def fetch(url):
        calls.append(url)
        return metadata if "index.crates.io/" in url else package

    monkeypatch.setattr(shipped, "_read_url", fetch)
    spdx, text, source, digest = shipped._crate("demo", "1.2", tmp_path)
    assert spdx == "MPL-2.0" and "Copyright upstream" in text
    assert source.endswith("demo-1.2.crate")
    assert digest == hashlib.sha256(package).hexdigest()
    assert len(calls) == 2
    # Reuse verified cache, but never accept a corrupted archive.
    shipped._crate("demo", "1.2", tmp_path)
    assert len(calls) == 2
    (tmp_path / "demo@1.2.crate").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        shipped._crate("demo", "1.2", tmp_path)


def test_nonpermissive_crate_source_is_retained_and_provenance_written(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "demo@1.0.crate").write_bytes(b"upstream source archive")
    monkeypatch.setattr(
        shipped,
        "_crate",
        lambda *args: ("MPL-2.0", "real notice", "https://source", "checksum"),
    )
    item = artifact("demo", "rust-crate", "opt/bin/tool", purl="pkg:cargo/demo@1.0")
    result = shipped.supplement(
        {"artifacts": [item]}, [], tmp_path, cache, set(), tmp_path / "legal"
    )
    assert result[0].license_text == "real notice"
    assert (
        tmp_path / "legal/source-archives/demo-1.0.crate"
    ).read_bytes() == b"upstream source archive"
    records = json.loads((tmp_path / "legal/shipped-provenance.json").read_text())
    assert records[0]["source_sha256"] == "checksum"
    assert records[0]["locations"] == [{"path": "opt/bin/tool"}]


def test_parent_package_does_not_cover_unknown_binary_or_other_ecosystem(tmp_path):
    item = artifact("cbc", "binary", "site/vendor/cbc.exe")
    existing = [Component("rust", "cbc", "1.0", "MIT")]
    result = shipped.supplement(
        {"artifacts": [item, owner("vendor/cbc.exe")]},
        existing,
        tmp_path,
        tmp_path / "cache",
        set(),
        tmp_path / "legal",
    )
    binary = next(c for c in result if c.name == "cbc")
    assert binary.ecosystem == "native" and binary.spdx == "UNKNOWN"


def test_solver_uses_own_license_with_scoped_exception(tmp_path):
    path = "pulp/solverdir/cbc/win/i64/cbc.exe"
    license_path = tmp_path / "site" / Path(path).parent / "coin-license.txt"
    license_path.parent.mkdir(parents=True)
    license_path.write_text("Eclipse Public License - v 1.0\nActual solver license")
    item = artifact("cbc", "binary", "site/" + path)
    spdx, text, source = shipped._resource(
        item, shipped._owners([owner(path, "pulp")]), tmp_path
    )
    assert spdx == "EPL-1.0" and "Actual solver license" in text
    policy = load_policy(Path(__file__).resolve().parents[1] / "policy/licenses.toml")
    assert validate_row(policy, "native", "cbc", "UNKNOWN", spdx) is None
    assert (
        validate_row(policy, "native", "unreviewed-solver", "UNKNOWN", spdx) is not None
    )
    assert validate_row(policy, "rust", "cbc", "0.1.2", spdx) is not None
    assert validate_row(policy, "native", "cbc", "UNKNOWN", "EPL-2.0") is not None
    assert validate_row(policy, "native", "cbc", "UNKNOWN", "UNKNOWN") is not None
    assert "coin-or" in source


def test_resource_requires_proof_for_every_location(tmp_path):
    item = artifact("cli", "binary", "site/setuptools/cli.exe")
    owners = shipped._owners([owner("setuptools/cli.exe")])
    assert shipped._resource(item, owners, tmp_path)[0] == "MIT"
    item["locations"].append({"path": "site/other/cli.exe"})
    assert shipped._resource(item, owners, tmp_path)[0] == "UNKNOWN"


def test_source_lockfile_keeps_dependency_and_local_manifest_license(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "local"\nlicense = "EPL-1.0"\n'
    )
    (directory / "LICENSE").write_text("upstream source notice")
    item = artifact("local", "rust-crate", "source/Cargo.lock")
    result = shipped.supplement(
        {"artifacts": [item]},
        [],
        tmp_path,
        tmp_path / "cache",
        set(),
        tmp_path / "legal",
    )
    assert result[0].spdx == "EPL-1.0"
    assert result[0].license_text == "upstream source notice"


def test_go_licenses_preserve_and_not_invented_or(tmp_path, monkeypatch):
    monkeypatch.setattr(shipped, "_read_url", lambda url: b"exact module source")
    item = artifact(
        "example.org/module",
        "go-module",
        "bin/server",
        purl="pkg:golang/example.org/module@1.0",
        licenses=[
            {"spdxExpression": "MIT", "contents": "copyright A"},
            {"spdxExpression": "EPL-1.0", "contents": "copyright B"},
        ],
    )
    result = shipped.supplement(
        {"artifacts": [item]},
        [],
        tmp_path,
        tmp_path / "cache",
        set(),
        tmp_path / "legal",
    )
    assert "AND" in result[0].spdx
    assert (
        "copyright A" in result[0].license_text
        and "copyright B" in result[0].license_text
    )


def test_existing_other_version_does_not_hide_shipped_dependency(tmp_path):
    item = artifact(
        "example.org/module",
        "go-module",
        "bin/server",
        version="2.0",
        licenses=[{"spdxExpression": "MIT"}],
    )
    result = shipped.supplement(
        {"artifacts": [item]},
        [Component("go", "example.org/module", "1.0", "MIT")],
        tmp_path,
        tmp_path / "cache",
        set(),
        tmp_path / "legal",
    )
    assert result[0].version == "2.0"


@pytest.mark.parametrize(
    "inventory,expected",
    [
        ("ecosystem,name,version\nrust,cbc,0.1.2\n", 1),
        ("ecosystem,name,version\nnative,cbc,UNKNOWN\n", 0),
    ],
)
def test_audit_does_not_confuse_solver_with_rust_crate(tmp_path, inventory, expected):
    sbom = tmp_path / "scan.json"
    sbom.write_text(json.dumps({"components": [{"name": "cbc", "version": "UNKNOWN"}]}))
    csv = tmp_path / "inventory.csv"
    csv.write_text(inventory)
    assert (
        audit(["--syft-sbom", str(sbom), "--our-csv", str(csv), "--fail-on-findings"])
        == expected
    )


def test_audit_python_normalization_preserves_go_module_paths(tmp_path):
    sbom = tmp_path / "scan.json"
    sbom.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "name": "ruamel-yaml",
                        "version": "1",
                        "purl": "pkg:pypi/ruamel-yaml@1",
                    }
                ]
            }
        )
    )
    csv = tmp_path / "inventory.csv"
    csv.write_text("ecosystem,name,version\npython,ruamel.yaml,1\n")
    assert (
        audit(["--syft-sbom", str(sbom), "--our-csv", str(csv), "--fail-on-findings"])
        == 0
    )
    assert shipped.identity("go", "github.com/a.b/c") != shipped.identity(
        "go", "github-com/a-b/c"
    )


def test_license_review_distinguishes_allowed_from_permissive(tmp_path):
    policy = Path(__file__).resolve().parents[1] / "policy/licenses.toml"
    shipped.write_license_review(
        [
            Component("rust", "copyleft", "1", "MPL-2.0"),
            Component("native", "cbc", "UNKNOWN", "EPL-1.0"),
            Component("go", "permissive", "1", "MIT"),
        ],
        policy,
        tmp_path,
    )
    records = {
        c["name"]: c for c in json.loads((tmp_path / "license-review.json").read_text())
    }
    assert records["copyleft"]["review_terms"] == ["MPL-2.0"]
    assert records["copyleft"]["policy_violation"] is None
    assert records["cbc"]["policy_violation"] is None
    assert records["cbc"]["review_terms"] == ["EPL-1.0"]
    assert "permissive" not in records


def test_go_license_templates_do_not_relicense_module():
    item = artifact(
        "example.com/module",
        "go-module",
        "bin/server",
        licenses=[
            {
                "spdxExpression": "Apache-2.0",
                "contents": "root license",
                "urls": ["https://proxy/module@v1/LICENSE.txt"],
            },
            {
                "spdxExpression": "GPL-3.0-only",
                "contents": "template string",
                "urls": ["https://proxy/module@v1/cmd/license_gpl.go"],
            },
        ],
    )
    assert shipped._licenses(item) == ("(Apache-2.0)", "root license")


def test_local_source_project_owner_cannot_cover_registry_dependencies(
    tmp_path, monkeypatch
):
    lock = "site/source/Cargo.lock"
    project = artifact("source", "rust-crate", lock, metadata={"source": ""})
    dependency = artifact(
        "dependency", "rust-crate", lock, purl="pkg:cargo/dependency@1.0"
    )
    owning = owner("source/Cargo.lock", "parent")
    monkeypatch.setattr(
        shipped,
        "_crate",
        lambda *args: ("EPL-1.0", "separate notice", "https://registry", "checksum"),
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "dependency@1.0.crate").touch()
    components = shipped.supplement(
        {"artifacts": [project, dependency, owning]},
        [],
        tmp_path,
        cache,
        set(),
        tmp_path / "legal",
    )
    result = {c.name: c for c in components}
    assert result["source"].spdx == "(MIT)"
    assert result["dependency"].spdx == "EPL-1.0"


def test_cli_merges_supplement_and_reports_scoped_license_exception(tmp_path):
    site = tmp_path / "site"
    dist = site / "pulp-3.3.2.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        "Name: pulp\nVersion: 3.3.2\nLicense-Expression: MIT\n"
    )
    path = "pulp/solverdir/cbc/win/i64/cbc.exe"
    notice = site / Path(path).parent / "coin-license.txt"
    notice.parent.mkdir(parents=True)
    notice.write_text("Eclipse Public License - v 1.0\nsolver copyright")
    scan = tmp_path / "scan.json"
    scan.write_text(
        json.dumps(
            {
                "artifacts": [
                    owner(path, "pulp"),
                    artifact("cbc", "binary", "site/" + path),
                ]
            }
        )
    )
    output = tmp_path / "legal"
    policy_path = Path(__file__).resolve().parents[1] / "policy/licenses.toml"
    assert (
        generate(
            [
                "--ecosystem",
                "python",
                "--site-packages",
                str(site),
                "--shipped-root",
                str(tmp_path),
                "--shipped-sbom",
                str(scan),
                "--registry-cache",
                str(tmp_path / "cache"),
                "--policy",
                str(policy_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert "EPL-1.0" in (output / "NOTICES-Native.txt").read_text()
    assert "solver copyright" in (output / "NOTICES-Native.txt").read_text()
    assert "cbc" in (output / "osrb-deps.csv").read_text()
    review = json.loads((output / "license-review.json").read_text())
    cbc = next(c for c in review if c["name"] == "cbc")
    assert cbc["policy_violation"] is None
    assert cbc["review_terms"] == ["EPL-1.0"]
    assert "Mere aggregation" in (output / "osrb-deps.csv").read_text()


@pytest.mark.parametrize(
    "framework,target,enabled",
    [
        ("dynamo", "frontend", True),
        ("vllm", "runtime", True),
        ("sglang", "runtime", True),
        ("trtllm", "runtime", True),
        ("dynamo", "planner", False),
        ("triton", "runtime", False),
    ],
)
def test_compliance_template_keeps_shell_valid_and_builder_scans_out(
    framework, target, enabled
):
    template = Path(__file__).resolve().parents[2] / "templates/compliance.Dockerfile"
    rendered = (
        Environment(undefined=StrictUndefined)
        .from_string(template.read_text())
        .render(
            framework=framework,
            target=target,
            device="cuda",
            make_efa=False,
            compliance_base_stage="pre_runtime",
            compliance_baseline_sbom="baseline",
            compliance_ecosystems="python,rust,dpkg,native",
            compliance_source_ecosystem_flags="--ecosystem rust",
        )
    )
    assert ("--shipped-sbom" in rendered) == enabled
    assert ("COPY --from=compliance_cataloger" in rendered) == enabled
    assert ("COPY --from=licenses /sboms/supplemental-sources" in rendered) == enabled
    assert "\\\n\n" not in rendered
    lines = rendered.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("RUN "):
            continue
        script = line[4:]
        while lines[index].endswith("\\"):
            index += 1
            script += "\n" + lines[index]
        result = subprocess.run(
            ["sh", "-n"], input=script, text=True, capture_output=True, timeout=10
        )
        assert result.returncode == 0, result.stderr


def test_local_crate_without_purl_keeps_rust_ecosystem():
    assert component_identity(
        {
            "name": "flaxlib",
            "properties": [{"name": "syft:package:type", "value": "rust-crate"}],
        }
    ) == ("rust", "flaxlib")


@pytest.mark.parametrize(
    "name,declaration,expected",
    [
        ("demo", 'license = "MIT/Apache-2.0"', "MIT OR Apache-2.0"),
        ("demo", 'license = "Apache-2.0/MIT"', "Apache-2.0 OR MIT"),
        ("jsonpath-rust", 'license-file = "LICENSE"', "MIT"),
        ("scoped-env", 'license-file = "LICENSE"', "MIT"),
        ("unreviewed", 'license-file = "LICENSE"', "UNKNOWN"),
    ],
)
def test_registry_license_resolution_uses_cargo_syntax_and_reviewed_overrides(
    tmp_path, name, declaration, expected
):
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w:gz") as archive:
        for filename, text in [
            (
                "Cargo.toml",
                f'[package]\nname = "{name}"\nversion = "1.0"\n{declaration}\n',
            ),
            ("LICENSE", "Upstream license-file notice"),
        ]:
            payload = text.encode()
            entry = tarfile.TarInfo(f"{name}-1.0/{filename}")
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
    payload = body.getvalue()
    (tmp_path / f"{name}@1.0.crate").write_bytes(payload)
    (tmp_path / f"{name}@1.0.json").write_text(
        json.dumps({"vers": "1.0", "cksum": hashlib.sha256(payload).hexdigest()})
    )
    spdx, text, _, _ = shipped._crate(name, "1.0", tmp_path)
    assert spdx == expected
    assert "Upstream license-file notice" in text


def test_embedded_go_main_uses_installed_wheel_metadata(tmp_path):
    directory = tmp_path / "site/mooncake-1.0.dist-info"
    directory.mkdir(parents=True)
    (directory / "METADATA").write_text(
        "Name: mooncake-transfer-engine-cuda13\nVersion: 1.0\n"
    )
    (directory / "LICENSE").write_text("Actual upstream notice")
    wheel = owner("mooncake/libetcd_wrapper.so", "mooncake-transfer-engine-cuda13")
    wheel["licenses"] = []  # Syft did not resolve the wheel license.
    wheel["locations"] = [{"path": "site/mooncake-1.0.dist-info/METADATA"}]
    item = artifact(
        "command-line-arguments", "go-module", "site/mooncake/libetcd_wrapper.so"
    )
    expr, text, _ = shipped._owning_project(item, shipped._owners([wheel]), tmp_path)
    assert expr == "Apache-2.0"
    assert text == "Actual upstream notice"
    # A different path cannot borrow the same wheel's license.
    item["locations"] = [{"path": "site/unrelated.so"}]
    assert (
        shipped._owning_project(item, shipped._owners([wheel]), tmp_path)[0]
        == "UNKNOWN"
    )


@pytest.mark.parametrize(
    "name,version,expected",
    [
        ("flaxlib", "0.0.1-a1", "Apache-2.0"),
        ("ittapi-integration-test", "0.1.0", "BSD-3-Clause"),
    ],
)
def test_local_crate_uses_its_reviewed_license(tmp_path, name, version, expected):
    directory = tmp_path / "site" / name
    directory.mkdir(parents=True)
    (directory / "Cargo.lock").write_text("")
    (directory / "Cargo.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "{version}"\n'
    )
    (directory / "LICENSE").write_text("Flaxlib upstream Apache notice")
    item = artifact(name, "rust-crate", f"site/{name}/Cargo.lock", version)
    result = shipped.supplement(
        {"artifacts": [item]},
        [],
        tmp_path,
        tmp_path / "cache",
        set(),
        tmp_path / "legal",
    )
    assert result[0].spdx == expected
    assert result[0].license_text == "Flaxlib upstream Apache notice"


@pytest.mark.parametrize(
    "metadata_name", ["system.egg-info/PKG-INFO", "system.egg-info"]
)
def test_system_python_uses_installed_metadata(tmp_path, metadata_name):
    metadata = tmp_path / "usr/lib/python3/dist-packages" / metadata_name
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "Name: system\nVersion: 1.0\n"
        "Classifier: License :: OSI Approved :: MIT License\n"
    )
    item = artifact("system", "python", str(metadata.relative_to(tmp_path)))
    assert shipped._python_metadata(item, tmp_path)[0] == "MIT"


def test_system_python_requires_exact_unique_dpkg_owner(tmp_path):
    path = "usr/lib/python3/dist-packages/system.egg-info/PKG-INFO"
    item = artifact("system", "python", path)
    item["locations"][0]["annotations"] = {"evidence": "primary"}
    package = artifact("python3-system", "deb", "var/lib/dpkg/status")
    package["metadata"] = {"files": [{"path": "/" + path}]}
    notice = tmp_path / "usr/share/doc/python3-system/copyright"
    notice.parent.mkdir(parents=True)
    notice.write_text(
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n\nFiles: *\nCopyright: Upstream\nLicense: LGPL-3+\n"
    )
    spdx, text, _ = shipped._system_python(item, [package], tmp_path)
    assert spdx == "LGPL-3.0-or-later"
    assert "Copyright: Upstream" in text
    assert shipped._system_python(item, [], tmp_path)[0] == "UNKNOWN"
    assert (
        shipped._system_python(item, [package, dict(package, name="other")], tmp_path)[
            0
        ]
        == "UNKNOWN"
    )


def test_system_python_policy_exceptions_stay_package_scoped(tmp_path):
    policy = load_policy(Path(__file__).resolve().parents[1] / "policy/licenses.toml")
    for name, expression in [
        ("python-apt", "GPL-2.0-or-later"),
        ("pygobject", "LGPL-2.1-or-later"),
    ]:
        assert validate_row(policy, "python", name, "1.0", expression) is None
        assert (
            validate_row(policy, "python", "unreviewed", "1.0", expression) is not None
        )
    components = [Component("python", "python-apt", "1.0", "GPL-2.0-or-later")]
    shipped.write_license_review(
        components,
        Path(__file__).resolve().parents[1] / "policy/licenses.toml",
        tmp_path,
    )
    assert json.loads((tmp_path / "license-review.json").read_text())[0][
        "review_terms"
    ] == ["GPL-2.0-or-later"]
