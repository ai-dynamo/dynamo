# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise exact lookup, strict catalogs, and applications versus observed packages."""

import json

import pytest
from compliance import overrides
from compliance.generators import python as python_generator
from compliance.generators import rust
from compliance.generators.common import Component

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    path = tmp_path / "overrides.yaml"
    path.write_text(
        'overrides:\n- {ecosystem: python, name: pkg, license: MIT, source: upstream}\n- {ecosystem: python, name: pkg, version: "2", license: Apache-2.0, source: upstream}\n- {ecosystem: rust, name: crate, license: MIT, source: upstream}\n'
    )
    monkeypatch.setattr(overrides, "_DEFAULT_OVERRIDES_PATH", path)
    overrides.reset_caches()
    yield path
    overrides.reset_caches()


def test_exact_version_wins_and_legacy_fallback_is_visible(catalog, tmp_path):
    assert overrides.lookup("python", "pkg", "2", raw_license="UNKNOWN") == "Apache-2.0"
    assert overrides.lookup("python", "pkg", "1", raw_license="UNKNOWN") == "MIT"
    output = tmp_path / "usage.json"
    overrides.write_usage(output, ["python"])
    rows = json.loads(output.read_text())["observations"]
    assert rows[0]["legacy_unversioned"] is True
    assert rows[1]["override_version"] == "2"
    assert all(row["applied"] for row in rows)


@pytest.mark.parametrize(
    "content",
    [
        "overrides:\n- {ecosystem: python, name: pkg, license: MIT, license: Apache-2.0, source: upstream}\n",
        "overrides:\n- {ecosystem: python, name: pkg, license: MIT, source: upstream}\n- {ecosystem: python, name: pkg, license: MIT, source: upstream}\n",
        "overrides:\n- {ecosystem: invalid, name: pkg, license: MIT, source: upstream}\n",
        "overrides:\n- {ecosystem: python, name: pkg, license: MIT}\n",
        'overrides:\n- {ecosystem: python, name: pkg, version: ">=1", license: MIT, source: upstream}\n',
    ],
)
def test_catalog_rejects_duplicates_schema_and_nonexact_versions(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        overrides.load_records(path)


def test_known_metadata_does_not_count_as_an_override_application(catalog, tmp_path):
    site = tmp_path / "site-packages"
    metadata = site / "pkg-1.dist-info/METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: pkg\nVersion: 1\nLicense-Expression: Apache-2.0\n")
    assert python_generator.collect_components([site])[0].spdx == "Apache-2.0"
    rust._component_from_sbom_entry(
        {"name": "crate", "version": "1", "licenses": [{"license": {"id": "MIT"}}]}
    )
    rust._component_from_sbom_entry({"name": "crate", "version": "2", "licenses": []})
    output = tmp_path / "usage.json"
    overrides.write_usage(output, ["python", "rust"])
    rows = {
        (r["ecosystem"], r["name"], r["version"]): r
        for r in json.loads(output.read_text())["observations"]
    }
    assert rows[("python", "pkg", "1")]["status"] == "matched_unused"
    assert rows[("rust", "crate", "1")]["applied"] is False
    assert rows[("rust", "crate", "2")]["applied"] is True


def test_associated_base_is_bundled_with_exact_hash(catalog, tmp_path):
    base = tmp_path / "base.cdx.json"
    base.write_text('{"components": []}')
    output = tmp_path / "usage.json"
    overrides.write_usage(output, [], base)
    report = json.loads(output.read_text())
    assert (tmp_path / report["baseline"]["file"]).read_bytes() == base.read_bytes()
    assert len(report["baseline"]["sha256"]) == 64


def test_supplemental_components_are_covered_without_erasing_lookup_evidence(
    catalog, tmp_path
):
    assert overrides.lookup("python", "pkg", "1", raw_license="UNKNOWN") == "MIT"
    components = [
        Component("python", "pkg", "1", "MIT"),
        Component("native", "bundled-extra", "2", "BSD-3-Clause"),
    ]
    overrides.ensure_observed(components)
    output = tmp_path / "usage.json"
    overrides.write_usage(output, ["python", "native"])
    rows = {
        (r["ecosystem"], r["name"]): r
        for r in json.loads(output.read_text())["observations"]
    }
    assert rows[("python", "pkg")]["applied"] is True
    assert rows[("python", "pkg")]["raw_license"] == "UNKNOWN"
    assert rows[("native", "bundled-extra")]["raw_license"] is None
    assert rows[("native", "bundled-extra")]["resolved_license"] == "BSD-3-Clause"
