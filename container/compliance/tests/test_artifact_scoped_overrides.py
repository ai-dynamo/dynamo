# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only reviewed artifact bytes may activate a version-scoped license correction."""

import hashlib

import pytest
import yaml
from compliance import overrides
from compliance.base_sboms import capture_baseline_sbom
from compliance.generators import dpkg
from compliance.policy.review import review_overrides

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]
COPYRIGHT = b"Format: https://example.test/dep5\n\nFiles: *\nLicense: GPL-3\n"


@pytest.fixture
def scoped_catalog(tmp_path, monkeypatch):
    path = tmp_path / "overrides.yaml"
    row = {
        "ecosystem": "dpkg",
        "name": "example",
        "version": "1",
        "license": "MIT",
        "source": "fixture",
        "usage": "fixture payload",
        "evidence": "fixture evidence",
        "reason": "fixture correction",
        "copyright_sha256": hashlib.sha256(COPYRIGHT).hexdigest(),
    }
    path.write_text(yaml.safe_dump({"overrides": [row]}))
    monkeypatch.setattr(overrides, "_DEFAULT_OVERRIDES_PATH", path)
    monkeypatch.setattr(
        capture_baseline_sbom.license_overrides, "_DEFAULT_OVERRIDES_PATH", path
    )
    overrides.reset_caches()
    capture_baseline_sbom.license_overrides.reset_caches()
    yield path, row
    overrides.reset_caches()
    capture_baseline_sbom.license_overrides.reset_caches()


def test_exact_copyright_bytes_activate_correction(tmp_path, scoped_catalog):
    copyright_path = tmp_path / "usr/share/doc/example/copyright"
    copyright_path.parent.mkdir(parents=True)
    copyright_path.write_bytes(COPYRIGHT)
    assert dpkg._resolve_license("example", "1", tmp_path) == "MIT"
    assert dpkg._resolve_license("example", "2", tmp_path) == "GPL-3.0-only"
    record = overrides._observations[("dpkg", "example", "1")]
    assert record["raw_license"] == "GPL-3.0-only"
    assert record["copyright_sha256"] == hashlib.sha256(COPYRIGHT).hexdigest()


def test_mismatch_or_missing_bytes_cannot_use_broader_fallback(
    tmp_path, scoped_catalog
):
    path, row = scoped_catalog
    legacy = {
        "ecosystem": "dpkg",
        "name": "example",
        "license": "Apache-2.0",
        "source": "legacy",
    }
    path.write_text(yaml.safe_dump({"overrides": [row, legacy]}))
    overrides.reset_caches()
    with pytest.raises(overrides.OverrideEvidenceUnavailable):
        overrides.lookup("dpkg", "example", "1")
    with pytest.raises(overrides.OverrideEvidenceUnavailable):
        overrides.lookup("dpkg", "example", "1", copyright_sha256="0" * 64)
    copyright_path = tmp_path / "usr/share/doc/example/copyright"
    copyright_path.parent.mkdir(parents=True)
    copyright_path.write_bytes(COPYRIGHT + b"changed")
    with pytest.raises(overrides.OverrideEvidenceMismatch):
        overrides.lookup("dpkg", "example", "1", artifact_root=tmp_path)
    copyright_path.unlink()
    with pytest.raises(overrides.OverrideEvidenceUnavailable):
        dpkg._resolve_license("example", "1", tmp_path)


def test_baseline_capture_without_files_validates_raw_scanner_license(
    tmp_path, scoped_catalog, caplog
):
    policy = tmp_path / "licenses.toml"
    policy.write_text(
        '[licenses]\nallow = ["MIT"]\ndeny = ["GPL-3.0-only"]\nunknown = "deny"\ncopyleft = "deny"\n'
    )
    component = {
        "name": "example",
        "version": "1",
        "purl": "pkg:deb/example@1",
        "licenses": [{"license": {"id": "GPL-3.0-only"}}],
    }
    count, reasons = capture_baseline_sbom.validate_delta([component], policy)
    assert count == 1 and "GPL-3.0-only" in reasons[0]
    assert "Copyright evidence unavailable" in caplog.text
    assert overrides._observations == {}


def test_non_dpkg_callers_cannot_accidentally_apply_dpkg_correction(scoped_catalog):
    assert overrides.lookup("rust", "example", "1") is None
    assert overrides.lookup("python", "example", "1") is None


def test_new_dpkg_corrections_require_hash_and_cannot_remove_predicate(
    tmp_path, scoped_catalog
):
    current, row = scoped_catalog
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"overrides": []}))
    assert review_overrides(current, base)["errors"] == []
    base.write_text(current.read_text())
    row.pop("copyright_sha256")
    current.write_text(yaml.safe_dump({"overrides": [row]}))
    assert (
        "predicate removal broadens scope"
        in review_overrides(current, base)["errors"][0]
    )


@pytest.mark.parametrize(
    "location", ["usr", "usr/share/doc/example", "usr/share/doc/example/copyright"]
)
def test_scoped_lookup_never_follows_artifact_symlinks(
    tmp_path, scoped_catalog, location
):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "copyright").write_bytes(COPYRIGHT)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    candidate = artifact / location
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.symlink_to(
        outside / "copyright" if location.endswith("/copyright") else outside
    )
    with pytest.raises(overrides.OverrideEvidenceMismatch, match="Symlinked"):
        dpkg._resolve_license("example", "1", artifact)


def test_dangling_changelog_symlink_counts_as_present(tmp_path, scoped_catalog):
    path, row = scoped_catalog
    row["absent_paths"] = ["/usr/share/doc/example/changelog.gz"]
    path.write_text(yaml.safe_dump({"overrides": [row]}))
    overrides.reset_caches()
    notice = tmp_path / "usr/share/doc/example/copyright"
    notice.parent.mkdir(parents=True)
    notice.write_bytes(COPYRIGHT)
    notice.with_name("changelog.gz").symlink_to("missing")
    with pytest.raises(overrides.OverrideEvidenceMismatch, match="still ships"):
        dpkg._resolve_license("example", "1", tmp_path)


def test_removing_absence_predicate_is_scope_expansion(tmp_path, scoped_catalog):
    current, row = scoped_catalog
    row["absent_paths"] = ["/usr/share/doc/example/changelog.gz"]
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"overrides": [row]}))
    row.pop("absent_paths")
    current.write_text(yaml.safe_dump({"overrides": [row]}))
    assert "preserved absent_paths" in review_overrides(current, base)["errors"][0]


def test_deleting_scoped_record_cannot_expose_versionless_fallback(
    tmp_path, scoped_catalog
):
    current, row = scoped_catalog
    legacy = {
        "ecosystem": "dpkg",
        "name": "example",
        "license": "MIT",
        "source": "legacy",
    }
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"overrides": [row, legacy]}))
    current.write_text(yaml.safe_dump({"overrides": [legacy]}))
    assert (
        "broader versionless fallback" in review_overrides(current, base)["errors"][0]
    )
