# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reviewed binary-license corrections are conditional on the actual shipped files."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from compliance import overrides
from compliance.generators import dpkg
from compliance.policy.review import classify
from compliance.policy.validate import load_policy, validate_row
from compliance.trim_packaging_docs import trim_reviewed_docs

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]
ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "container/compliance/license_overrides.yaml"
EVIDENCE = ROOT / "container/compliance/evidence/dpkg-license-corrections.json"


@pytest.fixture
def prepared(tmp_path):
    notice = b"reviewed copyright\n"
    document = b"reviewed changelog\n"
    doc_path = "/usr/share/doc/example/changelog.gz"
    copyright_path = tmp_path / "usr/share/doc/example/copyright"
    copyright_path.parent.mkdir(parents=True)
    copyright_path.write_bytes(notice)
    changelog = tmp_path / doc_path.lstrip("/")
    changelog.write_bytes(document)
    row = {
        "ecosystem": "dpkg",
        "name": "example",
        "version": "1",
        "license": "MIT",
        "source": "fixture",
        "copyright_sha256": hashlib.sha256(notice).hexdigest(),
        "absent_paths": [doc_path],
    }
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(yaml.safe_dump({"overrides": [row]}))
    review = {
        "name": "example",
        "version": "1",
        "license": "MIT",
        "copyright_sha256": row["copyright_sha256"],
        "absent_paths": [doc_path],
        "binary_packages": [
            {
                "removed_document": {
                    "path": doc_path,
                    "sha256": hashlib.sha256(document).hexdigest(),
                }
            }
        ],
    }
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"schema_version": 1, "packages": [review]}))
    return copyright_path, changelog, catalog, evidence, review


def test_removes_only_reviewed_doc_after_hash_check(tmp_path, prepared):
    copyright_path, changelog, catalog, evidence, _ = prepared
    license_path = copyright_path.with_name("LICENSE")
    license_path.write_text("preserve upstream license")
    assert trim_reviewed_docs(tmp_path, {"example": "1"}, catalog, evidence) == [
        "/usr/share/doc/example/changelog.gz"
    ]
    assert not changelog.exists()
    assert copyright_path.read_bytes() == b"reviewed copyright\n"
    assert license_path.read_text() == "preserve upstream license"
    assert trim_reviewed_docs(tmp_path, {"example": "1"}, catalog, evidence) == []


def test_unknown_version_is_untouched(tmp_path, prepared):
    _, changelog, catalog, evidence, _ = prepared
    assert trim_reviewed_docs(tmp_path, {"example": "2"}, catalog, evidence) == []
    assert changelog.exists()


@pytest.mark.parametrize("target", ["copyright", "changelog"])
def test_changed_bytes_abort_before_deletion(tmp_path, prepared, target):
    copyright_path, changelog, catalog, evidence, _ = prepared
    (copyright_path if target == "copyright" else changelog).write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch|Unreviewed"):
        trim_reviewed_docs(tmp_path, {"example": "1"}, catalog, evidence)
    assert changelog.exists()


def test_reviewed_symlink_is_unlinked_without_touching_target(tmp_path, prepared):
    _, changelog, catalog, evidence, review = prepared
    target = changelog.parent.parent / "dependency/changelog.gz"
    target.parent.mkdir()
    target.write_text("dependency documentation stays")
    changelog.unlink()
    changelog.symlink_to("../dependency/changelog.gz")
    review["binary_packages"][0]["removed_document"] = {
        "path": "/usr/share/doc/example/changelog.gz",
        "symlink_target": "../dependency/changelog.gz",
    }
    evidence.write_text(json.dumps({"schema_version": 1, "packages": [review]}))
    trim_reviewed_docs(tmp_path, {"example": "1"}, catalog, evidence)
    assert not changelog.is_symlink()
    assert target.read_text() == "dependency documentation stays"


def test_symlinked_package_directory_is_rejected(tmp_path, prepared):
    copyright_path, changelog, catalog, evidence, _ = prepared
    parent = copyright_path.parent
    moved = parent.with_name("real-package")
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked"):
        trim_reviewed_docs(tmp_path, {"example": "1"}, catalog, evidence)
    assert (moved / changelog.name).exists()


def test_exact_committed_payload_evidence_matches_all_six_corrections(tmp_path):
    evidence = json.loads(EVIDENCE.read_text())
    records = {(row.name, row.version): row for row in overrides.load_records(CATALOG)}
    policy = load_policy(ROOT / "container/compliance/policy/licenses.toml")
    for entry in evidence["packages"]:
        record = records[(entry["name"], entry["version"])]
        assert record.license == entry["license"]
        assert (
            record.copyright_sha256
            == hashlib.sha256(entry["copyright_text"].encode()).hexdigest()
        )
        assert set(record.absent_paths) == set(entry["absent_paths"])
        assert {p["architecture"] for p in entry["binary_packages"]} == {
            "amd64",
            "arm64",
        }
        path = tmp_path / f"usr/share/doc/{entry['name']}/copyright"
        path.parent.mkdir(parents=True)
        path.write_text(entry["copyright_text"])
        resolved = dpkg._resolve_license(entry["name"], entry["version"], tmp_path)
        assert resolved == record.license
        assert (
            validate_row(policy, "dpkg", entry["name"], entry["version"], resolved)
            is None
        )
        assert " OR " in resolved
        if entry["name"] == "libgraphite2-3":
            assert classify(resolved) == "reciprocal"
