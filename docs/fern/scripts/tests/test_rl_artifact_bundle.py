# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for digest closure of RL publication artifact bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rl_artifact_bundle

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

SCRIPTS = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _passing_checkers(monkeypatch: pytest.MonkeyPatch) -> None:
    for schema in rl_artifact_bundle.RECORD_SCHEMAS.values():
        monkeypatch.setitem(
            rl_artifact_bundle.PUBLICATION_CHECKERS, schema, lambda _record: []
        )


def _bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
    _passing_checkers(monkeypatch)
    root = tmp_path / "bundle"
    records = root / "records"
    artifacts = root / "artifacts"
    records.mkdir(parents=True)
    artifacts.mkdir()
    for name in ("framework.log", "program.log", "clean-room.log"):
        (artifacts / name).write_text(f"checked {name}\n", encoding="utf-8")

    framework_path = records / "framework.json"
    program_path = records / "program.json"
    clean_path = records / "clean-room.json"
    _write_json(
        framework_path,
        {
            "schema": rl_artifact_bundle.RECORD_SCHEMAS["framework_validation"],
            "record_state": "passed",
            "artifacts": ["artifact://bundle/artifacts/framework.log"],
        },
    )
    _write_json(
        program_path,
        {
            "schema": rl_artifact_bundle.RECORD_SCHEMAS["program_evidence"],
            "record_state": "passed",
            "artifacts": ["artifact://bundle/artifacts/program.log"],
        },
    )
    _write_json(
        clean_path,
        {
            "schema": rl_artifact_bundle.RECORD_SCHEMAS["clean_room_review"],
            "record_state": "passed",
            "linked_records": {
                "framework_validation": {
                    "uri": "artifact://bundle/records/framework.json",
                    "sha256": rl_artifact_bundle._sha256(framework_path),
                },
                "program_evidence": {
                    "uri": "artifact://bundle/records/program.json",
                    "sha256": rl_artifact_bundle._sha256(program_path),
                },
            },
            "artifacts": ["artifact://bundle/artifacts/clean-room.log"],
        },
    )
    return root, [framework_path, program_path, clean_path]


def test_complete_bundle_builds_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    index, findings = rl_artifact_bundle.build_index(root, records)
    assert findings == []
    assert index["closure_status"] == "complete"
    assert len(index["records"]) == 3
    assert len(index["artifacts"]) == 5
    assert index["referenced_uri_count"] == 5
    index_path = root / "bundle-index.json"
    _write_json(index_path, index)
    assert (
        rl_artifact_bundle.verify_index(
            root, index_path, rl_artifact_bundle._sha256(index_path)
        )
        == []
    )


def test_nested_environment_and_hardware_artifacts_are_digest_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    environment = root / "artifacts" / "environment-preflight.json"
    allocation = root / "artifacts" / "scheduler-allocation.json"
    _write_json(environment, {"schema": "dynamo.rl.environment-preflight.v1"})
    _write_json(allocation, {"visible_gpus": 8})
    framework = json.loads(records[0].read_text(encoding="utf-8"))
    framework["environment"] = {
        "artifacts": ["artifact://bundle/artifacts/environment-preflight.json"]
    }
    framework["hardware"] = {
        "artifacts": [
            "artifact://bundle/artifacts/environment-preflight.json",
            "artifact://bundle/artifacts/scheduler-allocation.json",
        ]
    }
    _write_json(records[0], framework)
    clean_room = json.loads(records[2].read_text(encoding="utf-8"))
    clean_room["linked_records"]["framework_validation"]["sha256"] = (
        rl_artifact_bundle._sha256(records[0])
    )
    _write_json(records[2], clean_room)
    index, findings = rl_artifact_bundle.build_index(root, records)
    assert findings == []
    assert index["closure_status"] == "complete"
    assert {
        "artifact://bundle/artifacts/environment-preflight.json",
        "artifact://bundle/artifacts/scheduler-allocation.json",
    }.issubset({entry["uri"] for entry in index["artifacts"]})
    assert len(index["artifacts"]) == 7


def test_cli_build_and_verify_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    build_args = ["build", "--artifact-root", str(root)]
    for record in records:
        build_args.extend(["--record", str(record)])
    build_args.extend(["--index-json", "bundle-index.json", "--strict"])
    assert rl_artifact_bundle.main(build_args) == 0
    index_sha256 = rl_artifact_bundle._sha256(root / "bundle-index.json")
    assert (
        rl_artifact_bundle.main(
            [
                "verify",
                "--artifact-root",
                str(root),
                "--index-json",
                "bundle-index.json",
                "--expected-index-sha256",
                index_sha256,
            ]
        )
        == 0
    )


def test_artifact_mutation_breaks_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    index, findings = rl_artifact_bundle.build_index(root, records)
    assert findings == []
    index_path = root / "bundle-index.json"
    _write_json(index_path, index)
    (root / "artifacts" / "framework.log").write_text(
        "mutated after review\n", encoding="utf-8"
    )
    findings = rl_artifact_bundle.verify_index(
        root, index_path, rl_artifact_bundle._sha256(index_path)
    )
    assert any(
        "artifact artifact://bundle/artifacts/framework.log sha256 does not match"
        in finding
        for finding in findings
    )


def test_index_requires_an_external_digest_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    index, findings = rl_artifact_bundle.build_index(root, records)
    assert findings == []
    index_path = root / "bundle-index.json"
    _write_json(index_path, index)
    original_digest = rl_artifact_bundle._sha256(index_path)
    index["referenced_uri_count"] = 4
    _write_json(index_path, index)
    assert rl_artifact_bundle.verify_index(root, index_path, original_digest) == [
        "index SHA-256 does not match the externally anchored digest"
    ]


def test_missing_artifact_is_incomplete_and_fails_strict_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    (root / "artifacts" / "program.log").unlink()
    index, findings = rl_artifact_bundle.build_index(root, records)
    assert index["closure_status"] == "incomplete"
    assert index["missing_uris"] == ["artifact://bundle/artifacts/program.log"]
    assert any("missing local artifacts" in finding for finding in findings)
    args = ["build", "--artifact-root", str(root)]
    for record in records:
        args.extend(["--record", str(record)])
    args.extend(["--index-json", "bundle-index.json", "--strict"])
    assert rl_artifact_bundle.main(args) == 1


def test_external_artifact_uri_is_not_claimed_as_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    framework = json.loads(records[0].read_text(encoding="utf-8"))
    framework["artifacts"].extend(
        ["artifact://external-store/run.log", "https://logs.example/run.json"]
    )
    _write_json(records[0], framework)
    clean = json.loads(records[2].read_text(encoding="utf-8"))
    clean["linked_records"]["framework_validation"]["sha256"] = (
        rl_artifact_bundle._sha256(records[0])
    )
    _write_json(records[2], clean)
    index, findings = rl_artifact_bundle.build_index(root, records)
    assert index["external_uris"] == [
        "artifact://external-store/run.log",
        "https://logs.example/run.json",
    ]
    assert any("external artifact URIs are not closed" in finding for finding in findings)


def test_noncanonical_or_traversing_uri_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    framework = json.loads(records[0].read_text(encoding="utf-8"))
    framework["artifacts"] = ["artifact://bundle/artifacts/%2E%2E/secret.log"]
    _write_json(records[0], framework)
    with pytest.raises(rl_artifact_bundle.BundleError, match="unsafe path"):
        rl_artifact_bundle.build_index(root, records)


def test_record_outside_root_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    outside = tmp_path / "outside.json"
    _write_json(outside, {"schema": "dynamo.rl.validation.v1"})
    with pytest.raises(rl_artifact_bundle.BundleError, match="escapes artifact root"):
        rl_artifact_bundle.build_index(root, [*records, outside])


def test_symlinked_record_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    alias = root / "records" / "framework-alias.json"
    alias.symlink_to(records[0])
    with pytest.raises(rl_artifact_bundle.BundleError, match="symbolic link"):
        rl_artifact_bundle.build_index(root, [alias, *records[1:]])


def test_symlinked_artifact_does_not_close_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    target = root / "artifacts" / "framework-target.log"
    target.write_text("target\n", encoding="utf-8")
    (root / "artifacts" / "framework.log").unlink()
    (root / "artifacts" / "framework.log").symlink_to(target)
    index, findings = rl_artifact_bundle.build_index(root, records)
    assert index["missing_uris"] == ["artifact://bundle/artifacts/framework.log"]
    assert any("missing local artifacts" in finding for finding in findings)


def test_clean_room_link_digest_must_match_bundled_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    clean = json.loads(records[2].read_text(encoding="utf-8"))
    clean["linked_records"]["program_evidence"]["sha256"] = "0" * 64
    _write_json(records[2], clean)
    _, findings = rl_artifact_bundle.build_index(root, records)
    assert (
        "clean-room linked_records.program_evidence.sha256 does not match the bundled record"
        in findings
    )


def test_real_planned_templates_cannot_form_a_complete_bundle(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    records_dir = root / "records"
    records_dir.mkdir(parents=True)
    sources = [
        SCRIPTS / "rl_validation_record.template.json",
        SCRIPTS / "rl_program_record.template.json",
        SCRIPTS / "rl_clean_room_record.template.json",
    ]
    records: list[Path] = []
    for source in sources:
        target = records_dir / source.name
        target.write_bytes(source.read_bytes())
        records.append(target)
    index, findings = rl_artifact_bundle.build_index(root, records)
    assert index["closure_status"] == "incomplete"
    assert all(not entry["publication_gate_passed"] for entry in index["records"])
    assert any("record_state must be passed for publication" in item for item in findings)
    assert "records do not reference any artifact URIs" in findings


def test_index_output_must_remain_inside_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, records = _bundle(tmp_path, monkeypatch)
    args = ["build", "--artifact-root", str(root)]
    for record in records:
        args.extend(["--record", str(record)])
    args.extend(["--index-json", str(tmp_path / "outside-index.json"), "--strict"])
    assert rl_artifact_bundle.main(args) == 2
