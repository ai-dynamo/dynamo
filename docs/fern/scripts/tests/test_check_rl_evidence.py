# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the RL evidence and release audit."""

from __future__ import annotations

import base64
import copy
import subprocess
from datetime import date
from pathlib import Path

import check_rl_evidence
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def test_repository_manifest_matches_current_sources_and_docs() -> None:
    manifest = check_rl_evidence.load_manifest(check_rl_evidence.DEFAULT_MANIFEST)
    assert check_rl_evidence.check_offline(manifest, check_rl_evidence.REPO_ROOT) == []


def test_missing_source_contract_is_actionable() -> None:
    manifest = copy.deepcopy(
        check_rl_evidence.load_manifest(check_rl_evidence.DEFAULT_MANIFEST)
    )
    claim = next(item for item in manifest["claims"] if item["id"] == "RL-CONTRACT-005")
    claim["sources"][0]["contains"].append("route-that-does-not-exist")
    findings = check_rl_evidence.check_offline(manifest, check_rl_evidence.REPO_ROOT)
    assert any(
        "RL-CONTRACT-005" in finding and "route-that-does-not-exist" in finding
        for finding in findings
    )


def test_every_ledger_row_requires_one_machine_record() -> None:
    manifest = copy.deepcopy(
        check_rl_evidence.load_manifest(check_rl_evidence.DEFAULT_MANIFEST)
    )
    removed = manifest["claims"].pop()
    findings = check_rl_evidence.check_offline(manifest, check_rl_evidence.REPO_ROOT)
    assert any(
        f"ledger records without audit rules: {removed['id']}" in finding
        for finding in findings
    )


def test_every_non_diagram_fence_requires_a_snippet_record() -> None:
    manifest = copy.deepcopy(
        check_rl_evidence.load_manifest(check_rl_evidence.DEFAULT_MANIFEST)
    )
    manifest["snippet_coverage"]["pages"][0]["blocks"].pop()
    findings = check_rl_evidence.check_offline(manifest, check_rl_evidence.REPO_ROOT)
    assert any("has no snippet evidence record" in finding for finding in findings)


def test_snippet_content_drift_requires_evidence_review() -> None:
    manifest = copy.deepcopy(
        check_rl_evidence.load_manifest(check_rl_evidence.DEFAULT_MANIFEST)
    )
    manifest["snippet_coverage"]["pages"][0]["blocks"][0]["sha256"] = "0" * 64
    findings = check_rl_evidence.check_offline(manifest, check_rl_evidence.REPO_ROOT)
    assert any(
        "content changed; review its command/config evidence" in finding
        for finding in findings
    )


def test_snippet_records_require_known_evidence_owner_and_expiration() -> None:
    manifest = copy.deepcopy(
        check_rl_evidence.load_manifest(check_rl_evidence.DEFAULT_MANIFEST)
    )
    page = manifest["snippet_coverage"]["pages"][0]
    page["owner"] = ""
    page["expiration_trigger"] = ""
    page["blocks"][0]["evidence_ids"] = ["RL-UNKNOWN-001"]
    findings = check_rl_evidence.check_offline(manifest, check_rl_evidence.REPO_ROOT)
    assert any(".owner must be a non-empty string" in finding for finding in findings)
    assert any(
        ".expiration_trigger must be a non-empty string" in finding
        for finding in findings
    )
    assert any(
        "references unknown evidence ID RL-UNKNOWN-001" in finding
        for finding in findings
    )


def test_snippet_inventory_handles_long_backtick_and_tilde_fences() -> None:
    text = '````bash\necho checked\n````\n\n~~~json\n{"checked": true}\n~~~\n'
    blocks = check_rl_evidence._snippet_blocks(text, set())
    assert [language for language, _digest in blocks] == ["bash", "json"]
    assert all(
        check_rl_evidence.SHA256.fullmatch(digest) for _language, digest in blocks
    )


def test_inline_contract_tokens_require_evidence_records() -> None:
    manifest = copy.deepcopy(
        check_rl_evidence.load_manifest(check_rl_evidence.DEFAULT_MANIFEST)
    )
    manifest["contract_surface"]["records"][0]["tokens"].pop()
    findings = check_rl_evidence.check_offline(manifest, check_rl_evidence.REPO_ROOT)
    assert any(
        "unrecorded environment contract tokens" in finding for finding in findings
    )


def test_contract_records_require_known_evidence_owner_and_expiration() -> None:
    manifest = copy.deepcopy(
        check_rl_evidence.load_manifest(check_rl_evidence.DEFAULT_MANIFEST)
    )
    record = manifest["contract_surface"]["records"][0]
    record["evidence_ids"] = ["RL-UNKNOWN-002"]
    record["owner"] = ""
    record["expiration_trigger"] = ""
    findings = check_rl_evidence.check_offline(manifest, check_rl_evidence.REPO_ROOT)
    assert any(
        "references unknown evidence ID RL-UNKNOWN-002" in finding
        for finding in findings
    )
    assert any(".owner must be a non-empty string" in finding for finding in findings)
    assert any(
        ".expiration_trigger must be a non-empty string" in finding
        for finding in findings
    )


def test_contract_token_extraction_normalizes_headers_and_url_ports() -> None:
    tokens = check_rl_evidence._contract_tokens(
        "X-Dynamo-Session-ID x-dynamo-session-id http://localhost:8001/v1/rl/workers"
    )
    assert tokens["header"] == {"x-dynamo-session-id"}
    assert tokens["port"] == {"8001"}
    assert tokens["route"] == {"/v1/rl/workers"}


def test_review_age_can_be_enforced_without_using_wall_clock() -> None:
    manifest = check_rl_evidence.load_manifest(check_rl_evidence.DEFAULT_MANIFEST)
    findings = check_rl_evidence.check_offline(
        manifest,
        check_rl_evidence.REPO_ROOT,
        max_age_days=2,
        today=date(2026, 8, 30),
    )
    assert any("evidence review is 3 days old" in finding for finding in findings)


def test_release_drift_reports_only_watched_changes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "config", "user.email", "rl-audit@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "RL audit test"], cwd=tmp_path, check=True
    )
    watched = tmp_path / "watched.txt"
    watched.write_text("reviewed\n", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("reviewed\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    baseline = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()

    (tmp_path / "unrelated.txt").write_text("changed\n", encoding="utf-8")
    changed, error = check_rl_evidence._git_changed_paths(
        tmp_path, baseline, ["watched.txt"]
    )
    assert error is None
    assert changed == []

    watched.write_text("changed\n", encoding="utf-8")
    changed, error = check_rl_evidence._git_changed_paths(
        tmp_path, baseline, ["watched.txt"]
    )
    assert error is None
    assert changed == ["watched.txt"]


def test_online_audit_accepts_recorded_branch_and_pull_state() -> None:
    manifest = {
        "github": {
            "branches": [
                {
                    "id": "RL-FW-BRANCH-001",
                    "repo": "example/framework",
                    "ref": "main",
                    "expected_sha": "a" * 40,
                    "files": [
                        {
                            "path": "recipe/config.yaml",
                            "contains": ["router_mode: kv"],
                        }
                    ],
                }
            ],
            "pulls": [
                {
                    "id": "RL-FW-PR-001",
                    "repo": "example/framework",
                    "number": 7,
                    "expected_state": "open",
                    "expected_draft": True,
                    "expected_merged": False,
                    "expected_head_prefix": "b17ceea",
                }
            ],
        }
    }

    def fetch(path: str) -> dict:
        if path.endswith("/commits/main"):
            return {"sha": "a" * 40}
        if "/contents/recipe/config.yaml" in path:
            return {
                "type": "file",
                "encoding": "base64",
                "content": base64.b64encode(b"router_mode: kv\n").decode(),
            }
        return {
            "state": "open",
            "draft": True,
            "merged_at": None,
            "head": {"sha": "b17ceea" + "0" * 33},
        }

    assert check_rl_evidence.check_online(manifest, fetch) == []


def test_online_audit_reports_pinned_framework_file_drift() -> None:
    manifest = {
        "github": {
            "branches": [
                {
                    "id": "RL-FW-BRANCH-001",
                    "repo": "example/framework",
                    "ref": "main",
                    "expected_sha": "a" * 40,
                    "files": [
                        {
                            "path": "recipe/config.yaml",
                            "contains": ["router_mode: kv"],
                        }
                    ],
                }
            ],
            "pulls": [],
        }
    }

    def fetch(path: str) -> dict:
        if path.endswith("/commits/main"):
            return {"sha": "a" * 40}
        return {
            "type": "file",
            "encoding": "base64",
            "content": base64.b64encode(b"router_policy: round_robin\n").decode(),
        }

    findings = check_rl_evidence.check_online(manifest, fetch)
    assert any(
        "recipe/config.yaml no longer contains 'router_mode: kv'" in finding
        for finding in findings
    )


def test_online_audit_reports_state_and_head_drift() -> None:
    manifest = {
        "github": {
            "branches": [],
            "pulls": [
                {
                    "id": "RL-FW-PR-001",
                    "repo": "example/framework",
                    "number": 7,
                    "expected_state": "open",
                    "expected_draft": False,
                    "expected_merged": False,
                    "expected_head_prefix": "oldhead",
                }
            ],
        }
    }

    def fetch(_path: str) -> dict:
        return {
            "state": "closed",
            "draft": False,
            "merged_at": "2026-08-28T00:00:00Z",
            "head": {"sha": "newhead"},
        }

    findings = check_rl_evidence.check_online(manifest, fetch)
    assert any(
        "state is 'closed', recorded as 'open'" in finding for finding in findings
    )
    assert any("merged is True, recorded as False" in finding for finding in findings)
    assert any("head moved from oldhead to newhead" in finding for finding in findings)
