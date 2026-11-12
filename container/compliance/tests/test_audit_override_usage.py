# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Missing or mismatched evidence cannot authorize override cleanup."""

import base64
import hashlib
import io
import json
import subprocess
from zipfile import ZipFile

import pytest
from compliance import audit_override_usage
from compliance.audit_override_usage import inspect_archive, summarize
from compliance.overrides import Override

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]
SHA = "a" * 40
IMAGE = {"prefix": "vllm-runtime", "cuda": "13.0", "architectures": ["amd64"]}
CSV = "ecosystem,name,version,spdx,source_url,notes\npython,pkg,1,MIT,,\n"


def observation(**kwargs):
    return {
        "ecosystem": "python",
        "name": "pkg",
        "version": "1",
        "raw_license": "UNKNOWN",
        "resolved_license": "MIT",
        "override_version": None,
        "override_license": "MIT",
        "applied": True,
        **kwargs,
    }


def artifact(
    *,
    telemetry=True,
    catalog="catalog",
    baseline=False,
    corrupt_base=False,
    unsafe=False,
):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr(f"linux_amd64/osrb-vllm-runtime-13.0-amd64-{SHA[:8]}.csv", CSV)
        if unsafe:
            archive.writestr("../bad", "x")
        if telemetry:
            base_bytes = b'{"components":[{"name":"basepkg","version":"1","purl":"pkg:deb/ubuntu/basepkg@1"}]}'
            base = (
                {
                    "file": "override-base.cdx.json",
                    "sha256": hashlib.sha256(base_bytes).hexdigest(),
                }
                if baseline
                else None
            )
            archive.writestr(
                "linux_amd64/override-usage.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "override_catalog_sha256": catalog,
                        "observations": [observation()],
                        "baseline": base,
                    }
                ),
            )
            if baseline:
                archive.writestr(
                    "linux_amd64/override-base.cdx.json",
                    b"tampered" if corrupt_base else base_bytes,
                )
    return stream.getvalue()


def test_complete_coverage_combines_actual_use_and_associated_base():
    obs, inventory, problems = inspect_archive(
        artifact(baseline=True), IMAGE, SHA, "catalog"
    )
    assert problems == []
    assert any(r["name"] == "basepkg" for r in inventory)
    report = summarize(
        (Override("python", "pkg", "MIT", "upstream"),), obs, inventory, problems
    )
    assert report["coverage_complete"] is True
    assert report["overrides"][0]["status"] == "applied"
    assert report["automatic_deletion"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"telemetry": False},
        {"catalog": "wrong-source"},
        {"baseline": True, "corrupt_base": True},
    ],
)
def test_missing_telemetry_wrong_source_or_base_hash_prevents_cleanup(kwargs):
    obs, inventory, problems = inspect_archive(
        artifact(**kwargs), IMAGE, SHA, "catalog"
    )
    report = summarize(
        (Override("python", "unused", "MIT", "upstream"),), obs, inventory, problems
    )
    assert report["coverage_complete"] is False
    assert report["overrides"][0]["cleanup_review_candidate"] is False


def test_unsafe_archive_is_rejected_without_extracting():
    with pytest.raises(ValueError, match="unsafe"):
        inspect_archive(artifact(unsafe=True), IMAGE, SHA, "catalog")


def test_required_arm64_cannot_be_replaced_by_amd64():
    image = {**IMAGE, "architectures": ["amd64", "arm64"]}
    _, _, problems = inspect_archive(artifact(), image, SHA, "catalog")
    assert any("arm64" in problem for problem in problems)


def test_inventory_match_is_not_application_or_proof_of_unused_override():
    record = Override("python", "pkg", "MIT", "upstream")
    report = summarize((record,), [], [observation()], [])
    assert report["overrides"][0]["status"] == "incomplete_coverage"
    assert report["coverage_complete"] is False
    assert report["overrides"][0]["applied"] is False
    assert report["overrides"][0]["cleanup_review_candidate"] is False


def test_metadata_disagreement_is_reviewable_conflict_not_silent_approval():
    record = Override("python", "pkg", "MIT", "upstream")
    row = observation(raw_license="GPL-3.0-only")
    report = summarize((record,), [row], [row], [])
    assert report["overrides"][0]["status"] == "conflict"
    assert report["overrides"][0]["applied"] is True


def test_latest_red_scheduled_run_uses_its_own_artifact(monkeypatch):
    raw = artifact()
    run = {
        "id": 90,
        "head_sha": SHA,
        "head_branch": "main",
        "event": "schedule",
        "conclusion": "failure",
        "head_repository": {"full_name": "org/repo"},
        "path": ".github/workflows/nightly-ci.yml",
    }
    manual = {**run, "id": 95, "event": "workflow_dispatch", "conclusion": "success"}
    metadata = {
        "id": 7,
        "name": f"compliance-{SHA}-vllm-runtime",
        "expired": False,
        "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "workflow_run": {"id": 90, "head_sha": SHA},
    }
    calls = []

    def api(endpoint):
        calls.append(endpoint)
        if "/workflows/" in endpoint:
            assert "status=success" not in endpoint
            return {"workflow_runs": [manual, run]}
        if "/artifacts?" in endpoint:
            assert "/runs/90/" in endpoint
            return {"total_count": 1, "artifacts": [metadata]}
        assert f"ref={SHA}" in endpoint
        return {"content": base64.b64encode(b"catalog").decode()}

    monkeypatch.setattr(audit_override_usage, "_api", api)
    monkeypatch.setattr(
        audit_override_usage.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=raw),
    )
    selected, archives, problems = audit_override_usage.download_latest(
        "org/repo", [IMAGE]
    )
    assert selected["id"] == 90
    assert archives == {"vllm-runtime": raw}
    assert problems == []
    assert len(calls) == 3


@pytest.mark.parametrize(
    "images", [[], [{**IMAGE, "architectures": []}], [IMAGE, IMAGE]]
)
def test_empty_or_duplicate_matrix_cannot_claim_complete_coverage(tmp_path, images):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps({"schema_version": 1, "images": images}))
    with pytest.raises(ValueError):
        audit_override_usage.load_matrix(path)
