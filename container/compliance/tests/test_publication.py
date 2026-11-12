# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publication accepts only complete evidence for the immutable images it copies."""

import io
import re
import shlex
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml
from compliance import image_provenance as provenance
from compliance import publication
from compliance.prepare_diff import inventory_name

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

SHA = "a" * 40
INDEX = "sha256:" + "b" * 64
AMD = "sha256:" + "c" * 64
ARM = "sha256:" + "d" * 64
IMAGE = f"registry.example/project:{SHA}-sglang-runtime"


def registry_manifest():
    return {
        "digest": INDEX,
        "manifests": [
            {"digest": AMD, "platform": {"os": "linux", "architecture": "amd64"}},
            {"digest": ARM, "platform": {"os": "linux", "architecture": "arm64"}},
            {
                "digest": "sha256:" + "e" * 64,
                "platform": {"os": "unknown", "architecture": "unknown"},
                "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
            },
        ],
    }


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "inspect_image", lambda ref: registry_manifest())
    for arch in ("amd64", "arm64"):
        directory = tmp_path / f"linux_{arch}"
        directory.mkdir()
        name = inventory_name("sglang-runtime", arch, SHA, "13.0")
        (directory / name).write_text(
            "ecosystem,name,version,spdx,source_url,notes\npython,demo,1,MIT,,\n"
        )
    provenance.create_provenance(
        IMAGE,
        tmp_path,
        "sglang-runtime",
        SHA,
        "ai-dynamo/dynamo",
        "123",
        "1",
        {"linux/amd64", "linux/arm64"},
        "13.0",
    )
    return tmp_path


def validate(root, **changes):
    args = {
        "root": root,
        "repository": "ai-dynamo/dynamo",
        "run_id": "123",
        "source_sha": SHA,
        "prefix": "sglang-runtime",
        "image_ref": IMAGE,
        "platforms": {"linux/amd64", "linux/arm64"},
    }
    args.update(changes)
    return provenance.validate_provenance(**args)


def test_provenance_covers_every_platform_and_ignores_attestations(inventory):
    manifest = validate(inventory)
    assert manifest["index_digest"] == INDEX
    assert {r["digest"] for r in manifest["platforms"]} == {AMD, ARM}
    assert manifest["run_attempt"] == "1"


@pytest.mark.parametrize(
    "change", ["missing-arm", "extra-platform", "duplicate-amd", "invalid-digest"]
)
def test_published_platform_set_cannot_silently_shrink(change):
    manifest = registry_manifest()
    if change == "missing-arm":
        manifest["manifests"].pop(1)
    elif change == "extra-platform":
        manifest["manifests"].append(
            {"digest": ARM, "platform": {"os": "windows", "architecture": "amd64"}}
        )
    elif change == "duplicate-amd":
        manifest["manifests"].append(manifest["manifests"][0])
    else:
        manifest["digest"] = "tag:mutable"
    with pytest.raises(ValueError):
        provenance.platform_digests(manifest, {"linux/amd64", "linux/arm64"})


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "456"},
        {"source_sha": "f" * 40},
        {"repository": "fork/dynamo"},
        {"prefix": "sglang-runtime-efa"},
        {"image_ref": IMAGE + "-nightly"},
        {"expected_attempt": "2"},
    ],
)
def test_same_sha_does_not_authorize_another_run_or_variant(inventory, changes):
    with pytest.raises(ValueError):
        validate(inventory, **changes)


def test_tag_change_and_modified_inventory_are_rejected(inventory, monkeypatch):
    current = registry_manifest()
    current["digest"] = "sha256:" + "f" * 64
    monkeypatch.setattr(provenance, "inspect_image", lambda ref: current)
    with pytest.raises(ValueError, match="index digest"):
        validate(inventory)
    monkeypatch.setattr(provenance, "inspect_image", lambda ref: registry_manifest())
    csv = (
        inventory
        / "linux_arm64"
        / inventory_name("sglang-runtime", "arm64", SHA, "13.0")
    )
    csv.write_text(csv.read_text().replace("demo,1", "demo,2"))
    with pytest.raises(ValueError, match="hash mismatch"):
        validate(inventory)


def test_missing_provenance_requires_rebuilding_legacy_artifacts(inventory):
    (inventory / "build-provenance.json").unlink()
    with pytest.raises(ValueError, match="evidence file"):
        validate(inventory)


def test_rootfs_inventory_must_equal_producer_inventory(inventory, tmp_path):
    rootfs = tmp_path / "rootfs"
    (rootfs / "legal").mkdir(parents=True)
    (rootfs / "legal/osrb-deps.csv").write_text("different shipped inventory")
    with pytest.raises(ValueError, match="differs from its producer"):
        provenance.scan_rootfs(rootfs, inventory, "linux/arm64", tmp_path / "out")


def test_stale_scan_result_cannot_make_an_empty_export_succeed(
    inventory, tmp_path, monkeypatch
):
    manifest = validate(inventory)
    output = tmp_path / "audit"
    result = output / "linux_amd64/accepted.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"stale": true}')
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
    )
    with pytest.raises(ValueError, match="without a successful audit result"):
        provenance.audit_platforms(inventory, manifest, output, "test-builder")
    assert not result.exists()


def test_manual_release_does_not_omit_selected_images():
    all_images = publication.selected_images("all", SHA, "registry/repo", True, {})
    assert len(all_images) == 10
    skipped = publication.selected_images(
        "all", SHA, "registry/repo", True, {"operator": "failure"}
    )
    assert set(all_images) - set(skipped) == {"operator"}
    assert (
        len(
            publication.selected_images(
                "all", SHA, "registry/repo", False, {"operator": "failure"}
            )
        )
        == 10
    )
    with pytest.raises(ValueError, match="Unknown"):
        publication.selected_images("vllm-runtime,typo", SHA, "registry/repo", True, {})


def test_preflight_never_leaves_partial_acceptance(tmp_path, monkeypatch):
    manifest = tmp_path / "publication-manifest.json"
    manifest.write_text('{"images": {"stale": "accepted"}}')
    calls = []

    def audit(repo, run, sha, prefix, ref, builder, out, nightly):
        calls.append(prefix)
        if prefix == "sglang-runtime":
            raise ValueError("ARM64 attribution failed")
        return "registry/repo@" + INDEX

    monkeypatch.setattr(publication, "audit_one", audit)
    with pytest.raises(ValueError, match="ARM64"):
        publication.preflight(
            "ai-dynamo/dynamo",
            "123",
            SHA,
            {
                "vllm-runtime": "registry/repo:one",
                "sglang-runtime": "registry/repo:two",
            },
            True,
            "builder",
            tmp_path,
        )
    assert calls == ["vllm-runtime", "sglang-runtime"]
    assert not manifest.exists()


def test_copy_source_remains_pinned_after_tag_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        publication, "audit_one", lambda *args: "registry/repo@" + INDEX
    )
    publication.preflight(
        "ai-dynamo/dynamo",
        "123",
        SHA,
        {"sglang-runtime": IMAGE},
        True,
        "builder",
        tmp_path,
    )
    # No registry request happens at copy time: changing the tag cannot change
    # the digest admitted by preflight, and an unselected source is rejected.
    manifest = tmp_path / "publication-manifest.json"
    assert publication.pinned_source(manifest, IMAGE) == "registry/repo@" + INDEX
    with pytest.raises(ValueError, match="not accepted"):
        publication.pinned_source(manifest, IMAGE + "-efa")


@pytest.mark.parametrize(
    "path,branch,nightly",
    [
        (".github/workflows/pr.yaml", "pull-request/1", True),
        (".github/workflows/nightly-ci.yml", "release/1.7.0", True),
        (".github/workflows/post-merge-ci.yml", "main", False),
    ],
)
def test_release_rejects_wrong_producer_workflow(monkeypatch, path, branch, nightly):
    monkeypatch.setattr(
        publication,
        "api",
        lambda endpoint: {
            "head_sha": SHA,
            "head_repository": {"full_name": "ai-dynamo/dynamo"},
            "path": path,
            "head_branch": branch,
        },
    )
    with pytest.raises(ValueError):
        publication.run_metadata("ai-dynamo/dynamo", "123", SHA, nightly)


def test_artifact_download_rejects_path_escape(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("../outside", "unsafe")
    artifact = {
        "id": 456,
        "name": f"compliance-{SHA}-operator",
        "expired": False,
        "workflow_run": {"id": 123},
    }
    monkeypatch.setattr(publication, "api", lambda endpoint: {"artifacts": [artifact]})
    monkeypatch.setattr(
        publication.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, archive.getvalue()
        ),
    )
    with pytest.raises(ValueError, match="Unsafe"):
        publication.download_inventory(
            "ai-dynamo/dynamo", "123", SHA, "operator", tmp_path
        )


def test_release_cannot_copy_before_gate_or_read_gate_from_source_commit():
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/release.yml").read_text())
    steps = workflow["jobs"]["release-publish"]["steps"]
    names = [step.get("name") for step in steps]
    assert names.index(
        "Require independent audits for every selected image"
    ) < names.index("Copy images to NGC")
    checkout = steps[names.index("Checkout trusted compliance gate")]
    assert checkout["with"]["ref"] == "${{ github.workflow_sha }}"
    copy = steps[names.index("Copy images to NGC")]["run"]
    assert copy.index("compliance.publication source") < copy.index(
        'crane copy "${SRC}"'
    )
    assert "|| exit 1" in copy


def test_release_manifest_path_uses_runner_context_only_after_runner_assignment():
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/release.yml").read_text())
    job = workflow["jobs"]["release-publish"]
    # GitHub rejects runner.* in jobs.<job_id>.env before any job is created.
    # The path belongs in step env, where runner.temp is available.
    assert not any("runner." in str(value) for value in job.get("env", {}).values())
    copy = next(step for step in job["steps"] if step.get("id") == "copy_images")
    assert (
        copy["env"]["PUBLICATION_MANIFEST"]
        == "${{ runner.temp }}/publication-audit/publication-manifest.json"
    )


@pytest.mark.parametrize(
    "workflow_name,operator_build",
    [
        ("pr.yaml", "operator"),
        ("post-merge-ci.yml", "operator"),
        ("nightly-ci.yml", "operator-build"),
    ],
)
def test_all_published_variants_have_independent_audits(workflow_name, operator_build):
    root = Path(__file__).resolve().parents[3]
    jobs = yaml.safe_load((root / ".github/workflows" / workflow_name).read_text())[
        "jobs"
    ]
    for token in (
        "vllm",
        "sglang",
        "trtllm",
        "vllm-efa",
        "sglang-efa",
        "trtllm-efa",
        "frontend",
        "planner",
        "sidecar",
        "operator",
    ):
        audit = jobs[token + "-compliance-audit"]
        assert audit["uses"].endswith("shared-compliance-audit.yml")
        assert audit["with"]["image_ref"]
        if workflow_name == "pr.yaml":
            assert token + "-compliance-audit" in jobs["backend-status-check"]["needs"]
    assert jobs["operator-compliance-audit"]["needs"] == [operator_build]


@pytest.mark.parametrize(
    "workflow_name", ["shared-build-image.yml", "shared-build-sidecar.yml"]
)
def test_image_provenance_inputs_reach_the_extractor(workflow_name):
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows" / workflow_name).read_text())
    steps = workflow["jobs"]["build"]["steps"]
    extracts = [
        step
        for step in steps
        if step.get("uses", "").endswith("actions/compliance-extract")
    ]
    assert len(extracts) == 1
    assert (
        extracts[0]["with"]["pushed_image_ref"]
        == "${{ steps.calculate-target-tag.outputs.image_uri }}"
    )
    assert extracts[0]["with"]["baseline_sbom_file"]
    assert extracts[0]["with"]["diff_base_run_id"]
    # Supplying these to the build action instead is accepted by YAML but never
    # binds the extracted inventory to the image (a real integration regression).
    for step in steps:
        if step.get("uses", "").endswith("actions/docker-remote-build"):
            assert "pushed_image_ref" not in step.get("with", {})


def test_operator_exports_and_binds_both_architectures():
    root = Path(__file__).resolve().parents[3]
    action = yaml.safe_load(
        (root / ".github/actions/build-deploy-component/action.yml").read_text()
    )
    steps = {step["name"]: step for step in action["runs"]["steps"]}
    assert (
        "compliance_platform=linux/amd64,linux/arm64"
        in steps["Determine build settings"]["run"]
    )
    assert "for ARCH in amd64 arm64" in steps["Compliance extract"]["run"]
    bind = steps["Bind final image digests to the producer inventory"]
    assert (
        bind["env"]["PLATFORMS"] == "${{ steps.settings.outputs.compliance_platform }}"
    )
    assert "compliance.image_provenance create" in bind["run"]
    upload = steps["Upload compliance artifact (NOTICES + CSV + CycloneDX)"]
    assert "always()" not in upload["if"]
    assert "success()" in upload["if"]
    sources = steps["Upload /sources/ archive (opt-in via archive_sources)"]
    assert sources["with"]["path"] == "/tmp/sources/"


def test_operator_main_binaries_keep_their_first_party_package_identity():
    root = Path(__file__).resolve().parents[3]
    for relative in ("deploy/operator/Dockerfile", "deploy/operator/Makefile"):
        commands = re.findall(r"go build[^\n]+", (root / relative).read_text())
        assert commands
        for command in commands:
            arguments = shlex.split(command.rstrip("\\").strip())
            targets = [arg for arg in arguments if arg.startswith(("./cmd", "cmd/"))]
            assert targets, (relative, command)
            # Loose-file compilation embeds command-line-arguments instead of
            # the module's first-party package path, hiding binary ownership.
            assert not any(target.endswith(".go") for target in targets), (
                relative,
                targets,
            )


@pytest.mark.parametrize("workflow_name", ["pr.yaml", "post-merge-ci.yml"])
@pytest.mark.parametrize("audit_result", ["success", "failure", "cancelled", "skipped"])
def test_successful_build_cannot_copy_after_unsuccessful_audit(
    workflow_name, audit_result
):
    root = Path(__file__).resolve().parents[3]
    jobs = yaml.safe_load((root / ".github/workflows" / workflow_name).read_text())[
        "jobs"
    ]
    for token in ("vllm", "sglang", "trtllm", "frontend", "planner"):
        job = jobs[token + "-copy-to-acr"]
        expression = job["if"].replace("${{", "").replace("}}", "")
        expression = expression.replace("always()", "True").replace(
            "!cancelled()", "True"
        )
        expression = expression.replace("github.event_name", repr("push"))
        expression = re.sub(
            r"needs\.changed-files\.outputs\.[a-z_]+", repr("true"), expression
        )
        expression = re.sub(
            r"needs\.([a-z-]+)\.result",
            lambda m: repr(
                audit_result if m[1].endswith("-compliance-audit") else "success"
            ),
            expression,
        )
        expression = (
            " ".join(expression.split()).replace("&&", "and").replace("||", "or")
        )
        # Evaluate the actual workflow predicate after replacing its contexts;
        # this catches the always() + ordering-only needs regression.
        assert eval(expression, {"__builtins__": {}}) is (audit_result == "success")


def test_reusable_release_and_audit_callers_allow_artifact_reads():
    root = Path(__file__).resolve().parents[3]
    checked = []
    for path in (root / ".github/workflows").glob("*.y*ml"):
        workflow = yaml.safe_load(path.read_text())
        for name, job in workflow.get("jobs", {}).items():
            target = job.get("uses", "")
            if not target.endswith(("/release.yml", "/shared-compliance-audit.yml")):
                continue
            permissions = job.get("permissions", workflow.get("permissions", {}))
            assert permissions.get("actions") in {"read", "write"}, (path.name, name)
            assert permissions.get("contents") in {"read", "write"}, (path.name, name)
            checked.append((path.name, name))
    assert ("stage-release-rc.yml", "invoke-release") in checked
    assert ("nightly-ci.yml", "release") in checked


@pytest.mark.parametrize(
    "workflow,job,cleanup_name,create_name",
    [
        (
            "shared-compliance-audit.yml",
            "audit",
            "Remove this audit builder",
            "Initialize native audit builder",
        ),
        (
            "release.yml",
            "release-publish",
            "Remove this publication audit builder",
            "Initialize publication audit builder",
        ),
    ],
)
def test_audit_cleanup_removes_only_its_own_builder(
    workflow, job, cleanup_name, create_name
):
    root = Path(__file__).resolve().parents[3]
    steps = yaml.safe_load((root / ".github/workflows" / workflow).read_text())["jobs"][
        job
    ]["steps"]
    by_name = {step.get("name"): step for step in steps}
    cleanup, create = by_name[cleanup_name], by_name[create_name]
    assert "always()" in cleanup["if"]
    assert cleanup["env"]["AUDIT_BUILDER"] == create["with"]["builder_name"]
    assert cleanup["run"] == 'docker buildx rm --force "${AUDIT_BUILDER}" || true'
