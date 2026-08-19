# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static contracts for Grype workflow wiring and report privacy."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_pr_scanner_is_wired_to_both_compliance_producers() -> None:
    """Both archived CSV producers must enforce the same Python PR policy."""
    repository = Path(__file__).resolve().parents[3]
    extract = (repository / ".github/actions/compliance-extract/action.yml").read_text(
        encoding="utf-8"
    )
    deploy = (
        repository / ".github/actions/build-deploy-component/action.yml"
    ).read_text(encoding="utf-8")

    assert "uses: ./.github/actions/grype-pr-scan" in extract
    assert "uses: ./.github/actions/grype-pr-scan" in deploy
    assert "inputs.vulnerability_scan_mode == 'pr'" in extract
    assert "inputs.diff_event_context == 'pr'" in deploy


def test_post_merge_has_no_github_vulnerability_report_sink() -> None:
    """Post-merge wiring must keep full reports and finding details out of GitHub."""
    repository = Path(__file__).resolve().parents[3]
    workflow = (repository / ".github/workflows/post-merge-ci.yml").read_text(
        encoding="utf-8"
    )
    start = workflow.index("  grype-reconcile:")
    end = workflow.index("  # IMAGE COPY JOBS", start)
    grype_job = workflow[start:end]

    assert "compliance.vulnerability_scan post-merge" in grype_job
    assert "LINEAR_API_KEY" in grype_job
    assert "upload-sarif" not in grype_job
    assert "upload-artifact" not in grype_job
    assert "GITHUB_STEP_SUMMARY" not in grype_job
    assert "Artifactory" in grype_job


def test_post_merge_enforcement_requires_explicit_rollout_gate() -> None:
    """Post-merge reconciliation must stay off until its Linear secret is ready."""
    repository = Path(__file__).resolve().parents[3]
    workflow = (repository / ".github/workflows/post-merge-ci.yml").read_text(
        encoding="utf-8"
    )
    start = workflow.index("  grype-reconcile:")
    end = workflow.index("  # IMAGE COPY JOBS", start)
    grype_job = workflow[start:end]

    assert "vars.GRYPE_POST_MERGE_ENFORCEMENT_ENABLED == 'true'" in grype_job
    assert "only after the LINEAR_API_KEY" in grype_job
    assert "PR scanning remains" in grype_job


def test_scanner_actions_pin_uv_and_expected_current_matrix() -> None:
    """Scanner wiring must pin uv and require all 18 current architecture pairs."""
    repository = Path(__file__).resolve().parents[3]
    pr_action = (repository / ".github/actions/grype-pr-scan/action.yml").read_text(
        encoding="utf-8"
    )
    workflow = (repository / ".github/workflows/post-merge-ci.yml").read_text(
        encoding="utf-8"
    )
    start = workflow.index("  grype-reconcile:")
    end = workflow.index("  # IMAGE COPY JOBS", start)
    grype_job = workflow[start:end]

    assert "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b" in (pr_action)
    assert "version: 0.11.15" in pr_action
    assert "--expected-pairs 18" in grype_job
    for producer in (
        "operator",
        "snapshot-agent",
        "vllm-build",
        "vllm-efa-build",
        "sglang-build",
        "sglang-efa-build",
        "trtllm-build",
        "trtllm-efa-build",
        "planner-build",
        "frontend-build",
    ):
        assert f"      - {producer}\n" in grype_job


def test_grype_archives_are_version_and_checksum_pinned() -> None:
    """Grype installation must retain the verified official release digests."""
    repository = Path(__file__).resolve().parents[3]
    action = (repository / ".github/actions/setup-grype/action.yml").read_text(
        encoding="utf-8"
    )

    assert "grype_0.116.1_linux_${GRYPE_ARCH}.tar.gz" in action
    assert "0122df7b655981abe547ad3d2190d65551dac6a2bfc80b4dc2a989b5d0587458" in action
    assert "a8d7504a149629324eb5f4ce3dc25dfd211bbfe047e64ee2bf7844b466c3d84d" in action


def test_full_image_audit_archives_native_syft_for_both_architectures() -> None:
    """The completeness audit must retain full native Syft data per arch."""
    repository = Path(__file__).resolve().parents[3]
    workflow = (repository / ".github/workflows/shared-compliance-audit.yml").read_text(
        encoding="utf-8"
    )

    assert "platform: [amd64, arm64]" in workflow
    assert "prod-tester-arm-v2" in workflow
    assert "${STEM}-${ARCH}.cdx.json" in workflow
    assert '"*linux_${ARCH}*"' in workflow
    assert "json=/tmp/syft-${ARCH}.syft.json" in workflow
    assert (
        "compliance-audit-${{ steps.resolve.outputs.image_tag }}-${{ matrix.platform }}"
        in workflow
    )
    assert "/tmp/syft-${{ matrix.platform }}.syft.json" in workflow
