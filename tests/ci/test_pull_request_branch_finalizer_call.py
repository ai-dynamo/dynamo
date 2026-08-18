# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for the pull-request branch finalizer caller."""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github/workflows"
PR_WORKFLOW_PATH = WORKFLOW_DIR / "pr.yaml"
FINALIZER = "ai-dynamo/dynamo/.github/workflows/finalize-pull-request-branch.yml@main"

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def load_workflow(path: Path):
    """Load one GitHub Actions workflow with PyYAML's YAML 1.1 semantics."""

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_pr_workflow_finalizes_after_changed_files():
    """Run the default-branch finalizer even when change detection fails."""

    workflow = load_workflow(PR_WORKFLOW_PATH)
    job = workflow["jobs"]["finalize-pull-request-branch"]

    assert job == {
        "name": "Finalize pull request branch",
        "needs": "changed-files",
        "if": "${{ always() }}",
        "uses": FINALIZER,
        "permissions": {"actions": "read", "contents": "read"},
    }


def test_only_pr_workflow_calls_the_finalizer():
    """Avoid redundant deletion races from other trusted workflows."""

    callers = []
    for path in (*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")):
        workflow = load_workflow(path)
        for job_name, job in workflow.get("jobs", {}).items():
            if job.get("uses") == FINALIZER:
                callers.append((path.name, job_name))

    assert callers == [("pr.yaml", "finalize-pull-request-branch")]
