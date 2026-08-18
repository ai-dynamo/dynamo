# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for trusted pull-request branch finalization."""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github/workflows"
FINALIZER_PATH = WORKFLOW_DIR / "finalize-pull-request-branch.yml"
BRANCH_PATTERN = "pull-request/[0-9]+"

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def load_workflow(path: Path):
    """Load one GitHub Actions workflow with PyYAML's YAML 1.1 semantics."""

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_finalizer_keeps_privileged_logic_on_the_default_branch():
    """Require a reusable, scoped finalizer with no caller-controlled inputs."""

    workflow = load_workflow(FINALIZER_PATH)
    assert workflow[True] == {"workflow_call": None}
    assert workflow["permissions"] == {"actions": "read", "contents": "read"}

    job = workflow["jobs"]["finalize"]
    assert job["environment"] == "pull-request-branch-cleanup"
    assert "github.repository == 'ai-dynamo/dynamo'" in job["if"]
    assert "github.event_name == 'push'" in job["if"]
    assert "refs/heads/pull-request/" in job["if"]

    steps = {step["name"]: step for step in job["steps"]}
    token_step = steps["Create a short-lived maintenance token"]
    assert re.fullmatch(
        r"actions/create-github-app-token@[0-9a-f]{40}", token_step["uses"]
    )
    assert token_step["with"] == {
        "client-id": "${{ vars.PR_BRANCH_CLEANER_CLIENT_ID }}",
        "private-key": "${{ secrets.PR_BRANCH_CLEANER_PRIVATE_KEY }}",
        "permission-contents": "write",
    }

    checkout_step = steps["Checkout the exact workflow commit"]
    assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", checkout_step["uses"])
    assert checkout_step["with"] == {
        "ref": "${{ github.sha }}",
        "token": "${{ steps.maintenance-token.outputs.token }}",
    }


def test_finalizer_uses_exact_ref_identity_for_deletion():
    """Prevent an older workflow run from deleting a branch that moved."""

    workflow = load_workflow(FINALIZER_PATH)
    steps = {step["name"]: step for step in workflow["jobs"]["finalize"]["steps"]}
    delete_script = steps["Delete the unchanged branch"]["run"]

    assert 'delete_ref="refs/heads/${GITHUB_REF_NAME}"' in delete_script
    assert '--force-with-lease="${delete_ref}:${GITHUB_SHA}"' in delete_script
    assert 'origin ":${delete_ref}"' in delete_script
    assert '[[ "$current_sha" != "$GITHUB_SHA" ]]' in delete_script


def test_expected_workflows_match_numeric_pull_request_push_triggers():
    """Keep the materialization guard aligned with every matching workflow."""

    finalizer = load_workflow(FINALIZER_PATH)
    expected = set(finalizer["env"]["EXPECTED_WORKFLOWS"].splitlines())
    matching = set()

    for path in (*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")):
        workflow = load_workflow(path)
        triggers = workflow.get(True, {})
        if not isinstance(triggers, dict) or "push" not in triggers:
            continue
        push = triggers["push"]
        branches = push.get("branches", []) if isinstance(push, dict) else []
        if push is None or BRANCH_PATTERN in branches:
            matching.add(str(path.relative_to(REPO_ROOT)))

    assert matching == expected
