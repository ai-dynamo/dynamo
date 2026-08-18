# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-workflow contract test for pull-request branch finalization."""

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
