# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""License policy failures must reach an already-required merge gate."""

from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]
ROOT = Path(__file__).resolve().parents[3]


def read_yaml(path):
    return yaml.load((ROOT / path).read_text(), Loader=yaml.BaseLoader)


def test_policy_review_is_a_dependency_of_required_premerge_aggregate():
    workflow = read_yaml(".github/workflows/pre-merge.yml")
    jobs = workflow["jobs"]
    assert "compliance-policy-review" in jobs["pre-merge-status-check"]["needs"]
    assert jobs["pre-merge-status-check"]["if"] == "always()"
    review = jobs["compliance-policy-review"]
    assert review["uses"] == "./.github/workflows/compliance-policy-review.yml"
    assert "continue-on-error" not in review
    assert review["with"]["base_sha"] == "${{ github.event.pull_request.base.sha }}"
    assert "outputs.compliance_policy == 'true'" in review["if"]
    called = read_yaml(".github/workflows/compliance-policy-review.yml")
    assert set(called["on"]) == {"workflow_call"}
    assert "continue-on-error" not in called["jobs"]["policy-review"]


@pytest.mark.parametrize(
    "path",
    [
        "container/compliance/policy/licenses.toml",
        "container/compliance/license_overrides.yaml",
        "container/compliance/policy/review.py",
        ".github/workflows/compliance-policy-review.yml",
        ".github/workflows/pre-merge.yml",
    ],
)
def test_policy_inputs_and_gate_changes_activate_the_required_job(path):
    filters = read_yaml(".github/filters.yaml")["compliance_policy"]
    assert any(fnmatchcase(path, pattern) for pattern in filters)
    action = read_yaml(".github/actions/changed-files/action.yml")
    assert (
        action["outputs"]["compliance_policy"]["value"]
        == "${{ steps.filter.outputs.compliance_policy_any_modified }}"
    )
    workflow = read_yaml(".github/workflows/pre-merge.yml")
    assert (
        workflow["jobs"]["changed-files"]["outputs"]["compliance_policy"]
        == "${{ steps.changes.outputs.compliance_policy }}"
    )
