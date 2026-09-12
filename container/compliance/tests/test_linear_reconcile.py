# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for private Linear vulnerability reconciliation."""

from __future__ import annotations

import datetime as dt

import pytest
from compliance.linear_reconcile import (
    LinearClient,
    LinearSettings,
    TrackedFinding,
    finding_key,
    reconcile_findings,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _finding() -> TrackedFinding:
    """Return one deterministic tracked finding."""
    package_purl = "pkg:pypi/urllib3"
    return TrackedFinding(
        key=finding_key("main", "CVE-2026-1234", package_purl),
        branch_line="main",
        vulnerability="CVE-2026-1234",
        package="urllib3",
        package_purl=package_purl,
        severity="Critical",
        installed_versions=("1.0",),
        fixed_versions=("2.0",),
        fix_states=("fixed",),
        targets=("vllm-amd64",),
        commit_sha="a" * 40,
        database_built="2026-07-29T00:00:00Z",
    )


class FakeTransport:
    """Stateful fake Linear GraphQL transport used by reconciliation tests."""

    def __init__(self, issues: list[dict[str, object]] | None = None) -> None:
        """Initialize fake issue state and mutation recording."""
        self.issues = issues or []
        self.mutations: list[tuple[str, dict[str, object]]] = []

    def __call__(self, query: str, variables: dict[str, object]) -> dict[str, object]:
        """Return fixture query data or record a fake mutation."""
        if "GrypeLinearContext" in query:
            return {
                "teams": {
                    "nodes": [
                        {
                            "id": "team-1",
                            "name": "Operations",
                            "states": {
                                "nodes": [
                                    {"id": "todo", "name": "Todo", "type": "unstarted"},
                                    {"id": "done", "name": "Done", "type": "completed"},
                                ]
                            },
                        }
                    ]
                },
                "projects": {
                    "nodes": [{"id": "project-1", "name": "Governance Project"}]
                },
            }
        if "GrypeProjectIssues" in query:
            return {
                "issues": {
                    "nodes": self.issues,
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        self.mutations.append((query, variables))
        field = "issueCreate" if "GrypeIssueCreate" in query else "issueUpdate"
        return {field: {"success": True, "issue": {"id": "issue-1"}}}


def test_new_finding_is_created_before_failure_decision() -> None:
    """A new finding should create an urgent issue and remain policy-active."""
    transport = FakeTransport()
    client = LinearClient("", transport=transport)

    active = reconcile_findings(
        client,
        LinearSettings(),
        [_finding()],
        complete_scan=True,
        today=dt.date(2026, 7, 29),
    )

    assert active == {_finding().key}
    assert len(transport.mutations) == 1
    _, variables = transport.mutations[0]
    issue_input = variables["input"]
    assert isinstance(issue_input, dict)
    assert issue_input["priority"] == 1
    assert issue_input["dueDate"] == "2026-07-31"
    assert issue_input["teamId"] == "team-1"
    assert issue_input["projectId"] == "project-1"


def test_absent_issue_closes_only_after_complete_scan() -> None:
    """Partial scans must never close an issue that may exist in a missing target."""
    finding = _finding()
    issue = {
        "id": "issue-1",
        "title": "managed",
        "description": f"<!-- dynamo-grype-key:{finding.key} -->",
        "state": {"id": "todo", "name": "Todo", "type": "unstarted"},
    }
    partial_transport = FakeTransport([issue])
    reconcile_findings(
        LinearClient("", transport=partial_transport),
        LinearSettings(),
        [],
        complete_scan=False,
    )
    assert partial_transport.mutations == []

    complete_transport = FakeTransport([issue])
    reconcile_findings(
        LinearClient("", transport=complete_transport),
        LinearSettings(),
        [],
        complete_scan=True,
    )
    assert len(complete_transport.mutations) == 1
    _, variables = complete_transport.mutations[0]
    assert variables["input"] == {"stateId": "done"}


def test_wont_fix_is_private_optional_exception() -> None:
    """Won't Fix should suppress only when the private policy switch is enabled."""
    finding = _finding()
    issue = {
        "id": "issue-1",
        "title": "managed",
        "description": f"<!-- dynamo-grype-key:{finding.key} -->",
        "state": {"id": "canceled", "name": "Won't Fix", "type": "canceled"},
    }
    transport = FakeTransport([issue])

    active = reconcile_findings(
        LinearClient("", transport=transport),
        LinearSettings(honor_wont_fix=True),
        [finding],
        complete_scan=True,
    )

    assert active == set()
    assert transport.mutations == []
