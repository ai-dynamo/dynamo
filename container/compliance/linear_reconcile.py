# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private Linear reconciliation for post-merge container vulnerability findings."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

LINEAR_ENDPOINT = "https://api.linear.app/graphql"
MANAGED_MARKER_RE = re.compile(r"<!-- dynamo-grype-key:([0-9a-f]{64}) -->")


class LinearError(RuntimeError):
    """Raised when Linear configuration or a GraphQL operation is invalid."""


@dataclass(frozen=True)
class TrackedFinding:
    """A deduplicated High/Critical finding to track privately in Linear."""

    key: str
    branch_line: str
    vulnerability: str
    package: str
    package_purl: str
    severity: str
    installed_versions: tuple[str, ...]
    fixed_versions: tuple[str, ...]
    fix_states: tuple[str, ...]
    targets: tuple[str, ...]
    commit_sha: str
    database_built: str


@dataclass(frozen=True)
class LinearSettings:
    """Names and behavior used to resolve Dynamo's Linear destination."""

    team_name: str = "Operations"
    project_name: str = "Governance Project"
    honor_wont_fix: bool = False
    critical_business_days: int = 2
    high_business_days: int = 5


@dataclass(frozen=True)
class LinearIssue:
    """The subset of a Linear issue needed for deterministic reconciliation."""

    issue_id: str
    title: str
    description: str
    state_id: str
    state_name: str
    state_type: str
    due_date: str


@dataclass(frozen=True)
class LinearContext:
    """Resolved Linear team, project, and workflow-state identifiers."""

    team_id: str
    project_id: str
    unstarted_state_id: str
    completed_state_id: str


Transport = Callable[[str, dict[str, object]], dict[str, object]]


def _default_transport(api_key: str, endpoint: str = LINEAR_ENDPOINT) -> Transport:
    """Create an authenticated stdlib HTTP transport for Linear GraphQL."""

    def send(query: str, variables: dict[str, object]) -> dict[str, object]:
        """Send one GraphQL request and return its decoded response."""
        body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LinearError("Linear request failed") from exc
        if payload.get("errors"):
            raise LinearError("Linear GraphQL returned an error")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise LinearError("Linear GraphQL response had no data")
        return data

    return send


class LinearClient:
    """Minimal Linear GraphQL client with an injectable transport for tests."""

    def __init__(
        self,
        api_key: str,
        *,
        transport: Transport | None = None,
        endpoint: str = LINEAR_ENDPOINT,
    ) -> None:
        """Initialize the client without making a network request."""
        if not api_key and transport is None:
            raise LinearError("LINEAR_API_KEY is required")
        self._transport = transport or _default_transport(api_key, endpoint)

    def resolve_context(self, settings: LinearSettings) -> LinearContext:
        """Resolve team, project, and workflow states by configured names."""
        query = """
        query GrypeLinearContext($teamName: String!, $projectName: String!) {
          teams(filter: {name: {eq: $teamName}}, first: 2) {
            nodes {
              id
              name
              states { nodes { id name type } }
            }
          }
          projects(filter: {name: {eq: $projectName}}, first: 2) {
            nodes { id name }
          }
        }
        """
        data = self._transport(
            query,
            {
                "teamName": settings.team_name,
                "projectName": settings.project_name,
            },
        )
        team_nodes = _nodes(data, "teams")
        project_nodes = _nodes(data, "projects")
        if len(team_nodes) != 1 or len(project_nodes) != 1:
            raise LinearError("Linear team/project names were missing or ambiguous")
        states = _nodes(team_nodes[0], "states")
        unstarted = _single_state(states, "unstarted")
        completed = _single_state(states, "completed")
        return LinearContext(
            team_id=_required_text(team_nodes[0], "id"),
            project_id=_required_text(project_nodes[0], "id"),
            unstarted_state_id=_required_text(unstarted, "id"),
            completed_state_id=_required_text(completed, "id"),
        )

    def list_project_issues(self, context: LinearContext) -> list[LinearIssue]:
        """Return all issues in the configured team/project, following pagination."""
        query = """
        query GrypeProjectIssues(
          $teamId: ID!,
          $projectId: ID!,
          $after: String
        ) {
          issues(
            filter: {
              team: {id: {eq: $teamId}},
              project: {id: {eq: $projectId}}
            },
            first: 100,
            after: $after,
            includeArchived: true
          ) {
            nodes {
              id
              title
              description
              dueDate
              state { id name type }
            }
            pageInfo { hasNextPage endCursor }
          }
        }
        """
        issues: list[LinearIssue] = []
        after: str | None = None
        while True:
            data = self._transport(
                query,
                {
                    "teamId": context.team_id,
                    "projectId": context.project_id,
                    "after": after,
                },
            )
            connection = data.get("issues")
            if not isinstance(connection, dict):
                raise LinearError("Linear issues response was malformed")
            for node in _nodes(data, "issues"):
                state = node.get("state")
                if not isinstance(state, dict):
                    raise LinearError("Linear issue state was missing")
                issues.append(
                    LinearIssue(
                        issue_id=_required_text(node, "id"),
                        title=_required_text(node, "title"),
                        description=str(node.get("description") or ""),
                        state_id=_required_text(state, "id"),
                        state_name=_required_text(state, "name"),
                        state_type=_required_text(state, "type"),
                        due_date=str(node.get("dueDate") or ""),
                    )
                )
            page_info = connection.get("pageInfo")
            if not isinstance(page_info, dict) or not page_info.get("hasNextPage"):
                return issues
            after = str(page_info.get("endCursor") or "")
            if not after:
                raise LinearError("Linear pagination cursor was missing")

    def create_issue(
        self,
        context: LinearContext,
        finding: TrackedFinding,
        *,
        due_date: str,
    ) -> None:
        """Create one urgent Linear issue for a newly observed finding."""
        mutation = """
        mutation GrypeIssueCreate($input: IssueCreateInput!) {
          issueCreate(input: $input) { success issue { id } }
        }
        """
        data = self._transport(
            mutation,
            {
                "input": {
                    "teamId": context.team_id,
                    "projectId": context.project_id,
                    "stateId": context.unstarted_state_id,
                    "priority": 1,
                    "title": _issue_title(finding),
                    "description": _issue_description(finding),
                    "dueDate": due_date,
                }
            },
        )
        _require_mutation_success(data, "issueCreate")

    def update_issue(
        self,
        issue_id: str,
        finding: TrackedFinding,
        *,
        due_date: str | None,
        state_id: str | None,
    ) -> None:
        """Refresh one managed issue and optionally reopen its workflow state."""
        mutation = """
        mutation GrypeIssueUpdate($issueId: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $issueId, input: $input) { success issue { id } }
        }
        """
        issue_input: dict[str, object] = {
            "priority": 1,
            "title": _issue_title(finding),
            "description": _issue_description(finding),
        }
        if due_date:
            issue_input["dueDate"] = due_date
        if state_id:
            issue_input["stateId"] = state_id
        data = self._transport(
            mutation,
            {"issueId": issue_id, "input": issue_input},
        )
        _require_mutation_success(data, "issueUpdate")

    def close_issue(self, issue_id: str, completed_state_id: str) -> None:
        """Move an absent managed issue to the team's completed state."""
        mutation = """
        mutation GrypeIssueClose($issueId: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $issueId, input: $input) { success issue { id } }
        }
        """
        data = self._transport(
            mutation,
            {
                "issueId": issue_id,
                "input": {"stateId": completed_state_id},
            },
        )
        _require_mutation_success(data, "issueUpdate")


def finding_key(branch_line: str, vulnerability: str, package_purl: str) -> str:
    """Return the stable branch/vulnerability/package reconciliation key."""
    source = f"{branch_line}\0{vulnerability}\0{package_purl}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def reconcile_findings(
    client: LinearClient,
    settings: LinearSettings,
    findings: list[TrackedFinding],
    *,
    complete_scan: bool,
    today: dt.date | None = None,
) -> set[str]:
    """Create/update findings, close absent issues only after a complete scan."""
    context = client.resolve_context(settings)
    managed: dict[str, LinearIssue] = {}
    for issue in client.list_project_issues(context):
        marker = MANAGED_MARKER_RE.search(issue.description)
        if marker:
            managed[marker.group(1)] = issue

    current_keys: set[str] = set()
    accepted_keys: set[str] = set()
    current_date = today or dt.datetime.now(tz=dt.timezone.utc).date()
    for finding in findings:
        current_keys.add(finding.key)
        issue = managed.get(finding.key)
        if issue and settings.honor_wont_fix and _is_wont_fix(issue):
            accepted_keys.add(finding.key)
            continue
        due_date = _due_date(finding.severity, settings, current_date).isoformat()
        if issue is None:
            client.create_issue(context, finding, due_date=due_date)
            continue
        reopen_state = (
            context.unstarted_state_id
            if issue.state_type in {"completed", "canceled"}
            else None
        )
        updated_due_date = due_date if reopen_state or not issue.due_date else None
        client.update_issue(
            issue.issue_id,
            finding,
            due_date=updated_due_date,
            state_id=reopen_state,
        )

    if complete_scan:
        for key, issue in managed.items():
            if key in current_keys or issue.state_type in {"completed", "canceled"}:
                continue
            client.close_issue(issue.issue_id, context.completed_state_id)
    return current_keys - accepted_keys


def _nodes(parent: dict[str, object], field: str) -> list[dict[str, object]]:
    """Extract a validated GraphQL connection's node dictionaries."""
    connection = parent.get(field)
    if not isinstance(connection, dict):
        raise LinearError(f"Linear response field {field} was missing")
    nodes = connection.get("nodes")
    if not isinstance(nodes, list) or not all(isinstance(node, dict) for node in nodes):
        raise LinearError(f"Linear response field {field}.nodes was malformed")
    return nodes


def _required_text(node: dict[str, object], field: str) -> str:
    """Return one required non-empty text field from a GraphQL node."""
    value = node.get(field)
    if not isinstance(value, str) or not value:
        raise LinearError(f"Linear response field {field} was missing")
    return value


def _single_state(
    states: list[dict[str, object]], state_type: str
) -> dict[str, object]:
    """Return the single preferred workflow state for a Linear state type."""
    matches = [state for state in states if state.get("type") == state_type]
    if not matches:
        raise LinearError(f"Linear team has no {state_type} workflow state")
    return min(matches, key=lambda state: str(state.get("name") or ""))


def _require_mutation_success(data: dict[str, object], field: str) -> None:
    """Validate a Linear mutation's standard success envelope."""
    payload = data.get(field)
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise LinearError(f"Linear mutation {field} failed")


def _issue_title(finding: TrackedFinding) -> str:
    """Return a concise private issue title."""
    return (
        f"[Grype][{finding.branch_line}] {finding.severity} "
        f"{finding.vulnerability} in {finding.package}"
    )


def _issue_description(finding: TrackedFinding) -> str:
    """Render the complete private remediation context for a Linear issue."""
    fixed = ", ".join(finding.fixed_versions) or "No fix currently available"
    return "\n".join(
        (
            f"<!-- dynamo-grype-key:{finding.key} -->",
            "Automated Dynamo container dependency vulnerability.",
            "",
            f"- Branch line: `{finding.branch_line}`",
            f"- Vulnerability: `{finding.vulnerability}`",
            f"- Package: `{finding.package_purl}`",
            f"- Severity: `{finding.severity}`",
            f"- Installed versions: `{', '.join(finding.installed_versions)}`",
            f"- Fix states: `{', '.join(finding.fix_states)}`",
            f"- Fixed versions: `{fixed}`",
            f"- Affected targets: `{', '.join(finding.targets)}`",
            f"- Last seen commit: `{finding.commit_sha}`",
            f"- Grype database built: `{finding.database_built}`",
        )
    )


def _is_wont_fix(issue: LinearIssue) -> bool:
    """Return whether a Linear issue is in the private Won't Fix state."""
    normalized = issue.state_name.lower().replace("’", "'")
    return issue.state_type == "canceled" and normalized in {"won't fix", "wont fix"}


def _due_date(
    severity: str,
    settings: LinearSettings,
    today: dt.date,
) -> dt.date:
    """Calculate the configured business-day remediation due date."""
    days = (
        settings.critical_business_days
        if severity.lower() == "critical"
        else settings.high_business_days
    )
    due = today
    added = 0
    while added < days:
        due += dt.timedelta(days=1)
        if due.weekday() < 5:
            added += 1
    return due
