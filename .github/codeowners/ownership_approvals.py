"""Determine who must approve a change to the ownership policy."""

from __future__ import annotations

from dataclasses import dataclass

from codeowners_match import (
    DELEGATED_POLICY_PATHS,
    compute_resolution,
    parse_codeowners,
    resolve_owners,
)
from emit_codeowners import _render_codeowners

PROCESS = "@ai-dynamo/dynamo-process-codeowners"
OPS = "@ai-dynamo/dynamo-ops-codeowners"
ENFORCEMENT_WORKFLOWS = frozenset(
    {".github/workflows/codeowners.yml", ".github/workflows/codeowners-approval.yml"}
)


@dataclass(frozen=True)
class OwnershipChange:
    path: str
    before: tuple[str, ...]
    after: tuple[str, ...]


def rendered_policy(spec: dict, external: list[dict]) -> str:
    lines, _ = _render_codeowners(
        compute_resolution(spec), group=True, external=external
    )
    return "\n".join(lines) + "\n"


def governance(spec: dict) -> dict:
    """Routing is delegated; contracts, defaults and schema remain governance."""
    return {key: value for key, value in spec.items() if key not in {"areas", "shared"}}


def enforcement_changed(paths: list[str]) -> bool:
    return any(
        path in ENFORCEMENT_WORKFLOWS
        or (path.startswith(".github/codeowners/") and path.endswith(".py"))
        for path in paths
    )


def broad(glob: str) -> bool:
    return (
        not glob.startswith("/") or glob.endswith("/") or any(c in glob for c in "*?[")
    )


def required_approvals(
    before: str, after: str, paths: list[str], *, protected: bool = False
) -> tuple[set[str], list[OwnershipChange]]:
    """Require every old/new principal for each changed effective owner set."""
    old_rules, new_rules = parse_codeowners(before), parse_codeowners(after)
    changes = []
    required = {OPS, PROCESS} if protected else set()
    for path in sorted(set(paths) - DELEGATED_POLICY_PATHS):
        old = tuple(sorted(set(resolve_owners(old_rules, path))))
        new = tuple(sorted(set(resolve_owners(new_rules, path))))
        if old != new:
            changes.append(OwnershipChange(path, old, new))
            required.update(old)
            required.update(new)

    # A new/removed scope can change ownership of files that do not exist yet.
    old_rules = [
        (g, owners)
        for g, owners in old_rules
        if g.lstrip("/") not in DELEGATED_POLICY_PATHS
    ]
    new_rules = [
        (g, owners)
        for g, owners in new_rules
        if g.lstrip("/") not in DELEGATED_POLICY_PATHS
    ]
    old_globs, new_globs = set(dict(old_rules)), set(dict(new_rules))
    changed_scopes = old_globs ^ new_globs
    old_map = {glob: set(owners) for glob, owners in old_rules}
    new_map = {glob: set(owners) for glob, owners in new_rules}
    for glob in old_globs & new_globs:
        principals = old_map[glob] | new_map[glob]
        if (
            broad(glob)
            and old_map[glob] != new_map[glob]
            and not principals <= required
        ):
            required.add(PROCESS)
    # Scope/order edits can affect future files absent from the tracked tree.
    common = old_globs & new_globs
    old_order = list(
        dict.fromkeys(g for g, _ in reversed(old_rules) if g in common and broad(g))
    )
    new_order = list(
        dict.fromkeys(g for g, _ in reversed(new_rules) if g in common and broad(g))
    )
    if old_order != new_order or any(broad(glob) for glob in changed_scopes):
        required.add(PROCESS)
    return required, changes


def approved_reviewers(reviews: list[dict], author: str, head: str) -> set[str]:
    """Require a current-head approval; comments preserve it, dismissals revoke it."""
    latest = {}
    for review in sorted(reviews, key=lambda r: (r.get("submitted_at") or "", r["id"])):
        user = review["user"]
        login = user["login"].lower()
        if login == author.lower() or user["type"] != "User":
            continue
        if review["state"] in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            latest[login] = review
    return {
        login
        for login, review in latest.items()
        if review["state"] == "APPROVED" and review["commit_id"] == head
    }


def missing_approvals(
    required: set[str], approved: set[str], members: dict[str, set[str]]
) -> set[str]:
    missing = set()
    for principal in required:
        if not principal.startswith("@"):
            raise ValueError(
                f"Approval routing requires a GitHub team or login, got {principal!r}"
            )
        eligible = members[principal] if "/" in principal else {principal[1:].lower()}
        if not eligible & approved:
            missing.add(principal)
    return missing
