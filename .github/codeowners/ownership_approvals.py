"""Determine who must approve a change to the ownership policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from codeowners_match import (
    DELEGATED_POLICY_PATHS,
    compute_resolution,
    match,
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


def policy_fingerprint(
    spec: dict, external: list[dict], enforcement: dict | None = None
) -> str:
    rules = [
        (glob, sorted(owners))
        for glob, owners in parse_codeowners(rendered_policy(spec, external))
    ]
    payload = json.dumps([rules, governance(spec), enforcement], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def enforcement_changed(paths: list[str]) -> bool:
    return any(
        path in ENFORCEMENT_WORKFLOWS
        or (path.startswith(".github/codeowners/") and path.endswith(".py"))
        for path in paths
    )


def broad(glob: str) -> bool:
    return glob.endswith("/") or any(c in glob for c in "*?[")


def may_overlap(left: str, right: str) -> bool:
    if left.startswith("/") and not broad(left):
        return match(right, left[1:])
    if right.startswith("/") and not broad(right):
        return match(left, right[1:])
    prefixes = []
    for glob in (left, right):
        prefix = glob if glob.startswith("/") else ""
        for char in "*?[":
            prefix = prefix.split(char, 1)[0]
        prefixes.append(prefix)
    return prefixes[0].startswith(prefixes[1]) or prefixes[1].startswith(prefixes[0])


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
    # Compare surviving pattern order separately from owner values. An unrelated
    # tracked transfer must not conceal a changed, currently empty intersection.
    old_order = list(dict.fromkeys(g for g, _ in reversed(old_rules)))[::-1]
    new_order = list(dict.fromkeys(g for g, _ in reversed(new_rules)))[::-1]
    old_pos, new_pos = {g: i for i, g in enumerate(old_order)}, {
        g: i for i, g in enumerate(new_order)
    }
    common = sorted(old_globs & new_globs)
    for i, left in enumerate(common):
        for right in common[i + 1 :]:
            if (
                (old_pos[left] < old_pos[right]) != (new_pos[left] < new_pos[right])
                and may_overlap(left, right)
                and not (
                    old_map[left] | old_map[right] | new_map[left] | new_map[right]
                )
                <= required
            ):
                required.add(PROCESS)
    if any(broad(glob) for glob in changed_scopes):
        required.add(PROCESS)
    return required, changes


def approved_reviewers(reviews: list[dict], author: str) -> dict[str, dict]:
    """Comments preserve the last decisive review; dismissals revoke it."""
    latest = {}
    for review in sorted(reviews, key=lambda r: (r.get("submitted_at") or "", r["id"])):
        user = review["user"]
        login = user["login"].lower()
        if login == author.lower() or user["type"] != "User":
            continue
        if review["state"] in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            latest[login] = review
    return {login: r for login, r in latest.items() if r["state"] == "APPROVED"}


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
