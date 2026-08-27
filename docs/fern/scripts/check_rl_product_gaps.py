#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the prioritized, issue-ready Dynamo RL product-gap register."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA = "dynamo.rl.product-gaps.v1"
GAP_ID = re.compile(r"DYN-RL-GAP-[0-9]{3}")
COMMIT = re.compile(r"[0-9a-f]{40}")
PRIORITIES = {"P0": 0, "P1": 1, "P2": 2}
STATUSES = {"proposed", "accepted", "in_progress", "resolved", "wont_fix"}
PROPOSAL_STATUSES = {"issue_ready", "decision_required", "filed"}
DECISION_VEHICLES = {"implementation_issue", "DEP"}
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = Path(__file__).with_name("rl_product_gaps.json")
EVIDENCE_MANIFEST = Path(__file__).with_name("rl_evidence.json")

GAP_SHAPE: dict[str, Any] = {
    "id": str,
    "title": str,
    "priority": str,
    "status": str,
    "proposal_status": str,
    "decision_vehicle": str,
    "owner_team": str,
    "problem": str,
    "current_boundary": str,
    "desired_contract": str,
    "affected_requirements": list,
    "depends_on": list,
    "source_assertions": list,
    "affected_docs": list,
    "acceptance_evidence": list,
    "docs_behavior": str,
    "expiration_trigger": str,
}
DECISION_SHAPE: dict[str, Any] = {
    "decision_id": str,
    "status": str,
    "current_scope": str,
    "decision_vehicle": str,
    "package_owner": str,
    "owner_team": str,
    "rationale": str,
    "prerequisite_gap_ids": list,
    "acceptance_conditions": list,
}


class GapRegistryError(ValueError):
    """Raised when a registry file cannot be loaded."""


def load_registry(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GapRegistryError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GapRegistryError(f"{path} must contain a JSON object")
    return value


def _check_shape(
    value: Any, shape: dict[str, Any], location: str, findings: list[str]
) -> None:
    if not isinstance(value, dict):
        findings.append(f"{location} must be an object")
        return
    for key, expected in shape.items():
        child_location = f"{location}.{key}" if location else key
        if key not in value:
            findings.append(f"{child_location} is required")
        elif not isinstance(value[key], expected):
            findings.append(f"{child_location} must be {expected.__name__}")


def _nonempty(value: Any, location: str, findings: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        findings.append(f"{location} must be a non-empty string")


def _string_list(
    value: Any,
    location: str,
    findings: list[str],
    *,
    minimum: int = 0,
    sorted_unique: bool = False,
) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        findings.append(f"{location} must contain only non-empty strings")
        return []
    if len(value) < minimum:
        findings.append(f"{location} must contain at least {minimum} items")
    if sorted_unique and value != sorted(set(value)):
        findings.append(f"{location} must be sorted and unique")
    return value


def _requirements(value: Any, location: str, findings: list[str]) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        findings.append(f"{location} must contain requirement integers")
        return
    if value != sorted(set(value)):
        findings.append(f"{location} must be sorted and unique")
    invalid = [item for item in value if item < 1 or item > 10]
    if invalid:
        findings.append(f"{location} contains values outside 1..10: {invalid}")


def _repo_file(raw_path: Any, location: str, findings: list[str]) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        findings.append(f"{location}.path must be a non-empty string")
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        findings.append(f"{location}.path must be repository-relative")
        return None
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError:
        findings.append(f"{location}.path escapes the repository")
        return None
    if not path.is_file():
        findings.append(f"{location}.path does not exist: {raw_path}")
        return None
    return path


def _assertions(
    value: Any,
    location: str,
    findings: list[str],
    *,
    require_contains: bool,
) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) for item in value)
    ):
        findings.append(f"{location} must contain at least one assertion object")
        return
    for index, assertion in enumerate(value):
        item_location = f"{location}[{index}]"
        path = _repo_file(assertion.get("path"), item_location, findings)
        contains = _string_list(
            assertion.get("contains", []), f"{item_location}.contains", findings
        )
        not_contains = _string_list(
            assertion.get("not_contains", []),
            f"{item_location}.not_contains",
            findings,
        )
        if require_contains and not contains:
            findings.append(f"{item_location}.contains must not be empty")
        elif not contains and not not_contains:
            findings.append(f"{item_location} must check contains or not_contains")
        if path is None:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in contains:
            if needle not in text:
                findings.append(
                    f"{item_location}: {assertion['path']} no longer contains {needle!r}"
                )
        for needle in not_contains:
            if needle in text:
                findings.append(
                    f"{item_location}: {assertion['path']} now contains gap-closing signal {needle!r}"
                )


def _dependency_findings(gaps: list[dict[str, Any]], findings: list[str]) -> None:
    by_id = {gap["id"]: gap for gap in gaps if isinstance(gap.get("id"), str)}
    for gap in gaps:
        gap_id = gap["id"]
        dependencies = gap["depends_on"]
        for dependency in dependencies:
            if dependency == gap_id:
                findings.append(f"{gap_id}.depends_on must not contain itself")
            elif dependency not in by_id:
                findings.append(
                    f"{gap_id}.depends_on references unknown gap {dependency}"
                )
            elif PRIORITIES.get(by_id[dependency]["priority"], 99) > PRIORITIES.get(
                gap["priority"], -1
            ):
                findings.append(
                    f"{gap_id} cannot have higher urgency than dependency {dependency}"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(gap_id: str, chain: list[str]) -> None:
        if gap_id in visiting:
            findings.append("dependency cycle: " + " -> ".join([*chain, gap_id]))
            return
        if gap_id in visited or gap_id not in by_id:
            return
        visiting.add(gap_id)
        for dependency in by_id[gap_id]["depends_on"]:
            visit(dependency, [*chain, gap_id])
        visiting.remove(gap_id)
        visited.add(gap_id)

    for gap_id in sorted(by_id):
        visit(gap_id, [])


def _baseline_findings(registry: dict[str, Any], findings: list[str]) -> None:
    commit = registry.get("baseline_dynamo_commit")
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        findings.append("baseline_dynamo_commit must be a full lowercase commit SHA")
        return
    try:
        manifest = json.loads(EVIDENCE_MANIFEST.read_text(encoding="utf-8"))
        evidence_commit = manifest["baseline"]["dynamo_commit"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        findings.append(f"cannot read evidence-manifest baseline: {exc}")
        return
    if commit != evidence_commit:
        findings.append(
            "baseline_dynamo_commit must match rl_evidence.json baseline.dynamo_commit"
        )


def validate_registry(registry: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if registry.get("schema") != SCHEMA:
        findings.append(f"schema must be {SCHEMA}")
    if registry.get("register_state") != "issue_ready":
        findings.append("register_state must be issue_ready")
    reviewed_on = registry.get("reviewed_on")
    try:
        date.fromisoformat(reviewed_on)
    except (TypeError, ValueError):
        findings.append("reviewed_on must be an ISO-8601 date")
    _baseline_findings(registry, findings)

    priority_policy = registry.get("priority_policy")
    if not isinstance(priority_policy, dict) or set(priority_policy) != set(PRIORITIES):
        findings.append("priority_policy must define exactly P0, P1, and P2")
    else:
        for priority in PRIORITIES:
            _nonempty(
                priority_policy[priority], f"priority_policy.{priority}", findings
            )

    decision = registry.get("closed_loop_decision")
    _check_shape(decision, DECISION_SHAPE, "closed_loop_decision", findings)
    gaps = registry.get("gaps")
    if (
        not isinstance(gaps, list)
        or not gaps
        or any(not isinstance(gap, dict) for gap in gaps)
    ):
        findings.append("gaps must contain at least one object")
        return findings
    for index, gap in enumerate(gaps):
        _check_shape(gap, GAP_SHAPE, f"gaps[{index}]", findings)
    if findings:
        return findings

    ids = [gap["id"] for gap in gaps]
    if len(ids) != len(set(ids)):
        findings.append("gap IDs must be unique")
    titles = [gap["title"] for gap in gaps]
    if len(titles) != len(set(titles)):
        findings.append("gap titles must be unique")
    for index, gap in enumerate(gaps):
        location = f"gaps[{index}]"
        if not GAP_ID.fullmatch(gap["id"]):
            findings.append(f"{location}.id must match DYN-RL-GAP-NNN")
        if gap["priority"] not in PRIORITIES:
            findings.append(f"{location}.priority must be P0, P1, or P2")
        if gap["status"] not in STATUSES:
            findings.append(f"{location}.status is invalid")
        if gap["proposal_status"] not in PROPOSAL_STATUSES:
            findings.append(f"{location}.proposal_status is invalid")
        if gap["decision_vehicle"] not in DECISION_VEHICLES:
            findings.append(
                f"{location}.decision_vehicle must be implementation_issue or DEP"
            )
        for field in (
            "title",
            "owner_team",
            "problem",
            "current_boundary",
            "desired_contract",
            "docs_behavior",
            "expiration_trigger",
        ):
            _nonempty(gap[field], f"{location}.{field}", findings)
        _requirements(
            gap["affected_requirements"],
            f"{location}.affected_requirements",
            findings,
        )
        _string_list(
            gap["depends_on"],
            f"{location}.depends_on",
            findings,
            sorted_unique=True,
        )
        _string_list(
            gap["acceptance_evidence"],
            f"{location}.acceptance_evidence",
            findings,
            minimum=4,
        )
        _assertions(
            gap["source_assertions"],
            f"{location}.source_assertions",
            findings,
            require_contains=False,
        )
        _assertions(
            gap["affected_docs"],
            f"{location}.affected_docs",
            findings,
            require_contains=True,
        )
    _dependency_findings(gaps, findings)

    if decision["status"] != "follow_on_dep_required":
        findings.append("closed_loop_decision.status must be follow_on_dep_required")
    if decision["current_scope"] != "request_plane_only":
        findings.append("closed_loop_decision.current_scope must be request_plane_only")
    if decision["decision_vehicle"] != "DEP":
        findings.append("closed_loop_decision.decision_vehicle must be DEP")
    if decision["package_owner"] != "unassigned":
        findings.append(
            "closed_loop_decision.package_owner must remain unassigned until DEP approval"
        )
    for field in ("decision_id", "owner_team", "rationale"):
        _nonempty(decision[field], f"closed_loop_decision.{field}", findings)
    prerequisites = _string_list(
        decision["prerequisite_gap_ids"],
        "closed_loop_decision.prerequisite_gap_ids",
        findings,
        minimum=1,
        sorted_unique=True,
    )
    _string_list(
        decision["acceptance_conditions"],
        "closed_loop_decision.acceptance_conditions",
        findings,
        minimum=4,
    )
    by_id = {gap["id"]: gap for gap in gaps}
    closed_loop_gap = by_id.get("DYN-RL-GAP-005")
    if closed_loop_gap is None:
        findings.append("DYN-RL-GAP-005 must record the closed-loop package gap")
    else:
        if prerequisites != closed_loop_gap["depends_on"]:
            findings.append(
                "closed_loop_decision prerequisites must equal DYN-RL-GAP-005 dependencies"
            )
        if (
            closed_loop_gap["priority"] != "P2"
            or closed_loop_gap["proposal_status"] != "decision_required"
            or closed_loop_gap["decision_vehicle"] != "DEP"
        ):
            findings.append(
                "DYN-RL-GAP-005 must remain a P2 decision-required DEP proposal"
            )
    unknown_prerequisites = sorted(set(prerequisites) - set(by_id))
    if unknown_prerequisites:
        findings.append(
            "closed_loop_decision references unknown prerequisites: "
            + ", ".join(unknown_prerequisites)
        )
    return findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path, nargs="?", default=DEFAULT_REGISTRY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        registry = load_registry(args.registry)
    except GapRegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    findings = validate_registry(registry)
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}", file=sys.stderr)
        return 1
    counts = {
        priority: sum(gap["priority"] == priority for gap in registry["gaps"])
        for priority in PRIORITIES
    }
    print(
        "RL product-gap register passed "
        f"({len(registry['gaps'])} gaps; "
        + "; ".join(f"{priority}={counts[priority]}" for priority in PRIORITIES)
        + ")."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
