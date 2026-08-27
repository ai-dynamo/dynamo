#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate an independent clean-room review of the RL documentation journey."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "dynamo.rl.clean-room-review.v1"
COMMIT = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
GUIDE_PREFIX = "docs/fern/pages/use-cases/reinforcement-learning/"
OWNER_ROLES = (
    "program_dri",
    "framework",
    "dynamo_contract",
    "routing",
    "weight_updates",
    "observability",
    "replay_simulation",
)
JOURNEY_GATES = (
    "navigation_and_pin",
    "clean_install_and_launch",
    "generation_and_training",
    "weight_update_and_recovery",
    "observability_and_diagnosis",
    "replay_and_simulation",
    "troubleshooting_and_security",
)

OWNER_SHAPE = {"name": str, "github": str, "accepted": bool}
LINKED_RECORD_SHAPE = {
    "record_id": str,
    "uri": str,
    "sha256": str,
    "checker_command": str,
    "checker_output_artifact": str,
    "publication_gate_passed": bool,
}
GATE_SHAPE = {"status": str, "conclusion": str, "artifacts": list}
REQUIRED_SHAPE: dict[str, Any] = {
    "schema": str,
    "record_state": str,
    "review_id": str,
    "scope": {
        "guide_path": str,
        "framework_name": str,
        "maturity_target": str,
        "integration_artifact": str,
        "recipe_commit": str,
        "core_commit": str,
        "dynamo_commit": str,
    },
    "linked_records": {
        "framework_validation": LINKED_RECORD_SHAPE,
        "program_evidence": LINKED_RECORD_SHAPE,
    },
    "reviewer": {
        "name": str,
        "github": str,
        "organization": str,
        "independence_attested": bool,
        "conflicts": list,
    },
    "owners": {role: OWNER_SHAPE for role in OWNER_ROLES},
    "environment": {
        "fresh_workspace": bool,
        "base_image": str,
        "base_image_digest": str,
        "model_name": str,
        "model_revision": str,
        "hardware_summary": str,
        "preexisting_dependencies": list,
    },
    "run": {
        "started_at": str,
        "completed_at": str,
        "entry_page": str,
        "navigation_clicks": int,
        "commands_executed": list,
        "undocumented_steps": list,
        "artifact_root": str,
    },
    "journey": {gate: GATE_SHAPE for gate in JOURNEY_GATES},
    "findings": list,
    "broken_links": {
        "command": str,
        "rl_errors": int,
        "unrelated_errors": int,
        "baseline_decision": str,
        "waiver_owner": str,
        "waiver_expires_at": str,
        "artifact": str,
    },
    "decision": {
        "outcome": str,
        "summary": str,
        "limitations": list,
        "signed_at": str,
        "artifact": str,
    },
    "last_validated": str,
}


class RecordError(ValueError):
    """Raised when a clean-room record cannot be loaded."""


def load_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordError(f"cannot load {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise RecordError(f"{path} must contain a JSON object")
    return record


def _matches_type(value: Any, expected: type | tuple[type, ...]) -> bool:
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, tuple) and int in expected and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _check_shape(value: Any, expected: Any, location: str, findings: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(value, dict):
            findings.append(f"{location} must be an object")
            return
        for key, child in expected.items():
            child_location = f"{location}.{key}" if location else key
            if key not in value:
                findings.append(f"{child_location} is required")
            else:
                _check_shape(value[key], child, child_location, findings)
        return
    if not _matches_type(value, expected):
        names = (
            ", ".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        findings.append(f"{location} must be {names}")


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _nonempty(value: str, location: str, findings: list[str]) -> None:
    if not value.strip():
        findings.append(f"{location} must be recorded for publication")


def _string_list(
    value: Any,
    location: str,
    findings: list[str],
    *,
    require_nonempty: bool = False,
) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        findings.append(f"{location} must contain only non-empty strings")
    elif require_nonempty and not value:
        findings.append(f"{location} must contain at least one item")


def _artifact_list(value: Any, location: str, findings: list[str]) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        findings.append(f"{location} must contain at least one artifact URI")


def _validate_finding_shape(finding: Any, index: int, findings: list[str]) -> None:
    _check_shape(
        finding,
        {
            "id": str,
            "severity": str,
            "status": str,
            "description": str,
            "resolution": str,
            "owner": str,
            "artifact": str,
        },
        f"findings[{index}]",
        findings,
    )


def validate_structure(record: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    _check_shape(record, REQUIRED_SHAPE, "", findings)
    if findings:
        return findings
    if record["schema"] != SCHEMA:
        findings.append(f"schema must be {SCHEMA!r}")
    if record["record_state"] not in {"planned", "passed", "failed"}:
        findings.append("record_state must be planned, passed, or failed")
    for gate_name, gate in record["journey"].items():
        if gate["status"] not in {"not_run", "passed", "failed"}:
            findings.append(
                f"journey.{gate_name}.status must be not_run, passed, or failed"
            )
    for index, finding in enumerate(record["findings"]):
        _validate_finding_shape(finding, index, findings)
        if isinstance(finding, dict):
            if finding.get("severity") not in {"blocking", "major", "minor", "nit"}:
                findings.append(
                    f"findings[{index}].severity must be blocking, major, minor, or nit"
                )
            if finding.get("status") not in {"open", "resolved", "waived"}:
                findings.append(
                    f"findings[{index}].status must be open, resolved, or waived"
                )
    for location in (
        "run.started_at",
        "run.completed_at",
        "broken_links.waiver_expires_at",
        "decision.signed_at",
        "last_validated",
    ):
        value: Any = record
        for part in location.split("."):
            value = value[part]
        if value and _parse_time(value) is None:
            findings.append(f"{location} must be ISO-8601 with a UTC offset when set")
    return findings


def _common_publication(record: dict[str, Any], findings: list[str]) -> None:
    if record["record_state"] != "passed":
        findings.append("record_state must be passed for publication")
    _nonempty(record["review_id"], "review_id", findings)
    scope = record["scope"]
    if not scope["guide_path"].startswith(GUIDE_PREFIX) or not scope[
        "guide_path"
    ].endswith(".md"):
        findings.append("scope.guide_path must identify an RL Markdown guide")
    for field in ("framework_name", "maturity_target", "integration_artifact"):
        _nonempty(scope[field], f"scope.{field}", findings)
    if scope["maturity_target"] not in {"experimental", "supported"}:
        findings.append("scope.maturity_target must be experimental or supported")
    for field in ("recipe_commit", "core_commit", "dynamo_commit"):
        if not COMMIT.fullmatch(scope[field]):
            findings.append(f"scope.{field} must be a full lowercase commit SHA")


def _linked_records_publication(record: dict[str, Any], findings: list[str]) -> None:
    for name, linked in record["linked_records"].items():
        location = f"linked_records.{name}"
        for field in ("record_id", "uri", "checker_command", "checker_output_artifact"):
            _nonempty(linked[field], f"{location}.{field}", findings)
        if not SHA256.fullmatch(linked["sha256"]):
            findings.append(f"{location}.sha256 must be a full lowercase digest")
        if "--publication-gate" not in linked["checker_command"]:
            findings.append(f"{location}.checker_command must use --publication-gate")
        if not linked["publication_gate_passed"]:
            findings.append(f"{location}.publication_gate_passed must be true")


def _ownership_publication(record: dict[str, Any], findings: list[str]) -> None:
    reviewer = record["reviewer"]
    for field in ("name", "github", "organization"):
        _nonempty(reviewer[field], f"reviewer.{field}", findings)
    if not reviewer["independence_attested"]:
        findings.append("reviewer.independence_attested must be true")
    if reviewer["conflicts"]:
        findings.append("reviewer.conflicts must be empty")
    reviewer_identities = {
        reviewer["name"].strip().lower(),
        reviewer["github"].strip().lower(),
    } - {""}
    for role, owner in record["owners"].items():
        location = f"owners.{role}"
        for field in ("name", "github"):
            _nonempty(owner[field], f"{location}.{field}", findings)
        if not owner["accepted"]:
            findings.append(f"{location}.accepted must be true")
        owner_identities = {
            owner["name"].strip().lower(),
            owner["github"].strip().lower(),
        } - {""}
        if reviewer_identities.intersection(owner_identities):
            findings.append(f"reviewer must be independent of {location}")


def _run_publication(record: dict[str, Any], findings: list[str]) -> None:
    environment = record["environment"]
    if not environment["fresh_workspace"]:
        findings.append("environment.fresh_workspace must be true")
    for field in ("base_image", "model_name", "model_revision", "hardware_summary"):
        _nonempty(environment[field], f"environment.{field}", findings)
    if not IMAGE_DIGEST.fullmatch(environment["base_image_digest"]):
        findings.append(
            "environment.base_image_digest must be an immutable sha256 digest"
        )
    _string_list(
        environment["preexisting_dependencies"],
        "environment.preexisting_dependencies",
        findings,
    )

    run = record["run"]
    started = _parse_time(run["started_at"])
    completed = _parse_time(run["completed_at"])
    if started is None:
        findings.append("run.started_at must be a recorded timestamp with offset")
    if completed is None:
        findings.append("run.completed_at must be a recorded timestamp with offset")
    if started is not None and completed is not None and completed < started:
        findings.append("run.completed_at must not precede started_at")
    if run["entry_page"] != record["scope"]["guide_path"]:
        findings.append("run.entry_page must equal scope.guide_path")
    if run["navigation_clicks"] < 0 or run["navigation_clicks"] > 2:
        findings.append("run.navigation_clicks must be between 0 and 2")
    _string_list(
        run["commands_executed"],
        "run.commands_executed",
        findings,
        require_nonempty=True,
    )
    if run["undocumented_steps"]:
        findings.append("run.undocumented_steps must be empty")
    _nonempty(run["artifact_root"], "run.artifact_root", findings)


def _journey_publication(record: dict[str, Any], findings: list[str]) -> None:
    for gate_name, gate in record["journey"].items():
        location = f"journey.{gate_name}"
        if gate["status"] != "passed":
            findings.append(f"{location}.status must be passed")
        _nonempty(gate["conclusion"], f"{location}.conclusion", findings)
        _artifact_list(gate["artifacts"], f"{location}.artifacts", findings)


def _findings_publication(record: dict[str, Any], findings: list[str]) -> None:
    for index, item in enumerate(record["findings"]):
        location = f"findings[{index}]"
        for field in ("id", "description", "resolution", "owner", "artifact"):
            _nonempty(item[field], f"{location}.{field}", findings)
        if item["status"] == "open":
            findings.append(f"{location}.status must not be open")
        if item["severity"] in {"blocking", "major"} and item["status"] != "resolved":
            findings.append(f"{location} blocking/major finding must be resolved")


def _decision_publication(record: dict[str, Any], findings: list[str]) -> None:
    links = record["broken_links"]
    _nonempty(links["command"], "broken_links.command", findings)
    if links["rl_errors"] != 0:
        findings.append("broken_links.rl_errors must be zero")
    if links["unrelated_errors"] < 0:
        findings.append("broken_links.unrelated_errors must be nonnegative")
    if links["unrelated_errors"] == 0:
        if links["baseline_decision"] != "resolved":
            findings.append(
                "broken_links.baseline_decision must be resolved when unrelated_errors is zero"
            )
    else:
        if links["baseline_decision"] != "waived_with_owner":
            findings.append(
                "broken_links.baseline_decision must be waived_with_owner when unrelated errors remain"
            )
        _nonempty(links["waiver_owner"], "broken_links.waiver_owner", findings)
        expiry = _parse_time(links["waiver_expires_at"])
        if expiry is None:
            findings.append(
                "broken_links.waiver_expires_at must be a recorded timestamp with offset"
            )
        signed = _parse_time(record["decision"]["signed_at"])
        if expiry is not None and signed is not None and expiry <= signed:
            findings.append("broken_links waiver must expire after decision.signed_at")
    _nonempty(links["artifact"], "broken_links.artifact", findings)

    decision = record["decision"]
    if decision["outcome"] != "approved":
        findings.append("decision.outcome must be approved")
    _nonempty(decision["summary"], "decision.summary", findings)
    _string_list(decision["limitations"], "decision.limitations", findings)
    signed = _parse_time(decision["signed_at"])
    if signed is None:
        findings.append("decision.signed_at must be a recorded timestamp with offset")
    completed = _parse_time(record["run"]["completed_at"])
    if signed is not None and completed is not None and signed < completed:
        findings.append("decision.signed_at must not precede run.completed_at")
    _nonempty(decision["artifact"], "decision.artifact", findings)
    validated = _parse_time(record["last_validated"])
    if validated is None:
        findings.append("last_validated must be a recorded timestamp with offset")
    if validated is not None and signed is not None and validated < signed:
        findings.append("last_validated must not precede decision.signed_at")


def publication_findings(record: dict[str, Any]) -> list[str]:
    findings = validate_structure(record)
    if findings:
        return findings
    _common_publication(record, findings)
    _linked_records_publication(record, findings)
    _ownership_publication(record, findings)
    _run_publication(record, findings)
    _journey_publication(record, findings)
    _findings_publication(record, findings)
    _decision_publication(record, findings)
    return findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--publication-gate", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        record = load_record(args.record)
    except RecordError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    findings = (
        publication_findings(record)
        if args.publication_gate
        else validate_structure(record)
    )
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}", file=sys.stderr)
        return 1
    mode = "publication" if args.publication_gate else "structure"
    print(f"RL clean-room review passed ({mode}; schema={SCHEMA}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
