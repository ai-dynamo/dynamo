# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply reviewed dispositions and evaluate separate static gates."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..common.contracts import Contract, Finding
from ..inputs.validation import validate_policy, validate_previous, validate_registry
from .comparison import direct_compare, investigation_notes
from .lifecycle import classify, contract_history

DISPOSITIONS = frozenset(
    {
        "tracked_gap",
        "intentional_divergence",
        "unsupported",
        "non_material",
        "needs_runtime_evidence",
    }
)


def apply_decisions(
    findings: list[Finding],
    registry: dict[str, Any],
    revisions: dict[str, str],
    scope: dict[str, Any],
) -> None:
    """Exact fact/scope matching; old B1 pair decisions are deliberately rejected."""
    validate_registry(registry, DISPOSITIONS)
    decisions = {record["identity"]: record for record in registry["decisions"]}
    for item in findings:
        record = decisions.get(item.identity)
        if record is None:
            continue
        item.decision = record
        if record["fingerprint"] != item.fingerprint:
            item.decision_status = "stale_facts"
        elif record["scope"] != scope:
            item.decision_status = "stale_scope"
        elif record["reviewed_revisions"] != revisions and not record.get(
            "carry_static_disposition", False
        ):
            item.decision_status = "stale_revisions"
        else:
            item.decision_status = "applicable"
            item.next_action = record["next_action"]
        # Runtime evidence is never promoted across revision/config boundaries,
        # even when the reviewer permits a static disposition to follow facts.
        item.decision = {
            **record,
            "current_runtime_evidence": [
                evidence
                for evidence in record["evidence"]
                if item.decision_status == "applicable"
                and evidence.get("kind") == "runtime"
                and evidence.get("revisions") == revisions
                and evidence.get("scope") == scope
            ],
        }


def build_assessment(
    native: Contract,
    dynamo: Contract,
    *,
    scope: dict[str, Any],
    previous: dict[str, Any] | None = None,
    decisions: dict[str, Any] | None = None,
    behavior: list[Finding] | None = None,
    policy: dict[str, Any] | None = None,
    investigate: bool = True,
) -> dict[str, Any]:
    if previous is not None:
        validate_previous(previous, scope)
    previous = contract_history(previous)
    findings = direct_compare(native, dynamo)
    notes = (
        investigation_notes(native, dynamo) + (behavior or []) if investigate else []
    )
    old_notes = {
        item["identity"]: item
        for item in (
            (
                previous.get("investigation", {}).get("notes", [])
                + previous.get("investigation", {}).get("unobserved_notes", [])
            )
            if previous
            else []
        )
    }
    for note in notes:
        note.decision_status = "advisory"
        note.next_action = (
            "Use source references to select or investigate behavioral tests."
        )
        prior = old_notes.get(note.identity)
        if prior:
            note.lifecycle = (
                "unchanged" if prior["fingerprint"] == note.fingerprint else "changed"
            )
    retired = classify(findings, previous, native, dynamo)
    revisions = {"dynamo": dynamo.revision, "vllm": native.revision}
    if decisions is not None:
        apply_decisions(findings, decisions, revisions, scope)
    required = [item for item in findings if item.category != "dynamo_specific"]
    pending = [
        item.identity for item in required if item.decision_status != "applicable"
    ]
    lost = [
        item["identity"]
        for item in retired
        if item["lifecycle"] == "no_longer_assessable"
    ]
    complete = (
        native.complete()
        and dynamo.complete()
        and not any(item.category == "coverage" for item in findings)
    )
    release: dict[str, Any] = {
        "status": "not_evaluated",
        "reason": "No agreed support policy supplied.",
    }
    if policy is not None:
        validate_policy(policy)
        blocked = [
            item.identity
            for item in required
            if item.identity in policy["required_findings_absent"]
        ]
        release = {
            "status": "blocked"
            if blocked or pending or lost or not complete
            else "static_policy_satisfied",
            "blocking_findings": blocked,
            "runtime_acceptance": "not_evaluated",
        }
    return {
        "schema": "dynamo-native-assessment/v2",
        "history_migration": previous.get("history_migration") if previous else None,
        "revisions": revisions,
        "previous_revisions": previous["revisions"] if previous else None,
        "scope": scope,
        "findings": [asdict(item) for item in findings],
        "retired_findings": retired,
        "gates": {
            "layer": "contract",
            "extraction": "complete" if complete else "incomplete",
            "review": "complete" if not pending and not lost else "action_required",
            "pending": pending,
            "lost_coverage": lost,
            "runtime_conformance": "not_established_by_static_assessment",
            "release": release,
        },
        "investigation": {
            "status": "available" if investigate else "disabled",
            "gating": False,
            "notes": [asdict(item) for item in notes],
            "unobserved_notes": [
                item
                for identity, item in old_notes.items()
                if identity not in {note.identity for note in notes}
            ],
        },
        "behavioral_conformance": {
            "status": "not_assessed",
            "reason": "Run a separately scoped behavioral test suite; source inspection is not execution evidence.",
        },
        "exit_code": 1
        if pending or lost or not complete or release["status"] == "blocked"
        else 0,
        "contracts": {"native": native.to_dict(), "dynamo": dynamo.to_dict()},
    }
