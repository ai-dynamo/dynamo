# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct comparison, baseline lifecycle, and separate review/release gates.

This module compares extracted observations, not implementation support lists.
Decisions belong to a separate registry; historical reports never grant approval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from protocol_assessment_inputs import (
    validate_policy,
    validate_previous,
    validate_registry,
)
from protocol_contract import Contract, FieldContract
from protocol_drift import digest

ASPECTS = ("wire_type", "required", "nullable", "default", "constraints")
DISPOSITIONS = frozenset(
    {
        "tracked_gap",
        "intentional_divergence",
        "unsupported",
        "non_material",
        "needs_runtime_evidence",
    }
)


@dataclass
class Finding:
    identity: str
    endpoint: str
    path: str
    aspect: str
    observation: str
    fingerprint: str
    native: Any
    dynamo: Any
    category: str = "compatibility"
    lifecycle: str = "new"
    decision: dict[str, Any] | None = None
    decision_status: str = "missing"
    next_action: str = "Review the observed difference and record a disposition."
    sources: list[dict[str, Any]] = field(default_factory=list)


def finding(
    endpoint: str,
    path: str,
    aspect: str,
    observation: str,
    native: Any,
    dynamo: Any,
    *,
    category: str = "compatibility",
    sources: list[dict[str, Any]] | None = None,
) -> Finding:
    return Finding(
        identity=f"{endpoint}#{path}:{aspect}",
        endpoint=endpoint,
        path=path,
        aspect=aspect,
        observation=observation,
        fingerprint=digest({"native": native, "dynamo": dynamo}),
        native=native,
        dynamo=dynamo,
        category=category,
        sources=sources or [],
    )


def direct_compare(native: Contract, dynamo: Contract) -> list[Finding]:
    """Always assess the baseline, including when neither revision changed."""
    if native.target != "vllm" or dynamo.target != "dynamo":
        raise ValueError("initial adapter supports only Dynamo versus vLLM")
    findings = []
    for endpoint in sorted(set(native.endpoints) | set(dynamo.endpoints)):
        upstream = native.endpoints.get(endpoint)
        local = dynamo.endpoints.get(endpoint)
        if upstream is None or local is None:
            findings.append(
                finding(
                    endpoint,
                    "*",
                    "endpoint",
                    "Endpoint extraction unavailable.",
                    upstream is not None,
                    local is not None,
                    category="coverage",
                )
            )
            continue
        matched = set()
        for path, original in sorted(upstream.fields.items()):
            # Aliases identify the same input slot; field placement remains
            # explicit in each wire name (e.g. nvext.ignore_eos versus ignore_eos).
            candidates = [
                item
                for item in local.fields.values()
                if set(item.wire_names) & set(original.wire_names)
            ]
            if not candidates and path in local.fields:
                candidates = [local.fields[path]]
            if len(candidates) > 1:
                findings.append(
                    finding(
                        endpoint,
                        path,
                        "ambiguous_input",
                        "Multiple Dynamo input slots match.",
                        original.facts(),
                        [item.facts() for item in candidates],
                        category="coverage",
                    )
                )
                matched.update(item.path for item in candidates)
                continue
            if not candidates:
                nested = [
                    item
                    for item in local.nested_inputs
                    if item["wire_path"][-1] in original.wire_names
                ]
                if nested:
                    findings.append(
                        finding(
                            endpoint,
                            path,
                            "placement_candidate",
                            "No matching top-level declaration; same-named nested Dynamo inputs exist. "
                            "These are different public locations, not established aliases or semantic equivalents.",
                            original.facts(),
                            {
                                "nested_inputs": [
                                    {
                                        key: value
                                        for key, value in item.items()
                                        if key != "source"
                                    }
                                    for item in nested
                                ]
                            },
                            category="coverage",
                            sources=original.source
                            + [source for item in nested for source in item["source"]],
                        )
                    )
                wildcard = local.additional_properties
                admission = local.untyped_handling.get(path)
                if admission is not None:
                    findings.append(
                        finding(
                            endpoint,
                            path,
                            "handling",
                            "Explicit admission rejection for this untyped native input."
                            if "reject" in admission.effects
                            else "Untyped input has a conditional handling path.",
                            original.facts(),
                            admission.facts(),
                            category="compatibility"
                            if admission.complete
                            else "coverage",
                            sources=original.source + admission.evidence,
                        )
                    )
                    continue
                unresolved = not local.fields_complete or wildcard is not None
                findings.append(
                    finding(
                        endpoint,
                        path,
                        "handling",
                        "No typed input slot; passthrough/custom handling is unresolved."
                        if unresolved
                        else "No Dynamo input or handling path identified.",
                        original.facts(),
                        wildcard.facts() if wildcard else None,
                        category="coverage" if unresolved else "compatibility",
                        sources=original.source
                        + (wildcard.evidence if wildcard else []),
                    )
                )
                continue
            current = candidates[0]
            matched.add(current.path)
            findings.extend(compare_field(endpoint, original, current))
        for path, current in sorted(local.fields.items()):
            if path not in matched:
                findings.append(
                    finding(
                        endpoint,
                        path,
                        "dynamo_specific",
                        "Dynamo-only input; not by itself a compatibility defect."
                        if upstream.fields_complete
                        else "Native input inventory incomplete; ownership unresolved.",
                        None,
                        current.facts(),
                        category="dynamo_specific"
                        if upstream.fields_complete
                        else "coverage",
                        sources=current.source,
                    )
                )
        for side, contract in (("native", upstream), ("dynamo", local)):
            if not contract.fields_complete:
                findings.append(
                    finding(
                        endpoint,
                        "*",
                        f"{side}_fields",
                        "Field enumeration is incomplete.",
                        None,
                        {"side": side},
                        category="coverage",
                    )
                )
    for side, contract in (("native", native), ("dynamo", dynamo)):
        for item in contract.diagnostics:
            # Reasons participate in the fingerprint, not the logical identity.
            findings.append(
                finding(
                    item.endpoint,
                    item.path,
                    f"{side}_coverage_{item.aspect}",
                    item.reason,
                    None,
                    {"side": side, "reason": item.reason},
                    category="coverage",
                    sources=[item.source] if item.source else [],
                )
            )
    identities = [item.identity for item in findings]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "duplicate finding identity; aggregate diagnostics by field/aspect"
        )
    return findings


def compare_field(
    endpoint: str, native: FieldContract, dynamo: FieldContract
) -> list[Finding]:
    results = []

    def add(aspect: str, message: str, category: str = "compatibility") -> None:
        results.append(
            finding(
                endpoint,
                native.path,
                aspect,
                message,
                native.facts(),
                dynamo.facts(),
                category=category,
                sources=native.source
                + dynamo.source
                + (dynamo.handling.evidence if dynamo.handling else []),
            )
        )

    if not native.wire_names or not dynamo.wire_names:
        add(
            "input_names",
            "Accepted input names or placement are unresolved.",
            "coverage",
        )
    elif set(native.wire_names) != set(dynamo.wire_names):
        add("input_names", "Accepted input names or placement differ.")
    for aspect in ASPECTS:
        left, right = native.facts()[aspect], dynamo.facts()[aspect]
        if left is None or right is None:
            add(aspect, f"{aspect} comparison is unresolved.", "coverage")
        elif left != right:
            add(aspect, f"Declared {aspect} differs; intent is not inferred.")
    handling = dynamo.handling
    if handling is None or not handling.complete:
        add(
            "handling",
            "Identified effects: "
            + (", ".join(handling.effects) if handling and handling.effects else "none")
            + ". Handling coverage is incomplete; acceptance is not support.",
            "coverage",
        )
    elif "reject" in handling.effects:
        add("handling", "An explicit rejection path exists; inspect its conditions.")
    elif handling.conditions:
        add(
            "handling",
            "Handling is conditional; inspect pipeline/backend/version requirements.",
        )
    elif not handling.effects:
        add("handling", "No handling path identified.")
    return results


def assessable(
    old: dict[str, Any], current: list[Finding], native: Contract, dynamo: Contract
) -> bool:
    """A disappeared finding is resolved only if its comparison remains covered."""
    endpoint, path = old["endpoint"], old["path"]
    related_paths = {path}
    if old["aspect"] == "placement_candidate" and isinstance(old.get("dynamo"), dict):
        for item in old["dynamo"].get("nested_inputs", []):
            related_paths.add(".".join(item["wire_path"]))
    for contract in (native, dynamo):
        scope = contract.endpoints.get(endpoint)
        if scope is None or not scope.fields_complete:
            return False
    return not any(
        item.category == "coverage"
        and item.endpoint == endpoint
        and (
            item.path == "*"
            or any(
                item.path == related
                or related.startswith(item.path + ".")
                or item.path.startswith(related + ".")
                for related in related_paths
            )
        )
        for item in current
    )


def classify(
    findings: list[Finding],
    previous: dict[str, Any] | None,
    native: Contract,
    dynamo: Contract,
) -> list[dict[str, Any]]:
    """Return retired findings without erasing lost-coverage history."""
    if previous is None:
        return []
    if previous.get("schema") != "dynamo-native-assessment/v1":
        raise ValueError("previous assessment is not a direct comparison report")
    old = {
        item["identity"]: item
        for item in (
            previous["findings"]
            + [
                item
                for item in previous.get("retired_findings", [])
                if item["lifecycle"] == "no_longer_assessable"
            ]
        )
    }
    current_ids = {item.identity for item in findings}
    for item in findings:
        prior = old.get(item.identity)
        if prior is not None:
            item.lifecycle = (
                "unchanged" if prior["fingerprint"] == item.fingerprint else "changed"
            )
    return [
        {
            **item,
            "lifecycle": "resolved"
            if assessable(item, findings, native, dynamo)
            else "no_longer_assessable",
        }
        for identity, item in sorted(old.items())
        if identity not in current_ids
    ]


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
) -> dict[str, Any]:
    if previous is not None:
        validate_previous(previous, scope)
    findings = direct_compare(native, dynamo) + (behavior or [])
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
    complete = native.complete() and dynamo.complete()
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
        "schema": "dynamo-native-assessment/v1",
        "revisions": revisions,
        "previous_revisions": previous["revisions"] if previous else None,
        "scope": scope,
        "findings": [asdict(item) for item in findings],
        "retired_findings": retired,
        "gates": {
            "extraction": "complete" if complete else "incomplete",
            "review": "complete" if not pending and not lost else "action_required",
            "pending": pending,
            "lost_coverage": lost,
            "runtime_conformance": "not_established_by_static_assessment",
            "release": release,
        },
        "exit_code": 1
        if pending or lost or not complete or release["status"] == "blocked"
        else 0,
        "contracts": {"native": native.to_dict(), "dynamo": dynamo.to_dict()},
    }
