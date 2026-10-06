# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare extracted contract observations, not runtime support claims."""

from __future__ import annotations

from ..common.contracts import Contract, FieldContract, Finding, finding

ASPECTS = ("wire_type", "required", "nullable", "default", "constraints")


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
                        original.contract_facts(),
                        [item.contract_facts() for item in candidates],
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
                            original.contract_facts(),
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
                unresolved = not local.fields_complete or wildcard is not None
                findings.append(
                    finding(
                        endpoint,
                        path,
                        "input_slot",
                        "No typed input slot; passthrough/custom input constraints are unresolved."
                        if unresolved
                        else "No matching Dynamo input declaration identified.",
                        original.contract_facts(),
                        {
                            "additional_properties": wildcard is not None,
                            "fields_complete": local.fields_complete,
                        },
                        category="coverage" if unresolved else "compatibility",
                        sources=original.source,
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
                        current.contract_facts(),
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
                native.contract_facts(),
                dynamo.contract_facts(),
                category=category,
                sources=native.source + dynamo.source,
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
    return results


def investigation_notes(native: Contract, dynamo: Contract) -> list[Finding]:
    """Source observations are advisory, never contract or runtime verdicts."""
    notes = []
    for endpoint, upstream in sorted(native.endpoints.items()):
        local = dynamo.endpoints.get(endpoint)
        if local is None:
            continue
        for path, original in sorted(upstream.fields.items()):
            matches = [
                item
                for item in local.fields.values()
                if set(item.wire_names) & set(original.wire_names)
            ]
            current = matches[0] if len(matches) == 1 else local.fields.get(path)
            handling = (
                current.handling
                if current
                else local.untyped_handling.get(path, local.additional_properties)
            )
            notes.append(
                finding(
                    endpoint,
                    path,
                    "handling",
                    "Source investigation only; invocation and runtime effects are unverified.",
                    original.contract_facts(),
                    handling.facts() if handling else None,
                    category="investigation",
                    sources=original.source + (handling.evidence if handling else []),
                )
            )
            notes[-1].decision_status = "advisory"
            notes[
                -1
            ].next_action = (
                "Use source references to select or investigate behavioral tests."
            )
    return notes
