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
