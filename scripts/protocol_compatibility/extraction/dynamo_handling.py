# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-derived admission/transport observations, independent of support claims.

Field sets come from runtime constants and the actual passthrough loop. Rule
bodies and capability resolvers are fingerprinted. A source-pattern mismatch
remains unresolved; a vocabulary entry alone never grants forwarding support.
"""

from __future__ import annotations

import json
import re

from scripts.protocol_compatibility.common.contracts import EndpointContract, Handling
from scripts.protocol_compatibility.extraction.rust_source import (
    RustItem,
    RustSources,
    group,
    split,
)


def named(sources: RustSources, name: str, suffix: str, kind: str) -> RustItem | None:
    matches = [
        item
        for item in sources.items
        if item.name == name and item.kind == kind and item.source.endswith(suffix)
    ]
    return matches[0] if len(matches) == 1 else None


def vocabulary(item: RustItem | None) -> set[str]:
    if item is None or "=" not in item.header:
        return set()
    value = item.header[item.header.index("=") + 1 :]
    if value[:2] != ["&", "["]:
        return set()
    body, end = group(value, 1)
    if end != len(value):
        return set()
    entries = split(body)
    if not all(len(entry) == 1 and entry[0].startswith('"') for entry in entries):
        return set()
    return {json.loads(entry[0]) for entry in entries}


def predicates(item: RustItem) -> list[str]:
    """Expose condition text, not an evaluated support decision."""
    result = []
    for index, token in enumerate(item.body):
        if token == "if":
            end = index + 1
            while end < len(item.body) and item.body[end] != "{":
                end += 1
            result.append(" ".join(item.body[index + 1 : end]))
    return result


def apply_admission(
    sources: RustSources, contract: EndpointContract, request: str
) -> None:
    if contract.additional_properties is None:
        return  # No discovered public catch-all; do not add fictitious inputs.
    declaration = named(
        sources, "PASSTHROUGH_EXTRA_FIELDS", "/openai/validate.rs", "const"
    )
    accepted = vocabulary(declaration)
    validator = named(
        sources, "validate_no_unsupported_fields_observed", "/openai/validate.rs", "fn"
    )
    if validator is None:
        validator = named(
            sources,
            "validate_no_unsupported_fields_with_ignore",
            "/openai/validate.rs",
            "fn",
        )
    if declaration is None or validator is None or not accepted:
        return
    callers = [
        item
        for item in sources.items
        if item.kind == "fn"
        and item.name == "validate"
        and "ValidateRequest" in item.owner
        and request in item.owner.split()
        and any(
            name in item.body
            for name in {
                "validate_no_unsupported_fields",
                "validate_no_unsupported_fields_for_endpoint",
            }
        )
    ]
    if not callers:
        return
    text = " ".join(validator.body)
    # Recognize the runtime exclusion from unsupported fields. A mere mention
    # in an unrelated function is not sufficient for admission classification.
    exclusion = "! PASSTHROUGH_EXTRA_FIELDS . contains ( & k . as_str ( ) )"
    if exclusion not in text or "unsupported_fields" not in validator.body:
        return
    passthrough = named(sources, "sampling_passthrough_args", "/preprocessor.rs", "fn")
    forward_names = set()
    if passthrough is not None:
        loop = re.search(r"for key in \[([^]]+)\]", " ".join(passthrough.body))
        if loop and {"get", "insert", "clone", "unsupported_fields"} <= set(
            passthrough.body
        ):
            forward_names = set(re.findall(r'"([^"\\]+)"', loop[1]))
    sampling = named(
        sources, "SAMPLING_FIELDS", "/common/backend_extensions.rs", "const"
    )
    sampled = vocabulary(sampling)
    capability = [
        item
        for item in sources.items
        if item.kind == "fn"
        and item.source.endswith("/common/backend_extensions.rs")
        and (
            item.name in {"lower_sampling_passthrough_for_target", "validate_field"}
            or (item.name == "resolve" and "SamplingTarget" in item.owner)
        )
    ]
    # Runtime admission is investigation evidence, not a typed declaration.
    # Never add inferred fields to the contract inventory: toggling source
    # investigation must not change schema coverage or contract fingerprints.
    for name in sorted(accepted):
        if name in contract.fields:
            continue
        evidence = [
            declaration.evidence(),
            validator.evidence(),
            *(item.evidence() for item in callers),
        ]
        effects = {"interpret"}
        conditions = [
            {"stage": "public_input_validation", "predicates": predicates(validator)}
        ]
        if name in forward_names:
            effects.add("forward")
            evidence.append(passthrough.evidence())
            conditions.append(
                {"transport": "preprocessed_rpc", "stage": "sampling_passthrough_args"}
            )
        if name in sampled and capability:
            effects.add("reject")
            evidence.extend(item.evidence() for item in [sampling, *capability])
            conditions.append(
                {
                    "stage": "worker_capability_and_legacy_target",
                    "rules": [
                        {"symbol": item.name, "predicates": predicates(item)}
                        for item in capability
                    ],
                }
            )
        contract.untyped_handling[name] = Handling(
            sorted(effects), conditions, evidence, complete=False
        )
    known_decl = named(
        sources, "VLLM_REQUEST_FIELDS", "/compatibility/vllm_fields.rs", "const"
    )
    protected = vocabulary(known_decl)
    known_guard = re.search(r"if ! known \. is_empty \( \) \{", text)
    if (
        protected
        and known_guard
        and "filter_map ( known_native_field )" in text
        and {"UnsupportedField", "Err"} <= set(validator.body)
    ):
        for name in sorted(protected - accepted - set(contract.fields)):
            contract.untyped_handling[name] = Handling(
                ["reject"],
                [
                    {
                        "stage": "known_native_field_admission",
                        "field": name,
                        "condition": "not owned by a typed field or explicit passthrough rule",
                    }
                ],
                [
                    validator.evidence(),
                    known_decl.evidence(),
                    declaration.evidence(),
                    *(item.evidence() for item in callers),
                ],
                complete=contract.fields_complete,
            )
    contract.additional_properties.evidence.extend(
        [declaration.evidence(), validator.evidence()]
    )
