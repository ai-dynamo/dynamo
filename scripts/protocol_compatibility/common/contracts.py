# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Language-neutral, source-derived request contracts used by both extractors.

Unknown is represented by None for facts (not for JSON defaults, which have an
explicit kind). Source locations and source-language spellings are evidence,
not wire types. No object in this module asserts runtime conformance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from scripts.protocol_compatibility.common.provenance import digest


@dataclass
class Handling:
    effects: list[str] = field(default_factory=list)
    conditions: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = False
    projection: str | None = None

    def __post_init__(self) -> None:
        allowed = {"interpret", "forward", "reject"}
        if not set(self.effects) <= allowed:
            raise ValueError(f"unknown handling effects: {self.effects}")
        self.effects = sorted(set(self.effects))

    def facts(self) -> dict[str, Any]:
        return {
            "effects": self.effects,
            "conditions": self.conditions,
            "complete": self.complete,
            "projection": self.projection,
            # A body change invalidates decisions even if the inferred label is
            # unchanged. File/line moves alone do not invalidate a decision.
            "implementation_hashes": sorted(
                item["semantic_sha256"]
                for item in self.evidence
                if "semantic_sha256" in item
            ),
        }


@dataclass
class FieldContract:
    path: str
    wire_names: list[str]
    wire_type: dict[str, Any] | None = None
    required: bool | None = None
    nullable: bool | None = None
    default: dict[str, Any] | None = None
    constraints: dict[str, Any] | None = None
    source: list[dict[str, Any]] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    handling: Handling | None = None

    def facts(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "wire_names": sorted(set(self.wire_names)),
            "wire_type": self.wire_type,
            "required": self.required,
            "nullable": self.nullable,
            "default": self.default,
            "constraints": self.constraints,
            "handling": self.handling.facts() if self.handling else None,
        }

    def contract_facts(self) -> dict[str, Any]:
        """Declaration-only identity; source handling has its own evidence layer."""
        return {key: value for key, value in self.facts().items() if key != "handling"}


@dataclass
class EndpointContract:
    fields: dict[str, FieldContract] = field(default_factory=dict)
    fields_complete: bool = False
    # Wildcard/passthrough is not proof that all unknown fields reach an engine.
    additional_properties: Handling | None = None
    # Explicit admission rules for untyped inputs, not fabricated Rust fields.
    untyped_handling: dict[str, Handling] = field(default_factory=dict)
    # Declaration-derived nested locations, not aliases for top-level inputs.
    # Segment arrays distinguish a literal dotted key from an object path.
    nested_inputs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CoverageDiagnostic:
    endpoint: str
    path: str
    aspect: str
    reason: str
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class Contract:
    target: str
    revision: str
    endpoints: dict[str, EndpointContract]
    diagnostics: list[CoverageDiagnostic] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def complete(self) -> bool:
        """Contract coverage only, independent of optional source investigation."""
        return (
            bool(self.endpoints)
            and not self.diagnostics
            and all(
                endpoint.fields_complete
                and all(
                    bool(item.wire_names)
                    and all(
                        value is not None
                        for value in (
                            item.wire_type,
                            item.required,
                            item.nullable,
                            item.default,
                            item.constraints,
                        )
                    )
                    for item in endpoint.fields.values()
                )
                for endpoint in self.endpoints.values()
            )
        )


def canonical_union(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten and sort wire unions, independent of source-language spelling."""
    flattened = []
    for member in members:
        flattened.extend(member["any_of"] if "any_of" in member else [member])
    unique = {digest(member): member for member in flattened}
    values = [unique[key] for key in sorted(unique)]
    return values[0] if len(values) == 1 else {"any_of": values}


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
