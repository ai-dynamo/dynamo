# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract Dynamo serde declarations and bounded implementation handling evidence.

No runtime support is inferred from a Rust field alone. Custom deserialization,
unknown macros/types, and unidentified forwarding paths remain explicit gaps in
coverage. The evidence is source-derived at the selected commit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.protocol_compatibility.common.contracts import (
    Contract,
    CoverageDiagnostic,
    EndpointContract,
    FieldContract,
    Handling,
    canonical_union,
)
from scripts.protocol_compatibility.common.source import commit
from scripts.protocol_compatibility.extraction.dynamo_handling import (
    apply_admission,
    predicates,
)
from scripts.protocol_compatibility.extraction.rust_source import (
    RustItem,
    RustSources,
    RustUnknown,
    attributes,
    group,
    load_sources,
    serde_options,
    split,
    string_option,
    tokens,
)

REQUESTS = {
    "/v1/chat/completions": "NvCreateChatCompletionRequest",
    "/v1/completions": "NvCreateCompletionRequest",
}


def rename(name: str, style: str | None) -> str:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower().split("_")
    if style is None:
        return name
    if style == "snake_case":
        return "_".join(words)
    if style == "camelCase":
        return words[0] + "".join(word.capitalize() for word in words[1:])
    if style == "PascalCase":
        return "".join(word.capitalize() for word in words)
    if style == "SCREAMING_SNAKE_CASE":
        return "_".join(words).upper()
    if style == "kebab-case":
        return "-".join(words)
    if style == "lowercase":
        return name.lower()
    if style == "UPPERCASE":
        return name.upper()
    raise RustUnknown(f"unsupported serde rename_all: {style}")


def field_parts(body: list[str]):
    for declaration in split(body):
        attrs, offset = attributes(declaration)
        if offset < len(declaration) and declaration[offset] == "pub":
            offset += 1
            if declaration[offset] == "(":
                _, offset = group(declaration, offset)
        if offset + 2 >= len(declaration) or declaration[offset + 1] != ":":
            raise RustUnknown("unnamed or macro-generated struct fields")
        yield declaration[offset].removeprefix("r#"), declaration[offset + 2 :], attrs


class DynamoContractExtractor:
    def __init__(self, sources: RustSources):
        self.sources = sources
        self.diagnostics: dict[tuple[str, str, str], set[str]] = {}
        self.structural_problems: dict[tuple[str, str], str] = {}

    def problem(self, endpoint: str, path: str, aspect: str, reason: str) -> None:
        self.diagnostics.setdefault((endpoint, path, aspect), set()).add(reason)

    def wire_type(
        self, annotation: list[str], source: str, stack: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        name = "".join(annotation)
        canonical = self.sources.imported_name(name, source)
        primitive = {
            "String": "string",
            "str": "string",
            "bool": "boolean",
            "f32": "number",
            "f64": "number",
            "serde_json::Value": "any",
        }
        if canonical in primitive:
            return {"type": primitive[canonical]}
        if re.fullmatch(r"[iu](?:8|16|32|64|128|size)", name):
            return {"type": "integer"}
        if "<" in annotation:
            start = annotation.index("<")
            inner, end = group(annotation, start)
            if end != len(annotation):
                raise RustUnknown(f"unresolved generic suffix: {name}")
            outer = "".join(annotation[:start]).split("::")[-1]
            arguments = split(inner)
            if outer in {"Option", "Box", "Arc"} and len(arguments) == 1:
                value = self.wire_type(arguments[0], source, stack)
                return (
                    canonical_union([value, {"type": "null"}])
                    if outer == "Option"
                    else value
                )
            if outer == "Vec" and len(arguments) == 1:
                return {
                    "type": "array",
                    "items": self.wire_type(arguments[0], source, stack),
                }
            if outer in {"HashMap", "BTreeMap", "Map"} and len(arguments) == 2:
                if self.wire_type(arguments[0], source, stack) != {"type": "string"}:
                    raise RustUnknown("non-string map key normalization")
                return {
                    "type": "object",
                    "additional_properties": self.wire_type(
                        arguments[1], source, stack
                    ),
                }
            raise RustUnknown(f"unsupported generic: {name}")
        item = self.sources.resolve(name, source)
        identity = item.source + ":" + item.name
        if identity in stack:
            raise RustUnknown(f"recursive schema reference: {identity}")
        stack = (*stack, identity)
        if item.kind == "type":
            if "=" not in item.header:
                raise RustUnknown(f"unresolved alias: {identity}")
            return self.wire_type(
                item.header[item.header.index("=") + 1 :], item.source, stack
            )
        if any(
            "Deserialize" in function.owner and item.name in function.owner.split()
            for function in self.sources.items
            if function.kind == "fn"
        ):
            raise RustUnknown(f"custom Deserialize implementation: {identity}")
        options = serde_options(item.attrs)
        unsupported = set(options) - {
            "rename_all",
            "untagged",
            "tag",
            "content",
            "deny_unknown_fields",
            "default",
            "rename",
        }
        if unsupported:
            raise RustUnknown(f"custom container serde options: {sorted(unsupported)}")
        if item.kind == "struct":
            fields, wildcard = self.fields(item, stack)
            if wildcard is not None:
                raise RustUnknown(f"nested flattened map/custom handler: {identity}")
            if any(
                any(
                    value is None
                    for value in (
                        entry.wire_type,
                        entry.required,
                        entry.nullable,
                        entry.default,
                        entry.constraints,
                    )
                )
                for entry in fields.values()
            ):
                raise RustUnknown(f"nested field contract incomplete: {identity}")
            return {
                "type": "object",
                "properties": {
                    wire_name: {
                        "wire_type": entry.wire_type,
                        "required": entry.required,
                        "default": entry.default,
                        "constraints": entry.constraints,
                    }
                    for entry in fields.values()
                    for wire_name in entry.wire_names
                },
            }
        if item.kind == "enum":
            style = next(iter(string_option(options, "rename_all")), None)
            choices, members = [], []
            if "tag" in options or "content" in options:
                raise RustUnknown(
                    f"tagged enum schema requires discriminant resolution: {identity}"
                )
            for variant in split(item.body):
                attrs, offset = attributes(variant)
                variant_options = serde_options(attrs)
                if "skip" in variant_options or "skip_deserializing" in variant_options:
                    continue
                variant_name = next(
                    iter(string_option(variant_options, "rename")),
                    rename(variant[offset], style),
                )
                offset += 1
                if offset == len(variant):
                    choices.append(variant_name)
                elif variant[offset] == "(" and "untagged" in options:
                    inner, _ = group(variant, offset)
                    if len(split(inner)) != 1:
                        raise RustUnknown("tuple enum variant with multiple values")
                    members.append(self.wire_type(inner, item.source, stack))
                else:
                    raise RustUnknown(f"unsupported enum variant: {identity}")
            if choices:
                members.append({"enum": sorted(choices)})
            if not members:
                raise RustUnknown("empty enum")
            return canonical_union(members)
        raise RustUnknown(f"not a schema declaration: {identity}")

    def fields(
        self, item: RustItem, stack: tuple[str, ...] = ()
    ) -> tuple[dict[str, FieldContract], Handling | None]:
        options = serde_options(item.attrs)
        if any(attr[0] in {"cfg", "cfg_attr"} for attr in item.attrs if attr):
            raise RustUnknown(f"conditional request declaration: {item.name}")
        if "Deserialize" not in [token for attr in item.attrs for token in attr]:
            raise RustUnknown(
                f"request fields use custom/missing Deserialize: {item.name}"
            )
        if set(options) - {
            "rename_all",
            "deny_unknown_fields",
            "default",
            "rename",
            "remote",
        }:
            raise RustUnknown(f"unresolved struct serde policy: {item.name}")
        if "remote" in options and string_option(options, "remote") != ["Self"]:
            raise RustUnknown(f"external remote serde target: {item.name}")
        style = next(iter(string_option(options, "rename_all")), None)
        result, wildcard = {}, None
        for name, annotation, attrs in field_parts(item.body):
            policy = serde_options(attrs)
            if "skip" in policy or "skip_deserializing" in policy:
                continue
            if any(attr[0] in {"cfg", "cfg_attr"} for attr in attrs if attr):
                raise RustUnknown(f"conditional field: {item.name}.{name}")
            if "flatten" in policy:
                if any(token in {"HashMap", "BTreeMap", "Map"} for token in annotation):
                    wildcard = Handling(evidence=[item.evidence()], complete=False)
                    continue
                nested = self.sources.resolve("".join(annotation), item.source)
                key = nested.source + ":" + nested.name
                if key in stack:
                    raise RustUnknown("cyclic flattened field")
                flattened, extra = self.fields(nested, (*stack, key))
                if set(flattened) & set(result):
                    raise RustUnknown("colliding flattened wire fields")
                result.update(flattened)
                wildcard = extra or wildcard
                continue
            renamed = next(iter(string_option(policy, "rename")), rename(name, style))
            names = [renamed, *string_option(policy, "alias")]
            field_contract = FieldContract(
                renamed,
                sorted(set(names)),
                source=[item.evidence()],
                references=["".join(annotation)],
            )
            result[renamed] = field_contract
            try:
                if "deserialize_with" in policy or "with" in policy:
                    raise RustUnknown("field has custom deserializer")
                field_contract.wire_type = self.wire_type(
                    annotation, item.source, stack
                )
                variants = field_contract.wire_type.get(
                    "any_of", [field_contract.wire_type]
                )
                field_contract.nullable = any(
                    value.get("type") in {"null", "any"} for value in variants
                )
                optional = annotation[0] == "Option"
                defaulted = "default" in policy or "default" in options
                field_contract.required = not optional and not defaulted
                field_contract.default = (
                    {"kind": "value", "value": None} if optional else {"kind": "absent"}
                )
                if defaulted and not optional:
                    raise RustUnknown(
                        "custom or container Default requires implementation resolution"
                    )
                field_contract.constraints = self.integer_bounds(annotation)
                if any(attr and attr[0] == "validate" for attr in attrs):
                    field_contract.constraints = None
                unknown_options = set(policy) - {
                    "rename",
                    "alias",
                    "default",
                    "skip_serializing",
                    "skip_serializing_if",
                }
                if unknown_options or string_option(policy, "default"):
                    field_contract.default = None
                    raise RustUnknown("unresolved field serde policy")
            except RustUnknown as error:
                # Preserve the field even when its structural details are only
                # partially known. The endpoint adapter emits fact diagnostics.
                self.structural_problems[(item.source, renamed)] = str(error)
        return result, wildcard

    @staticmethod
    def integer_bounds(annotation: list[str]) -> dict[str, Any]:
        inner = annotation[2:-1] if annotation[:2] == ["Option", "<"] else annotation
        name = "".join(inner)
        match = re.fullmatch(r"([iu])(8|16|32|64|128)", name)
        if not match:
            return {}
        width = int(match[2])
        return {
            "ge": -(2 ** (width - 1)) if match[1] == "i" else 0,
            "le": 2 ** (width - (1 if match[1] == "i" else 0)) - 1,
        }

    def handling(self, item: FieldContract, endpoint: str) -> Handling:
        evidence, effects, conditions = [], set(), []
        projection = None
        field_name = item.path
        for function in self.sources.items:
            if function.kind != "fn" or function.crate != "dynamo_llm":
                continue
            # Request-specific impls must not leak between chat and completion.
            if (
                "NvCreate" in function.owner
                and REQUESTS[endpoint] not in function.owner
            ):
                continue
            text = " ".join(function.body)
            access = re.search(r"\.\s*" + re.escape(field_name) + r"\b", text)
            keyed = json.dumps(field_name) in function.body
            if not access and not keyed:
                continue
            request_method = REQUESTS[endpoint] in function.owner.split()
            if request_method and (
                function.name.startswith("get_")
                or function.name in {"validate", "response_generator"}
            ):
                effects.add("interpret")
                evidence.append(function.evidence())
                conditions.append(
                    {
                        "stage": "request_accessor_or_transformation",
                        "function": function.name,
                        "owner": function.owner,
                        "predicates": predicates(function),
                        "detail": "source access identified; invocation and downstream semantics are not proven",
                    }
                )
                if function.name == "validate" and any(
                    token in function.body for token in {"bail", "Err", "map_err"}
                ):
                    effects.add("reject")
                if function.name == "response_generator":
                    projection = self.projection_evidence(
                        field_name, function, evidence, conditions
                    )
            if function.name.startswith(("validate_", "normalize_", "extract_")):
                effects.add("interpret")
                evidence.append(function.evidence())
                if any(token in function.body for token in {"bail", "Err"}):
                    effects.add("reject")
                    conditions.append(
                        {
                            "kind": "source_predicate",
                            "function": function.name,
                            "detail": "validation is conditional; inspect the cited implementation",
                        }
                    )
            if (
                function.name in {"sampling_passthrough_args", "backend_extra_args"}
                and "insert" in function.body
            ):
                effects.add("forward")
                evidence.append(function.evidence())
                conditions.append(
                    {
                        "transport": "preprocessed_rpc",
                        "stage": function.name,
                        "detail": "forwarding path identified, downstream behavior unverified",
                    }
                )
        # These are observations of selected implementation paths, never a
        # whole-program proof. Unidentified paths remain part of coverage.
        return Handling(
            sorted(effects), conditions, evidence, complete=False, projection=projection
        )

    def nested_inputs(
        self,
        fields: dict[str, FieldContract],
        prefix: tuple[str, ...] = (),
        stack: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        """Retain nested declaration locations even if sibling types are unknown.

        Only ordinary named structs/aliases and transparent Option/Box/Arc
        wrappers are followed. A custom deserializer is not a declaration of
        its accepted object paths. Arrays, maps and enums stay in the structural
        schema, not guessed as named-object locations.
        """
        result = []
        for field in fields.values():
            if not field.source or len(field.references) != 1:
                continue
            source = field.source[0]["path"]
            problem = self.structural_problems.get((source, field.path), "")
            if problem in {
                "field has custom deserializer",
                "unresolved field serde policy",
            }:
                continue
            annotation = tokens(field.references[0])
            seen = stack
            try:
                while True:
                    if annotation[:2] in (["Option", "<"], ["Box", "<"], ["Arc", "<"]):
                        annotation, _ = group(annotation, 1)
                        continue
                    if "<" in annotation:
                        break
                    item = self.sources.resolve("".join(annotation), source)
                    identity = item.source + ":" + item.name
                    if identity in seen:
                        break
                    seen = (*seen, identity)
                    if item.kind == "type" and "=" in item.header:
                        annotation = item.header[item.header.index("=") + 1 :]
                        source = item.source
                        continue
                    if item.kind != "struct" or any(
                        function.kind == "fn"
                        and "Deserialize" in function.owner
                        and item.name in function.owner.split()
                        for function in self.sources.items
                    ):
                        break
                    children, _ = self.fields(item, seen)
                    for parent_name in field.wire_names:
                        location = (*prefix, parent_name)
                        for child in children.values():
                            for child_name in child.wire_names:
                                result.append(
                                    {
                                        "wire_path": [*location, child_name],
                                        "declaration": child.facts(),
                                        "source": field.source + child.source,
                                        "scope": "nested declaration; ancestor validation and handling are not proven",
                                    }
                                )
                        result.extend(self.nested_inputs(children, location, seen))
                    break
            except RustUnknown:
                # Parent structural extraction already retains the unresolved
                # type/serde diagnostic. Never manufacture nested names.
                continue
        return result

    def projection_evidence(
        self,
        field_name: str,
        generator: RustItem,
        evidence: list[dict[str, Any]],
        conditions: list[dict[str, Any]],
    ) -> str:
        """Link a request's response-generator read to bounded source candidates.

        Same-named payloads are useful review evidence, not a proven data-flow
        edge. Keep that distinction in the report and never mark complete. Only
        the generator's endpoint module is searched, so chat evidence cannot
        silently stand in for completion behavior (or vice versa).
        """
        directory = generator.source.rsplit("/", 1)[0]
        if not generator.source.endswith("/delta.rs"):
            return "Request field is read by response_generator; downstream projection unresolved."
        pattern = re.compile(r"\.\s*(?:internal_)?" + re.escape(field_name) + r"\b")
        candidates = [
            function
            for function in self.sources.items
            if function.kind == "fn"
            and function.crate == "dynamo_llm"
            and function is not generator
            and function.source
            in {directory + "/delta.rs", directory + "/aggregator.rs"}
            and pattern.search(" ".join(function.body))
        ]
        for function in candidates:
            evidence.append(function.evidence())
            conditions.append(
                {
                    "stage": "response_projection_candidate",
                    "function": function.name,
                    "source": function.source,
                    "predicates": predicates(function),
                    "detail": "same-named payload access in the endpoint delta/aggregation path; not a proven data-flow edge",
                }
            )
        if not candidates:
            return "Request field is read by response_generator; downstream projection unresolved."
        return (
            "Request field is read by response_generator; same-named payload accesses "
            "are linked in this endpoint's delta/aggregation implementation. "
            "Wire placement, streaming serialization and runtime behavior remain unverified."
        )

    def extract(self, revision: str, *, investigate: bool = True) -> Contract:
        endpoints = {}
        for endpoint, name in REQUESTS.items():
            result = EndpointContract()
            endpoints[endpoint] = result
            try:
                root = self.sources.resolve(name, "")
                result.fields, result.additional_properties = self.fields(root)
                result.fields_complete = True
                if "remote" in serde_options(root.attrs):
                    wrappers = [
                        item
                        for item in self.sources.items
                        if item.kind == "fn"
                        and "Deserialize" in item.owner
                        and name in item.owner.split()
                    ]
                    result.fields_complete = False
                    self.problem(
                        endpoint,
                        "*",
                        "custom_deserialize",
                        "Remote-derived declarations retained; custom wrapper may alter the effective input contract. "
                        + str([item.evidence() for item in wrappers]),
                    )
            except RustUnknown as error:
                self.problem(endpoint, "*", "fields", str(error))
                continue
            if investigate:
                apply_admission(self.sources, result, name)
            result.nested_inputs = self.nested_inputs(result.fields)
            for path, item in result.fields.items():
                if investigate and item.handling is None:
                    item.handling = self.handling(item, endpoint)
                unknown = [
                    aspect
                    for aspect, value in item.facts().items()
                    if value is None and aspect != "handling"
                ]
                if unknown:
                    self.problem(
                        endpoint,
                        path,
                        "structural",
                        "Unresolved serde/type facts: " + ", ".join(unknown),
                    )
                for source in item.source:
                    reason = self.structural_problems.get((source["path"], path))
                    if reason:
                        self.problem(endpoint, path, "structural", reason)
            for problem in self.sources.problems:
                self.problem(endpoint, "*", "source", problem)
        return Contract(
            "dynamo",
            revision,
            endpoints,
            [
                CoverageDiagnostic(endpoint, path, aspect, "; ".join(sorted(reasons)))
                for (endpoint, path, aspect), reasons in sorted(
                    self.diagnostics.items()
                )
            ],
            {
                "extraction": "bounded Rust source analysis; no compilation or macro execution",
                "dependencies": self.sources.dependencies,
                "handling_scope": "selected validation/normalization/passthrough, request accessors and endpoint response-generator/projection candidates; not end-to-end support",
            },
        )


def extract_dynamo(
    repo: Path,
    revision: str,
    crate_cache: Path | None = None,
    *,
    investigate: bool = True,
) -> Contract:
    revision = commit(repo, revision)
    return DynamoContractExtractor(load_sources(repo, revision, crate_cache)).extract(
        revision, investigate=investigate
    )
