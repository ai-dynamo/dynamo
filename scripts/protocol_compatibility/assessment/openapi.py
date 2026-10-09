# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-only projection and coverage guards around the pinned oasdiff engine.

This module selects schemas, normalizes a bounded equivalent representation,
and reports unsupported constructs. It does not implement a source parser,
schema merger, or general semantic compatibility algorithm.
"""

import copy
import json
import subprocess
from pathlib import Path

import jsonpointer
from openapi_spec_validator import validate

ENDPOINTS = ("/v1/chat/completions", "/v1/completions")
OASDIFF_VERSION = "1.33.0"
SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "dependentSchemas")
SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
SCHEMA_SINGLE = (
    "items",
    "additionalProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "propertyNames",
    "contentSchema",
)
# These either are unsupported by oasdiff or cannot safely pass through its
# allOf merger. Retain their originals and disable flattening, never drop them.
UNSUPPORTED = {
    "$dynamicRef",
    "$dynamicAnchor",
    "$anchor",
    "$id",
    "$schema",
    "$defs",
    "if",
    "then",
    "else",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
    "patternProperties",
    "prefixItems",
    "contains",
    "contentSchema",
}
KNOWN = (
    UNSUPPORTED
    | set(SCHEMA_MAPS + SCHEMA_LISTS + SCHEMA_SINGLE)
    | {
        "$ref",
        "$comment",
        "type",
        "enum",
        "const",
        "required",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "dependentRequired",
        "minContains",
        "maxContains",
        "contentEncoding",
        "contentMediaType",
        "description",
        "title",
        "default",
        "example",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "discriminator",
        "externalDocs",
        "xml",
    }
)


def pointer(*parts: str) -> str:
    return "#" + jsonpointer.JsonPointer.from_parts(parts).path


def request_pointer(endpoint: str) -> str:
    return pointer(
        "paths",
        endpoint,
        "post",
        "requestBody",
        "content",
        "application/json",
        "schema",
    )


def schema_nodes(document: dict, schema: dict | bool, location: str, seen=None):
    """Walk schema positions and local component refs; never fetch references.

    Locations refer to retained input documents, including referenced components.
    Literal '$ref' keys in defaults/examples or property names remain data.
    """
    seen = set() if seen is None else seen
    if location in seen:
        return
    seen.add(location)
    yield location, schema
    if not isinstance(schema, dict):
        return
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/components/schemas/"):
            raise ValueError(f"Unsupported request reference at {location}: {ref}")
        yield from schema_nodes(
            document, jsonpointer.resolve_pointer(document, ref[1:]), ref, seen
        )
    for reference in schema.get("discriminator", {}).get("mapping", {}).values():
        if not reference.startswith("#/components/schemas/"):
            raise ValueError(
                f"Unsupported discriminator reference at {location}: {reference}"
            )
        yield from schema_nodes(
            document,
            jsonpointer.resolve_pointer(document, reference[1:]),
            reference,
            seen,
        )
    for key in SCHEMA_MAPS:
        for name, child in schema.get(key, {}).items():
            yield from schema_nodes(
                document, child, location + pointer(key, name)[1:], seen
            )
    for key in SCHEMA_LISTS:
        for index, child in enumerate(schema.get(key, [])):
            yield from schema_nodes(
                document, child, location + pointer(key, str(index))[1:], seen
            )
    for key in SCHEMA_SINGLE:
        if key in schema:
            yield from schema_nodes(
                document, schema[key], location + pointer(key)[1:], seen
            )


def request_document(document: dict) -> tuple[dict, list[dict]]:
    """Keep both POST JSON request contracts and their schema reference closure."""
    if document.get("openapi") not in {"3.1.0", "3.1.1"}:
        raise ValueError("This workflow requires OpenAPI 3.1 server exports")
    if (
        document.get(
            "jsonSchemaDialect", "https://spec.openapis.org/oas/3.1/dialect/base"
        )
        != "https://spec.openapis.org/oas/3.1/dialect/base"
    ):
        raise ValueError(
            "Custom OpenAPI schema dialects need explicit comparison support"
        )
    result = {
        "openapi": document["openapi"],
        "info": {"title": "Request contracts", "version": "1"},
        "paths": {},
        "components": {"schemas": {}},
    }
    gaps = []
    for endpoint in ENDPOINTS:
        body = copy.deepcopy(document["paths"][endpoint]["post"]["requestBody"])
        if set(body.get("content", {})) != {"application/json"}:
            raise ValueError(f"Expected one JSON request media type for {endpoint}")
        schema = body["content"]["application/json"]["schema"]
        for location, node in schema_nodes(document, schema, request_pointer(endpoint)):
            if location.startswith("#/components/schemas/"):
                name = jsonpointer.JsonPointer(location[1:]).parts[2]
                result["components"]["schemas"][name] = copy.deepcopy(
                    document["components"]["schemas"][name]
                )
            if isinstance(node, dict):
                if "x-dynamo-schema-import" in node:
                    raise ValueError(f"Unresolved dependency schema at {location}")
                unsupported = UNSUPPORTED.intersection(node) | {
                    key for key in node if key not in KNOWN and not key.startswith("x-")
                }
                for keyword in sorted(unsupported):
                    gaps.append(
                        {
                            "endpoint": endpoint,
                            "location": location,
                            "reason": f"Comparison/flattening support is incomplete for {keyword}",
                        }
                    )
        # No responses, parameters, security, operation IDs or unrelated endpoints
        # are compared. A fixed empty response is required by OpenAPI itself.
        result["paths"][endpoint] = {
            "post": {
                "requestBody": body,
                "responses": {"200": {"description": "Outside assessment scope"}},
            }
        }
    validate(result)
    return result, gaps


def normalize_nullable_strings(document: dict) -> tuple[dict, list[dict]]:
    """Canonicalize only bare string/null unions on a copy of scoped requests.

    ``anyOf: [{type: string}, {type: null}]`` and a string/null type array
    have the same validation meaning. Keep sibling constraints and annotations;
    do not simplify constrained/ref branches, oneOf, or other primitive unions.
    The schema-aware walker leaves defaults, examples and extensions untouched.
    """
    result = copy.deepcopy(document)
    changes = []
    seen = set()
    for endpoint in ENDPOINTS:
        location = request_pointer(endpoint)
        schema = jsonpointer.resolve_pointer(result, location[1:])
        for location, node in schema_nodes(result, schema, location, seen):
            if not isinstance(node, dict):
                continue
            branches = node.get("anyOf")
            if "type" not in node and branches in (
                [{"type": "string"}, {"type": "null"}],
                [{"type": "null"}, {"type": "string"}],
            ):
                before = {"anyOf": node["anyOf"]}
            elif node.get("type") == ["null", "string"]:
                before = {"type": node["type"]}
            else:
                continue
            normalized = dict(node)
            if "anyOf" in before:
                del normalized["anyOf"]
            normalized["type"] = ["string", "null"]
            # Replace the schema position, not an aliased object that may also
            # occur as literal default/example data (e.g. through YAML aliases).
            jsonpointer.set_pointer(result, location[1:], normalized)
            changes.append(
                {
                    "rule": "nullable-string-encoding/v1",
                    "location": location,
                    "before": before,
                    "after": {"type": ["string", "null"]},
                }
            )
    validate(result)
    return result, changes


def compare(
    binary: Path, native: Path, dynamo: Path, *, flatten: bool
) -> tuple[dict, list[str]]:
    version = subprocess.run(
        [str(binary), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if version.stdout.strip() != f"oasdiff version {OASDIFF_VERSION}":
        raise ValueError(
            f"Expected oasdiff {OASDIFF_VERSION}, got {version.stdout.strip()}"
        )
    command = [
        str(binary),
        "diff",
        "--allow-external-refs=false",
        "--auto-upgrade",
        "--exclude-elements",
        "description,examples,extensions,summary,title",
        "--format",
        "json",
    ]
    if flatten:
        command.append("--flatten-allof")
    command.extend([str(native), str(dynamo)])
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=120
    )
    return json.loads(result.stdout or "{}"), command


def request_fields(document: dict, endpoint: str) -> list[dict]:
    """Index instance paths, not component names or union branch indexes.

    Union and parent constraints stay attached as context; a name match never
    establishes whole-object equivalence. Recursive refs are not expanded forever.
    """
    fields = []

    def walk(node, location, path, context, refs):
        if not isinstance(node, dict):
            return
        if "$ref" in node:
            ref = node["$ref"]
            if ref not in refs:
                walk(
                    jsonpointer.resolve_pointer(document, ref[1:]),
                    ref,
                    path,
                    context,
                    refs | {ref},
                )
        parent_constraints = {
            key: node[key]
            for key in (
                "required",
                "dependentRequired",
                "dependentSchemas",
                "not",
                "if",
                "then",
                "else",
                "minProperties",
                "maxProperties",
                "propertyNames",
                "additionalProperties",
                "unevaluatedProperties",
            )
            if key in node
        }
        if parent_constraints:
            context = context + [
                {"location": location, "constraints": parent_constraints}
            ]
        for name, child in node.get("properties", {}).items():
            child_location = location + pointer("properties", name)[1:]
            fields.append(
                {
                    "path": [*path, name],
                    "location": child_location,
                    "schema": child,
                    "context": context,
                }
            )
            walk(child, child_location, (*path, name), context, refs)
        if "items" in node:
            # None denotes an array item and cannot collide with a JSON key.
            walk(node["items"], location + "/items", (*path, None), context, refs)
        for keyword in ("allOf", "anyOf", "oneOf"):
            for index, child in enumerate(node.get(keyword, [])):
                child_location = f"{location}/{keyword}/{index}"
                branch_context = context
                if keyword != "allOf":
                    branch_context = context + [
                        {"location": child_location, "union": keyword}
                    ]
                walk(child, child_location, path, branch_context, refs)

    location = request_pointer(endpoint)
    walk(jsonpointer.resolve_pointer(document, location[1:]), location, (), [], set())
    return fields


def match_input_aliases(dynamo: dict, native: dict) -> tuple[list[dict], list[dict]]:
    """Match explicit Dynamo aliases at the same request instance path.

    Does not rewrite schemas, infer aliases from similar names, or discard the
    original comparator delta. Multiple declarations are ambiguous, not a match.
    x-dynamo-input-aliases implies rejection of multiple spellings together.
    """
    matches, gaps = [], []
    for endpoint in ENDPOINTS:
        own_fields, backend_fields = request_fields(dynamo, endpoint), request_fields(
            native, endpoint
        )
        for own in own_fields:
            schema = own["schema"]
            if not isinstance(schema, dict) or "x-dynamo-input-aliases" not in schema:
                continue
            aliases = schema["x-dynamo-input-aliases"]
            if (
                not isinstance(aliases, list)
                or not aliases
                or any(not isinstance(name, str) or not name for name in aliases)
                or len(set(aliases)) != len(aliases)
                or own["path"][-1] in aliases
                # Older captures carried a redundant conflict marker. Accept
                # "reject", but fail closed if it contradicts alias semantics.
                or schema.get("x-dynamo-alias-conflict", "reject") != "reject"
            ):
                raise ValueError(
                    f"Invalid or unsupported alias metadata at {own['location']}"
                )
            for alias in aliases:
                path = [*own["path"][:-1], alias]
                candidates = [
                    field for field in backend_fields if field["path"] == path
                ]
                owners = [
                    field
                    for field in own_fields
                    if field["path"] == path
                    or (
                        field["path"][:-1] == path[:-1]
                        and isinstance(field["schema"], dict)
                        and isinstance(
                            field["schema"].get("x-dynamo-input-aliases", []), list
                        )
                        and alias in field["schema"].get("x-dynamo-input-aliases", [])
                    )
                ]
                if len(owners) != 1 or len(candidates) > 1:
                    gaps.append(
                        {
                            "side": "dynamo",
                            "endpoint": endpoint,
                            "field": ".".join(
                                "[]" if part is None else part for part in path
                            ),
                            "location": own["location"],
                            "reason": "Ambiguous alias ownership or backend union declarations; no name match assumed.",
                        }
                    )
                    continue
                if not candidates:
                    continue
                backend = candidates[0]
                matches.append(
                    {
                        "endpoint": endpoint,
                        "path": path,
                        "backend_name": alias,
                        "dynamo_name": own["path"][-1],
                        "dynamo_location": own["location"],
                        "backend_location": backend["location"],
                        "name_coverage": "matched_via_declared_input_alias",
                        "simultaneous_names": "Dynamo rejects canonical and alias together",
                        "parent_context": {
                            "dynamo": own["context"],
                            "backend": backend["context"],
                        },
                        "compatibility": "not_established_by_name_match",
                    }
                )
    return matches, gaps


def alias_value_document(document: dict, location: str) -> dict:
    """Isolate one field's value for oasdiff, keeping its reference closure.

    Parent constraints are deliberately NOT claimed covered by this probe; they
    remain in the complete request diff and alias match's parent_context.
    """
    result = copy.deepcopy(document)
    value = copy.deepcopy(jsonpointer.resolve_pointer(document, location[1:]))
    for endpoint in ENDPOINTS:
        jsonpointer.set_pointer(result, request_pointer(endpoint)[1:], value)
    return result


def request_differences(diff: dict) -> list[dict]:
    """Expose complete request-body deltas, not response/component rename noise."""
    findings = []
    for endpoint, change in sorted(diff.get("paths", {}).get("modified", {}).items()):
        operation = change.get("operations", {}).get("modified", {}).get("POST", {})
        if operation.get("requestBody"):
            findings.append(
                {
                    "endpoint": endpoint,
                    "location": request_pointer(endpoint),
                    "delta": operation["requestBody"],
                }
            )
    if diff.get("paths", {}).get("added") or diff.get("paths", {}).get("deleted"):
        raise ValueError("Scoped endpoint disappeared during comparison")
    return findings
