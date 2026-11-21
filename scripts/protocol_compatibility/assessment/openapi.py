# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare request schemas and run the shared oasdiff comparison engine.

request_document selects POST JSON bodies for chat and legacy completions, with
their referenced components and comparison gaps. Explicit input aliases get
separate value probes so name matching cannot hide constraint differences.

The response pipeline shares the schema walker, validator, nullable-string
normalization, and compare. compare runs framework-to-Dynamo; its raw deltas are
not runtime compatibility verdicts. openapi_workflow.py owns file retention and
reporting, keeping those effects separate from schema projection.
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


def pointer(*parts: str | int) -> str:
    """Encode path segments as a local JSON Pointer, escaping slashes and tildes."""
    return "#" + jsonpointer.JsonPointer.from_parts(parts).path


def request_pointer(endpoint: str) -> str:
    """Return the POST JSON request-schema pointer for an endpoint path."""
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
    Yield (pointer, schema) pairs once per location, updating `seen` if supplied.
    Unsupported or unresolved references raise; document contents are unchanged.
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


def check_schema_dialect(document: dict) -> None:
    """Reject unsupported document versions/dialects before comparison projection.

    Accept the standard 3.1 and 3.2 dialects, both based on JSON Schema 2020-12.
    Keyword coverage is checked separately on each reachable schema. This does not
    validate document structure or fetch a caller-provided dialect.
    """
    version = document.get("openapi", "")
    if version not in {"3.1.0", "3.1.1", "3.1.2", "3.2.0", "3.2.1"}:
        raise ValueError("This workflow requires OpenAPI 3.1 or 3.2 server exports")
    dialect = (
        "https://spec.openapis.org/oas/3.2/dialect/2025-09-17"
        if version.startswith("3.2.")
        else "https://spec.openapis.org/oas/3.1/dialect/base"
    )
    if document.get("jsonSchemaDialect", dialect) != dialect:
        raise ValueError(
            "Custom OpenAPI schema dialects need explicit comparison support"
        )


def validate_comparison_document(document: dict, roots: list[str]) -> None:
    """Validate a projected document without interpreting default data as refs.

    `roots` contains its schema-root JSON pointers. The schema-aware walk verifies
    local references and removes only default annotations from a validation copy;
    the original defaults remain in comparator inputs. Defaults are not JSON Schema
    assertions. The pinned validator's extra default-value check incorrectly follows
    literal `$ref` keys in that data, so it must not see them. Missing schema refs,
    unsupported external refs and malformed documents raise; input is unchanged.
    """
    checked = copy.deepcopy(document)
    seen = set()
    for location in roots:
        schema = jsonpointer.resolve_pointer(checked, location[1:])
        for location, node in schema_nodes(checked, schema, location, seen):
            if isinstance(node, dict) and "default" in node:
                replacement = {
                    key: value for key, value in node.items() if key != "default"
                }
                jsonpointer.set_pointer(checked, location[1:], replacement)
    validate(checked)


def request_document(document: dict) -> tuple[dict, list[dict]]:
    """Return a request comparison document and gaps for the two ENDPOINTS.

    Both paths must declare POST requestBody with only application/json content.
    Standard 3.2 request schema keywords use the same 2020-12 semantics here. The
    projection is not a server export: responses and other API surfaces are removed.
    Original documents remain untouched. Missing contracts or unresolved imports
    raise; unsupported reachable schema keywords are retained with coverage gaps.
    """
    check_schema_dialect(document)
    result = {
        "openapi": "3.1.0",
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
    validate_comparison_document(
        result, [request_pointer(endpoint) for endpoint in ENDPOINTS]
    )
    return result, gaps


def normalize_nullable_strings(
    document: dict, *, locations: list[str] | None = None
) -> tuple[dict, list[dict]]:
    """Canonicalize bare string/null unions on a copy of a comparison document.

    ``anyOf: [{type: string}, {type: null}]`` and a string/null type array
    have the same validation meaning. Keep sibling constraints and annotations;
    do not simplify constrained/ref branches, oneOf, or other primitive unions.
    The schema-aware walker leaves defaults, examples and extensions untouched.
    `locations` selects schema-root JSON pointers; omitted means the two request
    roots. Return the copied document and an audit of rewrites. Missing pointers
    or invalid output documents raise rather than returning a partial result.
    """
    result = copy.deepcopy(document)
    changes = []
    seen = set()
    roots = (
        locations
        if locations is not None
        else [request_pointer(endpoint) for endpoint in ENDPOINTS]
    )
    for location in roots:
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
    validate_comparison_document(result, roots)
    return result, changes


def compare(
    binary: Path, native: Path, dynamo: Path, *, flatten: bool
) -> tuple[dict, list[str]]:
    """Run pinned oasdiff from the target (`native`) to Dynamo; return delta and argv.

    Inputs must be retained comparator projections. Set `flatten` only when their
    schema coverage permits allOf merging. The process cannot fetch external refs.
    Version mismatches, subprocess failures/timeouts, and invalid JSON raise;
    a failed diff raises ValueError with its exit status and up to 2,000 stderr
    characters.
    Files are not changed. Empty output means an empty delta, not runtime parity.
    """
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
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=120
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()[-2000:]
        raise ValueError(
            f"oasdiff exited with status {error.returncode}: {detail}"
        ) from error
    return json.loads(result.stdout or "{}"), command


def request_fields(document: dict, endpoint: str) -> list[dict]:
    """Index fields beneath endpoint's POST JSON body for alias matching.

    Records contain instance path segments (None for array items), schema pointers,
    declarations, and parent context; component names are not instance paths.
    Union and parent constraints stay attached as context; a name match never
    establishes whole-object equivalence. Recursive refs are not expanded forever.
    Missing pointers raise. Returned schema/context objects refer to input data;
    callers must treat them as read-only.
    """
    fields = []

    def walk(node, location, path, context, refs):
        """Append fields with parent/union context; stop at references already on this path."""
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
    Return matches and ambiguity gaps. Invalid alias metadata or unresolved
    references raise; neither document is modified.
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
                            "kind": "request",
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
    `location` must resolve in the input; resolution failures propagate.
    The input is unchanged.
    """
    result = copy.deepcopy(document)
    value = copy.deepcopy(jsonpointer.resolve_pointer(document, location[1:]))
    for endpoint in ENDPOINTS:
        jsonpointer.set_pointer(result, request_pointer(endpoint)[1:], value)
    return result


def request_differences(diff: dict) -> list[dict]:
    """Select request-body deltas, excluding response/component-only changes.

    Return records referencing the raw delta; callers must not mutate them.
    Added/deleted scoped paths raise ValueError instead of reporting an empty diff.
    """
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
