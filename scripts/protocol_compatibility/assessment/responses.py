# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project unary bodies and decoded SSE events for the pinned comparison engine.

Call response_document on each retained server export, then compare the returned
documents with assessment.openapi.compare. These are comparison-only documents:
SSE schema denotes a parsed event whose JSON data has been decoded, not a wire
body. Original itemSchema/contentSchema locations accompany coverage gaps.

Only explicit string data alternatives (JSON contentSchema or a string const)
are decoded. Unsupported encodings/constraints never become a compatibility pass.
Neither this projection nor a structural diff proves runtime behavior or ordering.
"""

import copy
import json

import jsonpointer
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from .openapi import (
    ENDPOINTS,
    KNOWN,
    UNSUPPORTED,
    check_schema_dialect,
    pointer,
    schema_nodes,
    validate_comparison_document,
)


def decoded_data_schema(schema: dict) -> dict | bool:
    """Return a comparison schema for a string constant or JSON-decoded SSE data.

    The caller supplies the data property's schema, not an event or payload schema.
    Accept only the bounded representation used by the pinned native exports;
    reject unknown string constraints rather than losing their encoded semantics.
    JSON content is copied unchanged, including its references. Input is not mutated.
    Raises ValueError if decoding would discard an assertion or has ambiguous meaning.
    """
    if not isinstance(schema, dict):
        raise ValueError("SSE data must explicitly describe its string encoding")
    if schema.get("type", "string") != "string":
        raise ValueError("SSE data must be a string before decoding")
    annotations = {"description", "title", "$comment", "examples", "example"}
    keys = set(schema) - annotations - {"type"}
    if keys == {"const"} and isinstance(schema["const"], str):
        encoding, value = "literal", {"const": schema["const"]}
    elif keys == {"contentMediaType", "contentSchema"}:
        if schema["contentMediaType"] != "application/json":
            raise ValueError("Only application/json SSE data decoding is supported")
        encoding, value = "json", copy.deepcopy(schema["contentSchema"])
    elif keys == {"anyOf"} and schema["anyOf"]:
        return {"anyOf": [decoded_data_schema(branch) for branch in schema["anyOf"]]}
    else:
        raise ValueError(
            "Unsupported SSE data encoding or encoded-string constraints: "
            + ", ".join(sorted(keys))
        )
    # Preserve the encoding distinction: JSON "[DONE]" is not the literal
    # terminator. These tagged objects exist only in the comparison projection.
    return {
        "type": "object",
        "required": ["encoding", "value"],
        "properties": {"encoding": {"const": encoding}, "value": value},
    }


def decoded_event_schema(document: dict, schema: dict, refs: tuple = ()) -> dict:
    """Copy an itemSchema, replacing supported data schemas with decoded equivalents.

    Resolve local event-envelope refs and anyOf/oneOf event variants, retaining
    envelope constraints. Payload refs stay refs. Missing/recursive/external event
    refs, allOf envelopes, or unsupported data encodings raise ValueError (or a
    JsonPointerException for an unresolved ref); callers must report a coverage gap.
    """
    if not isinstance(schema, dict):
        raise ValueError("SSE itemSchema must describe event objects")
    if "$ref" in schema:
        ref = schema["$ref"]
        if (
            not ref.startswith("#/components/schemas/")
            or ref in refs
            or set(schema) != {"$ref"}
        ):
            raise ValueError("Unsupported event reference or reference siblings")
        return decoded_event_schema(
            document, jsonpointer.resolve_pointer(document, ref[1:]), (*refs, ref)
        )
    result = copy.deepcopy(schema)
    if "allOf" in result:
        raise ValueError("allOf event envelopes need explicit decoding support")
    found = False
    for keyword in ("anyOf", "oneOf"):
        if keyword in result:
            result[keyword] = [
                decoded_event_schema(document, branch, refs)
                for branch in result[keyword]
            ]
            found = True
    properties = result.get("properties", {})
    if "data" in properties:
        properties["data"] = decoded_data_schema(properties["data"])
        found = True
    if not found:
        raise ValueError("SSE event has no explicitly described data payload")
    return result


def validate_sse_event(document: dict, item_schema: dict, event: dict) -> None:
    """Validate one parsed SSE event, explicitly checking its JSON-encoded data.

    `event` is the SSE parser's object with string `data`, not a chunk or raw frame.
    `item_schema` belongs to `document`, which supplies local payload references.
    Return None on success; raise ValidationError for invalid event/data, ValueError
    for unsupported encoding declarations, or a reference-resolution error for gaps.
    No mutation, network fetch, ordering or termination check is performed. Ignore
    comment-only frames before calling. Both literal and JSON interpretations are
    tried because the same wire text may satisfy either declared alternative.
    """
    schema = decoded_event_schema(document, item_schema)
    for location, node in schema_nodes(document, schema, "#/event"):
        if isinstance(node, dict) and "x-dynamo-schema-import" in node:
            raise ValueError(f"Unresolved dependency schema at {location}")
    resource = Resource.from_contents(
        {"event": schema, "components": document.get("components", {})},
        default_specification=DRAFT202012,
    )
    validator = Draft202012Validator(
        {"$ref": "urn:response-event#/event"},
        registry=Registry().with_resource("urn:response-event", resource),
    )
    if not isinstance(event, dict) or not isinstance(event.get("data"), str):
        raise ValidationError("Parsed SSE event must contain string data")
    candidates = [("literal", event["data"])]

    def reject_nonfinite(value: str):
        """Reject Python's non-standard NaN/Infinity JSON extension."""
        raise ValueError(f"Non-JSON numeric constant: {value}")

    try:
        candidates.append(
            ("json", json.loads(event["data"], parse_constant=reject_nonfinite))
        )
    except ValueError:
        pass  # Invalid JSON may still be a valid explicitly declared literal.
    for encoding, value in candidates:
        candidate = {**event, "data": {"encoding": encoding, "value": value}}
        if validator.is_valid(candidate):
            return
    raise ValidationError(
        "SSE event does not satisfy its literal or JSON data contract"
    )


def response_document(document: dict) -> tuple[dict, list[dict]]:
    """Copy scoped POST response contracts into comparator inputs and return gaps.

    Keep status/media distinctions and reachable payload components. JSON bodies
    retain their schemas; SSE itemSchema becomes a decoded-event comparison schema.
    Missing/unsupported schemas are retained as empty comparison placeholders with
    explicit gaps, never evidence of compatibility. Missing operations and invalid
    document dialects raise. Input is unchanged; references are never fetched.
    """
    check_schema_dialect(document)
    result = {
        "openapi": "3.1.0",
        "info": {
            "title": "Decoded response comparison (not a server export)",
            "version": "1",
        },
        "paths": {},
        "components": {"schemas": {}},
    }
    gaps = []
    for endpoint in ENDPOINTS:
        responses = {}
        originals = document["paths"][endpoint]["post"].get("responses", {})
        if not originals:
            gaps.append(
                {
                    "endpoint": endpoint,
                    "location": pointer("paths", endpoint, "post", "responses"),
                    "reason": "No declared responses",
                }
            )
        for status, response in originals.items():
            if status.startswith("x-"):
                continue
            location = pointer("paths", endpoint, "post", "responses", status)
            projected = {"description": "Response comparison"}
            responses[status] = projected
            content = response.get("content", {})
            if "$ref" in response or (not content and status not in {"204", "304"}):
                gaps.append(
                    {
                        "endpoint": endpoint,
                        "location": location,
                        "reason": "Response body contract absent or response reference unsupported",
                    }
                )
            for media, contract in content.items():
                source = location + pointer("content", media)[1:]
                projected.setdefault("content", {})[media] = {"schema": {}}
                try:
                    if media == "text/event-stream":
                        if "schema" in contract or "itemSchema" not in contract:
                            raise ValueError(
                                "SSE comparison requires itemSchema without a whole-stream schema"
                            )
                        source += "/itemSchema"
                        schema = decoded_event_schema(document, contract["itemSchema"])
                    elif media == "application/json" and "schema" in contract:
                        source += "/schema"
                        schema = copy.deepcopy(contract["schema"])
                    else:
                        raise ValueError(
                            "Unsupported response media type or missing schema"
                        )
                    components = {}
                    if schema is True or schema == {}:
                        raise ValueError("Response schema has no payload constraints")
                    for node_location, node in schema_nodes(document, schema, source):
                        if node_location.startswith("#/components/schemas/"):
                            name = jsonpointer.JsonPointer(node_location[1:]).parts[2]
                            components[name] = copy.deepcopy(
                                document["components"]["schemas"][name]
                            )
                        if isinstance(node, dict):
                            if "x-dynamo-schema-import" in node:
                                raise ValueError(
                                    f"Unresolved dependency schema at {node_location}"
                                )
                            for keyword in sorted(
                                UNSUPPORTED.intersection(node)
                                | {
                                    key
                                    for key in node
                                    if key not in KNOWN and not key.startswith("x-")
                                }
                            ):
                                gaps.append(
                                    {
                                        "endpoint": endpoint,
                                        "location": (
                                            node_location
                                            if node_location.startswith(
                                                "#/components/schemas/"
                                            )
                                            else source
                                        ),
                                        "comparison_location": node_location,
                                        "reason": f"Response comparison support is incomplete for {keyword}",
                                    }
                                )
                    result["components"]["schemas"].update(components)
                    projected["content"][media]["schema"] = schema
                except (ValueError, jsonpointer.JsonPointerException) as error:
                    gaps.append(
                        {"endpoint": endpoint, "location": source, "reason": str(error)}
                    )
        result["paths"][endpoint] = {
            "post": {
                "responses": responses
                or {"default": {"description": "Missing contract"}}
            }
        }
    roots = [
        pointer(
            "paths", endpoint, "post", "responses", status, "content", media, "schema"
        )
        for endpoint, path in result["paths"].items()
        for status, response in path["post"]["responses"].items()
        for media in response.get("content", {})
    ]
    validate_comparison_document(result, roots)
    return result, gaps


def response_differences(diff: dict) -> list[dict]:
    """Select complete status/media/body deltas from the retained oasdiff result.

    Deltas follow framework-to-Dynamo orientation and remain candidates, not
    confirmed incompatibilities. Request and component-only noise is excluded.
    """
    findings = []
    for endpoint, change in sorted(diff.get("paths", {}).get("modified", {}).items()):
        operation = change.get("operations", {}).get("modified", {}).get("POST", {})
        if operation.get("responses"):
            findings.append(
                {
                    "endpoint": endpoint,
                    "location": pointer("paths", endpoint, "post", "responses"),
                    "delta": operation["responses"],
                }
            )
    if diff.get("paths", {}).get("added") or diff.get("paths", {}).get("deleted"):
        raise ValueError("Scoped endpoint disappeared during response comparison")
    return findings
