# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate response declarations into schemas that oasdiff can compare.

Unary JSON bodies remain body schemas. Server-sent events (SSE) need a different
representation: decoded event data carries an encoding tag so a JSON string and
a literal terminator remain distinct. Selected chunk payloads are compared
separately from event envelopes; neither projection describes raw wire bytes.

prepare_response_source checks explicit adapters against input hashes, then
response_document or chunk_document returns comparison documents and coverage gaps.
openapi_workflow.py retains them and invokes the comparator. validate_sse_event
checks one already-parsed event for fixture validation, not stream ordering or
termination. No function here starts a server or parses an SSE byte stream.
"""

import copy
import json
import re

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


def prepare_response_source(
    document: dict, config: dict | None, sha256: str
) -> tuple[dict, dict, dict]:
    """Copy a source and validate hash-qualified route/stream selections.

    `config` is one sources entry from the response-adapter YAML documented in
    README.md; None selects native itemSchema handling without route mappings.
    `sha256` must identify the retained raw bytes, including when `document` is
    their composed Dynamo derivative. Caller source/revision labels are retained,
    not attested. Only the two explicit OpenAI /v1 route mappings are supported.
    Return the projection source, stream selections keyed by media pointer, and
    provenance. Invalid selectors, hashes, paths or collisions raise ValueError;
    no input is mutated and no source URL is fetched.
    """
    result = copy.deepcopy(document)
    if config is None:
        return (
            result,
            {},
            {"sha256": sha256, "interpretation": "native-itemSchema-only"},
        )
    required = {"sha256", "source", "revision", "streams"}
    if (
        not isinstance(config, dict)
        or not required <= config.keys()
        or set(config) - required - {"path_mappings"}
    ):
        raise ValueError("Adapter source requires sha256, source, revision and streams")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(config["sha256"]))
        or config["sha256"] != sha256
    ):
        raise ValueError("Response adapter raw checksum mismatch")
    if any(
        not isinstance(config[key], str) or not config[key].strip()
        for key in ("source", "revision")
    ):
        raise ValueError("Response adapter source and revision must be explicit")
    mappings = config.get("path_mappings", {})
    allowed = {
        "/chat/completions": "/v1/chat/completions",
        "/completions": "/v1/completions",
    }
    if not isinstance(mappings, dict) or any(
        allowed.get(key) != value for key, value in mappings.items()
    ):
        raise ValueError("Only explicit completion /v1 path mappings are supported")
    routes = []
    for original, target in mappings.items():
        paths = result.get("paths", {})
        if (
            original not in paths
            or target in paths
            or not isinstance(paths[original].get("post"), dict)
        ):
            raise ValueError("Missing source operation or colliding mapped endpoint")
        routes.append(
            {
                "source_operation": pointer("paths", original, "post"),
                "projection_operation": pointer("paths", target, "post"),
                "server_context": {
                    "document": copy.deepcopy(document.get("servers", [])),
                    "path": copy.deepcopy(paths[original].get("servers", [])),
                    "operation": copy.deepcopy(
                        paths[original]["post"].get("servers", [])
                    ),
                },
            }
        )
        paths[target] = paths.pop(original)
    if not isinstance(config["streams"], list):
        raise ValueError("Adapter streams must be a list")
    streams = {}
    for entry in config["streams"]:
        if (
            not isinstance(entry, dict)
            or not {"pointer", "interpretation"} <= entry.keys()
            or set(entry) - {"pointer", "interpretation", "chunk_pointer"}
        ):
            raise ValueError("Stream selection requires pointer and interpretation")
        mode = entry["interpretation"]
        if mode not in {"event-schema", "event-item-schema", "decoded-chunk"}:
            raise ValueError("Unknown SSE interpretation")
        dialect = "3.2." if mode == "event-item-schema" else "3.1."
        if not str(document.get("openapi", "")).startswith(dialect):
            raise ValueError(
                "SSE interpretation does not match the source OpenAPI dialect"
            )
        location = entry["pointer"]
        if not isinstance(location, str) or not location.startswith("#/"):
            raise ValueError("Stream pointer must be a local JSON Pointer")
        parts = jsonpointer.JsonPointer(location[1:]).parts
        keyword = "itemSchema" if mode == "event-item-schema" else "schema"
        if (
            len(parts) != 8
            or parts[0] != "paths"
            or parts[2:4] != ["post", "responses"]
            or parts[5:] != ["content", "text/event-stream", keyword]
            or mappings.get(parts[1], parts[1]) not in ENDPOINTS
        ):
            raise ValueError("Stream pointer must select a scoped SSE media schema")
        jsonpointer.resolve_pointer(document, location[1:])
        projection_parts = [*parts]
        projection_parts[1] = mappings.get(parts[1], parts[1])
        projected_pointer = pointer(*projection_parts)
        media_pointer = pointer(*projection_parts[:-1])
        if media_pointer in streams:
            raise ValueError("Duplicate SSE stream selection")
        selection = {**copy.deepcopy(entry), "projection_pointer": projected_pointer}
        if "chunk_pointer" in entry:
            chunk = entry["chunk_pointer"]
            if (
                not isinstance(chunk, str)
                or not chunk.startswith(location + "/")
                or not chunk.endswith("/contentSchema")
                or mode == "decoded-chunk"
            ):
                raise ValueError(
                    "Chunk pointer must select JSON contentSchema within the chosen event"
                )
            selection["projection_chunk_pointer"] = (
                projected_pointer + chunk[len(location) :]
            )
        streams[media_pointer] = selection
    return (
        result,
        streams,
        {
            "sha256": sha256,
            "source": config["source"],
            "revision": config["revision"],
            "identity_verification": "raw_hash_only; source_and_revision_are_caller_assertions",
            "path_mappings": routes,
            "streams": list(streams.values()),
        },
    )


def decoded_chunk_schema(document: dict, schema: dict) -> dict:
    """Copy a source-qualified Chat chunk schema after checking its bounded shape.

    Only object declarations with choices and no envelope data property, directly
    or through pure local component refs, are supported. References and all
    payload constraints remain unchanged. Ambiguous forms raise ValueError or
    JsonPointerException so callers can retain a coverage gap.
    """
    root, refs = schema, set()
    while isinstance(root, dict) and "$ref" in root:
        ref = root["$ref"]
        if (
            not isinstance(ref, str)
            or not ref.startswith("#/components/schemas/")
            or ref in refs
            or set(root) != {"$ref"}
        ):
            raise ValueError("Unsupported decoded chunk reference")
        refs.add(ref)
        root = jsonpointer.resolve_pointer(document, ref[1:])
    if (
        not isinstance(root, dict)
        or root.get("type") != "object"
        or "choices" not in root.get("properties", {})
        or "data" in root.get("properties", {})
        or any(key in root for key in ("anyOf", "oneOf", "allOf"))
    ):
        raise ValueError(
            "Decoded chunk must explicitly describe a Chat choices object, not an event envelope"
        )
    return copy.deepcopy(schema)


def chunk_document(document: dict, selection: dict) -> tuple[dict, list[dict]]:
    """Project one explicitly selected decoded chunk for a separate oasdiff probe.

    `selection` comes from prepare_response_source. Event selections must point
    to a JSON data alternative, not an arbitrary nested schema; refs in that
    payload are preserved. Return a two-route comparison document and chunk gaps.
    Unsupported selectors/encodings raise ValueError or JsonPointerException.
    This probe does not compare envelopes or assert that the selected branch is
    the complete event union. Inputs are unchanged.
    """
    location = selection["projection_pointer"]
    if selection["interpretation"] == "decoded-chunk":
        chunk = decoded_chunk_schema(
            document, jsonpointer.resolve_pointer(document, location[1:])
        )
    else:
        chunk_location = selection.get("projection_chunk_pointer")
        if not chunk_location:
            raise ValueError("No explicit shared chunk selector")
        # Decode the full event first so no unsupported envelope/string assertion
        # disappears merely because its chunk can be selected.
        decoded_event_schema(
            document, jsonpointer.resolve_pointer(document, location[1:])
        )
        suffix = chunk_location[len(location) :]
        if not re.fullmatch(
            r"(?:/(?:anyOf|oneOf)/[0-9]+)*/properties/data(?:/anyOf/[0-9]+)*/contentSchema",
            suffix,
        ):
            raise ValueError("Chunk selector is not a direct JSON data alternative")
        data = jsonpointer.resolve_pointer(
            document, chunk_location[1:].rsplit("/", 1)[0]
        )
        decoded_data_schema(data)
        chunk = data["contentSchema"]
    parts = jsonpointer.JsonPointer(location[1:]).parts
    probe = {
        "openapi": "3.1.0",
        "info": {"title": "Selected chunk only", "version": "1"},
        "components": copy.deepcopy(document.get("components", {})),
        "paths": {
            endpoint: {"post": {"responses": {"204": {"description": "Not selected"}}}}
            for endpoint in ENDPOINTS
        },
    }
    probe["paths"][parts[1]]["post"]["responses"] = {
        parts[4]: {
            "description": "Selected chunk only",
            "content": {"application/json": {"schema": copy.deepcopy(chunk)}},
        }
    }
    projected, gaps = response_document(probe)
    return projected, [
        {
            **gap,
            "layer": "chunk",
            "source_pointer": selection.get("chunk_pointer", selection["pointer"]),
        }
        for gap in gaps
    ]


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
    """Copy an event-envelope schema, decoding its declared data alternatives.

    Accept a native itemSchema or an adapter-selected OpenAPI 3.1 event schema.
    `refs` tracks the active reference chain; callers normally leave it empty.
    Resolve local event-envelope refs and anyOf/oneOf event variants, retaining
    envelope constraints. Unwrap a bare single-alternative event union; payload
    refs stay refs. Missing/recursive/external event
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
    if result.get("type", "object") != "object":
        raise ValueError("SSE envelope must describe an object")
    found = False
    for keyword in ("anyOf", "oneOf"):
        if keyword in result:
            if not isinstance(result[keyword], list) or not result[keyword]:
                raise ValueError("SSE event alternatives must be a nonempty list")
            result[keyword] = [
                decoded_event_schema(document, branch, refs)
                for branch in result[keyword]
            ]
            if set(result) == {keyword} and len(result[keyword]) == 1:
                return result[keyword][0]
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


def response_document(
    document: dict, streams: dict | None = None
) -> tuple[dict, list[dict]]:
    """Copy scoped POST response contracts into comparator inputs and return gaps.

    Keep status/media distinctions and reachable payload components. Convert YAML
    integer status keys to strings in the projection. JSON bodies
    retain their schemas; SSE itemSchema becomes a decoded-event comparison schema.
    `streams` must contain hash-qualified selections from prepare_response_source.
    Explicit event-schema selections reuse the native decoder; decoded-chunk
    selections preserve payloads and record missing event layers separately.
    Missing/unsupported schemas are retained as empty comparison placeholders with
    explicit gaps, never evidence of compatibility. Missing operations and invalid
    document dialects raise. Input is unchanged; references are never fetched.
    """
    check_schema_dialect(document)
    streams = streams or {}
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
            status = str(status)
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
                        selection = streams.get(source)
                        mode = (
                            selection["interpretation"]
                            if selection
                            else "event-item-schema"
                        )
                        if mode == "event-item-schema" and not document[
                            "openapi"
                        ].startswith("3.2."):
                            raise ValueError(
                                "Native SSE itemSchema interpretation requires OpenAPI 3.2"
                            )
                        keyword = (
                            "itemSchema" if mode == "event-item-schema" else "schema"
                        )
                        if (
                            keyword not in contract
                            or ({"schema", "itemSchema"} - {keyword}) & contract.keys()
                        ):
                            raise ValueError(
                                "SSE comparison requires one explicitly selected event representation"
                            )
                        source += "/" + keyword
                        if mode == "decoded-chunk":
                            schema = decoded_chunk_schema(document, contract[keyword])
                            gaps.extend(
                                {
                                    "endpoint": endpoint,
                                    "location": selection["pointer"],
                                    "layer": layer,
                                    "reason": "Decoded chunk declaration does not specify this event layer",
                                }
                                for layer in (
                                    "envelope",
                                    "encoding",
                                    "error",
                                    "terminator",
                                    "extensions",
                                )
                            )
                        else:
                            schema = decoded_event_schema(document, contract[keyword])
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
                        {
                            "endpoint": endpoint,
                            "location": source,
                            "layer": "event"
                            if media == "text/event-stream"
                            else "body",
                            "reason": str(error),
                        }
                    )
        result["paths"][endpoint] = {
            "post": {
                "responses": responses
                or {"default": {"description": "Missing contract"}}
            }
        }
    validate_comparison_document(result, response_schema_roots(result))
    return result, gaps


def response_schema_roots(document: dict) -> list[str]:
    """List media schema pointers in a response projection, including empty placeholders.

    Accept output from response_document, not a raw export with itemSchema.
    Each declared media entry must contain a schema. The document is unchanged.
    """
    return [
        pointer(
            "paths", endpoint, "post", "responses", status, "content", media, "schema"
        )
        for endpoint, path in document["paths"].items()
        for status, response in path["post"]["responses"].items()
        for media in response.get("content", {})
    ]


def response_differences(diff: dict) -> list[dict]:
    """Select complete status/media/body deltas from the retained oasdiff result.

    Deltas follow framework-to-Dynamo orientation and remain candidates, not
    confirmed incompatibilities. Request and component-only noise is excluded.
    Records refer to raw deltas; treat them as read-only. Added/deleted scoped
    paths raise ValueError instead of reporting an empty diff.
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
