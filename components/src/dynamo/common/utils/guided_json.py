# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted checks for JSON schemas passed to constrained-decoding backends."""

import copy
import json
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any
from urllib.parse import unquote

from dynamo.llm import HttpError


def admits_only_empty_object(schema: Any) -> bool:
    """Return whether ``schema`` permits exactly the JSON value ``{}``."""
    if not isinstance(schema, dict):
        return False
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        return False
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or properties:
        return False
    if not isinstance(required, list) or required:
        return False
    for key, value in schema.items():
        if key in {
            "type",
            "properties",
            "required",
            "additionalProperties",
            "title",
            "description",
        }:
            continue
        if key == "minProperties" and type(value) is int and value == 0:
            continue
        if key == "maxProperties" and type(value) is int and value >= 0:
            continue
        if key in {
            "$comment",
            "default",
            "deprecated",
            "examples",
            "readOnly",
            "writeOnly",
        }:
            continue
        return False
    return True


# Keywords holding subschemas; literal keywords such as ``const`` are skipped.
# Mirrors ``rebase_schema_refs`` in lib/llm/src/protocols/openai/tools.rs.
_SCHEMA_MAP_KEYWORDS = frozenset(
    {
        "$defs",
        "definitions",
        "properties",
        "patternProperties",
        "dependentSchemas",
        "dependencies",
    }
)
_SCHEMA_KEYWORDS = frozenset(
    {
        "items",
        "prefixItems",
        "additionalItems",
        "contains",
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "propertyNames",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    }
)


def _rebase_schema_refs(schema: Any, root: str) -> bool:
    """Prefix fragment-only ``$ref`` values in schema positions with ``root``.

    An object with ``$id`` starts a separate resource, so its references keep
    that base and the walk stops there. Returns whether any reference changed.
    """
    rebased = False
    if isinstance(schema, list):
        for item in schema:
            rebased |= _rebase_schema_refs(item, root)
    elif isinstance(schema, dict):
        if "$id" in schema:
            # The engine owns reference resolution within this resource.
            return False
        reference = schema.get("$ref")
        if isinstance(reference, str) and reference.startswith("#"):
            fragment = reference[1:]
            if (
                not fragment
                or fragment.startswith("/")
                or fragment[:3].lower() == "%2f"
            ):
                schema["$ref"] = root + fragment
                rebased = True
        for keyword, value in schema.items():
            if keyword in _SCHEMA_MAP_KEYWORDS:
                if isinstance(value, dict):
                    for child in value.values():
                        rebased |= _rebase_schema_refs(child, root)
            elif keyword in _SCHEMA_KEYWORDS:
                rebased |= _rebase_schema_refs(value, root)
    return rebased


def build_required_tool_schema(
    tools: Iterable[tuple[str, dict[str, Any] | None]],
    parallel_tool_calls: bool | None,
) -> dict[str, Any]:
    """Build the ``tool_choice="required"`` JSON schema for an array of calls.

    ``tools`` holds ``(name, parameters)`` pairs. Nesting a parameter document
    under ``items.anyOf[i].properties.parameters`` would leave its local
    references (``#/definitions/...``, ``#/properties/...``, ``#``) pointing at
    the outer array, and would merge every tool's ``$defs`` into one map. A
    document that has local references or ``$defs`` therefore moves to
    ``$defs.tool_{i}`` with its references rebased onto that root; any other
    document stays inline. The callers' parameter dicts are not modified.
    Port of ``build_required_schema`` in lib/llm/src/protocols/openai/tools.rs.

    Raises ``ValueError`` when a tool's ``$defs`` is not an object.
    """
    defs: dict[str, Any] = {}
    any_of = []
    for index, (name, parameters) in enumerate(tools):
        # vLLM and SGLang treat empty parameters like missing ones.
        schema = (
            copy.deepcopy(parameters)
            if parameters
            else {"type": "object", "properties": {}}
        )
        if isinstance(schema, dict) and "$defs" in schema:
            if not isinstance(schema["$defs"], dict):
                raise ValueError(
                    f"`$defs` in the parameters of tool '{name}' must be an "
                    f"object, got {type(schema['$defs']).__name__}."
                )
        reference = f"#/$defs/tool_{index}"
        if _rebase_schema_refs(schema, reference) or (
            isinstance(schema, dict) and "$defs" in schema
        ):
            defs[f"tool_{index}"] = schema
            schema = {"$ref": reference}
        any_of.append(
            {
                "properties": {
                    "name": {"type": "string", "enum": [name]},
                    "parameters": schema,
                },
                "required": ["name", "parameters"],
            }
        )

    result: dict[str, Any] = {
        "type": "array",
        "minItems": 1,
        "items": {"type": "object", "anyOf": any_of},
    }
    if parallel_tool_calls is False:
        result["maxItems"] = 1
    if defs:
        result["$defs"] = defs
    return result


def _schema_error(message: str) -> HttpError:
    return HttpError(400, f"Invalid guided_json schema: {message}")


def _decode_pointer_token(token: str) -> str | None:
    decoded = []
    index = 0
    while index < len(token):
        char = token[index]
        if char != "~":
            decoded.append(char)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in ("0", "1"):
            return None
        decoded.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(decoded)


def _resolve_local_pointer(
    resource: dict[str, Any], ref: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    fragment = unquote(ref[1:])
    if not fragment:
        return resource, resource
    if not fragment.startswith("/"):
        return None

    target: Any = resource
    target_resource = resource
    for encoded_token in fragment[1:].split("/"):
        token = _decode_pointer_token(encoded_token)
        if token is None:
            return None
        if isinstance(target, dict):
            if token not in target:
                return None
            target = target[token]
        elif isinstance(target, list) and token.isdecimal():
            item_index = int(token)
            if item_index >= len(target):
                return None
            target = target[item_index]
        else:
            return None

        if isinstance(target, dict) and isinstance(target.get("$id"), str):
            target_resource = target

    if not isinstance(target, dict):
        return None
    return target, target_resource


def _child_resource(child: dict[str, Any], resource: dict[str, Any]) -> dict[str, Any]:
    return child if isinstance(child.get("$id"), str) else resource


def _iter_progressing_children(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for keyword in (
        "properties",
        "patternProperties",
        "dependentSchemas",
        "dependencies",
    ):
        children = node.get(keyword)
        if isinstance(children, dict):
            for child in children.values():
                if isinstance(child, dict):
                    yield child

    for keyword in (
        "additionalProperties",
        "unevaluatedProperties",
        "propertyNames",
        "items",
        "additionalItems",
        "prefixItems",
        "contains",
        "unevaluatedItems",
        "contentSchema",
    ):
        children = node.get(keyword)
        if isinstance(children, dict):
            yield children
        elif isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    yield child


def _has_cycle(
    edges: dict[tuple[int, int], set[tuple[int, int]]],
) -> bool:
    in_degree = {node: 0 for node in edges}
    for targets in edges.values():
        for target in targets:
            in_degree.setdefault(target, 0)
            in_degree[target] += 1

    ready = deque(node for node, degree in in_degree.items() if degree == 0)
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for target in edges.get(node, ()):
            in_degree[target] -= 1
            if in_degree[target] == 0:
                ready.append(target)

    return visited != len(in_degree)


def reject_nonprogressing_guided_json_ref_cycles(schema: Any) -> None:
    """Reject reachable local ``$ref`` cycles that cannot consume JSON.

    ``allOf`` preserves the current JSON instance, while object properties and
    array items cross a progress boundary. Choice applicators and schemas outside
    this narrow failure mode remain backend-owned.
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            return

    if not isinstance(schema, dict):
        return

    pending = [(schema, schema)]
    processed: set[tuple[int, int]] = set()
    edges: dict[tuple[int, int], set[tuple[int, int]]] = {}

    while pending:
        node, resource = pending.pop()
        node_key = (id(node), id(resource))
        if node_key in processed:
            continue
        processed.add(node_key)
        targets: set[tuple[int, int]] | None = None

        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#"):
            resolved = _resolve_local_pointer(resource, ref)
            if resolved is not None:
                target, target_resource = resolved
                targets = edges.setdefault(node_key, set())
                targets.add((id(target), id(target_resource)))
                pending.append((target, target_resource))

        all_of = node.get("allOf")
        if isinstance(all_of, list):
            for child in all_of:
                if not isinstance(child, dict):
                    continue
                child_resource = _child_resource(child, resource)
                if targets is None:
                    targets = edges.setdefault(node_key, set())
                targets.add((id(child), id(child_resource)))
                pending.append((child, child_resource))

        for child in _iter_progressing_children(node):
            pending.append((child, _child_resource(child, resource)))

    if _has_cycle(edges):
        raise _schema_error("non-progressing local $ref cycle detected")
