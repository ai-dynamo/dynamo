# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted checks for JSON schemas passed to constrained-decoding backends."""

import json
from collections import deque
from collections.abc import Iterator
from typing import Any
from urllib.parse import unquote

from dynamo.llm import HttpError


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
