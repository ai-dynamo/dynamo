# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation for JSON schemas passed to constrained-decoding backends."""

import json
from collections import deque
from typing import Any, Iterator
from urllib.parse import unquote

from dynamo.llm import HttpError

_NON_CONSUMING_ARRAY_KEYWORDS = ("allOf", "anyOf", "oneOf")
_NON_CONSUMING_SINGLE_KEYWORDS = ("not", "if", "then", "else")
_CONSUMING_ARRAY_KEYWORDS = ("prefixItems",)
_CONSUMING_SINGLE_KEYWORDS = (
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "items",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
)
_SCHEMA_MAP_KEYWORDS = (
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
    "patternProperties",
    "properties",
)
_NON_CONSUMING_MAP_KEYWORDS = frozenset(("dependentSchemas", "dependencies"))


def _schema_error(message: str) -> HttpError:
    return HttpError(400, f"Invalid guided_json schema: {message}")


def _is_schema(value: Any) -> bool:
    return isinstance(value, (dict, bool))


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _decode_pointer_token(token: str, ref: str) -> str:
    decoded = []
    index = 0
    while index < len(token):
        char = token[index]
        if char != "~":
            decoded.append(char)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in ("0", "1"):
            raise _schema_error(f"invalid JSON pointer escape in $ref {ref!r}")
        decoded.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(decoded)


def _resolve_local_ref(
    root: Any,
    ref: str,
    anchors: dict[str, tuple[Any, str]],
) -> tuple[Any, str] | None:
    if not ref.startswith("#"):
        # Preserve the backend's existing handling for external references.
        return None

    fragment = unquote(ref[1:])
    if not fragment:
        return root, "#"
    if not fragment.startswith("/"):
        anchor = anchors.get(fragment)
        if anchor is None:
            # Leave unsupported local-anchor behavior to the grammar backend.
            return None
        return anchor

    target = root
    canonical_path = "#"
    for encoded_token in fragment[1:].split("/"):
        token = _decode_pointer_token(encoded_token, ref)
        canonical_path += f"/{_escape_pointer_token(token)}"
        if isinstance(target, dict):
            if token not in target:
                raise _schema_error(f"unresolvable local $ref {ref!r}")
            target = target[token]
        elif isinstance(target, list):
            if not token.isdecimal():
                raise _schema_error(f"unresolvable local $ref {ref!r}")
            item_index = int(token)
            if item_index >= len(target):
                raise _schema_error(f"unresolvable local $ref {ref!r}")
            target = target[item_index]
        else:
            raise _schema_error(f"unresolvable local $ref {ref!r}")

    return target, canonical_path


def _schema_children(
    schema: dict[str, Any],
    path: str,
) -> Iterator[tuple[Any, str, bool]]:
    for keyword in _NON_CONSUMING_ARRAY_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for index, child in enumerate(children):
                if _is_schema(child):
                    yield child, f"{path}/{keyword}/{index}", True

    for keyword in _NON_CONSUMING_SINGLE_KEYWORDS:
        child = schema.get(keyword)
        if _is_schema(child):
            yield child, f"{path}/{keyword}", True

    for keyword in _CONSUMING_ARRAY_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for index, child in enumerate(children):
                if _is_schema(child):
                    yield child, f"{path}/{keyword}/{index}", False

    for keyword in _CONSUMING_SINGLE_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, list):
            for index, item in enumerate(child):
                if _is_schema(item):
                    yield item, f"{path}/{keyword}/{index}", False
        elif _is_schema(child):
            yield child, f"{path}/{keyword}", False

    for keyword in _SCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if not isinstance(children, dict):
            continue
        non_consuming = keyword in _NON_CONSUMING_MAP_KEYWORDS
        for name, child in children.items():
            if _is_schema(child):
                escaped_name = _escape_pointer_token(str(name))
                yield child, f"{path}/{keyword}/{escaped_name}", non_consuming


def _discover_schema_nodes(root: dict[str, Any]) -> dict[int, tuple[dict, str]]:
    nodes: dict[int, tuple[dict, str]] = {}
    pending = deque([(root, "#")])

    while pending:
        value, path = pending.popleft()
        if not isinstance(value, dict) or id(value) in nodes:
            continue
        nodes[id(value)] = (value, path)
        for child, child_path, _ in _schema_children(value, path):
            if isinstance(child, dict):
                pending.append((child, child_path))

    return nodes


def _collect_anchors(
    nodes: dict[int, tuple[dict, str]],
) -> dict[str, tuple[Any, str]]:
    anchors: dict[str, tuple[Any, str]] = {}
    for schema, path in nodes.values():
        anchor = schema.get("$anchor")
        if not isinstance(anchor, str):
            continue
        if anchor in anchors:
            raise _schema_error(f"duplicate local $anchor {anchor!r}")
        anchors[anchor] = (schema, path)
    return anchors


def _find_non_consuming_cycle(
    edges: dict[int, set[int]],
    paths: dict[int, str],
) -> list[str] | None:
    state: dict[int, int] = {}

    for start in edges:
        if state.get(start, 0) != 0:
            continue

        state[start] = 1
        active_nodes = [start]
        active_indexes = {start: 0}
        stack = [(start, iter(edges.get(start, ())))]

        while stack:
            node, neighbors = stack[-1]
            try:
                neighbor = next(neighbors)
            except StopIteration:
                state[node] = 2
                active_indexes.pop(node)
                active_nodes.pop()
                stack.pop()
                continue

            neighbor_state = state.get(neighbor, 0)
            if neighbor_state == 0:
                state[neighbor] = 1
                active_indexes[neighbor] = len(active_nodes)
                active_nodes.append(neighbor)
                stack.append((neighbor, iter(edges.get(neighbor, ()))))
            elif neighbor_state == 1:
                cycle_start = active_indexes[neighbor]
                cycle = active_nodes[cycle_start:] + [neighbor]
                return [paths[node_id] for node_id in cycle]

    return None


def validate_guided_json_schema(schema: Any) -> None:
    """Reject local-reference cycles that make no structural progress.

    Recursive object properties and array items are productive because they
    descend into a child JSON value. Direct references and same-instance schema
    combinators do not consume input, so a cycle containing only those edges can
    put a constrained-decoding backend into an invalid state.
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            # Preserve the backend's existing malformed-JSON error behavior.
            return

    if not isinstance(schema, dict):
        return

    nodes = _discover_schema_nodes(schema)
    anchors = _collect_anchors(nodes)
    edges: dict[int, set[int]] = {node_id: set() for node_id in nodes}
    paths = {node_id: path for node_id, (_, path) in nodes.items()}
    pending = deque(nodes)
    processed: set[int] = set()

    while pending:
        node_id = pending.popleft()
        if node_id in processed:
            continue
        processed.add(node_id)
        node, path = nodes[node_id]

        for child, child_path, non_consuming in _schema_children(node, path):
            if not isinstance(child, dict):
                continue
            child_id = id(child)
            if child_id not in nodes:
                nodes[child_id] = (child, child_path)
                paths[child_id] = child_path
                edges[child_id] = set()
                pending.append(child_id)
            if non_consuming:
                edges[node_id].add(child_id)

        if "$ref" not in node:
            continue
        ref = node["$ref"]
        if not isinstance(ref, str):
            raise _schema_error(f"$ref at {path} must be a string")
        resolved = _resolve_local_ref(schema, ref, anchors)
        if resolved is None:
            continue
        target, target_path = resolved
        if not _is_schema(target):
            raise _schema_error(f"local $ref {ref!r} does not target a schema")
        if not isinstance(target, dict):
            continue

        target_id = id(target)
        if target_id not in nodes:
            nodes[target_id] = (target, target_path)
            paths[target_id] = target_path
            edges[target_id] = set()
            pending.append(target_id)
        edges[node_id].add(target_id)

    cycle = _find_non_consuming_cycle(edges, paths)
    if cycle is not None:
        raise _schema_error(
            "unproductive circular $ref detected: " + " -> ".join(cycle)
        )
