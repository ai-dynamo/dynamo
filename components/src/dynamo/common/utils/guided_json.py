# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted checks for JSON schemas passed to constrained-decoding backends."""

import json
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


def reject_cyclic_guided_json_ref_chain(schema: Any) -> None:
    """Reject root-reachable cycles made only of local ``$ref`` hops.

    This intentionally does not interpret JSON Schema applicators. Productive
    recursion and schemas outside this narrow failure mode remain backend-owned.
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            return

    if not isinstance(schema, dict):
        return

    resource = schema
    node = schema
    seen: set[int] = set()

    while True:
        node_id = id(node)
        if node_id in seen:
            raise _schema_error("circular local $ref chain detected")
        seen.add(node_id)

        ref = node.get("$ref")
        if not isinstance(ref, str) or not ref.startswith("#"):
            return
        resolved = _resolve_local_pointer(resource, ref)
        if resolved is None:
            return

        node, resource = resolved
