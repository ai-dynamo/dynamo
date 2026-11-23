# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON Schema validation for decoded arguments, without schema retrieval."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterator
from itertools import islice
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from referencing import Registry, Resource
from referencing.exceptions import CannotDetermineSpecification, Unresolvable
from referencing.jsonschema import DRAFT202012


class SchemaValidationUnavailable(ValueError):
    """The verifier cannot safely evaluate this schema; it did not pass."""


def _reject_retrieval(uri: str) -> Resource:
    raise SchemaValidationUnavailable(f"external schema retrieval is disabled: {uri}")


def _schema_nodes(schema: Any, depth: int = 0) -> Iterator[dict[str, Any]]:
    if depth > 64:
        raise SchemaValidationUnavailable("schema nesting exceeds the verifier limit")
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise SchemaValidationUnavailable("a schema must be an object or boolean")
    yield schema
    for keyword in (
        "$defs",
        "definitions",
        "properties",
        "patternProperties",
        "dependentSchemas",
    ):
        children = schema.get(keyword)
        if isinstance(children, dict):
            for child in children.values():
                yield from _schema_nodes(child, depth + 1)
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                yield from _schema_nodes(child, depth + 1)
    for keyword in (
        "items",
        "additionalItems",
        "additionalProperties",
        "unevaluatedItems",
        "unevaluatedProperties",
        "propertyNames",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    ):
        child = schema.get(keyword)
        if isinstance(child, list) and keyword == "items":
            for item in child:
                yield from _schema_nodes(item, depth + 1)
        elif isinstance(child, (dict, bool)):
            yield from _schema_nodes(child, depth + 1)
    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        for child in dependencies.values():
            if isinstance(child, (dict, bool)):
                yield from _schema_nodes(child, depth + 1)


def _checked_nodes(schema: Any) -> list[dict[str, Any]]:
    nodes = list(islice(_schema_nodes(schema), 4097))
    if len(nodes) > 4096:
        raise SchemaValidationUnavailable("schema size exceeds the verifier limit")
    for node in nodes:
        if "$schema" in node and validator_for(node, default=None) is None:
            raise SchemaValidationUnavailable(
                f"unsupported JSON Schema dialect: {node['$schema']!r}"
            )
        if any(
            keyword in node
            for keyword in (
                "$id",
                "id",
                "$dynamicRef",
                "$recursiveRef",
                "$dynamicAnchor",
                "$recursiveAnchor",
            )
        ):
            raise SchemaValidationUnavailable(
                "schema scope changes and dynamic references are unsupported"
            )
        reference = node.get("$ref")
        if reference is not None and (
            not isinstance(reference, str) or not reference.startswith("#")
        ):
            raise SchemaValidationUnavailable(
                "only document-local schema references are supported"
            )
    return nodes


def _validator(schema: Any):
    nodes = _checked_nodes(schema)
    validator_class = validator_for(schema, default=Draft202012Validator)
    validator_class.check_schema(schema)
    resource = Resource.from_contents(schema, default_specification=DRAFT202012)
    registry = Registry(retrieve=_reject_retrieval).with_resource("", resource)
    resolver = registry.resolver()
    pending = deque(nodes)
    seen: set[int] = set()
    while pending:
        node = pending.popleft()
        node_identity = id(node)
        if node_identity in seen:
            continue
        seen.add(node_identity)
        if len(seen) > 4096:
            raise SchemaValidationUnavailable("schema size exceeds the verifier limit")
        if "$ref" in node:
            target = resolver.lookup(node["$ref"]).contents
            pending.extend(_checked_nodes(target))
            target_validator = validator_for(target, default=validator_class)
            target_validator.check_schema(target)
    return validator_class(schema, registry=registry)


def validate_json_schema(
    value: Any, schema: dict[str, Any] | bool, path: str = "$"
) -> list[str]:
    """Return instance violations, or raise when validation cannot be completed."""

    try:
        validator = _validator(schema)
        violations = list(validator.iter_errors(value))
    except (
        SchemaError,
        Unresolvable,
        CannotDetermineSpecification,
        RecursionError,
        re.error,
    ) as exc:
        raise SchemaValidationUnavailable(str(exc)) from exc
    problems = []
    for violation in violations:
        location = path + "".join(
            f"[{component}]" if isinstance(component, int) else f".{component}"
            for component in violation.absolute_path
        )
        problems.append(f"{location}: {violation.message}")
    return sorted(problems)
