# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve inherited request fields and accepted input names for source consumers."""

from __future__ import annotations

import ast
from typing import Any

REQUESTS = {
    "/v1/chat/completions": "ChatCompletionRequest",
    "/v1/completions": "CompletionRequest",
}


def wire_names(name: str, declaration: dict[str, Any]) -> list[str]:
    """Conservatively protect Python names and statically declared input aliases."""
    annotation = ast.parse(declaration["annotation"], mode="eval").body
    if name.startswith("_"):
        return []
    if isinstance(annotation, ast.Subscript):
        annotation = annotation.value
    if (isinstance(annotation, ast.Name) and annotation.id == "ClassVar") or (
        isinstance(annotation, ast.Attribute) and annotation.attr == "ClassVar"
    ):
        return []
    names = {name}
    default = declaration["default"]
    if default is None:
        return sorted(names)
    expression = ast.parse(default, mode="eval").body
    if not isinstance(expression, ast.Call):
        return sorted(names)
    for keyword in expression.keywords:
        if keyword.arg not in {"alias", "validation_alias"}:
            continue
        value = keyword.value
        if isinstance(value, ast.Constant) and value.value is None:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            names.add(value.value)
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "AliasChoices"
            and not value.keywords
            and all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in value.args
            )
        ):
            names.update(item.value for item in value.args)
        else:
            raise ValueError(f"unresolved input alias for {name}: {ast.unparse(value)}")
    return sorted(names)


def request_fields(
    inventory: dict[str, Any],
    class_name: str,
    *,
    tolerate_unresolved_aliases: bool = False,
) -> dict[str, Any]:
    """Resolve local upstream inheritance; fail on ambiguity or an unknown base.

    BaseModel is the sole external leaf. We do not pretend to evaluate arbitrary
    Python inheritance, model metaclasses or dynamic Pydantic validators.
    """
    classes = {
        (path, name): value
        for path, module in inventory["modules"].items()
        for name, value in module["contract"]["classes"].items()
    }

    def resolve(name: str, preferred: str | None = None) -> tuple[str, str]:
        binding = (
            inventory.get("dependency_coverage", {})
            .get("class_bindings", {})
            .get(f"{preferred}:{name}")
        )
        if binding and binding["source"] is not None:
            return binding["source"], binding["name"]
        if preferred is not None and (preferred, name) in classes:
            return preferred, name
        matches = [key for key in classes if key[1] == name]
        if len(matches) != 1:
            raise ValueError(
                f"unresolved or ambiguous upstream class {name}: {matches}"
            )
        return matches[0]

    def collect(key: tuple[str, str], stack: tuple = ()) -> dict[str, Any]:
        if key in stack:
            raise ValueError(f"cyclic upstream inheritance: {key}")
        source, name = key
        contract = classes[key]
        fields = {}
        for base in reversed(contract["bases"]):
            binding = (
                inventory.get("dependency_coverage", {})
                .get("class_bindings", {})
                .get(f"{source}:{base}")
            )
            if (base == "BaseModel" and binding is None) or binding == {
                "source": None,
                "name": "BaseModel",
            }:
                continue
            fields.update(collect(resolve(base, source), (*stack, key)))
        for field, declaration in contract["fields"].items():
            try:
                names = wire_names(field, declaration)
            except ValueError:
                if not tolerate_unresolved_aliases:
                    raise
                # The direct extractor retains this declaration and supplies a
                # field-level diagnostic rather than dropping the whole request.
                names = [field]
            if names:
                fields[field] = {
                    **declaration,
                    "wire_names": names,
                    "declared_in": name,
                    "source": source,
                }
            else:
                fields.pop(field, None)
        return fields

    return collect(resolve(class_name))
