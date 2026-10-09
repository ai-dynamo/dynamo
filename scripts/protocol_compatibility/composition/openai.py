# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve explicit import slots, never infer Dynamo contracts from a framework.

The manifest selects JSON Pointers in one pinned OpenAI document. Corrections
are standard JSON Patch with mandatory preceding test operations; they are
reviewed spec-to-code adjustments, not a source parser or permissive fallback.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import jsonpatch
import jsonpointer
import yaml

MARKER = "x-dynamo-schema-import"
PREFIX = "OpenAIBaseline."


def corrected(schema: Any, mapping: dict[str, Any], name: str) -> Any:
    """Apply reviewed JSON Patch, rejecting stale or unrelated guards."""
    operations = mapping.get("patch", [])
    for index, operation in enumerate(operations):
        if operation["op"] not in {"test", "add", "remove", "replace"}:
            raise ValueError(f"Unsupported correction operation: {operation['op']}")
        if operation["op"] == "test":
            continue
        if index == 0 or operations[index - 1]["op"] != "test":
            raise ValueError(f"Unguarded schema correction: {name}")
        expected_path = operation["path"]
        if operation["op"] == "add":
            expected_path = jsonpointer.JsonPointer.from_parts(
                jsonpointer.JsonPointer(expected_path).parts[:-1]
            ).path
        if operations[index - 1]["path"] != expected_path:
            raise ValueError(f"Correction test guards the wrong location: {name}")
        if not mapping.get("rationale"):
            raise ValueError(f"Schema correction needs a rationale: {name}")
    return jsonpatch.apply_patch(copy.deepcopy(schema), operations)


def compose(
    raw: dict[str, Any],
    manifest: dict[str, Any],
    *,
    baseline_bytes: bytes,
) -> dict[str, Any]:
    """Return a separate document, preserving raw inputs and endpoint definitions.

    Only schema components reachable from explicit slots are imported. All refs
    must be document-local; imports cannot overwrite Dynamo-owned definitions.
    Unmapped markers are errors, never a successful partially composed export.
    """
    if hashlib.sha256(baseline_bytes).hexdigest() != manifest["openai"]["sha256"]:
        raise ValueError("OpenAI specification checksum mismatch")
    baseline = yaml.safe_load(baseline_bytes)
    result = copy.deepcopy(raw)
    schemas = result["components"]["schemas"]
    imports: dict[str, dict[str, Any]] = {}
    mapped_pointers = {}
    for name, schema in schemas.items():
        if MARKER in schema:
            slot = schema[MARKER]
            mapping = manifest["imports"].get(slot["type"])
            if mapping and mapping.get("redirect_references", True):
                pointer = mapping["pointer"]
                if pointer in mapped_pointers:
                    raise ValueError(f"Ambiguous import mapping: {pointer}")
                mapped_pointers[pointer] = name

    def copy_schema(node: Any) -> Any:
        if isinstance(node, list):
            return [copy_schema(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$dynamicRef" in node or "$id" in node:
            raise ValueError(
                "Dynamic references and schema base-URI changes need explicit support"
            )
        copied = copy.deepcopy(node)
        # Walk schema positions only. Examples/defaults and property names may
        # themselves contain literal '$ref' keys that are not references.
        for keyword in (
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
        ):
            if keyword in node:
                copied[keyword] = {
                    key: copy_schema(value) for key, value in node[keyword].items()
                }
        for keyword in (
            "allOf",
            "anyOf",
            "oneOf",
            "prefixItems",
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
        ):
            if keyword in node:
                copied[keyword] = copy_schema(node[keyword])
        if "discriminator" in node and "mapping" in node["discriminator"]:
            copied["discriminator"]["mapping"] = {
                key: copy_schema({"$ref": reference})["$ref"]
                for key, reference in node["discriminator"]["mapping"].items()
            }
        if "$ref" in node:
            reference = node["$ref"]
            if not reference.startswith("#/components/schemas/"):
                raise ValueError(f"Unsupported imported reference: {reference}")
            if reference[1:] in mapped_pointers:
                copied["$ref"] = (
                    "#"
                    + jsonpointer.JsonPointer.from_parts(
                        ["components", "schemas", mapped_pointers[reference[1:]]]
                    ).path
                )
                return copied
            parts = jsonpointer.JsonPointer(reference[1:]).parts
            original = parts[2]
            name = PREFIX + original
            if name in schemas:
                raise ValueError(f"Imported component collides with raw schema: {name}")
            if name not in imports:
                # Reserve before recursion, so recursive references remain refs.
                imports[name] = {}
                root_pointer = jsonpointer.JsonPointer.from_parts(parts[:3]).path
                definition = jsonpointer.resolve_pointer(baseline, root_pointer)
                correction = manifest.get("components", {}).get(original, {})
                imports[name] = copy_schema(corrected(definition, correction, name))
            copied["$ref"] = (
                "#"
                + jsonpointer.JsonPointer.from_parts(
                    ["components", "schemas", name, *parts[3:]]
                ).path
            )
        return copied

    resolved = []
    for name, schema in list(schemas.items()):
        if MARKER not in schema:
            continue
        slot = schema[MARKER]
        if slot["crate"] != "async-openai":
            raise ValueError(f"Unsupported schema dependency: {slot['crate']}")
        mapping = manifest["imports"].get(slot["type"])
        if mapping is None:
            raise ValueError(f"Unmapped dependency type: {slot['type']}")
        replacement = corrected(
            jsonpointer.resolve_pointer(baseline, mapping["pointer"]), mapping, name
        )
        schemas[name] = copy_schema(replacement)
        resolved.append({"component": name, "mapping": mapping})
    schemas.update(imports)
    result["x-dynamo-schema-composition"] = {
        "openai": manifest["openai"],
        "dependencies": manifest["dependencies"],
        "resolved": resolved,
        "component_corrections": {
            name: mapping
            for name, mapping in manifest.get("components", {}).items()
            if PREFIX + name in imports
        },
        "exclusions": manifest["exclusions"],
    }
    return result
