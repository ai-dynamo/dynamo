# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Kustomize strategic-merge schema data from Dynamo CRDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CRD_DIR = REPO_ROOT / "deploy/operator/config/crd/bases"
OUTPUT_PATH = (
    REPO_ROOT / "recipes/kustomize/components/dynamo-openapi/dynamo-openapi.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the generated schema does not match the checked-in file",
    )
    return parser.parse_args()


def crd_paths() -> list[Path]:
    return sorted(CRD_DIR.glob("*.yaml"))


def pruned_schema(source: Any) -> dict[str, Any] | None:
    """Keep only paths needed to merge CRD map lists in Kustomize."""

    if not isinstance(source, dict):
        return None

    properties = {
        name: result
        for name, value in source.get("properties", {}).items()
        if (result := pruned_schema(value)) is not None
    }
    additional_properties = pruned_schema(source.get("additionalProperties"))
    items = pruned_schema(source.get("items"))

    list_type = source.get("x-kubernetes-list-type")
    map_keys = source.get("x-kubernetes-list-map-keys")
    if list_type == "map" and isinstance(map_keys, list) and len(map_keys) == 1:
        merge_key = map_keys[0]
        source_items = source.get("items")
        source_key_schema = (
            source_items.get("properties", {}).get(merge_key)
            if isinstance(source_items, dict)
            else None
        )
        item_schema = items or {"type": "object"}
        item_properties = item_schema.setdefault("properties", {})
        item_properties[merge_key] = {
            "type": source_key_schema.get("type", "string")
            if isinstance(source_key_schema, dict)
            else "string"
        }
        return {
            "type": "array",
            "items": item_schema,
            "x-kubernetes-patch-merge-key": merge_key,
            "x-kubernetes-patch-strategy": "merge",
        }

    if source.get("type") == "array" and items is not None:
        return {"type": "array", "items": items}

    if properties or additional_properties is not None:
        schema: dict[str, Any] = {"type": "object"}
        if properties:
            schema["properties"] = properties
        if additional_properties is not None:
            schema["additionalProperties"] = additional_properties
        return schema

    return None


def crd_definitions(crd_path: Path) -> dict[str, dict[str, Any]]:
    crd = yaml.safe_load(crd_path.read_text(encoding="utf-8"))
    if not isinstance(crd, dict):
        raise ValueError(f"{crd_path} must contain one CustomResourceDefinition")
    if crd.get("kind") != "CustomResourceDefinition":
        return {}

    group = crd.get("spec", {}).get("group")
    kind = crd.get("spec", {}).get("names", {}).get("kind")
    versions = crd.get("spec", {}).get("versions")
    if (
        not isinstance(group, str)
        or not isinstance(kind, str)
        or not isinstance(versions, list)
    ):
        raise ValueError(
            f"{crd_path} is missing CustomResourceDefinition identity fields"
        )

    definitions = {}
    for version_spec in versions:
        if not isinstance(version_spec, dict):
            continue
        version = version_spec.get("name")
        source_schema = version_spec.get("schema", {}).get("openAPIV3Schema")
        if not isinstance(version, str) or not isinstance(source_schema, dict):
            continue

        spec_schema = pruned_schema(source_schema.get("properties", {}).get("spec"))
        definition_name = f"{group}.{version}.{kind}"
        properties: dict[str, Any] = {
            "apiVersion": {"type": "string"},
            "kind": {"type": "string"},
            "metadata": {"type": "object"},
        }
        if spec_schema is not None:
            properties["spec"] = spec_schema
        definitions[definition_name] = {
            "type": "object",
            "properties": properties,
            "x-kubernetes-group-version-kind": [
                {"group": group, "version": version, "kind": kind}
            ],
        }

    return definitions


def generated_schema() -> str:
    definitions = {}
    for crd_path in crd_paths():
        definitions.update(crd_definitions(crd_path))
    if not definitions:
        raise ValueError("no Dynamo CRD definitions were generated")

    schema = {
        "x-generated-by": "scripts/generate_kustomize_openapi.py",
        "definitions": definitions,
    }
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    args = parse_args()
    try:
        rendered = generated_schema()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"generate_kustomize_openapi.py: {exc}", file=sys.stderr)
        return 1

    current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else None
    if args.check:
        if current == rendered:
            return 0
        print(
            f"Generated Kustomize OpenAPI schema is stale: {OUTPUT_PATH.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )
        print("Run: python3 scripts/generate_kustomize_openapi.py", file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
