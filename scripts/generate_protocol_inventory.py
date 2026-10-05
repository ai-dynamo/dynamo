# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate vLLM admission vocabulary from pinned native-server declarations.

This vocabulary identifies known fields, NOT supported fields. Admission uses it
only to prevent silent dropping. Backend/version capability checks remain separate.
No upstream modules are imported or executed. Run --check in version-bump CI.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from protocol_drift import digest, snapshot

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "lib/llm/src/protocols/openai/compatibility"
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


def request_fields(inventory: dict[str, Any], class_name: str) -> dict[str, Any]:
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
            names = wire_names(field, declaration)
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


def build_inventory(repo: Path, pins: dict[str, Any]) -> dict[str, Any]:
    if pins["format_version"] != 1 or pins["target"] != "vllm":
        raise ValueError("unsupported pin format or target")
    profiles = []
    for pin in pins["versions"]:
        upstream = snapshot(repo, pin["commit"], "vllm")
        profiles.append(
            {
                **pin,
                "source_inventory_sha256": digest(upstream),
                "endpoints": {
                    endpoint: request_fields(upstream, request)
                    for endpoint, request in REQUESTS.items()
                },
            }
        )
    return {
        "generated": "DO NOT EDIT: scripts/generate_protocol_inventory.py",
        "format_version": 1,
        "target": "vllm",
        "repository": pins["repository"],
        "meaning": "Declaration inventory only; membership does not claim support or parity.",
        "profiles": profiles,
    }


def rust_vocabulary(inventory: dict[str, Any]) -> str:
    fields = sorted(
        {
            name
            for profile in inventory["profiles"]
            for endpoint in profile["endpoints"].values()
            for field in endpoint.values()
            for name in field["wire_names"]
        }
    )
    versions = ", ".join(profile["version"] for profile in inventory["profiles"])
    lines = [
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        "// DO NOT EDIT: generated by scripts/generate_protocol_inventory.py.",
        f"// Inventory SHA256: {digest(inventory)}",
        "// Known native fields are not necessarily supported by Dynamo.",
        f'pub(super) const VLLM_VERSIONS: &str = "{versions}";',
        "pub(super) const VLLM_REQUEST_FIELDS: &[&str] = &[",
        *(f"    {json.dumps(field)}," for field in fields),
        "];",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--pins", type=Path, default=OUTPUT / "vllm_pins.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--inventory-output",
        type=Path,
        help="optionally write/check the detailed JSON at this path; not required in Git",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    inventory = build_inventory(args.upstream_repo, json.loads(args.pins.read_text()))
    outputs = {
        "vllm_fields.rs": rust_vocabulary(inventory),
    }
    if args.inventory_output:
        outputs[str(args.inventory_output.resolve())] = (
            json.dumps(inventory, indent=2, sort_keys=True) + "\n"
        )
    stale = []
    for name, contents in outputs.items():
        path = args.output_dir / name
        if args.check:
            if not path.exists() or path.read_text() != contents:
                stale.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
    if stale:
        parser.exit(1, "Stale generated inventory: " + ", ".join(stale) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
