#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

FERN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FERN_ROOT.parent
API_REFERENCE_DIR = REPO_ROOT / "docs/kubernetes/api-reference"
CRD_DIR = REPO_ROOT / "deploy/operator/config/crd/bases"
SCHEMA_DIR = FERN_ROOT / "kubectl-doc-schemas"
SCHEMA_SOURCES = FERN_ROOT / "components/kubectl-doc/schemaSources.generated.ts"
VERSION_FILE = FERN_ROOT / "components/kubectl-doc/VERSION"
COMPONENT_DIR = FERN_ROOT / "components/kubectl-doc"
DEFAULT_KUBECTL_DOC_REPO = "https://github.com/sttts/kubectl-doc.git"
VENDORED_FILES = [
    "KubeSchemaDoc.tsx",
    "kubectl-doc-runtime.d.ts",
    "kubectl-doc-runtime.js",
    "kubectl-doc-styles.ts",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)

    gen_parser = subcommands.add_parser("gen")
    gen_parser.add_argument(
        "--checkout",
        default=os.environ.get("KUBECTL_DOC_CHECKOUT"),
        help="kubectl-doc checkout to run for rendering",
    )

    checkout_parser = subcommands.add_parser("checkout")
    checkout_parser.add_argument("--checkout", required=True)
    checkout_parser.add_argument("--ref", required=True)
    checkout_parser.add_argument(
        "--repo",
        default=os.environ.get("KUBECTL_DOC_REPO", DEFAULT_KUBECTL_DOC_REPO),
    )

    vendor_parser = subcommands.add_parser("vendor")
    vendor_parser.add_argument(
        "--checkout", default=os.environ.get("KUBECTL_DOC_CHECKOUT")
    )
    vendor_parser.add_argument("--ref", default=os.environ.get("KUBECTL_DOC_REF"))
    vendor_parser.add_argument(
        "--repo",
        default=os.environ.get("KUBECTL_DOC_REPO", DEFAULT_KUBECTL_DOC_REPO),
    )

    args = parser.parse_args()
    if args.command == "gen":
        if not args.checkout:
            sys.exit(
                "KUBECTL_DOC_CHECKOUT or --checkout must point to a kubectl-doc checkout"
            )
        generate_api_reference(Path(args.checkout))
    elif args.command == "checkout":
        ensure_checkout(Path(args.checkout), args.ref, args.repo)
    elif args.command == "vendor":
        vendor_component(args.checkout, args.ref, args.repo)


def generate_api_reference(kubectl_doc_checkout: Path) -> None:
    API_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

    for old_page in API_REFERENCE_DIR.glob("*.mdx"):
        old_page.unlink()
    for old_schema in SCHEMA_DIR.glob("*.json"):
        old_schema.unlink()

    schema_sources: dict[str, tuple[str, str | None]] = {}
    for crd_path in discover_crds():
        page_sources = generate_crd(kubectl_doc_checkout, crd_path)
        write_page(API_REFERENCE_DIR / page_name(page_sources[0].kind), page_sources)
        schema_sources.update(
            {
                source.name: (source.initial_file, source.full_file)
                for source in page_sources
            }
        )

    write_schema_sources(schema_sources)


def discover_crds() -> list[Path]:
    return sorted(CRD_DIR.glob("*.yaml"))


def ensure_checkout(checkout_dir: Path, git_ref: str, repo: str) -> None:
    if not (checkout_dir / ".git").is_dir():
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", repo, str(checkout_dir)],
            check=True,
        )

    subprocess.run(
        [
            "git",
            "-C",
            str(checkout_dir),
            "fetch",
            "--tags",
            "--depth=1",
            "origin",
            git_ref,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout_dir), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )


def vendor_component(checkout: str | None, ref: str | None, repo: str) -> None:
    ref = ref or VERSION_FILE.read_text().strip()
    checkout_path = (
        Path(checkout) if checkout else REPO_ROOT / ".cache/kubectl-doc-vendor" / ref
    )
    if not checkout:
        ensure_checkout(checkout_path, ref, repo)

    subprocess.run(["make", "-C", str(checkout_path), "gen"], check=True)
    for file_name in VENDORED_FILES:
        shutil.copyfile(
            checkout_path / "react/kubectl-doc" / file_name,
            COMPONENT_DIR / file_name,
        )

    VERSION_FILE.write_text(f"{ref}\n")
    print(f"Vendored kubectl-doc component {ref}.")
    print("Run 'make -C fern gen' to refresh Dynamo CRD API reference artifacts.")


class PageSchema:
    def __init__(
        self,
        name: str,
        api_version: str,
        kind: str,
        initial_file: str,
        full_file: str | None,
    ):
        self.name = name
        self.api_version = api_version
        self.kind = kind
        self.initial_file = initial_file
        self.full_file = full_file


def generate_crd(kubectl_doc_checkout: Path, crd_path: Path) -> list[PageSchema]:
    with tempfile.TemporaryDirectory(prefix="kubectl-doc-schemas-") as temp_dir:
        command = [
            "go",
            "run",
            "./cmd/kubectl-doc",
            "-f",
            str(crd_path),
            "-o",
            "markdown-fern",
            "--fern-schema-dir",
            temp_dir,
            "--fern-schema-url-path",
            ".",
            "--all-versions",
        ]
        result = subprocess.run(
            command,
            cwd=kubectl_doc_checkout,
            check=True,
            text=True,
            capture_output=True,
        )
        schemas = extract_schemas(result.stdout)
        page_sources: list[PageSchema] = []
        for index, schema in enumerate(schemas):
            kind = schema["kind"]
            name = f"{kind}Schema{index}"
            full_file = schema_full_file(schema)
            initial_file = (
                full_file.removesuffix("-full.json") + ".json"
                if full_file
                else schema_file_name(kind, index)
            )
            if full_file:
                schema["fullPayloadURL"] = f"./{full_file}"
                shutil.copyfile(Path(temp_dir) / full_file, SCHEMA_DIR / full_file)
            write_json(SCHEMA_DIR / initial_file, schema)
            page_sources.append(
                PageSchema(name, schema["apiVersion"], kind, initial_file, full_file)
            )
        return page_sources


def extract_schemas(mdx: str) -> list[dict[str, Any]]:
    marker = "export const kubectlDocSchemas = "
    marker_index = mdx.find(marker)
    if marker_index < 0:
        raise ValueError("kubectl-doc output did not contain kubectlDocSchemas")

    start = mdx.find("[", marker_index)
    if start < 0:
        raise ValueError("kubectlDocSchemas did not contain a JSON array")

    end = find_matching_bracket(mdx, start)
    return json.loads(mdx[start : end + 1])


def find_matching_bracket(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unterminated kubectlDocSchemas JSON array")


def schema_full_file(schema: dict[str, Any]) -> str | None:
    full_payload_url = schema.get("fullPayloadURL")
    if not full_payload_url:
        return None
    return str(full_payload_url).rsplit("/", maxsplit=1)[-1]


def schema_file_name(kind: str, index: int) -> str:
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", kind)
    slug = "-".join(word.lower() for word in words)
    return f"{slug}-schema-{index}.json"


def page_name(kind: str) -> str:
    return f"{kind.lower()}.mdx"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
    )


def write_page(path: Path, page_sources: list[PageSchema]) -> None:
    if not page_sources:
        raise ValueError(f"no schemas generated for {path}")

    title = page_sources[0].kind
    lines = [
        "---",
        f'title: "{title}"',
        "---",
        "",
        'import { LazyKubeSchemaDoc } from "@/components/kubectl-doc/LazyKubeSchemaDoc";',
        "",
    ]
    if len(page_sources) == 1:
        lines.append(f'<LazyKubeSchemaDoc name={{"{page_sources[0].name}"}} />')
    else:
        lines.append("<Tabs>")
        for source in page_sources:
            lines.extend(
                [
                    f'  <Tab title="{source.api_version}">',
                    f'    <LazyKubeSchemaDoc name={{"{source.name}"}} />',
                    "  </Tab>",
                ]
            )
        lines.append("</Tabs>")
    path.write_text("\n".join(lines) + "\n")


def write_schema_sources(schema_sources: dict[str, tuple[str, str | None]]) -> None:
    lines = [
        "export type SchemaSource = {",
        "  initial: string;",
        "  full?: string;",
        "};",
        "",
        'export const schemaBaseURL = "https://raw.githubusercontent.com/ai-dynamo/dynamo/main/fern/kubectl-doc-schemas";',
        "",
        "export const schemaSources: Record<string, SchemaSource> = {",
    ]
    for name in sorted(schema_sources):
        initial, full = schema_sources[name]
        if full:
            lines.append(f'  "{name}": {{ initial: "{initial}", full: "{full}" }},')
        else:
            lines.append(f'  "{name}": {{ initial: "{initial}" }},')
    lines.extend(["};", ""])
    SCHEMA_SOURCES.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
