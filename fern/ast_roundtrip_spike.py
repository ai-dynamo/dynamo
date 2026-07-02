#!/usr/bin/env -S uv run --script

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mdformat==1.0.0",
#   "mdformat-frontmatter==2.1.2",
#   "mdformat-gfm==1.0.0",
#   "mdformat-gfm-alerts==2.0.0",
# ]
# ///

"""Measure Python Markdown AST round-trip compatibility across the docs tree.

This spike copies the input tree, parses and renders every Markdown/MDX file
with mdformat's markdown-it-py pipeline plus the local Fern MDX plugin, and
reports formatting churn, idempotence, and parser failures. It never edits the
source tree.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import mdformat
import mdformat.plugins

if __package__:
    from . import markdown_it_fern_mdx
else:
    import markdown_it_fern_mdx


MD_EXTENSIONS = {".md", ".mdx"}
PARSER_EXTENSIONS = ("frontmatter", "gfm", "gfm_alerts", "fern_mdx")


def register_fern_mdx_extension() -> None:
    """Register the local spike as an in-process mdformat extension."""

    mdformat.plugins.PARSER_EXTENSIONS.setdefault("fern_mdx", markdown_it_fern_mdx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output tree"
    )
    return parser.parse_args()


def round_trip(text: str) -> str:
    register_fern_mdx_extension()
    return mdformat.text(
        text,
        extensions=PARSER_EXTENSIONS,
        options={"wrap": "keep"},
    )


def new_extension_report() -> dict[str, Any]:
    return {
        "totalFiles": 0,
        "changedFiles": 0,
        "unchangedFiles": 0,
        "sourceBytes": 0,
        "outputBytes": 0,
    }


def main() -> int:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if (
        input_root == output_root
        or input_root in output_root.parents
        or output_root in input_root.parents
    ):
        raise SystemExit("input and output trees must not contain one another")

    if output_root.exists():
        if not args.force:
            raise SystemExit(
                f"output already exists: {output_root} (pass --force to replace it)"
            )
        shutil.rmtree(output_root)
    shutil.copytree(input_root, output_root)

    report: dict[str, Any] = {
        "parser": "mdformat/markdown-it-py",
        "parserExtensions": sorted(PARSER_EXTENSIONS),
        "input": str(input_root),
        "output": str(output_root),
        "byExtension": {
            extension: new_extension_report() for extension in sorted(MD_EXTENSIONS)
        },
        "changedFiles": [],
        "errors": [],
        "nonIdempotentFiles": [],
    }

    files = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix in MD_EXTENSIONS
    )
    for source_path in files:
        relative_path = source_path.relative_to(input_root)
        destination_path = output_root / relative_path
        extension_report = report["byExtension"][source_path.suffix]
        extension_report["totalFiles"] += 1

        source = source_path.read_text(encoding="utf-8")
        extension_report["sourceBytes"] += len(source.encode())
        try:
            rendered = round_trip(source)
            rerendered = round_trip(rendered)
        except Exception as error:  # noqa: BLE001 - the report records parser failures
            report["errors"].append(
                {
                    "file": str(relative_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        extension_report["outputBytes"] += len(rendered.encode())
        if rendered == source:
            extension_report["unchangedFiles"] += 1
        else:
            extension_report["changedFiles"] += 1
            report["changedFiles"].append(str(relative_path))
            destination_path.write_text(rendered, encoding="utf-8")

        if rerendered != rendered:
            report["nonIdempotentFiles"].append(str(relative_path))

    report_text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    return 1 if report["errors"] or report["nonIdempotentFiles"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
