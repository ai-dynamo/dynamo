#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Rust API reference page as GitHub-friendly Markdown.

Produces docs/api/rust/README.md with YAML frontmatter for Fern
compatibility but only standard Markdown constructs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import toml as tomllib  # type: ignore[no-redef]  -- fallback for Python <3.11

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fern_helpers import REPO_ROOT, render_markdown_table  # noqa: E402

FRONTMATTER = """\
---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Rust API
max-toc-depth: 2
---
"""

# Group classification constants
GROUP_CORE = "core"
GROUP_SUPPORTING = "supporting"
GROUP_DEV = "dev"
GROUP_BINDINGS = "bindings"

DISPLAYED_CRATES: set[str] = {
    "dynamo-runtime",
    "dynamo-llm",
    "dynamo-kv-router",
    "dynamo-memory",
    "kvbm-logical",
    "dynamo-config",
    "dynamo-async-openai",
    "dynamo-parsers",
    "dynamo-mocker",
}

CORE_CRATES: set[str] = {
    "dynamo-runtime",
    "dynamo-llm",
    "dynamo-kv-router",
    "dynamo-memory",
    "kvbm-logical",
}

DESCRIPTION_OVERRIDES: dict[str, str] = {
    "dynamo-runtime": "Distributed runtime for service discovery and RPC",
    "dynamo-llm": "LLM serving abstractions and pipeline orchestration",
    "dynamo-kv-router": "KV cache-aware request routing",
    "dynamo-memory": "Memory allocation and management for inference",
    "kvbm-logical": "Block allocation and lifecycle for KV cache",
    "dynamo-config": "Configuration parsing and environment utilities",
    "dynamo-async-openai": "Async OpenAI-compatible client for inference endpoints",
    "dynamo-parsers": "Output parsing for tool calling and reasoning",
    "dynamo-mocker": "Mock LLM scheduler and KV manager for testing",
}

CRATE_ICONS: dict[str, str] = {
    "dynamo-runtime": "regular microchip",
    "dynamo-llm": "regular brain",
    "dynamo-kv-router": "regular route",
    "dynamo-memory": "regular memory",
    "kvbm-logical": "regular cubes",
    "dynamo-config": "regular gear",
    "dynamo-parsers": "regular code",
    "dynamo-mocker": "regular vial",
    "dynamo-async-openai": "regular globe",
}
BINDINGS_INFO: list[dict[str, str]] = [
    {
        "member": "lib/bindings/python/codegen",
        "name": "dynamo-codegen",
        "language": "Python",
        "description": "PyO3 codegen (produces the `dynamo._core` module)",
    },
    {
        "member": "lib/bindings/c",
        "name": "libdynamo_llm",
        "language": "C",
        "description": "C bindings for the LLM library",
    },
]
GITHUB_BASE = "https://github.com/ai-dynamo/dynamo/tree/main"


@dataclass
class CrateInfo:
    """Metadata for a single Rust workspace crate."""

    name: str
    member_path: str
    description: str
    publish: bool
    icon: str
    group: str


def _load_toml(path: Path) -> dict:
    """Load and parse a TOML file."""
    try:
        return tomllib.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {path}: {exc}") from exc


def _classify_crate(name: str, member_path: str, desc: str) -> str:
    """Determine which group a crate belongs to."""
    if "bindings/" in member_path:
        return GROUP_BINDINGS
    if name in CORE_CRATES:
        return GROUP_CORE
    if any(kw in desc.lower() for kw in ("mock", "test", "bench")):
        return GROUP_DEV
    return GROUP_SUPPORTING


def parse_and_classify(root: Path) -> dict[str, list[CrateInfo]]:
    """Parse workspace Cargo.toml and classify crates into groups."""
    ws = _load_toml(root / "Cargo.toml")
    ws_desc = ws.get("workspace", {}).get("package", {}).get("description", "")
    members = list(dict.fromkeys(ws["workspace"]["members"]))

    groups: dict[str, list[CrateInfo]] = {
        GROUP_CORE: [],
        GROUP_SUPPORTING: [],
        GROUP_DEV: [],
        GROUP_BINDINGS: [],
    }
    for m in members:
        pkg = _load_toml(root / m / "Cargo.toml").get("package", {})
        name = pkg.get("name", m.split("/")[-1])
        if name not in DISPLAYED_CRATES:
            continue
        desc = DESCRIPTION_OVERRIDES.get(name, pkg.get("description", ws_desc))
        if isinstance(desc, dict):
            desc = ws_desc
        group = _classify_crate(name, m, desc)
        crate = CrateInfo(
            name=name,
            member_path=m,
            description=desc,
            publish=pkg.get("publish") is not False,
            icon=CRATE_ICONS.get(name, "regular box"),
            group=group,
        )
        groups[group].append(crate)
    return groups


def _crate_url(c: CrateInfo, version: str) -> str:
    """Build the URL for a crate (docs.rs or GitHub source)."""
    if not c.publish:
        return f"{GITHUB_BASE}/{c.member_path}"
    return f"https://docs.rs/{c.name}/{version}"


def _crates_to_cards(crates: list[CrateInfo], version: str) -> list[dict[str, str]]:
    """Convert CrateInfo list to card dicts for render_card_group."""
    return [
        {
            "title": c.name,
            "icon": c.icon,
            "href": _crate_url(c, version),
            "description": c.description,
        }
        for c in crates
    ]


def _render_bindings_table() -> str:
    """Render the bindings section as a Markdown table."""
    rows = [
        "| Crate | Language | Source | Description |",
        "| --- | --- | --- | --- |",
    ]
    for b in BINDINGS_INFO:
        src = f"[{b['member']}]({GITHUB_BASE}/{b['member']})"
        rows.append(f"| `{b['name']}` | {b['language']} | {src} | {b['description']} |")
    return "\n".join(rows)


def render_page(groups: dict[str, list[CrateInfo]], version: str) -> str:
    """Render the full Rust API reference page."""
    parts = [
        FRONTMATTER,
        "# Rust API Reference\n",
        "NVIDIA Dynamo's core infrastructure is implemented in Rust across multiple workspace crates.",
        "API documentation is hosted on [docs.rs](https://docs.rs) for published crates.\n",
        "## Core Crates\n",
        render_markdown_table(_crates_to_cards(groups[GROUP_CORE], version)),
        "## Supporting Crates\n",
        render_markdown_table(_crates_to_cards(groups[GROUP_SUPPORTING], version)),
    ]
    if groups[GROUP_DEV]:
        parts += [
            "## Development & Testing\n",
            render_markdown_table(_crates_to_cards(groups[GROUP_DEV], version)),
        ]
    parts += [
        "## Bindings\n",
        _render_bindings_table() + "\n",
        "## Building Locally\n",
        "```bash\ncargo doc --no-deps --workspace --open\n```\n",
        "This generates HTML documentation for all workspace crates and opens it in your browser.",
        "The output is written to `target/doc/`.\n",
    ]
    return "\n".join(parts)


def main() -> None:
    """Entry point: parse args and generate the Rust API reference page."""
    parser = argparse.ArgumentParser(description="Generate Rust API reference page")
    parser.add_argument("--version", default="latest")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "docs" / "api" / "rust"
    )
    args = parser.parse_args()

    groups = parse_and_classify(REPO_ROOT)
    content = render_page(groups, args.version)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text(content)
    print(f"Wrote {args.output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
