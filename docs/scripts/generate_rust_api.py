#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Rust API reference page for Fern docs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import toml as tomllib  # type: ignore[no-redef]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fern_helpers import AUTOGEN_WARNING, REPO_ROOT, SPDX_HEADER

CORE_CRATES: set[str] = {
    "dynamo-runtime",
    "dynamo-llm",
    "dynamo-kv-router",
    "dynamo-memory",
    "kvbm-logical",
}
CRATE_ICONS: dict[str, str] = {
    "dynamo-runtime": "regular microchip",
    "dynamo-llm": "regular brain",
    "dynamo-kv-router": "regular route",
    "dynamo-memory": "regular memory",
    "kvbm-logical": "regular cubes",
    "dynamo-tokens": "regular key",
    "dynamo-config": "regular gear",
    "dynamo-parsers": "regular code",
    "dynamo-bench": "regular gauge-high",
    "dynamo-mocker": "regular vial",
    "kvbm-kernels": "regular microchip",
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
    name: str
    member_path: str
    description: str
    publish: bool
    icon: str = ""
    group: str = ""


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def parse_crates(root: Path) -> list[CrateInfo]:
    ws = _load_toml(root / "Cargo.toml")
    ws_desc = ws.get("workspace", {}).get("package", {}).get("description", "")
    members = list(dict.fromkeys(ws["workspace"]["members"]))
    crates = []
    for m in members:
        pkg = _load_toml(root / m / "Cargo.toml").get("package", {})
        desc = pkg.get("description", ws_desc)
        if isinstance(desc, dict):
            desc = ws_desc
        crates.append(
            CrateInfo(
                name=pkg.get("name", m.split("/")[-1]),
                member_path=m,
                description=desc,
                publish=pkg.get("publish") is not False,
            )
        )
    return crates


def _is_dev(desc: str) -> bool:
    low = desc.lower()
    return any(kw in low for kw in ("mock", "test", "bench"))


def classify(crates: list[CrateInfo]) -> dict[str, list[CrateInfo]]:
    groups: dict[str, list[CrateInfo]] = {
        "core": [],
        "supporting": [],
        "dev": [],
        "bindings": [],
    }
    for c in crates:
        c.icon = CRATE_ICONS.get(c.name, "regular box")
        if "bindings/" in c.member_path:
            c.group = "bindings"
        elif c.name in CORE_CRATES:
            c.group = "core"
        elif _is_dev(c.description):
            c.group = "dev"
        else:
            c.group = "supporting"
        groups[c.group].append(c)
    return groups


def _crate_url(c: CrateInfo, version: str) -> str:
    if not c.publish:
        return f"{GITHUB_BASE}/{c.member_path}"
    return f"https://docs.rs/{c.name}/{version}"


def _render_card(c: CrateInfo, version: str) -> str:
    url = _crate_url(c, version)
    return f'  <Card title="{c.name}" icon="{c.icon}" href="{url}">\n    {c.description}\n  </Card>'


def _render_card_group(crates: list[CrateInfo], cols: int, version: str) -> str:
    cards = "\n\n".join(_render_card(c, version) for c in crates)
    return f"<CardGroup cols={{{cols}}}>\n\n{cards}\n\n</CardGroup>"


def _render_bindings_table() -> str:
    lines = ["| Crate | Language | Source | Description |", "| --- | --- | --- | --- |"]
    for b in BINDINGS_INFO:
        src = f"[{b['member']}]({GITHUB_BASE}/{b['member']})"
        lines.append(
            f"| `{b['name']}` | {b['language']} | {src} | {b['description']} |"
        )
    return "\n".join(lines)


def render_page(groups: dict[str, list[CrateInfo]], version: str) -> str:
    parts = [
        SPDX_HEADER.format(sidebar_title="Rust API"),
        AUTOGEN_WARNING,
        "# Rust API Reference\n",
        "NVIDIA Dynamo's core infrastructure is implemented in Rust across multiple workspace crates.",
        "API documentation is hosted on [docs.rs](https://docs.rs) for published crates.\n",
        "## Core Crates\n",
        _render_card_group(groups["core"], 2, version) + "\n",
        "## Supporting Crates\n",
        _render_card_group(groups["supporting"], 3, version) + "\n",
    ]
    if groups["dev"]:
        parts += [
            "## Development & Testing\n",
            _render_card_group(groups["dev"], 3, version) + "\n",
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
    parser = argparse.ArgumentParser(description="Generate Rust API reference page")
    parser.add_argument("--version", default="latest")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "docs" / "api" / "rust"
    )
    args = parser.parse_args()

    crates = parse_crates(REPO_ROOT)
    groups = classify(crates)
    content = render_page(groups, args.version)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text(content)
    print(f"Wrote {args.output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
