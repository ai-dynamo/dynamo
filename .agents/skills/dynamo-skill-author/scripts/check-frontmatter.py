# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict frontmatter validator for a Dynamo skill's SKILL.md.

Checks:
  - YAML frontmatter parses.
  - Required fields present: name, description, version, author, tags, tools.
  - `name` matches the parent directory name and starts with `dynamo-`.
  - `description` is 50-500 chars and contains no angle-bracket placeholders.
  - `version` matches MAJOR.MINOR.PATCH.
  - `author` is `NVIDIA`.
  - `tags` includes `dynamo` and is 3-6 entries.
  - `tools` is a non-empty list.

Exits 0 on all-pass, non-zero on any failure. Output uses PASS / FAIL lines
on stdout for the agent to parse.

Usage:
    python3 check-frontmatter.py <path-to-SKILL.md>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("FAIL|deps|pyyaml not installed. Install via: pip install pyyaml", flush=True)
    sys.exit(2)


REQUIRED_FIELDS = {"name", "description", "version", "author", "tags", "tools"}
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
XML_TAG_RE = re.compile(r"<[a-zA-Z][^>]*>")


def split_frontmatter(text: str) -> str:
    parts = text.split("---", 2)
    if len(parts) < 3 or parts[0].strip():
        raise ValueError("file does not start with --- frontmatter delimiters")
    return parts[1]


def check(path: Path) -> int:
    failures = 0
    passes = 0

    def pf(ok: bool, name: str, detail: str) -> None:
        nonlocal failures, passes
        if ok:
            passes += 1
            print(f"PASS|{name}|{detail}", flush=True)
        else:
            failures += 1
            print(f"FAIL|{name}|{detail}", flush=True)

    if not path.is_file():
        print(f"FAIL|exists|{path} not found", flush=True)
        return 1

    raw = path.read_text(encoding="utf-8")

    try:
        fm_text = split_frontmatter(raw)
    except ValueError as e:
        print(f"FAIL|frontmatter|{e}", flush=True)
        return 1

    try:
        fm = yaml.safe_load(fm_text)
    except yaml.YAMLError as e:
        print(f"FAIL|yaml|parse error: {e}", flush=True)
        return 1

    if not isinstance(fm, dict):
        print("FAIL|yaml|frontmatter is not a mapping", flush=True)
        return 1

    missing = REQUIRED_FIELDS - set(fm.keys())
    pf(not missing, "required-fields", f"missing: {sorted(missing) or 'none'}")
    if missing:
        return 1

    name = fm["name"]
    pf(
        isinstance(name, str) and name.startswith("dynamo-"),
        "name-prefix",
        f"name='{name}' (must start with 'dynamo-')",
    )

    parent = path.parent.name
    pf(name == parent, "name-matches-dir", f"name='{name}' vs directory='{parent}'")

    desc = fm["description"]
    pf(isinstance(desc, str), "description-type", f"type={type(desc).__name__}")
    if isinstance(desc, str):
        length = len(desc)
        pf(50 <= length <= 500, "description-length", f"{length} chars (target 50-500)")
        xml_hits = XML_TAG_RE.findall(desc)
        pf(
            not xml_hits,
            "description-xml",
            f"angle-bracket placeholders detected: {xml_hits[:3] or 'none'}",
        )

    version = fm["version"]
    pf(
        isinstance(version, str) and bool(VERSION_RE.match(version)),
        "version-format",
        f"version='{version}' (must match MAJOR.MINOR.PATCH)",
    )

    author = fm["author"]
    pf(author == "NVIDIA", "author", f"author='{author}' (must be 'NVIDIA')")

    tags = fm["tags"]
    pf(isinstance(tags, list) and "dynamo" in tags, "tags-dynamo", f"tags={tags}")
    if isinstance(tags, list):
        pf(3 <= len(tags) <= 6, "tags-count", f"{len(tags)} tags (target 3-6)")

    tools = fm["tools"]
    pf(isinstance(tools, list) and len(tools) > 0, "tools-nonempty", f"tools={tools}")

    print(f"\nFrontmatter check: {passes} passed, {failures} failed", flush=True)
    return 0 if failures == 0 else 1


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 check-frontmatter.py <path-to-SKILL.md>", file=sys.stderr)
        return 2
    return check(Path(sys.argv[1]))


if __name__ == "__main__":
    sys.exit(main())
