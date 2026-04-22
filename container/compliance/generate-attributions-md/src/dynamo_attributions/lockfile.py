# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lock file parsers for transitive dependency extraction.

Parses resolved dependency information from:
- Cargo.lock (Rust)
- go.mod (Go direct + indirect)
- pip freeze output (Python)
"""

from __future__ import annotations

import re

from .types import Ecosystem, ResolvedPackage


def parse_cargo_lock(
    content: str,
    workspace_members: list[str] | None = None,
) -> list[ResolvedPackage]:
    """Parse Cargo.lock into resolved packages with dependency edges."""
    if not content.strip():
        return []

    packages: list[ResolvedPackage] = []
    blocks = re.split(r"\n\[\[package\]\]\n", "\n" + content)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        name_m = re.search(r'^name\s*=\s*"([^"]+)"', block, re.MULTILINE)
        ver_m = re.search(r'^version\s*=\s*"([^"]+)"', block, re.MULTILINE)
        if not name_m or not ver_m:
            continue

        name = name_m.group(1)
        version = ver_m.group(1)

        source_m = re.search(r'^source\s*=\s*"([^"]+)"', block, re.MULTILINE)
        source = source_m.group(1) if source_m else ""

        deps: list[str] = []
        deps_m = re.search(
            r"^dependencies\s*=\s*\[(.*?)\]", block, re.MULTILINE | re.DOTALL
        )
        if deps_m:
            dep_entries = re.findall(r'"([^"]+)"', deps_m.group(1))
            for entry in dep_entries:
                dep_name = entry.split()[0]
                deps.append(dep_name)

        if workspace_members is not None:
            is_direct = name in workspace_members
        else:
            is_direct = source == ""

        packages.append(
            ResolvedPackage(
                name=name,
                version=version,
                ecosystem=Ecosystem.RUST,
                is_direct=is_direct,
                dependencies=deps,
                source=source,
            )
        )

    return packages


def _parse_go_require_line(
    line: str,
    seen: set[str],
    packages: list[ResolvedPackage],
) -> None:
    """Append one require entry from ``line`` (already stripped of comments)."""
    is_indirect = "// indirect" in line
    line_clean = re.sub(r"//.*$", "", line).strip()
    parts = line_clean.split()
    if len(parts) < 2:
        return

    module_path = parts[0]
    version = parts[1]

    if module_path in seen:
        return
    seen.add(module_path)

    packages.append(
        ResolvedPackage(
            name=module_path,
            version=version,
            ecosystem=Ecosystem.GO,
            is_direct=not is_indirect,
            dependencies=[],
            source="",
        )
    )


def parse_go_mod(content: str) -> list[ResolvedPackage]:
    """Parse go.mod into resolved packages.

    Handles three syntactic forms documented in
    https://go.dev/ref/mod#go-mod-file-require:

      * ``require ( ... )`` blocks (one entry per line)
      * Single-line ``require module/path v1.2.3`` directives
      * Lines with ``// indirect`` markers (preserves the direct/indirect split)
    """
    if not content.strip():
        return []

    packages: list[ResolvedPackage] = []
    seen: set[str] = set()

    # Strip block-style requires first so we don't double-process them.
    block_pattern = re.compile(r"require\s*\((.*?)\)", re.DOTALL)
    for block in block_pattern.findall(content):
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                _parse_go_require_line(line, seen, packages)

    # Now scan for single-line requires. Capture the rest of the line so the
    # `// indirect` marker (if any) flows through to _parse_go_require_line.
    content_no_blocks = block_pattern.sub("", content)
    single_line = re.compile(
        r"^\s*require\s+(\S+\s+\S+(?:\s*//.*)?)$",
        re.MULTILINE,
    )
    for match in single_line.finditer(content_no_blocks):
        _parse_go_require_line(match.group(1), seen, packages)

    return packages
