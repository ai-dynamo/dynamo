# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""go.mod parser and multi-module discovery.

Emits plain dict records per the §C2 lockfile package record contract:

    {
        "name": str,
        "version": str,
        "ecosystem": "golang",
        "is_direct": bool,
        "source": str,
    }

Handles the three require forms documented at
https://go.dev/ref/mod#go-mod-file-require:

  * ``require ( ... )`` blocks (one entry per line)
  * Single-line ``require module/path v1.2.3`` directives
  * Lines with ``// indirect`` markers (preserves the direct/indirect split)

Also filters out ``replace`` directives pointing to local paths — those are
first-party code, not third-party deps.
"""

from __future__ import annotations

import os
import re

ECOSYSTEM = "golang"


def _parse_replace_directives(content: str) -> set[str]:
    """Return module paths replaced with local filesystem paths.

    Matches both block-form ``replace ( ... )`` and single-line ``replace``
    directives. A replacement target is considered local when its path starts
    with ``.`` or ``/`` (see https://go.dev/ref/mod#go-mod-file-replace).
    """
    local_replacements: set[str] = set()

    def _record(line: str) -> None:
        line = re.sub(r"//.*$", "", line).strip()
        if not line:
            return
        # `orig => target` or `orig v1.2.3 => target` or `orig => target v1.2.3`
        if "=>" not in line:
            return
        lhs, rhs = line.split("=>", 1)
        lhs_parts = lhs.split()
        rhs_parts = rhs.split()
        if not lhs_parts or not rhs_parts:
            return
        orig = lhs_parts[0]
        target = rhs_parts[0]
        if target.startswith(".") or target.startswith("/"):
            local_replacements.add(orig)

    block_pattern = re.compile(r"replace\s*\((.*?)\)", re.DOTALL)
    for block in block_pattern.findall(content):
        for raw in block.splitlines():
            _record(raw)

    content_no_blocks = block_pattern.sub("", content)
    single_line = re.compile(r"^\s*replace\s+(.+)$", re.MULTILINE)
    for match in single_line.finditer(content_no_blocks):
        _record(match.group(1))

    return local_replacements


def _parse_go_require_line(
    line: str,
    seen: set[str],
    packages: list[dict],
    local_replacements: set[str],
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

    if module_path in local_replacements:
        # Replaced with a local filesystem path — first-party, not a dep.
        return

    packages.append(
        {
            "name": module_path,
            "version": version,
            "ecosystem": ECOSYSTEM,
            "is_direct": not is_indirect,
            "source": "",
        }
    )


def parse_go_mod(content: str) -> list[dict]:
    """Parse go.mod into third-party package records."""
    if not content.strip():
        return []

    local_replacements = _parse_replace_directives(content)

    packages: list[dict] = []
    seen: set[str] = set()

    # Strip block-style requires first so we don't double-process them.
    block_pattern = re.compile(r"require\s*\((.*?)\)", re.DOTALL)
    for block in block_pattern.findall(content):
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                _parse_go_require_line(line, seen, packages, local_replacements)

    # Single-line requires. Capture the rest of the line so the `// indirect`
    # marker (if any) flows through to _parse_go_require_line.
    content_no_blocks = block_pattern.sub("", content)
    single_line = re.compile(
        r"^\s*require\s+(\S+\s+\S+(?:\s*//.*)?)$",
        re.MULTILINE,
    )
    for match in single_line.finditer(content_no_blocks):
        _parse_go_require_line(match.group(1), seen, packages, local_replacements)

    return packages


def list_go_mod_files(repo_root: str) -> list[str]:
    """Return repo-relative paths of every ``go.mod`` file under ``repo_root``.

    Walks the working tree directly (no git indirection). Skips the ``.git``
    directory. Results are sorted for deterministic ordering.
    """
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        if "go.mod" in filenames:
            full = os.path.join(dirpath, "go.mod")
            rel = os.path.relpath(full, repo_root).replace(os.sep, "/")
            found.append(rel)
    found.sort()
    return found
