# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cargo.lock parser and workspace-member discovery.

Emits plain dict records per the §C2 lockfile package record contract:

    {
        "name": str,
        "version": str,
        "ecosystem": "cargo",
        "is_direct": bool,
        "source": str,
    }

Workspace members (first-party Dynamo crates) are filtered out of
``parse_cargo_lock`` by default so callers get only third-party crates.
"""

from __future__ import annotations

import os
import re

ECOSYSTEM = "cargo"


def parse_cargo_lock(
    content: str,
    workspace_members: list[str] | None = None,
) -> list[dict]:
    """Parse Cargo.lock into third-party package records.

    Workspace members are *excluded* from the returned list. When
    ``workspace_members`` is provided, entries with a matching ``name`` are
    dropped. When it is ``None``, the parser falls back to the source-based
    heuristic: entries with no ``source`` field are treated as workspace
    members and dropped (in Cargo.lock, workspace members are exactly the
    entries with no ``source``).
    """
    if not content.strip():
        return []

    packages: list[dict] = []
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

        if workspace_members is not None:
            is_workspace_member = name in workspace_members
        else:
            # Source-based heuristic: sourceless entries are workspace members.
            is_workspace_member = source == ""

        if is_workspace_member:
            # First-party workspace crates are not third-party deps; drop them.
            continue

        packages.append(
            {
                "name": name,
                "version": version,
                "ecosystem": ECOSYSTEM,
                # Surfaced entries are, by construction, non-workspace members.
                # ``is_direct`` here means "appears in a workspace member's
                # direct dependency set" — we can't cheaply distinguish that
                # without walking the dep graph, so leave False. Group C does
                # not currently use this field for cargo.
                "is_direct": False,
                "source": source,
            }
        )

    return packages


def _expand_workspace_member_paths(pattern: str, paths: list[str]) -> list[str]:
    """Expand a workspace member entry into concrete directory paths.

    Literal entries pass through unchanged. Glob entries (currently only the
    ``prefix/*`` form used by Dynamo's Cargo.toml) are matched against
    ``paths``, which should be the list of repo-relative file paths from the
    working tree. The expanded output lists every directory containing a
    ``Cargo.toml`` that matches the glob, sorted for deterministic ordering.
    """
    if "*" not in pattern:
        return [pattern]

    if not pattern.endswith("/*"):
        # Anything more exotic than `prefix/*` would need fnmatch; fail closed
        # so the caller can fall back rather than silently under-counting
        # workspace crates.
        raise ValueError(f"Unsupported workspace glob pattern: {pattern!r}")

    prefix = pattern[:-2]
    matched: set[str] = set()
    for path in paths:
        if not path.endswith("/Cargo.toml"):
            continue
        parent = path[: -len("/Cargo.toml")]
        if parent.startswith(prefix + "/") and "/" not in parent[len(prefix) + 1 :]:
            matched.add(parent)
    return sorted(matched)


def _walk_repo_files(repo_root: str) -> list[str]:
    """List repo-relative file paths in the working tree.

    Skips ``.git`` to stay fast on large repos. Returns POSIX-style paths so
    glob matching against ``prefix/*`` patterns behaves consistently regardless
    of platform.
    """
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Skip the git dir to avoid scanning the object store.
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, repo_root)
            # Normalize to POSIX separators.
            out.append(rel.replace(os.sep, "/"))
    return out


def get_workspace_members(repo_root: str) -> list[str]:
    """Return workspace member crate names from ``<repo_root>/Cargo.toml``.

    Supports both literal paths and ``prefix/*`` globs. Globs are expanded
    against the working tree under ``repo_root``. Returns an empty list if
    ``Cargo.toml`` is missing or lists no members.
    """
    root_cargo = os.path.join(repo_root, "Cargo.toml")
    try:
        with open(root_cargo, encoding="utf-8") as f:
            content = f.read()
    except (FileNotFoundError, OSError):
        return []

    if not content:
        return []

    members_m = re.search(r"members\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not members_m:
        return []

    raw_entries = re.findall(r'"([^"]+)"', members_m.group(1))
    repo_files = (
        _walk_repo_files(repo_root)
        if any("*" in entry for entry in raw_entries)
        else []
    )

    expanded_paths: list[str] = []
    for entry in raw_entries:
        expanded_paths.extend(_expand_workspace_member_paths(entry, repo_files))

    members: list[str] = []
    for path in expanded_paths:
        member_cargo = os.path.join(repo_root, path, "Cargo.toml")
        try:
            with open(member_cargo, encoding="utf-8") as f:
                member_content = f.read()
        except (FileNotFoundError, OSError):
            continue
        name_m = re.search(r'^name\s*=\s*"([^"]+)"', member_content, re.MULTILINE)
        if name_m:
            members.append(name_m.group(1))
    return members
