# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract transitive dependencies from Dynamo lock files via git-show."""

from __future__ import annotations

import logging
import re
import subprocess

from .lockfile import parse_cargo_lock, parse_go_mod
from .tree import DependencyTree
from .types import Ecosystem, ResolvedPackage

logger = logging.getLogger(__name__)


def _git_read(dynamo_path: str, branch: str, file_path: str) -> str:
    """Read file content from a git ref without checkout."""
    try:
        result = subprocess.run(
            ["git", "show", f"{branch}:{file_path}"],
            cwd=dynamo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            logger.debug(
                "git show %s:%s failed: %s", branch, file_path, exc.stderr.strip()
            )
        return ""


def _git_list_go_mod_files(dynamo_path: str, branch: str) -> list[str]:
    """List all go.mod files in the repo at a given branch."""
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", branch, "--name-only"],
            cwd=dynamo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return [line for line in result.stdout.splitlines() if line.endswith("/go.mod")]
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            logger.debug("git ls-tree failed: %s", exc.stderr.strip())
        return []


def _expand_workspace_member_paths(pattern: str, paths: list[str]) -> list[str]:
    """Expand a workspace member entry into concrete directory paths.

    Literal entries pass through unchanged. Glob entries (currently only the
    `prefix/*` form used by Dynamo's Cargo.toml) are matched against ``paths``,
    which should be the list of files known to the repo (typically from
    ``git ls-tree``). The expanded output lists every directory that contains
    a ``Cargo.toml`` and matches the glob, sorted for deterministic ordering.
    """
    if "*" not in pattern:
        return [pattern]

    if not pattern.endswith("/*"):
        # Anything more exotic than `prefix/*` would need fnmatch; fail closed
        # so the caller can fall back to `cargo metadata` rather than silently
        # under-counting workspace crates.
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


def _git_ls_tree(dynamo_path: str, branch: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", branch, "--name-only"],
            cwd=dynamo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.splitlines()
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            logger.debug("git ls-tree failed: %s", exc.stderr.strip())
        return []


def _get_cargo_workspace_members(dynamo_path: str, branch: str) -> list[str]:
    """Extract workspace member crate names from Cargo.toml.

    Supports both literal paths and ``prefix/*`` globs (finding #20). Globs are
    expanded against the file list returned by ``git ls-tree`` so we never
    touch the working tree.
    """
    content = _git_read(dynamo_path, branch, "Cargo.toml")
    if not content:
        return []

    members_m = re.search(r"members\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not members_m:
        return []

    raw_entries = re.findall(r'"([^"]+)"', members_m.group(1))
    repo_files = (
        _git_ls_tree(dynamo_path, branch)
        if any("*" in entry for entry in raw_entries)
        else []
    )

    expanded_paths: list[str] = []
    for entry in raw_entries:
        expanded_paths.extend(_expand_workspace_member_paths(entry, repo_files))

    members: list[str] = []
    for path in expanded_paths:
        member_content = _git_read(dynamo_path, branch, f"{path}/Cargo.toml")
        if not member_content:
            continue
        name_m = re.search(r'^name\s*=\s*"([^"]+)"', member_content, re.MULTILINE)
        if name_m:
            members.append(name_m.group(1))
    return members


def extract_transitive(
    dynamo_path: str,
    branch: str = "HEAD",
    ecosystem: Ecosystem | None = None,
) -> DependencyTree:
    """Extract Rust and Go transitive dependencies via git-show.

    Python packages are extracted separately via --image (container inspection).
    """
    all_packages: list[ResolvedPackage] = []

    if ecosystem is None or ecosystem == Ecosystem.RUST:
        cargo_lock = _git_read(dynamo_path, branch, "Cargo.lock")
        if cargo_lock:
            workspace_members = _get_cargo_workspace_members(dynamo_path, branch)
            rust_pkgs = parse_cargo_lock(cargo_lock, workspace_members or None)
            all_packages.extend(rust_pkgs)

    if ecosystem is None or ecosystem == Ecosystem.GO:
        go_mod_files = _git_list_go_mod_files(dynamo_path, branch)
        seen_go_pkgs: set[tuple[str, str]] = set()
        for go_mod_path in go_mod_files:
            go_mod = _git_read(dynamo_path, branch, go_mod_path)
            if go_mod:
                for pkg in parse_go_mod(go_mod):
                    key = (pkg.name, pkg.version)
                    if key not in seen_go_pkgs:
                        seen_go_pkgs.add(key)
                        all_packages.append(pkg)

    return DependencyTree(all_packages)
