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


def _get_cargo_workspace_members(dynamo_path: str, branch: str) -> list[str]:
    """Extract workspace member crate names from Cargo.toml.

    Note: only literal member paths are supported (no glob patterns like "lib/*").
    Dynamo's Cargo.toml uses literal paths, so this is sufficient.
    """
    content = _git_read(dynamo_path, branch, "Cargo.toml")
    if not content:
        return []

    members: list[str] = []
    members_m = re.search(r"members\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if members_m:
        paths = re.findall(r'"([^"]+)"', members_m.group(1))
        for path in paths:
            member_content = _git_read(dynamo_path, branch, f"{path}/Cargo.toml")
            if member_content:
                name_m = re.search(
                    r'^name\s*=\s*"([^"]+)"', member_content, re.MULTILINE
                )
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
        go_mod = _git_read(dynamo_path, branch, "deploy/operator/go.mod")
        if go_mod:
            go_pkgs = parse_go_mod(go_mod)
            all_packages.extend(go_pkgs)

    return DependencyTree(all_packages)
