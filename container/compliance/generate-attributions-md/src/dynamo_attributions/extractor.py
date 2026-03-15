# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract transitive dependencies from Dynamo lock files via git-show."""

from __future__ import annotations

import logging
import re
import subprocess

from .lockfile import parse_cargo_lock, parse_go_mod, parse_pip_freeze
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


def _get_direct_python_packages(dynamo_path: str, branch: str) -> list[str]:
    """Extract declared Python package names from requirements files."""
    req_files = [
        "container/deps/requirements.common.txt",
        "container/deps/requirements.frontend.txt",
        "container/deps/requirements.planner.txt",
        "container/deps/requirements.benchmark.txt",
        "container/deps/requirements.vllm.txt",
    ]
    names: set[str] = set()
    for req_file in req_files:
        content = _git_read(dynamo_path, branch, req_file)
        if not content:
            continue
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            m = re.match(r"^([a-zA-Z0-9][a-zA-Z0-9._-]*)", line)
            if m:
                names.add(m.group(1))
    return sorted(names)


def extract_transitive(
    dynamo_path: str,
    branch: str = "HEAD",
    ecosystem: Ecosystem | None = None,
    pip_freeze_content: str | None = None,
    direct_python_packages: list[str] | None = None,
) -> DependencyTree:
    """Extract transitive dependencies and build a dependency tree.

    Reads lock files via git-show (no checkout needed).
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

    if ecosystem is None or ecosystem == Ecosystem.PYTHON:
        if pip_freeze_content:
            py_pkgs = parse_pip_freeze(pip_freeze_content)
            if direct_python_packages:
                direct_set = {
                    re.sub(r"[-_.]+", "-", n).lower() for n in direct_python_packages
                }
                for pkg in py_pkgs:
                    norm = re.sub(r"[-_.]+", "-", pkg.name).lower()
                    if norm in direct_set:
                        pkg.is_direct = True
            all_packages.extend(py_pkgs)

    return DependencyTree(all_packages)
