# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from benchmarks.coding.common import dedupe_paths

IGNORED_FILENAMES = {"history.jsonl"}


def claude_project_dir_for_root(root: Path, home_dir: Path) -> Path:
    encoded = str(root.resolve()).replace("/", "-")
    return home_dir / ".claude" / "projects" / encoded


def is_trace_path(path: Path) -> bool:
    return path.suffix == ".jsonl" and path.name not in IGNORED_FILENAMES


def is_ignored_path(path: Path) -> bool:
    return "subagents" in path.parts or path.name in IGNORED_FILENAMES


def scan_trace_dir(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path.resolve()
        for path in root.rglob("*.jsonl")
        if path.is_file() and is_trace_path(path) and not is_ignored_path(path)
    )


def iter_ancestor_roots(start: Path) -> Iterator[Path]:
    current = start.resolve()
    while True:
        yield current
        parent = current.parent
        if parent == current:
            return
        current = parent


def discover_trace_files(explicit_inputs: list[str], script_dir: Path) -> list[Path]:
    home_dir = Path.home()
    claude_projects_root = home_dir / ".claude" / "projects"
    discovered: list[Path] = []

    if explicit_inputs:
        for raw_path in explicit_inputs:
            input_path = Path(raw_path).expanduser().resolve()
            if input_path.is_file():
                if not is_trace_path(input_path) or is_ignored_path(input_path):
                    raise FileNotFoundError(
                        f"Not a Claude session trace file: {input_path}"
                    )
                discovered.append(input_path)
                continue
            if not input_path.exists():
                raise FileNotFoundError(f"Input path does not exist: {input_path}")
            if not input_path.is_dir():
                raise FileNotFoundError(f"Unsupported input path: {input_path}")

            is_claude_trace_dir = (
                input_path == claude_projects_root
                or claude_projects_root in input_path.parents
            )
            if is_claude_trace_dir:
                directory_hits = scan_trace_dir(input_path)
                if directory_hits:
                    discovered.extend(directory_hits)
                    continue
            else:
                repo_hits = scan_trace_dir(
                    claude_project_dir_for_root(input_path, home_dir)
                )
                if repo_hits:
                    discovered.extend(repo_hits)
                    continue

                directory_hits = scan_trace_dir(input_path)
                if directory_hits:
                    discovered.extend(directory_hits)
                    continue

            raise FileNotFoundError(
                "No Claude session traces found under input path or its encoded "
                f"Claude project directory: {input_path}"
            )
        return dedupe_paths(discovered)

    for candidate_root in dedupe_paths(list(iter_ancestor_roots(script_dir))):
        discovered.extend(
            scan_trace_dir(claude_project_dir_for_root(candidate_root, home_dir))
        )

    discovered.extend(scan_trace_dir(claude_projects_root))
    return dedupe_paths(discovered)
