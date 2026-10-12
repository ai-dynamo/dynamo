#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate repository-scoped agent instruction file pairs.

Every tracked AGENTS.md must be a regular file with a sibling CLAUDE.md.
Every tracked CLAUDE.md must be a regular file containing exactly
"@AGENTS.md\n" and have a sibling AGENTS.md.

The validator reads the Git index so staged additions, removals, and symlink
changes are checked exactly as they would be committed.

Usage: validate_agent_instructions.py [repo_root]
"""

import subprocess
import sys
from pathlib import Path, PurePosixPath

AGENTS_FILE = "AGENTS.md"
CLAUDE_FILE = "CLAUDE.md"
EXPECTED_CLAUDE_CONTENT = b"@AGENTS.md\n"
REGULAR_FILE_MODES = {"100644", "100755"}


def git_output(root, *args):
    """Run Git in root and return stdout."""
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def tracked_instruction_files(root):
    """Return tracked instruction paths and their index modes."""
    output = git_output(root, "ls-files", "--stage", "-z")
    files = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, _, stage = metadata.decode("ascii").split()
        if stage != "0":
            continue
        path = PurePosixPath(raw_path.decode("utf-8", errors="surrogateescape"))
        if path.name in {AGENTS_FILE, CLAUDE_FILE}:
            files[path] = mode
    return files


def validate(root):
    """Return validation errors for tracked instruction files under root."""
    files = tracked_instruction_files(root)
    errors = []

    for path in sorted(files, key=str):
        mode = files[path]
        sibling_name = CLAUDE_FILE if path.name == AGENTS_FILE else AGENTS_FILE
        sibling = path.with_name(sibling_name)

        if mode not in REGULAR_FILE_MODES:
            errors.append(f"{path}: must be a regular file, found Git mode {mode}")
        if sibling not in files:
            errors.append(f"{path}: missing sibling {sibling_name}")

        if path.name == CLAUDE_FILE and mode in REGULAR_FILE_MODES:
            content = git_output(root, "show", f":{path}")
            if content != EXPECTED_CLAUDE_CONTENT:
                errors.append(
                    f"{path}: must contain exactly '@AGENTS.md' followed by a newline"
                )

    return files, errors


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1]
    try:
        files, errors = validate(root)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        print(f"error: Git command failed: {message}", file=sys.stderr)
        return 1

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(
            f"\n{len(errors)} agent instruction validation error(s)",
            file=sys.stderr,
        )
        return 1

    print(f"validated {len(files) // 2} agent instruction scopes: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
