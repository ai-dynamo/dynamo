#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the current Git publication candidates without local artifacts/history."""

import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    git_root = Path(
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True
        ).strip()
    ).resolve()

    def git_paths(*args):
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", *args]
        )
        return {Path(p.decode()) for p in output.split(b"\0") if p}

    destination = args.destination.expanduser().resolve()
    if (
        destination == git_root
        or git_root in destination.parents
        or destination.exists()
    ):
        parser.error("destination must not exist and must be outside the repository")
    ignored_tracked = git_paths("--cached", "--ignored", "--exclude-standard")
    if ignored_tracked:
        parser.error(
            "tracked files match ignore rules; remove them from the index before exporting"
        )
    candidates = git_paths("--cached", "--others", "--exclude-standard")
    files = []
    for relative in sorted(candidates):
        source = root / relative
        if source.is_symlink() or root not in source.resolve().parents:
            parser.error(f"refusing symlink or path outside repository: {relative}")
        if source.is_file():
            files.append(relative)
        elif source.exists():
            parser.error(f"refusing non-file entry: {relative}")
    destination.mkdir(parents=True)
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    print(f"Exported {len(files)} source files to {destination}")


if __name__ == "__main__":
    main()
