# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report tj-actions JSON file lists and reject files without a CI filter."""

import json
import os
from pathlib import Path


def load_files(path: Path) -> set[str]:
    """Read an unescaped JSON array written by tj-actions/changed-files."""
    filenames = json.loads(path.read_text())
    if not isinstance(filenames, list) or not all(
        isinstance(name, str) for name in filenames
    ):
        raise ValueError(f"Expected a JSON array of filenames in {path.name}")
    return set(filenames)


def report(output_dir: Path, base_sha: str = "") -> int:
    """Log filenames as JSON data and check the union of all explicit filters."""
    all_path = output_dir / "all_all_modified_files.json"
    all_files = load_files(all_path)
    print("Base SHA:", json.dumps(base_sha or "default (previous commit)"))
    print(
        f"All modified files ({len(all_files)} total):", json.dumps(sorted(all_files))
    )
    print("Files matching each filter:")
    covered: set[str] = set()
    for path in sorted(output_dir.glob("*_all_modified_files.json")):
        if path == all_path:
            continue
        filenames = load_files(path)
        filter_name = path.name.removesuffix("_all_modified_files.json")
        print(f"  {json.dumps(filter_name)}: {json.dumps(sorted(filenames))}")
        covered.update(filenames)

    uncovered = all_files - covered
    if uncovered:
        print("::error::The following files are not covered by any CI filter:")
        for filename in sorted(uncovered):
            print(json.dumps(filename))
        print("Add these paths to .github/filters.yaml. See .github/FILTERS.md.")
        return 1
    print("All modified files are covered by CI filters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        report(Path(os.environ["CHANGED_FILES_DIR"]), os.environ["BASE_SHA"])
    )
