# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report tj-actions JSON file lists and reject files without a CI filter."""

import json
import os
from pathlib import Path

# Each class must independently contain the entire modified-file set. Do not
# union classes: mixed changes require the existing full runtime selection.
SGLANG_UNRELATED_CHANGE_CLASSES = (
    {"components/src/dynamo/frontend/tests/test_vllm_processor_unit.py"},
)


def modified_only_files(output_dir: Path, all_files: set[str]) -> set[str]:
    """Return verified ordinary modifications; uncertainty keeps full tests."""
    try:
        modified = load_files(output_dir / "all_modified_files.json")
        complete = load_files(output_dir / "all_all_changed_and_modified_files.json")
        other_statuses = [
            load_files(output_dir / f"all_{status}_files.json")
            for status in (
                "added",
                "copied",
                "deleted",
                "renamed",
                "type_changed",
                "unmerged",
                "unknown",
            )
        ]
    except (OSError, ValueError):
        return set()
    if modified != all_files or complete != all_files or any(other_statuses):
        return set()
    return modified


def runtime_test_outputs(output_dir: Path, all_files: set[str]) -> dict[str, bool]:
    modified = modified_only_files(output_dir, all_files)
    return {
        "sglang_runtime": not (
            modified
            and any(modified <= allowed for allowed in SGLANG_UNRELATED_CHANGE_CLASSES)
        ),
    }


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
    outputs = runtime_test_outputs(output_dir, all_files)
    for name, required in outputs.items():
        value = str(bool(required)).lower()
        print(f"{name}={value}")
        if os.environ.get("GITHUB_OUTPUT"):
            with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
                output.write(f"{name}={value}\n")
    print("All modified files are covered by CI filters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        report(Path(os.environ["CHANGED_FILES_DIR"]), os.environ["BASE_SHA"])
    )
