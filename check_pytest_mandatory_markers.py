# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re
import sys
from pathlib import Path
from typing import List

CATEGORIES = {
    "pipeline": {"nightly", "pre_merge", "post_merge", "weekly", "release"},
    "type": {"unit", "integration", "e2e", "smoke", "performance"},
    "infra": {"a100", "gpu_1", "gpu_2", "gpu_0"},
}


def extract_file_level_markers(lines: List[str]):
    file_markers = set()
    marker_pat = re.compile(r"pytest\.mark\.([a-zA-Z0-9_]+)")
    in_pytestmark_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("pytestmark"):
            if "[" in stripped:  # multi-line, likely list assignment
                in_pytestmark_list = True
                # capture markers on the same line (e.g. pytestmark = [pytest.mark.a, pytest.mark.b])
                for m in marker_pat.findall(stripped):
                    file_markers.add(m)
                continue
            else:
                # single-line assignment: pytestmark = pytest.mark.xxx
                for m in marker_pat.findall(stripped):
                    file_markers.add(m)
        elif in_pytestmark_list:
            # Continue collecting all entries until we find the closing ]
            for m in marker_pat.findall(stripped):
                file_markers.add(m)
            if "]" in stripped:
                in_pytestmark_list = False
        # Only process top-of-file assignments (stop after first def/class, as before)
        if stripped.startswith("def ") or stripped.startswith("class "):
            break
    return file_markers


def extract_class_level_markers(lines: List[str], idx: int):
    for i in range(idx - 1, -1, -1):
        if re.match(r"^\s*class\s+\w+", lines[i]):
            class_markers = set()
            j = i - 1
            while j >= 0:
                line = lines[j].strip()
                if line.startswith("@pytest.mark."):
                    m = re.match(r"@pytest\.mark\.([A-Za-z0-9_]+)", line)
                    if m:
                        class_markers.add(m.group(1))
                    j -= 1
                elif not line or line.startswith("#"):
                    j -= 1
                else:
                    break
            return class_markers
        if re.match(r"^\s*def\s+\w+", lines[i]):
            break
    return set()


def get_markers_for_test_function_works_partially(
    lines: List[str], idx: int, file_level: set
):
    found = set(file_level)
    marker_pat = re.compile(r"@[\s]*pytest\.mark\.([A-Za-z0-9_]+)")
    j = idx - 1
    # Handle decorators above the function, even if multiline or separated by blanks/comments
    while j >= 0:
        line = lines[j].strip()
        # If line is empty or comment, skip up
        if not line or line.startswith("#"):
            j -= 1
            continue
        # If line is part of a decorator (starts with "@" OR is the continuation of a decorator line)
        if line.startswith("@") or (line.startswith(")") or line.startswith("(")):
            # Extract ALL markers present anywhere in the line (including continued/multiline)
            for m in marker_pat.findall(line):
                found.add(m)
            j -= 1
            continue
        # For decorators split across multiple indented lines, keep walking up until a code line
        if (
            line.endswith(",")
            or line.endswith("(")
            or line.endswith(")")
            or line.endswith("\\")
        ):
            # Still may be continuation of a decorator block
            for m in marker_pat.findall(line):
                found.add(m)
            j -= 1
            continue
        # On hitting code or docstring or anything else, stop.
        break
    return found


# Works for static marker collection.
# TODO: Update markers defined in custom config for tests/serve/ e2e tests.
def get_markers_for_test_function(lines: List[str], idx: int, file_level: set):
    found = set(file_level)
    marker_pat = re.compile(r"@pytest\.mark\.([A-Za-z0-9_]+)")
    max_lines = 20
    for j in range(idx - 1, max(idx - max_lines, -1), -1):
        line = lines[j]
        for m in marker_pat.findall(line):
            if m != "parametrize":
                found.add(m)
        # You may want to break early if you see another `def`, `class`, or docstring, but for decorator
        # stacks with argument blocks, it is better to overscan than underscan. Most projects never
        # have more than 10 contiguous decorator/data/comment lines above a test.
    return found


def check_categories(marker_set):
    missing = []
    for cat, allowed in CATEGORIES.items():
        if not (marker_set & allowed):
            missing.append(cat)
    return missing


def file_tests_with_missing_categories(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    file_level = extract_file_level_markers(lines)
    for idx, line in enumerate(lines):
        if re.match(r"^\s*def test_", line):
            marker_set = get_markers_for_test_function(lines, idx, file_level)
            missing = check_categories(marker_set)
            if missing:
                print(f"\nDEBUG {path}, line {idx + 1}")
                print("  Decorators and blank/comment lines above:")
                j = idx - 1
                while j >= 0:
                    small = lines[j].strip()
                    if small.startswith("@") or not small or small.startswith("#"):
                        print(f"    {lines[j].rstrip()}")
                        j -= 1
                    else:
                        break
                print(f"  Markers collected: {marker_set}")
                print(f"  Categories missing: {missing}\n")
                yield idx + 1, line.strip(), missing


def main():
    error = False
    found_any = False
    tests_dir = Path(__file__).parent / "tests"
    if not tests_dir.exists() or not tests_dir.is_dir():
        print(f"Could not find 'dynamo/tests' directory at: {tests_dir}")
        sys.exit(1)
    test_files = list(tests_dir.rglob("test_*.py"))
    if not test_files:
        print(f"No test_*.py files found recursively under {tests_dir}/")
        return
    for path in sorted(test_files):
        found_any = True
        for lineno, sig, missing in file_tests_with_missing_categories(path):
            print(
                f"File {path}, line {lineno}: '{sig}' is missing marker(s) from: {', '.join(missing)}"
            )
            error = True
    if not found_any:
        print(f"No test_*.py files found recursively under {tests_dir}/")
    if error:
        print("\nERROR: Some tests are missing required category markers.")
        sys.exit(1)


if __name__ == "__main__":
    main()
