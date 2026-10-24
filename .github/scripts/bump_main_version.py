#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bump main to the next minor version after a release branch is cut.

Given the release branch version (e.g. 1.6.0) this works out the version main
should move to (1.7.0), compares it with the version main has now, and either
rewrites every version pin or reports that nothing needs to change:

    release/1.6.0 cut, main at 1.6.0  -> bump to 1.7.0
    release/1.6.1 cut, main at 1.7.0  -> no-op (main is already ahead)
    release/1.7.0 cut, main at 1.7.0  -> bump to 1.8.0

Usage:
    bump_main_version.py plan  --release-version 1.6.0 [--root .]
    bump_main_version.py apply --release-version 1.6.0 [--root .]

`plan` prints JSON (`current`, `target`, `bump_needed`) and edits nothing.
`apply` does the same and rewrites the files when a bump is needed.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
HELM_CHART_DIR = "deploy/helm/charts/platform"


def parse(version: str) -> tuple[int, int, int]:
    m = SEMVER.match(version)
    if not m:
        sys.exit(f"error: '{version}' is not MAJOR.MINOR.PATCH")
    return int(m[1]), int(m[2]), int(m[3])


def next_minor(version: str) -> str:
    major, minor, _ = parse(version)
    return f"{major}.{minor + 1}.0"


def tracked(root: Path, *patterns: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *patterns],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [root / p for p in out.split("\0") if p]


def current_version(root: Path) -> str:
    text = (root / "Cargo.toml").read_text()
    m = re.search(
        r"^\[workspace\.package\]\n(?:[^\[]*?\n)?version = \"([^\"]+)\"", text, re.M
    )
    if not m:
        sys.exit("error: no version in [workspace.package] of Cargo.toml")
    return m[1]


def sub_lines(text: str, rules: list[tuple[str, str]]) -> str:
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text, flags=re.M)
    return text


def bump_cargo_toml(text: str, old: str, new: str) -> str:
    o = re.escape(old)
    return sub_lines(
        text,
        [
            # [package] / [workspace.package] version.
            (rf'^(version = "){o}(")$', rf"\g<1>{new}\2"),
            # Internal path dependencies carry the version for crates.io publishing.
            (rf'^(.*\bpath = "[^"]+".*\bversion = "){o}(")', rf"\g<1>{new}\2"),
        ],
    )


def bump_cargo_lock(text: str, old: str, new: str) -> str:
    # Only workspace crates (path packages) have no `source` line; leave
    # registry and git packages alone even if they share the version string.
    blocks = text.split("[[package]]")
    for i, block in enumerate(blocks[1:], start=1):
        if "\nsource = " in block:
            continue
        blocks[i] = re.sub(
            rf'^(version = "){re.escape(old)}(")$',
            rf"\g<1>{new}\2",
            block,
            count=1,
            flags=re.M,
        )
    return "[[package]]".join(blocks)


def bump_pyproject(text: str, old: str, new: str) -> str:
    o = re.escape(old)
    return sub_lines(
        text,
        [
            (rf'^(version = "){o}(")$', rf"\g<1>{new}\2"),
            (rf"(\bai-dynamo[\w-]*==){o}\b", rf"\g<1>{new}"),
        ],
    )


def bump_chart_yaml(text: str, old: str, new: str) -> str:
    o = re.escape(old)
    return sub_lines(
        text,
        [
            (rf"^(\s*version: ){o}$", rf"\g<1>{new}"),
            (rf'^(appVersion: "){o}(")$', rf"\g<1>{new}\2"),
        ],
    )


def bump_helm_readme(text: str, old: str, new: str) -> str:
    o = re.escape(old)
    return sub_lines(
        text,
        [
            (rf"(Version: ){o}\b", rf"\g<1>{new}"),
            (rf"(badge/Version-){o}(-)", rf"\g<1>{new}\2"),
            (rf"(\| file://[^|]+\| [\w-]+ \| ){o}( \|)", rf"\g<1>{new}\2"),
        ],
    )


def targets(root: Path):
    for p in tracked(root, "Cargo.toml", "**/Cargo.toml"):
        yield p, bump_cargo_toml
    for p in tracked(root, "Cargo.lock", "**/Cargo.lock"):
        yield p, bump_cargo_lock
    for p in tracked(root, "pyproject.toml", "**/pyproject.toml"):
        yield p, bump_pyproject
    for p in tracked(root, f"{HELM_CHART_DIR}/Chart.yaml", f"{HELM_CHART_DIR}/**/Chart.yaml"):
        yield p, bump_chart_yaml
    for p in tracked(root, f"{HELM_CHART_DIR}/README.md"):
        yield p, bump_helm_readme


def apply_bump(root: Path, old: str, new: str) -> list[str]:
    changed = []
    for path, fn in targets(root):
        before = path.read_text()
        after = fn(before, old, new)
        if after != before:
            path.write_text(after)
            changed.append(str(path.relative_to(root)))
    return sorted(set(changed))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["plan", "apply"])
    ap.add_argument("--release-version", required=True, help="e.g. 1.6.0")
    ap.add_argument("--root", default=".", type=Path)
    args = ap.parse_args()

    root = args.root.resolve()
    target = next_minor(args.release_version)
    current = current_version(root)
    bump_needed = parse(current) < parse(target)
    result = {
        "release": args.release_version,
        "current": current,
        "target": target,
        "bump_needed": bump_needed,
    }
    if args.command == "apply" and bump_needed:
        result["changed"] = apply_bump(root, current, target)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
