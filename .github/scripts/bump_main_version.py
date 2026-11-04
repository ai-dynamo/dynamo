#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move main to the next minor version once its release branch is cut.

Once release/X.Y.Z exists, main must be at X.(Y+1).0 so nightlies built from
main sort above the X.Y releases:

    newest release/1.6.0, main at 1.6.0 -> bump to 1.7.0
    newest release/1.6.1, main at 1.7.0 -> nothing to do
    newest release/1.7.0, main at 1.7.0 -> bump to 1.8.0

The operator chart's appVersion never moves here. The operator upgrade-history
gate (deploy/operator/hack/generate_upgrade_fixtures.py) accepts the next
appVersion only after the operator owners capture the released predecessor's
history; see "Provenance and release boundaries" in
deploy/operator/internal/controller/testdata/upgrade/README.md.
"""

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
RELEASE_HEAD = re.compile(r"\trefs/heads/release/(\d+\.\d+\.\d+)$")
HELM_CHART_DIR = "deploy/helm/charts/platform"
OPERATOR_CHART = f"{HELM_CHART_DIR}/components/operator/Chart.yaml"
# Indented, so the operator chart's top-level `name: dynamo-operator` never matches.
OPERATOR_PIN = re.compile(r"^[ \t]+(?:-[ \t]+)?name:[ \t]*dynamo-operator[ \t]*$", re.M)


def parse(version: str) -> tuple[int, int, int]:
    m = SEMVER.match(version)
    if not m:
        sys.exit(f"error: '{version}' is not MAJOR.MINOR.PATCH")
    return int(m[1]), int(m[2]), int(m[3])


def next_minor(version: str) -> str:
    major, minor, _ = parse(version)
    return f"{major}.{minor + 1}.0"


def pick_release(ls_remote: str) -> str:
    versions = [
        m[1] for line in ls_remote.splitlines() if (m := RELEASE_HEAD.search(line))
    ]
    if not versions:
        sys.exit("error: no release/X.Y.Z branch on origin")
    return max(versions, key=parse)


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if proc.returncode:
        sys.exit(f"error: git {args[0]} failed: {proc.stderr.strip()}")
    return proc.stdout


def latest_release(root: Path) -> str:
    return pick_release(
        git(root, "ls-remote", "--heads", "origin", "refs/heads/release/*")
    )


def needs_bump(current: str, release: str) -> bool:
    if parse(current) >= parse(next_minor(release)):
        return False
    if parse(current)[:2] != parse(release)[:2]:
        sys.exit(
            f"error: main is at {current}, which is not on the release/{release} "
            "line; a skipped minor or a major cut needs a manual bump"
        )
    return True


def tracked(root: Path, *patterns: str) -> list[Path]:
    out = git(root, "ls-files", "-z", "--", *patterns)
    return [root / p for p in out.split("\0") if p]


def workspace_version(cargo_toml: str) -> str:
    m = re.search(
        r"^\[workspace\.package\]\n(?:[^\[]*?\n)?version = \"([^\"]+)\"",
        cargo_toml,
        re.M,
    )
    if not m:
        sys.exit("error: no version in [workspace.package] of Cargo.toml")
    return m[1]


def current_version(root: Path) -> str:
    return workspace_version((root / "Cargo.toml").read_text())


def operator_app_version(root: Path) -> str:
    path = root / OPERATOR_CHART
    if not path.is_file():
        return ""
    m = re.search(r'^appVersion:[ \t]*"?([^"\s]+)"?[ \t]*$', path.read_text(), re.M)
    return m[1] if m else ""


def sub_lines(text: str, rules: list[tuple[str, str]]) -> str:
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text, flags=re.M)
    return text


def bump_cargo_toml(text: str, old: str, new: str) -> str:
    o = re.escape(old)
    return sub_lines(
        text,
        [
            (rf'^(version = "){o}(")$', rf"\g<1>{new}\2"),
            (rf'^(.*\bpath = "[^"]+".*\bversion = "){o}(")', rf"\g<1>{new}\2"),
        ],
    )


def bump_cargo_lock(text: str, old: str, new: str) -> str:
    """Move only packages without a `source` line, as `cargo update --workspace`
    would; CI's `cargo test --locked` fails if the two ever differ."""
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
            (rf"(\bai-dynamo[\w-]*==){o}(?![\w.])", rf"\g<1>{new}"),
        ],
    )


def bump_chart_yaml(text: str, old: str, new: str) -> str:
    o = re.escape(old)
    text = re.sub(rf"^(version: ){o}$", rf"\g<1>{new}", text, flags=re.M)
    text, pins = re.subn(
        r"^(\s*-\s+name:\s*dynamo-operator\s*\n(?:(?!\s*-\s)[^\n]*\n)*?"
        rf"\s*version: ){o}$",
        rf"\g<1>{new}",
        text,
        flags=re.M,
    )
    if OPERATOR_PIN.search(text) and pins != 1:
        sys.exit(f"error: the dynamo-operator dependency pin is not at {old}")
    return text


def bump_helm_readme(text: str, old: str, new: str) -> str:
    """Edit the generated README as `make generate-helm-docs` would; the operator
    `make check` in CI fails if the two ever differ."""
    o = re.escape(old)
    return sub_lines(
        text,
        [
            (rf"(!\[Version: ){o}(\])", rf"\g<1>{new}\2"),
            (rf"(badge/Version-){o}(-)", rf"\g<1>{new}\2"),
            (rf"(\| file://[^|]+\| [\w-]+ \| ){o}( \|)", rf"\g<1>{new}\2"),
        ],
    )


def targets(root: Path) -> Iterator[tuple[Path, Callable[[str, str, str], str]]]:
    for p in tracked(root, "Cargo.toml", "**/Cargo.toml"):
        yield p, bump_cargo_toml
    for p in tracked(root, "Cargo.lock", "**/Cargo.lock"):
        yield p, bump_cargo_lock
    for p in tracked(root, "pyproject.toml", "**/pyproject.toml"):
        yield p, bump_pyproject
    for p in tracked(
        root, f"{HELM_CHART_DIR}/Chart.yaml", f"{HELM_CHART_DIR}/**/Chart.yaml"
    ):
        yield p, bump_chart_yaml
    for p in tracked(root, f"{HELM_CHART_DIR}/README.md"):
        yield p, bump_helm_readme


def apply_bump(root: Path, old: str, new: str) -> list[str]:
    """Compute every edit before writing any, so a refusal leaves the tree as is."""
    edits = {}
    for path, fn in targets(root):
        before = path.read_text()
        after = fn(before, old, new)
        if after != before:
            edits[path] = after
    cargo_toml = root / "Cargo.toml"
    moved = workspace_version(edits.get(cargo_toml) or cargo_toml.read_text())
    if moved != new:
        sys.exit(f"error: Cargo.toml would still read {moved}")
    for path, text in edits.items():
        path.write_text(text)
    return sorted(str(path.relative_to(root)) for path in edits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["plan", "apply"])
    ap.add_argument(
        "--release-version",
        help="release branch to act on (default: newest release/X.Y.Z on origin)",
    )
    ap.add_argument("--root", default=".", type=Path)
    args = ap.parse_args()

    root = args.root.resolve()
    release = args.release_version or latest_release(root)
    current = current_version(root)
    target = next_minor(release)
    bump_needed = needs_bump(current, release)
    result = {
        "release": release,
        "current": current,
        "target": target,
        "bump_needed": bump_needed,
        "operator_app_version": operator_app_version(root),
    }
    if args.command == "apply" and bump_needed:
        result["changed"] = apply_bump(root, current, target)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
