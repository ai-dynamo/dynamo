#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply a dev-version suffix to every Dynamo package version and cross-ref.

Invoked by nightly CI on the runner, before `docker buildx build`. Takes one
argument -- a suffix like '.dev20260423' -- and rewrites, in place:
  - [project].version in every Dynamo pyproject.toml (PEP 440 form)
  - [package].version / [workspace.package].version in every Cargo.toml
    (SemVer form: dash instead of dot before 'dev', so '1.1.0-dev20260423')
  - The `ai-dynamo-runtime==1.1.0` pin in the root pyproject
  - The `version = "1.1.0"` pins on dynamo-*/kvbm-* path deps in root Cargo.toml

Empty suffix is a no-op, so safe to run unconditionally in every workflow.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PYPROJECT_TARGETS = [
    "pyproject.toml",
    "lib/bindings/python/pyproject.toml",
    "lib/bindings/kvbm/pyproject.toml",
    "lib/gpu_memory_service/pyproject.toml",
]

# Sub-crate Cargo files with an EXPLICIT [package].version (not workspace-inherited).
# kvbm-config uses `version.workspace = true`, so it's intentionally omitted.
# lib/runtime/examples/Cargo.toml is also omitted: it's a nested workspace (own
# [workspace.package]) used only for local example binaries, not shipped in any
# wheel, and nothing outside that workspace pins its version.
# Root Cargo.toml is handled separately by rewrite_root_cargo.
SUBCRATE_CARGO_TARGETS = [
    "lib/bindings/python/Cargo.toml",
    "lib/bindings/python/codegen/Cargo.toml",
    "lib/bindings/kvbm/Cargo.toml",
    "lib/kvbm-common/Cargo.toml",
    "lib/kvbm-engine/Cargo.toml",
    "lib/kvbm-kernels/Cargo.toml",
    "lib/kvbm-logical/Cargo.toml",
    "lib/kvbm-physical/Cargo.toml",
]

# Line-anchored: matches `version = "X.Y.Z"` lines. Skips `version.workspace = true`
# (no quotes) and `version = { ... }` (no string). Safe for sub-crate Cargo.tomls
# whose only `version = "..."` line is the [package] one; external-crate deps use
# the `name = { version = "..." }` inline-table form which this regex skips.
VERSION_LINE_RE = re.compile(r'^(\s*version\s*=\s*")([^"]+)(")\s*$', re.MULTILINE)

# Root pyproject cross-ref to the runtime wheel.
PY_RUNTIME_PIN_RE = re.compile(r'("ai-dynamo-runtime==)([^"]+)(")')


def pep440(suffix: str, base: str) -> str:
    # suffix already starts with '.' (dev release) or '+' (local-only).
    return base + suffix


def semver(suffix: str, base: str) -> str:
    # Convert a PEP 440-style '.devN' into SemVer '-devN'.
    if suffix.startswith("."):
        return base + "-" + suffix[1:]
    return base + suffix


def _pep440_tail(suffix: str) -> str:
    # The trailing text that pep440() appends; used to detect "already stamped".
    return suffix


def _semver_tail(suffix: str) -> str:
    # The trailing text that semver() appends; used to detect "already stamped".
    return "-" + suffix[1:] if suffix.startswith(".") else suffix


def rewrite_pyproject(path: Path, suffix: str, is_root: bool) -> None:
    text = path.read_text()

    current = VERSION_LINE_RE.search(text)
    if current is None:
        raise RuntimeError(f"no [project].version in {path}")
    if current.group(2).endswith(_pep440_tail(suffix)):
        return  # already stamped -- idempotent no-op

    def _bump(m: re.Match) -> str:
        return f"{m.group(1)}{pep440(suffix, m.group(2))}{m.group(3)}"

    text, n = VERSION_LINE_RE.subn(_bump, text, count=1)
    assert n == 1  # guaranteed by the search above

    if is_root:
        text = PY_RUNTIME_PIN_RE.sub(
            lambda m: f"{m.group(1)}{pep440(suffix, m.group(2))}{m.group(3)}",
            text,
        )
    path.write_text(text)


def rewrite_subcrate_cargo(path: Path, suffix: str) -> None:
    text = path.read_text()
    tail = _semver_tail(suffix)

    def _bump(m: re.Match) -> str:
        base = m.group(2)
        if base.endswith(tail):
            return m.group(0)  # already stamped
        return f"{m.group(1)}{semver(suffix, base)}{m.group(3)}"

    text = VERSION_LINE_RE.sub(_bump, text)
    path.write_text(text)


def rewrite_root_cargo(root: Path, suffix: str) -> None:
    """Root Cargo.toml has three kinds of `version = "..."` sites:
      1. [workspace.package].version                          -- bump
      2. Internal path-dep pins in [workspace.dependencies],  -- bump (must match (1))
         e.g. `dynamo-runtime = { path = "lib/runtime", version = "1.1.0" }`
      3. External-crate deps, e.g. `anyhow = { version = "1" }` -- leave alone

    (1) and (2) always use the SAME literal string. Anchor on it, then rewrite
    only `version = "<that exact string>"` occurrences. This bumps (1) and (2)
    in one pass while leaving (3) untouched (they hold other values like "1",
    "0.45.0", "=0.19.3", etc.). An explicit "already stamped" guard makes this
    idempotent -- re-running with the same suffix is a no-op.
    """
    path = root / "Cargo.toml"
    text = path.read_text()

    m = re.search(
        r'\[workspace\.package\][^\[]*?\n\s*version\s*=\s*"([^"]+)"',
        text,
    )
    if not m:
        raise RuntimeError("no [workspace.package].version in root Cargo.toml")
    base = m.group(1)
    if base.endswith(_semver_tail(suffix)):
        return  # already stamped -- idempotent no-op
    new = semver(suffix, base)

    text = re.sub(
        rf'(\bversion\s*=\s*"){re.escape(base)}(")',
        lambda mm: f"{mm.group(1)}{new}{mm.group(2)}",
        text,
    )
    path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("suffix", help="e.g. .dev20260423 (empty = no-op)")
    ap.add_argument("root", nargs="?", default=".", help="repo root")
    args = ap.parse_args()

    if not args.suffix:
        print("apply_dev_version: empty suffix, no-op", file=sys.stderr)
        return 0

    root = Path(args.root).resolve()
    for rel in PYPROJECT_TARGETS:
        rewrite_pyproject(root / rel, args.suffix, is_root=(rel == "pyproject.toml"))
    rewrite_root_cargo(root, args.suffix)
    for rel in SUBCRATE_CARGO_TARGETS:
        rewrite_subcrate_cargo(root / rel, args.suffix)

    print(f"apply_dev_version: stamped suffix '{args.suffix}'", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
