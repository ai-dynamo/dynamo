#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply a dev-version suffix to every Dynamo package version and cross-ref.

Invoked by nightly CI on the runner, before `docker buildx build`. Takes one
argument -- a suffix like '.dev20260423+g1234567' -- and rewrites, in place:
  - [project].version in every Dynamo pyproject.toml (PEP 440 form)
  - [package].version / [workspace.package].version in every Cargo.toml
    (SemVer form: dash instead of dot before 'dev', so '1.1.0-dev...+g...')
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
# Root Cargo.toml is handled separately by rewrite_root_cargo.
SUBCRATE_CARGO_TARGETS = [
    "lib/bindings/python/Cargo.toml",
    "lib/bindings/kvbm/Cargo.toml",
    "lib/kvbm-common/Cargo.toml",
    "lib/kvbm-engine/Cargo.toml",
    "lib/kvbm-kernels/Cargo.toml",
    "lib/kvbm-logical/Cargo.toml",
    "lib/kvbm-physical/Cargo.toml",
    "lib/runtime/examples/Cargo.toml",
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
    # Convert a PEP 440-style '.devN+g...' into SemVer '-devN+g...'.
    if suffix.startswith("."):
        return base + "-" + suffix[1:]
    return base + suffix


def rewrite_pyproject(path: Path, suffix: str, is_root: bool) -> None:
    text = path.read_text()

    def _bump(m: re.Match) -> str:
        return f"{m.group(1)}{pep440(suffix, m.group(2))}{m.group(3)}"

    text, n = VERSION_LINE_RE.subn(_bump, text, count=1)
    if n != 1:
        raise RuntimeError(f"no [project].version in {path}")

    if is_root:
        text = PY_RUNTIME_PIN_RE.sub(
            lambda m: f"{m.group(1)}{pep440(suffix, m.group(2))}{m.group(3)}",
            text,
        )
    path.write_text(text)


def rewrite_subcrate_cargo(path: Path, suffix: str) -> None:
    text = path.read_text()
    text = VERSION_LINE_RE.sub(
        lambda m: f"{m.group(1)}{semver(suffix, m.group(2))}{m.group(3)}",
        text,
    )
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
    "0.45.0", "=0.19.3", etc.). Idempotent: re-running with the same suffix is
    a no-op because `base` becomes the already-suffixed value, which never
    matches external deps.
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
    new = semver(suffix, base)

    text = re.sub(
        rf'(\bversion\s*=\s*"){re.escape(base)}(")',
        lambda mm: f"{mm.group(1)}{new}{mm.group(2)}",
        text,
    )
    path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("suffix", help="e.g. .dev20260423+g1234567 (empty = no-op)")
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
