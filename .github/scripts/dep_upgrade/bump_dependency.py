#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply a framework dependency version bump in-place.

Idempotent: re-running with the same --version is a no-op.
Only the framework's own pin lines are modified; torch ecosystem, recipe
YAMLs, release-artifact docs, and per-release pins in releases.data.ts stay put.

The support matrix now lives in docs/fern/components/releases.data.ts (the
single source of truth for the Reference pages); the development-head pin is
MAIN_TOT, the analog of the old support-matrix.md "main (ToT)" row.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Pattern

# (relative_path, regex, replacement_template). {ver} placeholder filled at apply time.
TRTLLM_TARGETS: list[tuple[str, Pattern[str], str]] = [
    (
        "container/context.yaml",
        # Match runtime_image_tag inside the trtllm: block (first sub-block,
        # e.g. cuda13.1) without colliding with vllm/sglang earlier in the file.
        re.compile(
            r"(?m)(^trtllm:\s*?\n(?:[ \t]+[^\n]*\n)*?[ \t]+runtime_image_tag:\s+)\S+",
        ),
        r"\g<1>{ver}",
    ),
    (
        "pyproject.toml",
        re.compile(r'"tensorrt-llm==[^"]+"'),
        '"tensorrt-llm=={ver}"',
    ),
    (
        "docs/fern/components/releases.data.ts",
        # Rewrite MAIN_TOT.trtllm — the development-head (ToT) TRT-LLM pin, the
        # releases.data.ts analog of the old support-matrix "main (ToT)" row.
        # Anchored to the MAIN_TOT literal and non-greedy to the first trtllm
        # key, so the per-release `trtllm:` pins in RELEASES are never touched.
        re.compile(r'(?s)(export const MAIN_TOT\b.*?\btrtllm:\s*")[^"]+(")'),
        r"\g<1>{ver}\g<2>",
    ),
]

FRAMEWORK_TARGETS: dict[str, list[tuple[str, Pattern[str], str]]] = {
    "trtllm": TRTLLM_TARGETS,
}

# Sibling of FRAMEWORK_TARGETS, kept separate because it is driven by a value
# only the compliance capture can produce (the new baseline's digest stem), not
# by the release version. Same block anchoring as runtime_image_tag above, so it
# cannot reach the vllm/sglang baseline_sbom lines earlier in the file.
FRAMEWORK_BASELINES: dict[str, tuple[str, Pattern[str], str]] = {
    "trtllm": (
        "container/context.yaml",
        re.compile(
            r"(?m)(^trtllm:\s*?\n(?:[ \t]+[^\n]*\n)*?[ \t]+baseline_sbom:\s+)\S+",
        ),
        r"\g<1>{stem}",
    ),
}


def set_baseline_stem(framework: str, stem: str, repo_root: Path) -> bool:
    """Point the framework's baseline_sbom at a freshly captured stem.

    The runtime image's licenses stage subtracts this baseline from NOTICES, so
    leaving it behind a runtime_image_tag bump makes every moved package version
    read as net-new and fails the license gate. Returns True if the file changed.
    """
    rel, pat, tmpl = FRAMEWORK_BASELINES[framework]
    path = repo_root / rel
    text = path.read_text()
    new_text, n = pat.subn(tmpl.replace("{stem}", stem), text)
    if n != 1:
        raise SystemExit(
            f"{rel}: matched {n} {framework} baseline_sbom pins; expected 1"
        )
    if new_text == text:
        return False
    path.write_text(new_text)
    return True


def apply(framework: str, version: str, repo_root: Path) -> int:
    """Apply all bumps for the framework. Returns count of files written."""
    written: set[Path] = set()
    for rel, pat, tmpl in FRAMEWORK_TARGETS[framework]:
        path = repo_root / rel
        text = path.read_text()
        replacement = tmpl.replace("{ver}", version)
        new_text, n = pat.subn(replacement, text)
        if n == 0:
            raise SystemExit(
                f"{rel}: regex matched 0 occurrences "
                f"(pattern broken or pin already non-conformant)"
            )
        if n > 1 and rel == "docs/fern/components/releases.data.ts":
            raise SystemExit(f"{rel}: matched {n} MAIN_TOT trtllm pins; expected 1")
        if new_text != text:
            path.write_text(new_text)
            written.add(path)
    return len(written)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", required=True, choices=sorted(FRAMEWORK_TARGETS))
    p.add_argument(
        "--version",
        required=True,
        help="version without 'v' prefix, e.g. 1.3.0rc12",
    )
    p.add_argument(
        "--baseline-stem",
        help=(
            "Compliance baseline stem from capture_baseline_sbom.py, e.g. "
            "release@2366e4b4. When given, also repoints the framework's "
            "baseline_sbom in container/context.yaml. Omit to bump versions only."
        ),
    )
    p.add_argument("--repo-root", type=Path, default=Path("."))
    args = p.parse_args()
    n = apply(args.framework, args.version, args.repo_root)
    print(f"changed {n} files")
    if args.baseline_stem:
        changed = set_baseline_stem(args.framework, args.baseline_stem, args.repo_root)
        print(
            f"baseline_sbom -> {args.baseline_stem}"
            f"{'' if changed else ' (already set)'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
