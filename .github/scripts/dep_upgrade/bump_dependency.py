#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply a framework dependency version bump in-place.

Idempotent: re-running with the same --version is a no-op.
Only the framework's own pin lines are modified; torch ecosystem, recipe
YAMLs, release-artifact docs, and historical support-matrix rows stay put.
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
        "docs/reference/support-matrix.md",
        re.compile(r"^(\| \*\*main \(ToT\)\*\* \| `[^`]+` \| `)[^`]+(` \|)", re.M),
        r"\g<1>{ver}\g<2>",
    ),
]

VLLM_TARGETS: list[tuple[str, Pattern[str], str]] = [
    (
        "container/context.yaml",
        # cuda13.0 sub-block: runtime_image_tag: v{ver}-ubuntu2404
        re.compile(
            r"(?m)(^vllm:\s*?\n(?:[ \t]+[^\n]*\n)*?[ \t]+cuda13\.0:\s*?\n"
            r"(?:[ \t]+[^\n]*\n)*?[ \t]+runtime_image_tag:\s+v)[^\s-]+(-ubuntu2404\s*$)",
        ),
        r"\g<1>{ver}\g<2>",
    ),
    (
        "container/context.yaml",
        # cpu sub-block: runtime_image_tag: v{ver}  (no image-tag suffix)
        re.compile(
            r"(?m)(^vllm:\s*?\n(?:[ \t]+[^\n]*\n)*?[ \t]+cpu:\s*?\n"
            r"(?:[ \t]+[^\n]*\n)*?[ \t]+runtime_image_tag:\s+v)\S+(\s*$)",
        ),
        r"\g<1>{ver}\g<2>",
    ),
    (
        "pyproject.toml",
        re.compile(r'"vllm\[flashinfer,runai,otel\]==[^"]+"'),
        '"vllm[flashinfer,runai,otel]=={ver}"',
    ),
    (
        "docs/reference/support-matrix.md",
        # ToT row, 3rd backtick column (vLLM), after SGLang and TRT-LLM columns.
        re.compile(
            r"^(\| \*\*main \(ToT\)\*\* \| `[^`]+` \| `[^`]+` \| `)[^`]+(`)",
            re.M,
        ),
        r"\g<1>{ver}\g<2>",
    ),
]

SGLANG_TARGETS: list[tuple[str, Pattern[str], str]] = [
    (
        "container/context.yaml",
        # cuda13.0 sub-block: runtime_image_tag: v{ver}-cu130-runtime
        re.compile(
            r"(?m)(^sglang:\s*?\n(?:[ \t]+[^\n]*\n)*?[ \t]+cuda13\.0:\s*?\n"
            r"(?:[ \t]+[^\n]*\n)*?[ \t]+runtime_image_tag:\s+v)[^\s-]+(-cu130-runtime\s*$)",
        ),
        r"\g<1>{ver}\g<2>",
    ),
    (
        "pyproject.toml",
        re.compile(r'"sglang\[diffusion\]==[^"]+"'),
        '"sglang[diffusion]=={ver}"',
    ),
    (
        "docs/reference/support-matrix.md",
        # ToT row, 1st backtick column (SGLang).
        re.compile(
            r"^(\| \*\*main \(ToT\)\*\* \| `)[^`]+(`)",
            re.M,
        ),
        r"\g<1>{ver}\g<2>",
    ),
]

FRAMEWORK_TARGETS: dict[str, list[tuple[str, Pattern[str], str]]] = {
    "trtllm": TRTLLM_TARGETS,
    "vllm": VLLM_TARGETS,
    "sglang": SGLANG_TARGETS,
}


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
        if n > 1 and rel == "docs/reference/support-matrix.md":
            raise SystemExit(f"{rel}: matched {n} ToT rows; expected 1")
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
    p.add_argument("--repo-root", type=Path, default=Path("."))
    args = p.parse_args()
    n = apply(args.framework, args.version, args.repo_root)
    print(f"changed {n} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
