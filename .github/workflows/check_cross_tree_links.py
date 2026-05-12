#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail CI if any docs link points outside docs/.

The published docs-website branch contains only fern/. Any link like
``[x](../../lib/foo.rs)`` becomes a 404 at docs.nvidia.com because the
referenced file does not exist on docs-website. Use a full GitHub URL
(https://github.com/ai-dynamo/dynamo/blob/main/...) instead.

Run from the repo root::

    python3 .github/workflows/check_cross_tree_links.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#?]+)(?:[#?][^)]*)?\)")
REF_DEF_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?([^>\s#?]+)(?:[#?][^>\s]*)?>?")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "#", "/")


def _check_target(
    target: str, md: Path, docs: Path, lineno: int, failures: list[str]
) -> None:
    """Append a CI annotation to ``failures`` if ``target`` resolves outside ``docs/``."""
    if target.startswith(SKIP_PREFIXES):
        return
    resolved = (md.parent / target).resolve()
    try:
        resolved.relative_to(docs)
    except ValueError:
        rel = md.relative_to(docs.parent)
        failures.append(
            f"::error file={rel},line={lineno}::"
            f"Link target outside docs/: {target!r}. "
            f"The docs-website branch ships only fern/, so this URL 404s on "
            f"docs.nvidia.com. Use a full GitHub URL instead: "
            f"https://github.com/ai-dynamo/dynamo/blob/main/<repo-path>"
        )


def main() -> int:
    docs = Path("docs").resolve()
    if not docs.is_dir():
        print(f"::error::docs/ not found at {docs}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for md in sorted(docs.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        in_fence = False
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for target in LINK_RE.findall(line):
                _check_target(target, md, docs, lineno, failures)
            ref_match = REF_DEF_RE.match(line)
            if ref_match:
                _check_target(ref_match.group(1), md, docs, lineno, failures)

    if failures:
        print("\n".join(failures))
        print(
            f"\nFound {len(failures)} cross-tree link(s) in docs/. "
            f"Fix by replacing the relative path with a full GitHub URL.",
            file=sys.stderr,
        )
        return 1

    print("No cross-tree docs links found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
