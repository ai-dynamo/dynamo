#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Point snapshot links at the shared Reference pages the snapshot no longer carries.

The release job drops reference/general/ files from pages-<tag>/ when pages-dev/
has the same file, because the versioned nav serves those pages from pages-dev/.
Links inside the snapshot still name the dropped copies, and Fern renders a link
to a missing file as a raw path that 404s. This rewrites each such link to the
shared pages-dev/ copy and keeps any anchor.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

# Inline links, JSX href attributes, and reference-style link definitions.
LINK = re.compile(
    r"""(\]\(|href=["']|^[ \t]*\[[^\]\n]+\]:[ \t]*)([^)\s"'#]+\.mdx?)(#[^)\s"']*)?""",
    re.MULTILINE,
)


def relink(text: str, page: Path, snapshot: Path, shared: Path, subdir: Path) -> str:
    """Return ``text`` with links to dropped shared pages pointed at ``shared``."""

    def swap(match: re.Match[str]) -> str:
        target = match.group(2)
        if "://" in target or target.startswith("/"):
            return match.group(0)
        resolved = Path(os.path.normpath(page.parent / target))
        if not resolved.is_relative_to(snapshot) or resolved.exists():
            return match.group(0)
        rel = resolved.relative_to(snapshot)
        if not rel.is_relative_to(subdir) or not (shared / rel).is_file():
            return match.group(0)
        new = os.path.relpath(shared / rel, page.parent)
        return f"{match.group(1)}{new}{match.group(3) or ''}"

    return LINK.sub(swap, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="fern/pages-<tag>")
    parser.add_argument("shared", type=Path, help="fern/pages-dev")
    parser.add_argument("--subdir", type=Path, default=Path("reference/general"))
    args = parser.parse_args()

    snapshot, shared = args.snapshot.resolve(), args.shared.resolve()
    changed = 0
    for page in sorted(snapshot.rglob("*.md*")):
        if page.suffix not in {".md", ".mdx"}:
            continue
        text = page.read_text(encoding="utf-8")
        rewritten = relink(text, page, snapshot, shared, args.subdir)
        if rewritten != text:
            page.write_text(rewritten, encoding="utf-8")
            changed += 1
    print(f"relink_shared_reference: updated {changed} file(s) under {args.snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
