#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render SiteStyles on every hosted Fern page before preview or publish.

The private NVIDIA global theme replaces the project's ``css:`` and custom
footer fields. Page-level MDX components survive that merge, so the docs
workflow runs this script against its disposable composed tree immediately
before ``fern generate``. Source pages and release snapshots remain unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMPORT = 'import { SiteStyles } from "@/components/SiteStyles";'
COMPONENT = "<SiteStyles />"
INJECTION = f"\n\n{IMPORT}\n\n{COMPONENT}\n"
PAGE_SUFFIXES = {".md", ".mdx"}
ROOT_PATTERNS = (
    "pages",
    "pages-*",
    "digest",
    "blogs",
    "translations/*/pages*",
)


def discover_pages(fern_root: Path) -> list[Path]:
    """Return every publishable Markdown page in a composed Fern tree."""
    pages: set[Path] = set()
    index = fern_root / "index.mdx"
    if index.is_file():
        pages.add(index)

    for pattern in ROOT_PATTERNS:
        for root in fern_root.glob(pattern):
            if root.is_file() and root.suffix in PAGE_SUFFIXES:
                pages.add(root)
            elif root.is_dir():
                pages.update(
                    path
                    for path in root.rglob("*")
                    if (
                        path.is_file()
                        and path.suffix in PAGE_SUFFIXES
                        and path.name.lower() != "readme.md"
                    )
                )
    return sorted(pages)


def inject(path: Path) -> bool:
    """Insert the import and component after frontmatter; return whether changed."""
    text = path.read_text(encoding="utf-8")
    if COMPONENT in text:
        return False
    if text.startswith("---\n"):
        closing = text.find("\n---\n", 4)
        if closing < 0:
            raise ValueError(f"{path}: Fern page has no closing frontmatter delimiter")
        insert_at = closing + len("\n---")
        rendered = text[:insert_at] + INJECTION + text[insert_at + 1 :]
    else:
        # Legacy snapshots include generated Markdown without frontmatter. MDX
        # imports are valid at the document start, so keep those pages covered.
        rendered = f"{IMPORT}\n\n{COMPONENT}\n\n{text}"
    path.write_text(rendered, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fern_root", type=Path, help="Composed Fern directory")
    args = parser.parse_args()

    pages = discover_pages(args.fern_root)
    if not pages:
        parser.error(f"no publishable pages found under {args.fern_root}")

    changed = sum(inject(path) for path in pages)
    print(f"inject_site_styles: updated {changed}/{len(pages)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
