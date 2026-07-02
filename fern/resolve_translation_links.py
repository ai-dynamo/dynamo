#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Resolve relative page links in translated docs to site URLs at build time.

Fern's localization (early access) pairs a translation with its base page by
mirrored path, but it does not yet resolve relative Markdown links inside
translated content against the nav -- it naively joins them onto the page URL,
producing dead links. Until Fern fixes that, this script rewrites relative
page links in fern/translations/<lang>/pages-dev/** to root-relative site
URLs, computed from the *current* nav on every publish so they cannot go
stale when pages move.

Source-repo convention (docs/translations/<lang>/<path> mirrors docs/<path>):
  - links to translated siblings stay shallow-relative (quickstart.mdx)
  - links to untranslated pages are deep-relative into the base tree
    (../../../reference/support-matrix.md), so they stay valid for the
    repo link checker and GitHub browsing
  - image refs are left alone (Fern resolves them against the base page)

Both link forms resolve to a base-tree page here; links whose target is
translated get the locale-prefixed URL so readers stay in their language.
Links to pages that are not in the nav are left unchanged (with a warning).

Usage:
    resolve_translation_links.py --nav docs/index.yml \
        --translations-root fern/translations --site-root /dynamo --version-slug dev

Delete this script (and re-shallow the deep-relative links) once Fern
resolves relative links in translated content natively.
"""

import argparse
import os
import re
import sys
from pathlib import Path, PurePosixPath

import yaml

LINK = re.compile(r"(!?)(\[[^\]]*\])\(([^)#\s]+)(#[^)]*)?\)")
PAGE_EXT = (".md", ".mdx")


def slugify(name: str) -> str:
    """Fern-style slug: camel humps split, non-alphanumerics collapse to '-'."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def build_slug_map(nav_file: Path) -> dict[str, str]:
    """Map nav 'path' entries (relative to docs/) to site slugs."""
    mapping: dict[str, str] = {}

    def walk(node, prefix):
        if isinstance(node, list):
            for item in node:
                walk(item, prefix)
            return
        if not isinstance(node, dict):
            return
        if "page" in node:
            slug = node.get("slug") or slugify(node["page"])
            full = prefix + [slug] if slug else prefix
            if "path" in node:
                mapping[node["path"]] = "/".join(full)
            return
        if "section" in node:
            slug = node.get("slug")
            if slug is None:
                slug = "" if node.get("skip-slug") else slugify(node["section"])
            new_prefix = prefix + [slug] if slug else prefix
            if "path" in node:
                mapping[node["path"]] = "/".join(new_prefix)
            walk(node.get("contents", []), new_prefix)
            return
        for key in ("tab", "contents", "layout", "navigation"):
            if key in node:
                walk(node[key], prefix)

    data = yaml.safe_load(nav_file.read_text(encoding="utf-8"))
    walk(data.get("navigation", data), [])
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--nav",
        type=Path,
        required=True,
        help="source docs/index.yml (paths relative to docs/)",
    )
    ap.add_argument(
        "--translations-root",
        type=Path,
        required=True,
        help="fern/translations directory to rewrite in place",
    )
    ap.add_argument(
        "--site-root",
        required=True,
        help="product slug the site is served under, e.g. /dynamo",
    )
    ap.add_argument(
        "--version-slug",
        required=True,
        help="version slug the translated pages belong to, e.g. dev",
    )
    args = ap.parse_args()

    slugs = build_slug_map(args.nav)
    rewritten = warned = 0

    for lang_dir in sorted(p for p in args.translations_root.iterdir() if p.is_dir()):
        lang = lang_dir.name
        pages_root = lang_dir / "pages-dev"
        if not pages_root.is_dir():
            continue
        for page in sorted(pages_root.rglob("*")):
            if page.suffix not in PAGE_EXT or not page.is_file():
                continue
            rel = page.relative_to(pages_root)  # mirrors docs/<rel>
            # links were authored from docs/translations/<lang>/<rel>
            virtual_dir = PurePosixPath("docs/translations") / lang / rel.parent

            def repl(m: re.Match) -> str:
                nonlocal rewritten, warned
                bang, label, target, anchor = (
                    m.group(1),
                    m.group(2),
                    m.group(3),
                    m.group(4) or "",
                )
                if bang or target.startswith(("http://", "https://", "mailto:", "/")):
                    return m.group(0)
                if not target.endswith(PAGE_EXT):
                    return m.group(0)
                q = PurePosixPath(os.path.normpath(str(virtual_dir / target)))
                mirror_prefix = PurePosixPath("docs/translations") / lang
                if q.is_relative_to(mirror_prefix):
                    doc_rel = str(q.relative_to(mirror_prefix))
                elif q.is_relative_to("docs"):
                    doc_rel = str(q.relative_to("docs"))
                else:
                    print(f"  [warn] {lang}/{rel}: {target} escapes docs/, left as-is")
                    warned += 1
                    return m.group(0)
                slug = slugs.get(doc_rel)
                if slug is None:
                    print(f"  [warn] {lang}/{rel}: {target} not in nav, left as-is")
                    warned += 1
                    return m.group(0)
                # Locale sits between product and version in Fern URLs
                # (/dynamo/zh-CN/dev/...); links starting with the product
                # slug pass through Fern's renderer unmodified.
                translated = (pages_root / doc_rel).exists()
                url = (
                    f"{args.site_root}/{lang}/{args.version_slug}/{slug}"
                    if translated
                    else f"{args.site_root}/{args.version_slug}/{slug}"
                )
                rewritten += 1
                return f"{bang}{label}({url}{anchor})"

            text = page.read_text(encoding="utf-8")
            new = LINK.sub(repl, text)
            if new != text:
                page.write_text(new, encoding="utf-8")

    print(
        f"resolve_translation_links: rewrote {rewritten} link(s), "
        f"{warned} left unresolved"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
