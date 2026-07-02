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

Source-repo convention (fern/translations/<lang>/pages-dev/<path> mirrors
docs/<path>, matching where the sync publishes both):
  - links to translated siblings stay shallow-relative (quickstart.mdx)
  - links to untranslated pages are deep-relative into the base tree
    (../../../../../docs/reference/support-matrix.md), so they stay valid for the
    repo link checker and GitHub browsing
  - image refs are left alone and NOT copied into the mirror -- Fern
    resolves them against the base page location (verified on hosted
    builds: they serve from the pages-dev assets), so copies would only
    drift

Both link forms resolve to a base-tree page here; links whose target is
translated get the locale-prefixed URL so readers stay in their language.
Links to pages that exist in the repo but are not published in the nav are
rewritten to their GitHub source URL (with a warning) -- a relative path
would naive-join into a dead link on the rendered page.

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
# Fallback for links whose target isn't published in the nav (no site URL).
GITHUB_BLOB = "https://github.com/ai-dynamo/dynamo/blob/main"


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
        if "tab" in node:
            # Tabs contribute a URL segment unless skip-slug'd (the docs tab
            # is; recipes isn't: /dynamo/dev/recipes/...).
            tab_cfg = tabs.get(node["tab"]) or {}
            slug = tab_cfg.get("slug")
            if slug is None:
                slug = "" if tab_cfg.get("skip-slug") else slugify(node["tab"])
            if slug:
                prefix = prefix + [slug]
        for key in ("contents", "layout", "navigation"):
            if key in node:
                walk(node[key], prefix)

    data = yaml.safe_load(nav_file.read_text(encoding="utf-8"))
    tabs = data.get("tabs") or {}
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
            # links were authored from fern/translations/<lang>/pages-dev/<rel>
            # in the source repo (same layout the sync publishes)
            virtual_dir = (
                PurePosixPath("fern/translations") / lang / "pages-dev" / rel.parent
            )

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
                mirror_prefix = PurePosixPath("fern/translations") / lang / "pages-dev"
                if q.is_relative_to(mirror_prefix):
                    doc_rel = str(q.relative_to(mirror_prefix))
                elif q.is_relative_to("docs"):
                    doc_rel = str(q.relative_to("docs"))
                else:
                    print(
                        f"::warning::{lang}/{rel}: {target} escapes docs/, left as-is"
                    )
                    warned += 1
                    return m.group(0)
                slug = slugs.get(doc_rel)
                if slug is None:
                    # Target exists in the repo but isn't published in the nav,
                    # so it has no site URL. A relative path would naive-join
                    # into a guaranteed 404 on the rendered page; link to the
                    # GitHub source instead so the reader still lands somewhere
                    # real.
                    print(
                        f"::warning::{lang}/{rel}: {target} not in nav, "
                        f"linking to GitHub source"
                    )
                    warned += 1
                    return f"{bang}{label}({GITHUB_BLOB}/docs/{doc_rel}{anchor})"
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
            # Skip link-shaped text inside fenced code blocks / inline code.
            parts = re.split(r"(```.*?```|~~~.*?~~~|`[^`\n]*`)", text, flags=re.S)
            new = "".join(
                p if i % 2 else LINK.sub(repl, p) for i, p in enumerate(parts)
            )
            if new != text:
                page.write_text(new, encoding="utf-8")

    print(
        f"resolve_translation_links: rewrote {rewritten} link(s) to site URLs, "
        f"{warned} to GitHub source (not in nav)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
