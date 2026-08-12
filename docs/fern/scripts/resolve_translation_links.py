#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
page links in docs/fern/translations/<lang>/pages/** to root-relative site
URLs, computed from the *current* nav on every publish so they cannot go
stale when pages move.

Source-repo convention (docs/fern/translations/<lang>/pages/<path> mirrors
docs/fern/pages/<path>):
  - links to translated pages are relative within the locale mirror
  - links to untranslated pages are relative from the translated source file
    back to the docs/fern/pages/ base tree
  - image refs are left alone and NOT copied into the mirror -- Fern
    resolves them against the base page location, so copies would only drift

Both link forms resolve to a base-tree page here; links whose target is
translated get the locale-prefixed URL so readers stay in their language.
Links to pages that exist in the repo but are not published in the nav are
rewritten to their GitHub source URL (with a warning) -- a relative path
would naive-join into a dead link on the rendered page.

The .md/.mdx extension is matched loosely: the PR renamed several base pages
from .md to .mdx (they now contain <Tabs>/<Tab> components), but authored
links still point at the old .md name. When the literal target is absent the
resolver retries with the sibling extension before falling back.

Usage:
    # dev sync (default --pages-dir pages)
    python3 docs/fern/scripts/resolve_translation_links.py \
        --nav docs/fern/index.yml --translations-root docs/fern/translations \
        --site-root /dynamo --version-slug dev
    # release snapshot
    python3 docs/fern/scripts/resolve_translation_links.py \
        --nav docs/fern/index.yml \
        --translations-root docs/fern/translations --site-root /dynamo \
        --version-slug vX.Y.Z --pages-dir pages-vX.Y.Z --github-ref vX.Y.Z

Delete this script (and re-shallow the deep-relative links) once Fern
resolves relative links in translated content natively.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path, PurePosixPath

import yaml

LINK = re.compile(r"(!?)(\[[^\]]*\])\(([^)#\s]+)(#[^)]*)?\)")
PAGE_EXT = (".md", ".mdx")
GITHUB_REPO_BLOB = "https://github.com/ai-dynamo/dynamo/blob"


def slugify(name: str) -> str:
    """Fern-style slug: camel humps split, non-alphanumerics collapse to '-'."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def build_slug_map(nav_file: Path) -> dict[str, str]:
    """Map nav `path` entries (relative to docs/fern/) to site slugs.

    Handles tabs, tab variants, sections, folders (by 'path'), and pages. The
    tab/variant walk mirrors how Fern composes URL segments: a tab or variant
    contributes a slug segment unless it is skip-slug'd; the first (default)
    variant with no explicit slug contributes nothing.
    """
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
            # Tabs contribute a URL segment unless skip-slug'd (the docs tabs
            # are; recipes isn't: /dynamo/dev/recipes/...).
            tab_cfg = tabs.get(node["tab"]) or {}
            slug = tab_cfg.get("slug")
            if slug is None:
                slug = "" if tab_cfg.get("skip-slug") else slugify(node["tab"])
            tab_prefix = prefix + [slug] if slug else prefix
            if "variants" in node:
                variants = node["variants"] or []
                for i, variant in enumerate(variants):
                    if not isinstance(variant, dict):
                        continue
                    vslug = variant.get("slug")
                    if variant.get("skip-slug"):
                        vslug = None
                    elif vslug is None:
                        # First variant is the default and adds no segment
                        # unless it declares an explicit slug (handled above).
                        is_default = variant.get("default") or i == 0
                        vslug = None if is_default else slugify(variant["title"])
                    vprefix = tab_prefix + [vslug] if vslug else tab_prefix
                    walk(variant.get("layout", []), vprefix)
                return
            for key in ("contents", "layout", "navigation"):
                if key in node:
                    walk(node[key], tab_prefix)
            return
        for key in ("contents", "layout", "navigation"):
            if key in node:
                walk(node[key], prefix)

    data = yaml.safe_load(nav_file.read_text(encoding="utf-8"))
    tabs = data.get("tabs") or {}
    walk(data.get("navigation", data), [])
    return mapping


def _alt_ext(rel: str) -> str | None:
    """Swap a .md target for .mdx (or vice versa); None if not a page path."""
    for ext in PAGE_EXT:
        if rel.endswith(ext):
            other = ".mdx" if ext == ".md" else ".md"
            return rel[: -len(ext)] + other
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--nav",
        type=Path,
        required=True,
        help="source docs/fern/index.yml (paths relative to docs/fern/; authored pages begin with pages/)",
    )
    ap.add_argument(
        "--translations-root",
        type=Path,
        required=True,
        help="docs/fern/translations directory to rewrite in place",
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
    ap.add_argument(
        "--pages-dir",
        default="pages",
        help="translation subtree to process, mirroring the base pages dir "
        "(pages, or pages-vX.Y.Z for a release snapshot)",
    )
    ap.add_argument(
        "--github-ref",
        default=os.environ.get("GITHUB_SHA", "main"),
        help="ref for GitHub-fallback links (targets not published in the "
        "nav). Defaults to the build's commit, which is correct for dev "
        "syncs and PR previews; release snapshots pass the tag so fallback "
        "links stay faithful to the release on workflow_dispatch rebuilds "
        "too.",
    )
    args = ap.parse_args()
    github_blob = f"{GITHUB_REPO_BLOB}/{args.github_ref}"

    slugs = build_slug_map(args.nav)
    rewritten = warned = 0

    for lang_dir in sorted(p for p in args.translations_root.iterdir() if p.is_dir()):
        lang = lang_dir.name
        pages_root = lang_dir / args.pages_dir
        if not pages_root.is_dir():
            continue
        for page in sorted(pages_root.rglob("*")):
            if page.suffix not in PAGE_EXT or not page.is_file():
                continue
            rel = page.relative_to(pages_root)  # mirrors docs/fern/<rel>
            # Model paths from the docs/fern root. A shallow relative link can
            # remain inside the locale mirror; a deeper relative link can walk
            # back to a base page under docs/fern/.
            virtual_dir = (
                PurePosixPath("translations") / lang / args.pages_dir / rel.parent
            )
            mirror_prefix = PurePosixPath("translations") / lang / args.pages_dir

            def resolve_doc_rel(target: str):
                """Normalize an authored target to a nav-relative doc path.

                Returns (doc_rel, escaped) where escaped is True when the link
                leaves the base tree entirely (left untouched by the caller).
                """
                q = PurePosixPath(os.path.normpath(str(virtual_dir / target)))
                if q.is_relative_to(mirror_prefix):
                    return str(q.relative_to(mirror_prefix)), False
                # A normalized path without a leading ``..`` is still inside
                # docs/fern and is therefore relative to the base nav root.
                if not q.parts or q.parts[0] != "..":
                    return str(q), False
                return None, True

            def lookup(doc_rel: str):
                """(slug, doc_rel_on_disk, translated) using extension fallback."""
                candidates = [doc_rel]
                if not doc_rel.startswith("pages/"):
                    candidates.append(f"pages/{doc_rel}")
                for base in list(candidates):
                    alt = _alt_ext(base)
                    if alt and alt not in candidates:
                        candidates.append(alt)
                for cand in candidates:
                    if cand in slugs:
                        mirror_rel = cand.removeprefix("pages/")
                        translated = (pages_root / mirror_rel).exists()
                        return slugs[cand], cand, translated
                for cand in candidates:
                    if (args.nav.parent / cand).is_file():
                        mirror_rel = cand.removeprefix("pages/")
                        return None, cand, (pages_root / mirror_rel).exists()
                return None, None, False

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
                doc_rel, escaped = resolve_doc_rel(target)
                if escaped:
                    print(
                        f"::warning::{lang}/{rel}: {target} escapes docs/, left as-is"
                    )
                    warned += 1
                    return m.group(0)
                slug, disk_rel, translated = lookup(doc_rel)
                if slug is None:
                    if disk_rel is None:
                        # Typo'd or missing target (even after ext fallback): a
                        # GitHub URL would be a plausible-looking 404, so leave
                        # the link untouched.
                        print(
                            f"::warning::{lang}/{rel}: {target} not in nav and "
                            f"not in docs/, left as-is"
                        )
                        warned += 1
                        return m.group(0)
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
                    return f"{bang}{label}({github_blob}/docs/fern/{disk_rel}{anchor})"
                # Locale sits between product and version in Fern URLs
                # (/dynamo/zh-CN/dev/...); links starting with the product
                # slug pass through Fern's renderer unmodified.
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
        f"{warned} warned (GitHub fallback or left as-is)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
