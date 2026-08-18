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
"""Reject site-absolute asset paths, which Fern never serves.

Fern does not host docs/fern/assets/ at a stable URL. It uploads each asset to
a content-hashed CDN path and rewrites the reference -- but only in MDX and in
docs.yml. A path written as:

    url("/dynamo/assets/img/dynamo-logo.svg")     in a <style> template literal
    src="/dynamo/assets/dynamo-demo.cast"          in TSX

is rewritten by nothing and reaches the browser verbatim, where it 404s. This
is invisible locally and in review: the path looks canonical, the file really
is in the repo, and only the published page is wrong. It shipped once already
(#12373 swapped a working absolute URL for one of these and blanked the Home
page hero mark), so this catches the shape rather than the instance.

Use instead:
  - MDX pages         a relative path -- src="../../assets/img/logo.svg"
  - docs.yml          a repo-relative path -- ./assets/NVIDIA_dark.svg
  - CSS in a component  no repo asset. Render an <img> from the page MDX and
                        style it, or inline the bytes as a data: URI.

Usage: python3 check_asset_paths.py [files...]
With no arguments, checks the docs/fern trees where this can occur.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # docs/fern
REPO = ROOT.parents[1]  # repo root, for readable paths
SELF = Path(__file__).resolve()  # skipped: the docstring must show the bad shape

# A site-absolute reference into the asset tree, anchored on the leading slash so
# the two forms Fern does rewrite stay clean: `../../assets/...` in MDX and
# docs.yml's `./assets/...`.
#
# Openers, each one a shape that showed up in review:
#   "..." '...'   quoted attributes and config values
#   `...`         template literals, which is where these files keep their CSS
#   url(...)      the CSS form the Home hero regressed on
#   key: /...     bare YAML scalars; docs.yml writes logo/favicon/fonts unquoted
#   - /...        YAML sequence entries
#
# MULTILINE so `^` still anchors per line if this is ever handed whole-file text
# rather than the line-at-a-time feed check() uses today.
ABSOLUTE_ASSET = re.compile(
    r"""(?:["'`(]|:\s|^\s*-\s)\s*(/(?:[\w.-]+/)*assets/[^"'`)\s]+)""",
    re.MULTILINE,
)

# translations/ carries the zh-CN pages, which publish through the locale
# configured in docs.yml and 404 the same way.
DEFAULT_GLOBS = (
    "components/**/*.tsx",
    "components/**/*.ts",
    # components/README.md is the declared source of truth for component usage,
    # so an example there is copy-pasted into pages verbatim. Without this glob
    # the canonical example is the one file this check cannot see.
    "components/**/*.md",
    "pages/**/*.mdx",
    "pages/**/*.md",
    "translations/**/*.mdx",
    "translations/**/*.md",
    "main.css",
    "custom.js",
    # logo:, favicon: and the font path: entries. Fern rewrites the ./assets/...
    # form here, but a site-absolute one would ship verbatim.
    "docs.yml",
)


def label(path: Path) -> str:
    """Repo-relative where possible; absolute for a path outside the repo."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def check(path: Path) -> list[str]:
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        for match in ABSOLUTE_ASSET.finditer(line):
            ref = match.group(1)
            problems.append(
                f"{label(path)}:{lineno}: site-absolute asset path\n"
                f"      {line.strip()[:72]}\n"
                f"      Fern serves no such URL, so {ref} 404s once published.\n"
                f"      Use a relative path from an MDX page, or inline the asset."
            )
    return problems


# Every case here is one a reviewer caught the pattern missing, or one an earlier
# revision wrongly flagged. Run with --test.
CASES: tuple[tuple[str, bool, str], ...] = (
    ('background: url("/dynamo/assets/img/x.svg")', True, "quoted url()"),
    ("background: url(/dynamo/assets/img/x.svg)", True, "bare url()"),
    ("const s = `/dynamo/assets/demo.cast`;", True, "template literal"),
    ('<img src="/dynamo/assets/img/y.png" />', True, "quoted attribute"),
    ("favicon: /dynamo/assets/NVIDIA_symbol.svg", True, "bare YAML scalar"),
    ("  dark: /dynamo/assets/NVIDIA_dark.svg", True, "indented YAML scalar"),
    ("  - path: /dynamo/assets/fonts/NVIDIASans.woff2", True, "YAML sequence"),
    ('src="../../assets/img/x.svg"', False, "relative, Fern rewrites it"),
    ("  dark: ./assets/NVIDIA_dark.svg", False, "docs.yml relative form"),
    ("x: https://cdn.example.com/assets/a.svg", False, "external URL"),
    ('url("https://cdn.example.com/assets/a.svg")', False, "quoted external URL"),
    ("/dynamo/assets/img/x.svg", False, "bare prose, no opener"),
)


def selftest() -> int:
    failures = 0
    for line, should_match, why in CASES:
        ok = bool(ABSOLUTE_ASSET.search(line)) == should_match
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}: {why}")
    total = len(CASES)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


def main() -> int:
    if "--test" in sys.argv[1:]:
        return selftest()

    args = [Path(a) for a in sys.argv[1:]]
    if args:
        targets = [a for a in args if a.exists()]
    else:
        targets = sorted({p for glob in DEFAULT_GLOBS for p in ROOT.glob(glob)})
    targets = [t for t in targets if t.resolve() != SELF]

    problems: list[str] = []
    for target in targets:
        problems.extend(check(target))

    if problems:
        print("site-absolute asset paths found:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"checked {len(targets)} file(s): no site-absolute asset paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
