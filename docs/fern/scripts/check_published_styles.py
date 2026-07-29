#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assert the published site actually carries its page-level component CSS.

Production sets `global-theme: nvidia`, which overrides the project `css:` and
`js:` entries and the custom `footer:` (Fern's docs.yml schema documents the
override). Components that deliver their CSS through a page-level <style>
block survive it; anything relying on main.css does not. That failure is
invisible before merge -- PR previews delete the theme -- so this runs against
the published site after publish.

Each check asserts a selector appears as a CSS *rule* (followed by `{` or `,`),
not merely as a class name in markup. A page can be full of `class="foo"` while
the rule that styles it is absent, which is exactly the failure mode.

Usage:
    python3 check_published_styles.py [--base URL] [--retries N] [--delay SEC]
Exits 1 with the failing page/selector pairs listed.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "https://docs.nvidia.com/dynamo/dev/"

# (page path, selector that must exist as a CSS rule, component that owns it)
CHECKS: list[tuple[str, str, str]] = [
    ("", ".dynamo-story-windowbar", "LandingStyles"),
    ("", ".dynamo-welcome__terminal", "LandingStyles"),
    ("community", ".dynamo-community-page", "LandingStyles"),
    ("digest", ".dynamo-blog-art__grid", "BlogStyles"),
    ("reference/compatibility", ".dynref-panel", "ReferenceStyles"),
]


def fetch(url: str, retries: int, delay: float) -> str:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read().decode("utf-8", "ignore")
        except (urllib.error.URLError, TimeoutError) as exc:  # noqa: PERF203
            last = exc
            if attempt < retries:
                time.sleep(delay)
    raise SystemExit(f"could not fetch {url}: {last}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--delay", type=float, default=30.0)
    args = parser.parse_args()

    base = args.base if args.base.endswith("/") else args.base + "/"
    failures: list[str] = []

    for path, selector, owner in CHECKS:
        url = base + path
        html = fetch(url, args.retries, args.delay)
        as_rule = len(re.findall(re.escape(selector) + r"\s*[,{]", html))
        in_markup = len(re.findall(r'class="[^"]*' + re.escape(selector[1:]), html))
        if as_rule:
            print(f"ok    {url}  {selector} ({as_rule} rules, {owner})")
            continue
        failures.append(
            f"{url}\n      {selector} appears in {in_markup} element(s) but has no CSS rule.\n"
            f"      {owner} did not reach the page. If the styles moved to main.css,\n"
            f"      the global theme drops them -- deliver them from the component instead."
        )

    if failures:
        print("\nFAIL: published pages are missing component CSS\n", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"\nall {len(CHECKS)} published style checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
