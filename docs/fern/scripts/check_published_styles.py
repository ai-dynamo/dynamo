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
"""Assert the published site actually carries its page-level component CSS.

Production sets `global-theme: nvidia`, which overrides the project `css:` and
`js:` entries and the custom `footer:` (Fern's docs.yml schema documents the
override). Components that deliver their CSS through a page-level <style>
block survive it; anything relying on main.css does not. Hosted pull request
previews use the same theme and should catch that failure before merge; this
published-site check remains the post-publish backstop.

Selector checks assert that a selector appears as a CSS *rule* (followed by
`{` or `,`), not merely as a class name in markup. Declaration checks assert
that required CSS custom properties are present too. A page can carry all of
its selectors while declarations that depend on an undefined custom property
silently become invalid, which is the community-widget failure this also guards.

CDN propagation and Fern's publish pipeline can lag, so the CDN returns HTTP
200 with the *previous* HTML while the new page is still propagating. `fetch()`
would return that stale body immediately and the selector check would fail on
a healthy publish. The retry loop therefore covers both transport errors *and*
a successful body that does not yet carry the rule. Only after every retry
still lacks the rule is the page recorded as a failure.

The regex helpers are the release gate's core assertion, so `--test` exercises
them against a small table of matching-minified, class-only, adjacent-name,
and both-rule-and-markup cases -- boundary or escaping changes cannot silently
invert the result.

Usage:
    python3 check_published_styles.py [--base URL] [--retries N] [--delay SEC]
    python3 check_published_styles.py --test

Exits 1 with the failing page/selector pairs listed.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE = "https://docs.nvidia.com/dynamo/dev/"

# Only allow --base to reach the published docs origin. Any other host is
# rejected by url_from_base(), giving urlopen() below an explicit HTTPS trust
# boundary so Ruff's S310 warning has something concrete to point at.
ALLOWED_ORIGINS: frozenset[str] = frozenset({"https://docs.nvidia.com"})

# (page path, selector that must exist as a CSS rule, component that owns it)
CHECKS: list[tuple[str, str, str]] = [
    ("", ".dynamo-story-windowbar", "LandingStyles"),
    ("", ".dynamo-welcome__terminal", "LandingStyles"),
    ("community", ".dynamo-community-page", "LandingStyles"),
    (
        "external-publications",
        "article:has(.dynamo-pubs__section)",
        "PublicationsStyles",
    ),
    ("digest", ".dynamo-blog-art__grid", "BlogStyles"),
    ("reference/compatibility", ".dynref-panel", "ReferenceStyles"),
    # URL from the nav's explicit slugs (section `benchmarks`, page
    # `llama-3-70b-topology`), not the page's file path.
    (
        "recipes/benchmarks/llama-3-70b-topology",
        ".dynamo-benchmark-grid",
        "RecipeStyles",
    ),
]

# (page path, CSS declaration that must exist, component that owns it)
DECLARATION_CHECKS: list[tuple[str, str, str]] = [
    ("community", "--dynamo-community-rule:", "LandingStyles"),
]

# Minified CSS keeps only mandatory whitespace, so match `.foo{` and `.foo,`
# with optional whitespace before the boundary. In-markup hits are handled
# separately by in_markup_count().
_RULE_BOUNDARY = r"\s*[,{]"


def as_rule_count(selector: str, html: str) -> int:
    """Return occurrences of ``selector`` used as a CSS rule."""
    return len(re.findall(re.escape(selector) + _RULE_BOUNDARY, html))


def in_markup_count(selector: str, html: str) -> int:
    """Return occurrences of ``selector``'s class name inside class=\"...\"."""
    if not selector.startswith("."):
        return 0
    return len(re.findall(r'class="[^"]*' + re.escape(selector[1:]), html))


def declaration_count(declaration: str, html: str) -> int:
    """Return definitions of a required CSS custom property."""
    return html.count(declaration)


def url_from_base(base: str, path: str) -> str:
    """Join ``base`` and ``path``, refuse anything outside ALLOWED_ORIGINS."""
    base = base if base.endswith("/") else base + "/"
    url = urllib.parse.urljoin(base, path)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise SystemExit(f"URL is not https: {url}")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in ALLOWED_ORIGINS:
        raise SystemExit(
            f"{origin} is not an allowed docs origin: {sorted(ALLOWED_ORIGINS)}"
        )
    return url


def fetch_once(url: str) -> str:
    # url_from_base() has confined url to ALLOWED_ORIGINS above, so the raw
    # urlopen() below has an explicit HTTPS trust boundary rather than trusting
    # its input.
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", "ignore")


def probe(url: str, selector: str, retries: int, delay: float) -> tuple[str, int]:
    """Fetch ``url`` and retry when transport fails or the response is stale."""
    last: Exception | None = None
    html = ""
    for attempt in range(1, retries + 1):
        try:
            html = fetch_once(url)
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(delay)
            continue
        count = as_rule_count(selector, html)
        if count or attempt == retries:
            return html, count
        time.sleep(delay)
    raise SystemExit(f"could not fetch {url}: {last}")


def probe_declaration(
    url: str, declaration: str, retries: int, delay: float
) -> tuple[str, int]:
    """Fetch ``url`` and retry until a required declaration is present."""
    last: Exception | None = None
    html = ""
    for attempt in range(1, retries + 1):
        try:
            html = fetch_once(url)
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(delay)
            continue
        count = declaration_count(declaration, html)
        if count or attempt == retries:
            return html, count
        time.sleep(delay)
    raise SystemExit(f"could not fetch {url}: {last}")


def report_declaration(
    check: tuple[str, str, str], base: str, retries: int, delay: float
) -> str | None:
    """Run one DECLARATION_CHECKS entry; return an error on failure."""
    path, declaration, owner = check
    url = url_from_base(base, path)
    _, count = probe_declaration(url, declaration, retries, delay)
    if count:
        print(f"ok    {url}  {declaration} ({count} declarations, {owner})")
        return None
    return (
        f"{url}\n      required declaration {declaration!r} is absent.\n"
        f"      {owner} reached the page without all CSS variables its rules use."
    )


def report(
    check: tuple[str, str, str], base: str, retries: int, delay: float
) -> str | None:
    """Run one CHECKS entry; return an error string or None on success."""
    path, selector, owner = check
    url = url_from_base(base, path)
    html, as_rule = probe(url, selector, retries, delay)
    if as_rule:
        print(f"ok    {url}  {selector} ({as_rule} rules, {owner})")
        return None
    return (
        f"{url}\n      {selector} appears in "
        f"{in_markup_count(selector, html)} element(s) but has no CSS rule.\n"
        f"      {owner} did not reach the page. If the styles moved to "
        f"main.css,\n      the global theme drops them -- deliver them "
        f"from the component instead."
    )


# CSS_RULE_CASES exercises the regex helpers against the real failure modes so
# a boundary or escaping change surfaces here instead of silently inverting the
# published-style gate.
CSS_RULE_CASES: list[tuple[str, str, str, int, int]] = [
    ("minified rule", ".foo", ".foo{color:red}", 1, 0),
    ("rule after comma", ".foo", "h1,.foo{color:red}", 1, 0),
    ("only markup", ".foo", '<div class="foo">x</div>', 0, 1),
    ("substring class is not a rule", ".foo", ".foo-bar{color:red}", 0, 0),
    (
        "rule and markup both present",
        ".foo",
        '.foo{color:red}<div class="foo">x</div>',
        1,
        1,
    ),
    ("class attribute list", ".foo", '<div class="bar foo baz">x</div>', 0, 1),
    ("double-underscore selector", ".dyn__pt", ".dyn__pt{color:red}", 1, 0),
    (
        "functional pseudo-class selector",
        "article:has(.dynamo-pubs__section)",
        "article:has(.dynamo-pubs__section){margin-inline:0 auto}",
        1,
        0,
    ),
]

DECLARATION_CASES: list[tuple[str, str, str, int]] = [
    (
        "custom property definition",
        "--dynamo-community-rule:",
        ".page{--dynamo-community-rule:rgba(0,0,0,.2)}",
        1,
    ),
    (
        "custom property usage is not a definition",
        "--dynamo-community-rule:",
        ".page{border:1px solid var(--dynamo-community-rule)}",
        0,
    ),
]


def run_tests() -> int:
    failed = 0
    for name, selector, html, want_rule, want_markup in CSS_RULE_CASES:
        rule = as_rule_count(selector, html)
        markup = in_markup_count(selector, html)
        if rule == want_rule and markup == want_markup:
            print(f"  PASS: {name}")
            continue
        failed += 1
        print(
            f"  FAIL: {name}\n"
            f"    as_rule_count:   expected {want_rule}, got {rule}\n"
            f"    in_markup_count: expected {want_markup}, got {markup}"
        )
    for name, declaration, html, want in DECLARATION_CASES:
        count = declaration_count(declaration, html)
        if count == want:
            print(f"  PASS: {name}")
            continue
        failed += 1
        print(
            f"  FAIL: {name}\n" f"    declaration_count: expected {want}, got {count}"
        )
    total = len(CSS_RULE_CASES) + len(DECLARATION_CASES)
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--delay", type=float, default=30.0)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        return run_tests()

    failures: list[str] = []
    for check in CHECKS:
        failure = report(check, args.base, args.retries, args.delay)
        if failure is not None:
            failures.append(failure)
    for check in DECLARATION_CHECKS:
        failure = report_declaration(check, args.base, args.retries, args.delay)
        if failure is not None:
            failures.append(failure)

    if failures:
        print("\nFAIL: published pages are missing component CSS\n", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(
        f"\nall {len(CHECKS) + len(DECLARATION_CHECKS)} published style checks passed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
