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
"""Guard against module specifiers Fern would try to bundle with rolldown.

docs.yml registers docs/fern/components via `experimental.mdx-components`. When
Fern publishes, it scans each component source for module specifiers that are
neither relative nor allowlisted, and treats what it finds as third-party
dependencies to bundle -- shelling out to `npx rolldown@<pin>` with the
component's directory as cwd. The docs-website branch ships no package.json
next to fern/, so that bundle cannot resolve anything and the build fails with:

    Failed to bundle third-party imports in fern/components/<file>.tsx:
    rolldown exited with code 127 ... sh: 1: rolldown: not found

The trap is that Fern's scan is a plain-text regex over the whole file -- it
does not parse the source, so it does not skip comments. A usage example in a
component's doc header, such as

    * <the ES module keyword> { RecipeStyles } from "@/components/RecipeStyles";

is collected as a real dependency on `@/components` and sends the publish down
the rolldown path. That is what broke the "Preview or publish docs" job: the
failing component rotated between RecipeStyles, ReferenceStyles and
TerminalDemo depending on which one Fern reached first, and none of the three
had an actual npm dependency. Document the specifier in prose instead.

The regex and allowlist below mirror the Fern CLI's own implementation (cli.cjs,
the helper behind "Failed to bundle third-party imports"). That copy can drift,
so a full scan also fails when fern.config.json is bumped past PORTED_FROM --
re-read the bundler in the new CLI, reconcile the two constants, then move
PORTED_FROM.

Usage: python3 check_component_imports.py [files...]
With no arguments, checks every source file under docs/fern/components and
verifies the Fern pin. Run with --test to self-test the matcher.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # docs/fern
REPO = ROOT.parents[1]  # repo root, for readable paths
COMPONENTS = ROOT / "components"

# The Fern CLI release SPECIFIER and ALLOWLIST below were read out of. Nothing
# makes the CLI keep that shape, so a bump that widens the allowlist -- or
# replaces the text scan with a real parser that skips comments -- would leave
# this check silently over- or under-reporting. Under-reporting is the costly
# direction: it breaks the docs publish exactly the way this script exists to
# prevent. Fail on the bump instead, while someone is already looking at Fern.
PORTED_FROM = "5.80.2"

# Fern only scans these extensions.
SOURCE_SUFFIXES = (".js", ".jsx", ".ts", ".tsx")

# Specifiers Fern resolves itself; everything else is "third party".
ALLOWLIST = ("react", "react-dom", "@mdx-js/react", "next")

# Fern's own specifier regex, verbatim. Note it is applied to raw text, which is
# why comments count.
SPECIFIER = re.compile(
    r"""(?:^|[^\w.])(?:import|export)\s+(?:[\w*\s{},$]*?from\s+)?["']([^"'\n]+)["']"""
    r"""|import\(\s*["']([^"'\n]+)["']\s*\)"""
    r"""|require\(\s*["']([^"'\n]+)["']\s*\)""",
    re.M,
)


def package_root(specifier: str) -> str:
    """Reduce a specifier to the package Fern would have to install."""
    parts = specifier.split("/")
    if specifier.startswith("@"):
        return "/".join(parts[:2])
    return parts[0] if parts else specifier


def is_relative(specifier: str) -> bool:
    return specifier.startswith(("./", "../")) or specifier in (".", "..")


def third_party(contents: str) -> list[tuple[int, str]]:
    """Return (line number, specifier) for everything Fern would bundle."""
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    for match in SPECIFIER.finditer(contents):
        group = next((i for i in (1, 2, 3) if match.group(i) is not None), None)
        if group is None:
            continue
        specifier = match.group(group)
        if is_relative(specifier):
            continue
        if package_root(specifier) in ALLOWLIST or specifier in seen:
            continue
        seen.add(specifier)
        # Anchor on the specifier, not on match.start(): the first alternation
        # opens with (?:^|[^\w.]), which consumes the preceding newline for a
        # statement at column 0 and would report the line above.
        found.append((contents[: match.start(group)].count("\n") + 1, specifier))
    return found


def check_pin_drift() -> list[str]:
    """Report when fern.config.json has moved past the ported-from release."""
    config_path = ROOT / "fern.config.json"
    try:
        pinned = json.loads(config_path.read_text(encoding="utf-8")).get("version", "")
    except (OSError, ValueError) as exc:
        return [
            f"{display(config_path)}: could not read the pinned Fern version: {exc}"
        ]
    if pinned == PORTED_FROM:
        return []
    return [
        f"{display(config_path)}: pins Fern {pinned}, but this check's matcher was"
        f" ported from {PORTED_FROM}.\n"
        f"      Re-read the third-party-import bundler in {pinned}, reconcile"
        f" SPECIFIER and ALLOWLIST, then update PORTED_FROM."
    ]


def display(path: Path) -> str:
    """Repo-relative when possible; a caller may pass a path from anywhere."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def check(path: Path) -> list[str]:
    # Explicit encoding: most component sources carry non-ASCII (em dashes in
    # the doc headers), and the locale default is not UTF-8 everywhere.
    text = path.read_text(encoding="utf-8")
    return [
        f"{display(path)}:{line}: {specifier}\n"
        f"      Fern reads this as a third-party dependency and bundles the file"
        f" with rolldown, which fails the docs publish.\n"
        f"      If it is a usage example or a comment, describe the specifier in"
        f" prose instead of writing a literal statement."
        for line, specifier in third_party(text)
    ]


def selftest() -> int:
    # (source, should_flag, expected line number or None to skip the check)
    cases: list[tuple[str, bool, int | None]] = [
        ('import { useState } from "react";', False, None),
        ('import ReactDOM from "react-dom/client";', False, None),
        ('import { useMDXComponents } from "@mdx-js/react";', False, None),
        ('import { CURRENT_TAG } from "./releases.data";', False, None),
        ('import { X } from "../shared/x";', False, None),
        # The regression this script exists for: an example inside a comment.
        (' *   import { RecipeStyles } from "@/components/RecipeStyles";', True, 1),
        ('So instead of `import "asciinema-player"` we load it from a CDN.', True, 1),
        ('export { Foo } from "@/components/Foo";', True, 1),
        ('const mod = await import("chart.js");', True, 1),
        ('const lodash = require("lodash");', True, 1),
        # A statement at column 0 must report its own line, not the one above:
        # the leading (?:^|[^\w.]) alternation eats the previous newline.
        ('header\nimport { X } from "@/components/X";\n', True, 2),
        ('a\nb\nexport { Y } from "@/components/Y";\n', True, 3),
        # Prose that merely contains the word must not trip the matcher.
        (
            '* fails with "Failed to bundle third-party imports" at publish.',
            False,
            None,
        ),
        (
            "* Every page must pull in the named export from `@/components/Foo`.",
            False,
            None,
        ),
    ]
    failures = 0
    for source, should_flag, expected_line in cases:
        hits = third_party(source)
        if bool(hits) != should_flag:
            failures += 1
            verb = "flagged" if hits else "missed"
            print(f"selftest: {verb} unexpectedly: {source!r}", file=sys.stderr)
            continue
        if expected_line is not None and hits[0][0] != expected_line:
            failures += 1
            print(
                f"selftest: reported line {hits[0][0]}, expected {expected_line}:"
                f" {source!r}",
                file=sys.stderr,
            )
    if failures:
        print(f"{failures} selftest case(s) failed", file=sys.stderr)
        return 1
    print(f"selftest: {len(cases)} case(s) passed")
    return 0


def main() -> int:
    if "--test" in sys.argv[1:]:
        return selftest()

    args = [Path(a) for a in sys.argv[1:]]
    targets = args or sorted(
        p for p in COMPONENTS.rglob("*") if p.suffix in SOURCE_SUFFIXES
    )
    targets = [t for t in targets if t.suffix in SOURCE_SUFFIXES and t.is_file()]

    problems: list[str] = []
    for target in targets:
        problems.extend(check(target))

    if problems:
        print("module specifiers Fern cannot bundle:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\n  Fern resolves only "
            + ", ".join(ALLOWLIST)
            + " and relative paths. A component that genuinely needs an npm"
            " package also needs a package.json on the docs-website branch,"
            " which does not exist -- load it from a CDN at runtime instead"
            " (see docs/fern/components/TerminalDemo.tsx).",
            file=sys.stderr,
        )
        return 1

    # Only on a full scan: an explicit file list is a targeted call, and the pin
    # is a property of the tree rather than of any one component.
    if not args:
        drift = check_pin_drift()
        if drift:
            print("Fern CLI pin has moved past this check:\n", file=sys.stderr)
            for problem in drift:
                print(f"  - {problem}", file=sys.stderr)
            return 1

    print(f"checked {len(targets)} component source(s): no unbundleable specifiers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
