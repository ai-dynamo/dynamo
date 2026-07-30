#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail on imports that make Fern shell out to rolldown during a docs build.

Fern's `experimental.mdx-components` bundler regex-scans every .js/.jsx/.ts/.tsx
file under docs/fern/components for import/export/require specifiers. Anything
that is neither relative (./, ../) nor in Fern's allowlist (react, react-dom,
@mdx-js/react, next) is treated as a third-party dependency, and Fern runs

    npx --quiet --yes rolldown@<pinned> -c <config>

once per offending file. That is a registry download on every docs build, and it
is the direct cause of the intermittent preview failures:

    rolldown exited with code 127 / sh: 1: rolldown: not found
    rolldown exited with code 1 / ERR_MODULE_NOT_FOUND ... _npx/.../rolldown/...

The scan does not skip comments, so a docblock usage example such as

    import { RecipeStyles } from "@/components/RecipeStyles";

is enough to trigger it even though no component actually depends on anything
outside the allowlist. Write such examples with backticks around the specifier
instead of quotes, and say so in the surrounding comment.

If a component ever needs a real third-party dependency, this check has to be
revisited together with a package.json + node_modules for the docs project —
which is exactly what Fern's error message asks for.

The constants below mirror fern-api's own bundler; keep them in sync when the
pin in docs/fern/fern.config.json moves.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
SUFFIXES = {".js", ".jsx", ".ts", ".tsx"}

# Verbatim from fern-api's cli.cjs. re.ASCII is not cosmetic: JavaScript's \w is
# ASCII-only, Python's is Unicode-aware, so without it a Unicode letter sitting
# directly against `import` is a word character here but not there — Fern would
# bundle the file and this check would stay silent.
SPECIFIER = re.compile(
    r"""(?:^|[^\w.])(?:import|export)\s+(?:[\w*\s{},$]*?from\s+)?["']([^"'\n]+)["']"""
    r"""|import\(\s*["']([^"'\n]+)["']\s*\)"""
    r"""|require\(\s*["']([^"'\n]+)["']\s*\)""",
    re.MULTILINE | re.ASCII,
)
ALLOWLIST = {"react", "react-dom", "@mdx-js/react", "next"}


def package_name(specifier: str) -> str:
    parts = specifier.split("/")
    if specifier.startswith("@"):
        return "/".join(parts[:2])
    return parts[0] if parts else specifier


def is_relative(specifier: str) -> bool:
    return specifier.startswith(("./", "../")) or specifier in {".", ".."}


def offenders(text: str) -> list[str]:
    found: list[str] = []
    for match in SPECIFIER.finditer(text):
        specifier = next((g for g in match.groups() if g is not None), None)
        if specifier is None or is_relative(specifier):
            continue
        if package_name(specifier) in ALLOWLIST:
            continue
        if specifier not in found:
            found.append(specifier)
    return found


# The two cases this check exists for are the first two: a quoted specifier in a
# comment must be caught, a backticked one must not. The rest pin the allowlist
# and the import()/require() branches of SPECIFIER.
CASES: list[tuple[str, str, list[str]]] = [
    (
        "quoted specifier in a comment",
        ' *   import { Foo } from "@/components/Foo";',
        ["@/components/Foo"],
    ),
    (
        "backtick example in a comment",
        " *   import { Foo } from `@/components/Foo`;",
        [],
    ),
    ("relative import", 'import { Foo } from "./Foo";', []),
    ("parent-relative import", 'import { Foo } from "../shared/Foo";', []),
    ("allowlisted bare package", 'import { useState } from "react";', []),
    ("allowlisted scoped package", 'import { X } from "@mdx-js/react";', []),
    (
        "non-allowlisted scoped package",
        'import { X } from "@scope/pkg";',
        ["@scope/pkg"],
    ),
    ("re-export", 'export { Foo } from "@/components/Foo";', ["@/components/Foo"]),
    ("dynamic import", 'const m = await import("some-pkg");', ["some-pkg"]),
    ("require", 'const m = require("some-pkg");', ["some-pkg"]),
    (
        "deduplicated",
        'import "@scope/pkg";\nimport { X } from "@scope/pkg";',
        ["@scope/pkg"],
    ),
    # Pins re.ASCII. Without it Python treats the accented letter as a word
    # character, [^\w.] fails to match, and the specifier is missed — while
    # Fern, whose \w is ASCII-only, bundles the file.
    (
        "unicode letter against the import keyword",
        'éimport { X } from "@scope/pkg";',
        ["@scope/pkg"],
    ),
]


def run_tests() -> int:
    failed = 0
    for name, source, expected in CASES:
        actual = offenders(source)
        if actual == expected:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}\n    expected: {expected}\n    actual:   {actual}")
            failed += 1
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


def main() -> int:
    if "--test" in sys.argv[1:]:
        return run_tests()
    failures = 0
    for path in sorted(COMPONENTS.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        found = offenders(path.read_text())
        if not found:
            continue
        failures += 1
        rel = path.relative_to(COMPONENTS.parents[2])
        print(
            f"{rel}: Fern will bundle this file with rolldown because it reads "
            f"{', '.join(found)} as a third-party import.",
            file=sys.stderr,
        )
    if failures:
        print(
            "\nIf the specifier is inside a comment, quote it with backticks "
            "instead. If it is real code, see this script's docstring.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
