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

The constants below are transcribed from fern-api's own bundler, so they are
only correct for one release of it. DERIVED_FROM_FERN records which, and the
run below refuses to report a clean tree when docs/fern/fern.config.json has
moved past it — a stale allowlist fails closed but silently, which is the worst
of both.

To re-derive after a Fern bump:

    npm pack fern-api@<version> && tar xzf fern-api-<version>.tgz
    grep -o 'TZu=\\[[^]]*\\]' package/cli.cjs     # the allowlist
    grep -o 'SGm="[^"]*"' package/cli.cjs        # the pinned rolldown version

then update SPECIFIER, ALLOWLIST and DERIVED_FROM_FERN together.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FERN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FERN_ROOT.parents[1]
COMPONENTS = FERN_ROOT / "components"
FERN_CONFIG = FERN_ROOT / "fern.config.json"
SUFFIXES = {".js", ".jsx", ".ts", ".tsx"}

# The fern-api release SPECIFIER and ALLOWLIST were transcribed from.
DERIVED_FROM_FERN = "5.80.2"

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


def pinned_fern_version() -> str | None:
    """The fern-api version this repo actually builds docs with, or None."""
    try:
        return json.loads(FERN_CONFIG.read_text())["version"]
    except (OSError, ValueError, KeyError):
        return None


def check_derivation() -> int:
    """Refuse to vouch for a tree when the constants predate the Fern in use."""
    if not FERN_CONFIG.is_file():
        print(
            f"check_component_imports: expected {FERN_CONFIG} — the script has "
            "moved away from docs/fern/scripts/ and its paths need updating",
            file=sys.stderr,
        )
        return 1
    version = pinned_fern_version()
    if version is None:
        print(
            f"check_component_imports: could not read a version from {FERN_CONFIG}",
            file=sys.stderr,
        )
        return 1
    if version != DERIVED_FROM_FERN:
        print(
            f"check_component_imports: SPECIFIER and ALLOWLIST were transcribed "
            f"from fern-api {DERIVED_FROM_FERN}, but fern.config.json now pins "
            f"{version}. Re-derive them (see this script's docstring) and update "
            f"DERIVED_FROM_FERN, or this check silently vouches for the wrong "
            f"bundler.",
            file=sys.stderr,
        )
        return 1
    return 0


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


USAGE = "usage: check_component_imports.py [--test]"


def main() -> int:
    args = sys.argv[1:]
    if args not in ([], ["--test"]):
        print(
            f"check_component_imports: unrecognized arguments {args}", file=sys.stderr
        )
        print(USAGE, file=sys.stderr)
        return 2
    if args == ["--test"]:
        return run_tests()
    stale = check_derivation()
    if stale:
        return stale
    failures = 0
    for path in sorted(COMPONENTS.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        found = offenders(path.read_text())
        if not found:
            continue
        failures += 1
        rel = path.relative_to(REPO_ROOT)
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
