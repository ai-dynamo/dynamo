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
"""Fail on imports that make Fern shell out to rolldown during a docs build.

Fern's `experimental.mdx-components` bundler scans every .js/.jsx/.ts/.tsx file
under docs/fern/components for import specifiers. Anything neither relative
(./, ../) nor in its allowlist is treated as a third-party dependency, and Fern
runs `npx rolldown` once per offending file — a registry download on every docs
build, and the direct cause of intermittent preview failures:

    rolldown exited with code 127 / sh: 1: rolldown: not found
    rolldown exited with code 1 / ERR_MODULE_NOT_FOUND ... _npx/.../rolldown/...

Fern's scan is a regex over raw text and does not skip comments, so a docblock
usage example is indistinguishable from a real dependency. That is why usage
examples live in components/README.md instead: markdown is outside the scan.

This check reads the same raw text Fern does, comments included, so it catches
an example that slips back into a docblock. SPECIFIER is transcribed from
fern-api rather than approximated: a looser pattern matching any quoted string
after an import/export keyword fires on ordinary value exports such as
`export const KIND = "all"`, which is why the real one insists on `from` or a
directly quoted specifier.

Being a transcription, it is only correct for one release. DERIVED_FROM_FERN
records which, and the scan refuses to report a clean tree once
docs/fern/fern.config.json moves past it — a stale pattern fails closed but
silently, which is the worst outcome for a gate. To re-derive after a bump:

    npm pack fern-api@<version> && tar xzf fern-api-<version>.tgz
    grep -o 'TZu=\\[[^]]*\\]' package/cli.cjs     # the allowlist
    grep -o 'SGm="[^"]*"' package/cli.cjs        # the pinned rolldown version

then update SPECIFIER, ALLOWLIST and DERIVED_FROM_FERN together.

If a component ever genuinely needs a third-party dependency, this check has to
be revisited alongside a package.json and node_modules for the docs project,
which is what Fern's error message asks for.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FERN_ROOT = Path(__file__).resolve().parents[1]
# Not FERN_ROOT.parents[1]: that raises IndexError at import time if the script
# is ever run from a shallower path, before check_derivation can say why. Nor a
# slice of .parents, which is 3.10+ only — this hook is `language: system`, so
# it runs under whichever python3 is on PATH, and crashing on an older one turns
# a gate that must fail loudly into a traceback.
REPO_ROOT = FERN_ROOT.parent.parent
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
# Fern resolves these itself and never bundles them.
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
    """Refuse to vouch for a tree when SPECIFIER predates the Fern in use."""
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


# The first case is the one this check exists for: an example in a comment is
# what broke docs previews, and Fern cannot tell it from a real dependency.
CASES: list[tuple[str, str, list[str]]] = [
    (
        "usage example in a comment",
        ' *   import { Foo } from "@/components/Foo";',
        ["@/components/Foo"],
    ),
    ("real third-party import", 'import x from "some-pkg";', ["some-pkg"]),
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
    ("prose with no specifier", " * We import the styles at build time.", []),
    # Value exports carrying string literals are the reason SPECIFIER insists on
    # `from` or a directly quoted specifier. A looser pattern flags all three of
    # these, which is how install-selector-data.ts and releases.data.ts fail.
    ("value export of a string", 'export const KIND = "all";', []),
    ("value export of an object", 'export const V = { channel: "stable" };', []),
    ("value export of a list", 'export const TAGS = ["sglang", "container"];', []),
    # Pins re.ASCII. Without it Python treats the accented letter as a word
    # character, [^\w.] fails to match, and the specifier is missed — while
    # Fern, whose \w is ASCII-only, bundles the file.
    (
        "unicode letter against the import keyword",
        'éimport { X } from "@scope/pkg";',
        ["@scope/pkg"],
    ),
]

USAGE = "usage: check_component_imports.py [--test]"


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
        print(
            f"{path.relative_to(REPO_ROOT)}: Fern will bundle this file with "
            f"rolldown because it reads {', '.join(found)} as a third-party import.",
            file=sys.stderr,
        )
    if failures:
        print(
            "\nUsage examples belong in docs/fern/components/README.md, where "
            "Fern's scan cannot see them. A real dependency needs the wider fix "
            "in this script's docstring.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
