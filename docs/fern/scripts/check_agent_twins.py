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
"""Require an agent-readable twin for data a React component renders.

Fern derives the Markdown twin and llms.txt from MDX. A page that renders its
data through a React component publishes that data to humans and to nobody
else: the component's output never reaches the twin, so an agent reading
llms.txt sees the surrounding prose with the table missing. Nothing about the
page looks wrong, which is what makes it easy to ship.

This has happened three times. The Python API reference was rebuilt on Fern's
own MDX components for exactly this reason (#12110). The compatibility page
replaced a generated support-matrix accordion with ReleaseSupportMatrix. The
pairwise interaction matrices on that page moved to InteractionStatus.

The expectation is derived from releases.data.ts, the same source the
components read, so it tracks the data instead of a hand-maintained keyword
list. Add a release and the twin must carry it; rename a feature and the twin
must follow. A hardcoded list would drift the moment someone shipped a release
without touching this file, which is precisely when the check needs to fire.

Coverage is a threshold, not a demand for every item. The twin summarizes: a
support matrix may legitimately group patch releases, and the feature tables
name features without repeating every one in prose. Requiring 100% would make
the check unusable and it would be turned off. Requiring most of them catches
a twin that was never written or that silently stopped generating, which is
the failure that actually occurs.

Usage: python3 check_agent_twins.py [files...]
With no arguments, checks every page under docs/fern/pages/.
Run with --test to exercise the matcher against its own cases.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # docs/fern
REPO = ROOT.parents[1]
DATA = ROOT / "components" / "releases.data.ts"

LLMS_ONLY = re.compile(r"<llms-only>(.*?)</llms-only>", re.S)

# Fraction of the derived items a twin must mention. Set from the current
# pages: the compatibility twin carries every feature and every stable
# release, so the slack is headroom for legitimate summarizing, not a
# tolerance for missing data.
THRESHOLD = 0.8


def _releases(source: str) -> set[str]:
    """Versions the support matrix renders, read from the array it reads.

    ReleaseSupportMatrix derives its rows from CUDA_HISTORY, and its own
    comment says only stable and patch releases carry those rows. Deriving
    from RELEASES instead would demand coverage of dev builds and of releases
    that predate the matrix, and the check would fail on a correct twin.
    """
    try:
        blk = source[source.index("export const CUDA_HISTORY") :]
    except ValueError:
        return set()
    blk = blk.split("\nexport const ", 1)[0]
    return set(re.findall(r'version:\s*"v?([0-9][^"]*)"', blk))


def _features(source: str) -> set[str]:
    """Feature names from the FEATURES block."""
    try:
        blk = source[source.index("export const FEATURES") :]
    except ValueError:
        return set()
    blk = blk.split("export const ", 2)[1] if "export const " in blk[1:] else blk
    return set(re.findall(r'name:\s*"([^"]+)"', blk))


def expectations(source: str) -> dict[str, object]:
    """What each component's twin has to account for.

    A set means every item must appear: the support matrix either carries a
    release row or it does not.

    The tuple-of-regexes shape is supported for a component whose data cannot
    be expressed as a name list, because the names also appear in a
    neighbouring table and would satisfy the check on their own. Nothing needs
    it today. The pairwise interaction matrices are plain markdown, which
    reaches the twin natively and needs no guard.
    """
    feats = _features(source)
    return {
        "ReleaseSupportMatrix": _releases(source),
        "FeatureHeatmap": feats,
    }


def _uses(text: str, component: str) -> bool:
    """True when the component is rendered, not merely imported or named."""
    return re.search(rf"<{re.escape(component)}[\s/>]", text) is not None


def check(path: Path, expected: dict[str, set[str]]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    used = [c for c in expected if _uses(text, c)]
    # A component that renders data but is not declared above would slip past
    # silently, so the naming convention is the tripwire: adding one forces a
    # decision about its twin instead of defaulting to no coverage.
    undeclared = sorted(
        set(re.findall(r"<([A-Z]\w*(?:Matrix|Heatmap|Status|Table))[\s/>]", text))
        - set(expected)
    )
    if undeclared:
        rel_u = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        return [
            f"{rel_u}: renders {', '.join(undeclared)}, which is not declared in "
            f"expectations(). Add it with what its twin must carry, or rename it "
            f"if it does not render data."
        ]
    if not used:
        return []

    rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
    twins = LLMS_ONLY.findall(text)
    if not twins:
        return [
            f"{rel}: renders {', '.join(sorted(used))} but has no <llms-only> "
            f"twin, so the data reaches humans and not llms.txt."
        ]

    blob = "\n".join(twins)
    problems: list[str] = []
    for component in sorted(used):
        spec = expected[component]

        if isinstance(spec, tuple):
            missing = [pat for pat in spec if not re.search(pat, blob)]
            if missing:
                problems.append(
                    f"{rel}: <llms-only> twin is missing the structure "
                    f"{component} renders. No match for: {', '.join(missing)}"
                )
            continue

        if not spec:
            continue
        # Word-boundary, so "1.3.0rc19" does not satisfy "1.3.0". A twin of
        # release candidates would otherwise score full marks while carrying
        # no released row at all.
        missing = sorted(
            i
            for i in spec
            if not re.search(rf"(?<![\w.]){re.escape(i)}(?![\w.])", blob, re.I)
        )
        covered = 1 - len(missing) / len(spec)
        if covered < THRESHOLD:
            shown = ", ".join(missing[:6])
            more = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
            problems.append(
                f"{rel}: <llms-only> twin covers {covered:.0%} of what "
                f"{component} renders, below {THRESHOLD:.0%}. Missing: {shown}{more}"
            )
    return problems


def _selftest() -> int:
    exp = {"ReleaseSupportMatrix": {"1.3.0", "1.3.1", "1.2.0", "1.2.1", "1.1.0"}}
    cases: list[tuple[str, str, bool]] = [
        ("no component", "Prose mentioning 1.3.0 and nothing else.", True),
        ("component, no twin", "<ReleaseSupportMatrix />", False),
        (
            "twin covering everything",
            "<ReleaseSupportMatrix />\n<llms-only>1.3.0 1.3.1 1.2.0 1.2.1 1.1.0</llms-only>",
            True,
        ),
        (
            "twin at threshold",
            "<ReleaseSupportMatrix />\n<llms-only>1.3.0 1.3.1 1.2.0 1.2.1</llms-only>",
            True,
        ),
        (
            "twin below threshold",
            "<ReleaseSupportMatrix />\n<llms-only>1.3.0 only</llms-only>",
            False,
        ),
        (
            "import alone is not use",
            'import { ReleaseSupportMatrix } from "@/components/ReleaseSupportMatrix";',
            True,
        ),
        ("open tag counts as use", "<ReleaseSupportMatrix>", False),
    ]
    tmp = Path("/tmp/_agent_twin_case.mdx")
    passed = 0
    for name, body, expect_ok in cases:
        tmp.write_text(body, encoding="utf-8")
        ok = not check(tmp, exp)
        if ok == expect_ok:
            passed += 1
        else:
            print(f"  FAIL {name}: expected {'pass' if expect_ok else 'fail'}")
    tmp.unlink(missing_ok=True)
    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


def main() -> int:
    if "--test" in sys.argv:
        return _selftest()

    if not DATA.exists():
        print(
            f"::error::{DATA} not found; cannot derive twin expectations",
            file=sys.stderr,
        )
        return 1
    expected = expectations(DATA.read_text(encoding="utf-8"))
    if not any(expected.values()):
        print(
            "::error::derived no releases or features from releases.data.ts",
            file=sys.stderr,
        )
        return 1

    args = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    targets = [a for a in args if a.exists()] or sorted((ROOT / "pages").rglob("*.mdx"))

    problems: list[str] = []
    for target in targets:
        problems.extend(check(target, expected))

    if problems:
        print("component-rendered data missing from the agent twin:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    counts = ", ".join(f"{k} {len(v)}" for k, v in expected.items() if v)
    print(
        f"checked {len(targets)} page(s) against releases.data.ts ({counts}): twins cover their components"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
