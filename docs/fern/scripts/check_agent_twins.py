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
replaced a generated support-matrix accordion with ReleaseSupportMatrix, and
its pairwise matrices moved to FeatureInteractions.

The expectation is derived from releases.data.ts, the same source the
components read, so it tracks the data instead of a hand-maintained keyword
list. Add a release and the twin must carry it; rename a feature and the twin
must follow. A hardcoded list would drift the moment someone shipped a release
without touching this file, which is precisely when the check needs to fire.

Coverage floors are measured against the current twins, not guessed. Releases,
artifacts and early-access models each sit at full coverage today, so anything
lower only buys room for a regression. Feature names keep a margin because the
twin names them in prose as well as in a table.

Usage: python3 check_agent_twins.py [files...]
With no arguments, checks the pages in IN_SCOPE.
Run with --test to exercise the matcher against its own cases.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # docs/fern
REPO = ROOT.parents[1]
DATA = ROOT / "components" / "releases.data.ts"

# Pages this check governs. Scoped rather than repo-wide because 18 other
# components are rendered across the docs and 9 of them read releases.data,
# each needing its own decision about whether its page owes a twin. Guessing
# at those would put unverified expectations into a merge-blocking gate.
# Expanding this list is the follow-up; the compatibility page is here because
# it is where component-rendered data replaced markdown that agents could read.
IN_SCOPE: frozenset[str] = frozenset(
    {
        "pages/reference/general/compatibility.mdx",
        "pages/reference/general/release-artifacts.mdx",
        "pages/reference/general/model-early-access-builds.mdx",
    }
)

# Rendered on an in-scope page but carrying no data of their own: styling, or
# a control whose values the twin already states through another component's
# expectation on the same page.
EXEMPT: frozenset[str] = frozenset(
    {
        "ReferenceStyles",
        "CompatibilityHero",
        "PinnedEnvironment",
        "TagLookup",
    }
)


def _components() -> set[str]:
    """Component names available to pages, from the components directory."""
    return {p.stem for p in (ROOT / "components").glob("*.tsx")}


LLMS_ONLY = re.compile(r"<llms-only>(.*?)</llms-only>", re.S)

# Measured against the current twins rather than guessed: releases, artifacts
# and EA models each sit at 100 percent coverage today, so a floor below that
# only buys room for a regression. Features are 100 percent too but keep a
# margin, because the twin names them in prose as well as in the table and an
# exact count there would break on a rewording that loses nothing.
THRESHOLD: dict[str, float] = {
    "ReleaseSupportMatrix": 1.0,
    "ArtifactBrowser": 1.0,
    "ModelEABuildCards": 1.0,
    "FeatureHeatmap": 0.8,
}
DEFAULT_THRESHOLD = 0.8


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


def _interaction_backends(source: str) -> list[str]:
    """Backend keys in FEATURE_INTERACTIONS, the array FeatureInteractions reads."""
    try:
        blk = source[source.index("export const FEATURE_INTERACTIONS") :]
    except ValueError:
        return []
    blk = blk.split("\nexport const ", 1)[0]
    return sorted(set(re.findall(r'backend:\s*"([^"]+)"', blk)))


INTERACTIONS_HEADING = r"[Ff]eature interactions by backend"
RELEASES_HEADING = r"\*\*CUDA toolkit and minimum driver per Dynamo release\*\*"
RELEASES_END = r"\n\*\*"


def _interaction_features(source: str) -> set[str]:
    """Row labels in the pairwise matrices.

    Bounded on the array literal, not on the next ``export const``. Two other
    exports sit between INTERACTION_FEATURES and the next const -- a type and
    an interface -- so splitting on ``export const`` swallowed both and
    admitted the backend names from ``backend: "SGLang" | "TensorRT-LLM" |
    "vLLM"``. That inflated the denominator in the "missing N of M row labels"
    message, and inconsistently: the uppercase filter kept SGLang and
    TensorRT-LLM but dropped vLLM.
    """
    try:
        blk = source[source.index("export const INTERACTION_FEATURES") :]
    except ValueError:
        return set()
    end = blk.find("]")
    if end == -1:
        return set()
    return set(re.findall(r'"([^"]+)"', blk[:end]))


def _interactions_segment(blob: str) -> str:
    """The twin from the interactions heading onward, or empty if absent.

    Scoping matters: these row labels also appear in the feature-support table
    above, so an unscoped search is satisfied by that table and every matrix
    row can be deleted without the check noticing.
    """
    m = re.search(INTERACTIONS_HEADING, blob)
    return blob[m.start() :] if m else ""


def _releases_segment(blob: str) -> str:
    """The release-table section of the twin, or empty if absent.

    Scoped for the same reason as the interactions section, and it matters
    more here. The backend-pins table in the same twin lists bare engine and
    NIXL versions, so an unscoped search finds "1.3.2" there and counts the
    release row as covered. Measured on this page, deleting every release row
    still scored 40% rather than 0%, which is above no floor but well above
    the zero the check should have reported.
    """
    m = re.search(RELEASES_HEADING, blob)
    if not m:
        return ""
    rest = blob[m.end() :]
    nxt = re.search(RELEASES_END, rest)
    return rest[: nxt.start()] if nxt else rest


def _named(source: str, const: str, field: str = "name") -> set[str]:
    """Values of one field inside one exported array."""
    try:
        blk = source[source.index(f"export const {const}") :]
    except ValueError:
        return set()
    blk = blk.split("\nexport const ", 1)[0]
    return set(re.findall(rf'{field}:\s*"([^"]+)"', blk))


def expectations(source: str) -> dict[str, object]:
    """What each component's twin has to account for.

    A set means every item must appear: the support matrix either carries a
    release row or it does not.

    A tuple of regexes means the twin must have a particular shape. The
    pairwise interaction matrices need this because their feature names also
    appear in the feature-support table directly above them, so a name list is
    satisfied by that table alone and the matrices could be deleted without
    the check noticing.
    """
    feats = _features(source)
    backends = _interaction_backends(source) or ["vLLM", "SGLang", "TensorRT-LLM"]
    return {
        "ReleaseSupportMatrix": _releases(source),
        "FeatureHeatmap": feats,
        "FeatureInteractions": tuple(
            [r"[Ff]eature interactions by backend"]
            + [rf"\*{re.escape(b)}\*" for b in backends]
        ),
        "ArtifactBrowser": _named(source, "ARTIFACTS"),
        "ModelEABuildCards": _named(source, "MODEL_EA_BUILDS", "model"),
    }


def _uses(text: str, component: str) -> bool:
    """True when the component is rendered, not merely imported or named."""
    return re.search(rf"<{re.escape(component)}[\s/>]", text) is not None


def check(path: Path, expected: dict[str, object], source: str = "") -> list[str]:
    text = path.read_text(encoding="utf-8")
    used = [c for c in expected if _uses(text, c)]
    # Enumerated from the components directory, not guessed from the name. A
    # suffix rule missed FeatureInteractions, which is exactly the component
    # that most needed catching, so the repo's own component list is the
    # source of truth and anything new is undeclared until someone decides.
    undeclared = sorted(
        {c for c in _components() if _uses(text, c)} - set(expected) - EXEMPT
    )
    if undeclared:
        rel_u = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        return [
            f"{rel_u}: renders {', '.join(undeclared)}, which is not declared in "
            f"expectations() or EXEMPT. Say what its twin must carry, or exempt "
            f"it if it renders no data."
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
            # Shape alone is not enough: keeping the heading and the backend
            # markers while deleting every table row under them passed. Row
            # labels are therefore required inside the section itself, where
            # the feature-support table above cannot satisfy them.
            segment = _interactions_segment(blob)
            rows = _interaction_features(source)
            absent = sorted(r for r in rows if r not in segment)
            if rows and absent:
                shown = ", ".join(absent[:6])
                more = f" (+{len(absent) - 6} more)" if len(absent) > 6 else ""
                problems.append(
                    f"{rel}: the interactions section of the twin is missing "
                    f"{len(absent)} of {len(rows)} row labels: {shown}{more}"
                )
            continue

        if not spec:
            continue
        # Releases are searched only inside their own section, for the same
        # reason as the interactions rows above: bare version strings appear
        # in the backend-pins table too, so an unscoped search scores a twin
        # whose release rows were all deleted.
        haystack = (
            _releases_segment(blob) if component == "ReleaseSupportMatrix" else blob
        )
        # Word-boundary, so "1.3.0rc19" does not satisfy "1.3.0". A twin of
        # release candidates would otherwise score full marks while carrying
        # no released row at all.
        missing = sorted(
            i
            for i in spec
            if not re.search(rf"(?<![\w.]){re.escape(i)}(?![\w.])", haystack, re.I)
        )
        covered = 1 - len(missing) / len(spec)
        floor = THRESHOLD.get(component, DEFAULT_THRESHOLD)
        if covered < floor:
            shown = ", ".join(missing[:6])
            more = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
            problems.append(
                f"{rel}: <llms-only> twin covers {covered:.0%} of what "
                f"{component} renders, below {floor:.0%}. Missing: {shown}{more}"
            )
    return problems


def _selftest() -> int:
    """Cases that have actually failed here, not cases invented to pass.

    Every one of these except the first two is a defect this file shipped with
    at some point: substring matching that accepted release candidates, a
    suffix-based tripwire that missed FeatureInteractions, and a structural
    expectation that could not fail because it guarded a deleted component.
    """
    exp = {
        "ReleaseSupportMatrix": {"1.3.0", "1.3.1", "1.2.0", "1.2.1", "1.1.0"},
        "FeatureInteractions": (r"[Ff]eature interactions", r"\*vLLM\*", r"\*SGLang\*"),
        "FeatureHeatmap": {"alpha", "bravo", "charlie", "delta", "echo"},
    }
    twin = "<llms-only>{}</llms-only>"

    def rel(versions: str) -> str:
        """A release twin shaped like the real one.

        The release search is scoped to this heading, so a fixture without it
        models a page the check would score at zero -- which is the point, but
        it has to be stated deliberately rather than by omission.
        """
        return twin.format(
            "**CUDA toolkit and minimum driver per Dynamo release**\n" + versions
        )

    full = rel("1.3.0 1.3.1 1.2.0 1.2.1 1.1.0")
    cases: list[tuple[str, str, bool]] = [
        ("no component", "Prose mentioning 1.3.0.", True),
        ("component, no twin", "<ReleaseSupportMatrix />", False),
        ("twin covering everything", "<ReleaseSupportMatrix />\n" + full, True),
        # Releases are all-or-nothing. One missing row must fail, because
        # adding a release the twin never gains is the change this catches,
        # and at a fractional threshold that change passes.
        (
            "one release missing fails",
            "<ReleaseSupportMatrix />\n" + rel("1.3.0 1.3.1 1.2.0 1.2.1"),
            False,
        ),
        (
            "twin well below",
            "<ReleaseSupportMatrix />\n" + rel("1.3.0"),
            False,
        ),
        # The release search is scoped to its own heading. Versions appear in
        # the backend-pins table too, so an unscoped search scored a twin whose
        # release rows had all been deleted -- measured at 40%, not 0%.
        (
            "versions outside the release section do not count",
            "<ReleaseSupportMatrix />\n"
            + twin.format(
                "NIXL 1.3.0 UCX 1.3.1 engine 1.2.0 pin 1.2.1 base 1.1.0\n"
                "**CUDA toolkit and minimum driver per Dynamo release**\n"
                "(rows deleted)"
            ),
            False,
        ),
        # Features keep headroom, since the twin names them in prose too.
        (
            "feature headroom holds",
            "<FeatureHeatmap />\n" + twin.format("alpha bravo charlie delta"),
            True,
        ),
        (
            "features below headroom",
            "<FeatureHeatmap />\n" + twin.format("alpha"),
            False,
        ),
        (
            "import alone is not use",
            'import { ReleaseSupportMatrix } from "@/x";',
            True,
        ),
        ("open tag counts as use", "<ReleaseSupportMatrix>", False),
        # rc and post builds must not satisfy a release row
        (
            "rc builds do not count",
            "<ReleaseSupportMatrix />\n"
            + twin.format("1.3.0rc19 1.3.1rc1 1.2.0rc4 1.2.1rc2 1.1.0rc9"),
            False,
        ),
        (
            "post builds do not count",
            "<ReleaseSupportMatrix />\n"
            + twin.format(
                "1.3.0.post1 1.3.1.post1 1.2.0.post1 1.2.1.post1 1.1.0.post1"
            ),
            False,
        ),
        # structural expectation: shape, not names
        (
            "structure present",
            "<FeatureInteractions />\n"
            + twin.format("Feature interactions *vLLM* *SGLang*"),
            True,
        ),
        (
            "structure missing a backend",
            "<FeatureInteractions />\n" + twin.format("Feature interactions *vLLM*"),
            False,
        ),
        (
            "structure heading gone",
            "<FeatureInteractions />\n" + twin.format("*vLLM* *SGLang*"),
            False,
        ),
    ]
    # A private temp dir, not a fixed /tmp path: two concurrent runs on a
    # shared runner would otherwise write the same file, and a redirected or
    # read-only TMPDIR fails the hook outright.
    passed = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "_agent_twin_case.mdx"
        for name, body, expect_ok in cases:
            tmp.write_text(body, encoding="utf-8")
            if (not check(tmp, exp)) == expect_ok:
                passed += 1
            else:
                print(f"  FAIL {name}: expected {'pass' if expect_ok else 'fail'}")
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
    data = DATA.read_text(encoding="utf-8")
    expected = expectations(data)
    if not any(expected.values()):
        print(
            "::error::derived no releases or features from releases.data.ts",
            file=sys.stderr,
        )
        return 1

    args = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    in_scope = [ROOT / rel for rel in sorted(IN_SCOPE)]
    if args:
        # Explicit paths (pre-commit passes none, but a human might) are still
        # filtered to the governed set, so running it on an arbitrary page
        # cannot produce a finding the gate would not also produce.
        wanted = {p.resolve() for p in args}
        targets = [p for p in in_scope if p.resolve() in wanted]
    else:
        targets = [p for p in in_scope if p.exists()]
    missing = [p for p in in_scope if not p.exists()]
    if missing:
        for p in missing:
            print(f"::error::in-scope page not found: {p}", file=sys.stderr)
        return 1

    problems: list[str] = []
    for target in targets:
        problems.extend(check(target, expected, data))

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
