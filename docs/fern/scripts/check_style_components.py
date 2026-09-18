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
"""Guard the CSS template literals in the *Styles.tsx components.

Each style component holds its CSS in a template literal:

    const LANDING_CSS = `
    ... ~50KB of CSS, comments included ...
    `;

A single raw backtick or `${` anywhere in that body closes or interpolates the
literal and breaks the build. It is easy to introduce by hand -- a CSS comment
quoting a class name as `node--k8s` is enough -- and nothing else catches it:
fern check does not parse TSX, and the CSS-level checks only read main.css.
sync_site_css.py already applies this rule to main.css before mirroring it;
this applies the same rule to the components that hold CSS directly.

Usage: python3 check_style_components.py [files...]
With no arguments, checks every docs/fern/components/*Styles.tsx.

Coverage stops at *Styles.tsx rather than every components/*.tsx because
roughly twenty components hold a CSS literal and three of them
(ModelEABuildCards, TagLookup, TerminalDemo) interpolate into it deliberately,
which this check reads as a defect. Widening it properly needs a way to mark
those as intentional. Until then, a component with its own CSS should keep it
in a *Styles.tsx file, as PublicationsStyles.tsx does.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # docs/fern
REPO = ROOT.parents[1]  # repo root, for readable paths
LITERAL = re.compile(r"const\s+(\w+)\s*=\s*`(.*?)`;", re.DOTALL)


def check(path: Path) -> list[str]:
    text = path.read_text()
    problems: list[str] = []
    matches = list(LITERAL.finditer(text))
    if not matches:
        return problems

    # Guarded like check_agent_twins does: pre-commit passes repo-relative
    # paths, so an unguarded relative_to raises ValueError before the message
    # prints -- on the one run that matters, the author gets a pathlib
    # traceback instead of the diagnostic this check exists to produce.
    rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
    for match in matches:
        name, body = match.group(1), match.group(2)
        start_line = text[: match.start(2)].count("\n") + 1
        for offence, label in (("`", "raw backtick"), ("${", "interpolation")):
            index = body.find(offence)
            if index == -1:
                continue
            line = start_line + body[:index].count("\n")
            snippet = body.splitlines()[body[:index].count("\n")].strip()[:72]
            problems.append(
                f"{rel}:{line}: {label} inside {name}\n"
                f"      {snippet}\n"
                f"      A {label} ends the template literal and breaks the build."
            )
    return problems


def main() -> int:
    args = [Path(a) for a in sys.argv[1:]]
    # The *Styles.tsx convention plus the two components that hold CSS under
    # another name. Not every component: this check forbids interpolation,
    # which holds for a pure CSS literal and not for ordinary TSX.
    targets = args or sorted(
        p
        for p in (ROOT / "components").glob("*.tsx")
        if p.stem.endswith("Styles")
        or p.stem in {"ReleaseSupportMatrix", "FeatureInteractions"}
    )
    targets = [t for t in targets if t.suffix == ".tsx" and t.exists()]

    problems: list[str] = []
    for target in targets:
        problems.extend(check(target))

    if problems:
        print("template literal defects found:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"checked {len(targets)} style component(s): literals intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
