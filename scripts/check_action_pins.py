#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that every external GitHub Action is pinned to a commit SHA.

A `uses:` reference to a mutable tag (`@v4`) or a branch resolves at run time
to whatever commit that ref points at. An upstream account compromise or a
re-pointed tag then executes attacker-controlled code inside CI, with access to
repository secrets (CWE-829). Pinning to a full 40-character commit SHA makes
the reference immutable.

The SHA alone is unreadable, so this also requires the trailing version comment
that makes a pin auditable and upgradable by a human:

    uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2

Exactly two spaces before `#`, one space after, and a full `vMAJOR.MINOR.PATCH`
version. A bare major (`# v6`) is rejected: it cannot be checked against the
SHA, which is the only reason to write the comment at all.

Not checked, because neither is an external tag reference:
  - local composite actions   (`uses: ./.github/actions/foo`)
  - same-repo reusable workflows and any `${{ }}` expression ref

Usage:
  check_action_pins.py [repo_root]   scan and report violations
  check_action_pins.py --test        run the self-test cases
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The single definition of what gets scanned. The pre-commit hook triggers on a
# coarser path pattern on purpose; duplicating this list there would be a second
# copy to keep in sync.
DEFAULT_GLOBS = (
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    ".github/actions/**/action.yml",
    ".github/actions/**/action.yaml",
)

# Upstreams that publish major-only tags (v8, v24) and no vMAJOR.MINOR.PATCH at
# all, so the full-semver comment is impossible rather than merely absent. The
# pin is still a real SHA; only the comment is relaxed. Keep this list short and
# justified - each entry is a check that is not being performed.
MAJOR_ONLY_UPSTREAMS = {
    # Tags v4 and later are major-only (v4 ... v24); patch-level tags exist
    # only up to v3.1.4, so no v8.Y.Z exists for the pinned major.
    "dawidd6/action-download-artifact",
}

USES = re.compile(
    r"^(?P<indent>[ \t]*(?:-[ \t]+)?)uses:(?P<sep>[ \t]*)(?P<ref>\S+)(?P<rest>.*)$"
)
ACTION = re.compile(
    r"^(?P<action>[A-Za-z0-9][\w.-]*/[\w.-]+(?:/[\w./-]+)?)@(?P<ref>.+)$"
)
SHA = re.compile(r"^[0-9a-f]{40}$")
FULL_SEMVER = re.compile(r"^ {2}# v\d+\.\d+\.\d+$")
MAJOR_ONLY = re.compile(r"^ {2}# v\d+$")


def check_line(line: str) -> str | None:
    """Return an error message for one line, or None if it is fine.

    Anything that is not an external action reference returns None.
    """
    m = USES.match(line.rstrip("\n"))
    if not m:
        return None
    ref, rest = m.group("ref"), m.group("rest")

    # A quoted YAML scalar keeps its quotes inside \S+, which would fall through the
    # ACTION match below and read as "not an external reference" - letting a quoted
    # mutable tag pass the check this script exists to enforce.
    if len(ref) > 1 and ref[0] in "\"'" and ref[-1] == ref[0]:
        ref = ref[1:-1]

    # Local composite actions and expression refs are not external tags.
    if ref.startswith("./") or ref.startswith("$") or "${{" in ref or "${{" in rest:
        return None

    am = ACTION.match(ref)
    if not am:
        return None
    action, at = am.group("action"), am.group("ref")

    if not SHA.match(at):
        hint = at if at.startswith("v") else "vX.Y.Z"
        return (
            f"{action} is pinned to the mutable ref '{at}'. "
            f"Pin it to the full 40-character commit SHA of the tag and keep the "
            f"version in a comment: uses: {action}@<sha>  # {hint}"
        )

    owner_repo = "/".join(action.split("/")[:2])
    if not rest.strip():
        return f"{action} is pinned but has no version comment. Append two spaces then '# vX.Y.Z'."

    if owner_repo in MAJOR_ONLY_UPSTREAMS:
        if FULL_SEMVER.match(rest) or MAJOR_ONLY.match(rest):
            return None
        return (
            f"{action} comment must be exactly two spaces then '# vN' or '# vX.Y.Z' "
            f"(upstream publishes major-only tags); found '{rest.strip()}'."
        )

    if FULL_SEMVER.match(rest):
        return None
    if MAJOR_ONLY.match(rest):
        return (
            f"{action} has a bare-major comment '{rest.strip()}'. Use the full version the "
            f"SHA resolves to ('# vX.Y.Z') so the comment can be checked against the pin."
        )
    return f"{action} comment must be exactly two spaces then '# vX.Y.Z'; found '{rest.strip()}'."


def scan(root: Path) -> list[str]:
    problems: list[str] = []
    for pattern in DEFAULT_GLOBS:
        for path in sorted(root.glob(pattern)):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                err = check_line(line)
                if err:
                    problems.append(f"{path.relative_to(root)}:{lineno}: {err}")
    return problems


CASES: list[tuple[str, bool]] = [
    # (line, should_report_a_problem)
    (
        "      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2",
        False,
    ),
    (
        "    uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2",
        False,
    ),
    (
        "      uses: github/codeql-action/init@42947a340483f03ba47bb1a039b2c519aab3df85  # v3.37.8",
        False,
    ),
    # Not external references.
    ("      - uses: ./.github/actions/pytest", False),
    (
        "    uses: ${{ github.repository }}/.github/workflows/shared-test.yml@main",
        False,
    ),
    ("      uses: ./.github/actions/docker-build", False),
    # The repo's self-repository shorthand; see .github/actionlint.yaml.
    ("    uses: $/.github/workflows/shared-test.yml", False),
    ("      - uses: $/.github/actions/pytest", False),
    ("        run: echo 'uses: actions/checkout@v4 in a string'", False),
    # Quoted YAML scalars are still external references.
    ('      uses: "actions/checkout@v4"', True),
    ("      uses: 'actions/checkout@main'", True),
    (
        '      uses: "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"  # v6.0.2',
        False,
    ),
    # Mutable refs.
    ("      - uses: actions/checkout@v4", True),
    ("      uses: actions/setup-node@v4.4.0", True),
    ("      uses: actions/checkout@main", True),
    # Comment problems.
    ("      uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd", True),
    (
        "      uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2",
        True,
    ),
    (
        "      uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  #v6.0.2",
        True,
    ),
    (
        "      uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6",
        True,
    ),
    (
        "      uses: actions/checkout@DE0FAC2E4500DABE0009E67214FF5F5447CE83DD  # v6.0.2",
        True,
    ),
    # Major-only upstreams keep a bare major, but still need the spacing.
    (
        "      uses: dawidd6/action-download-artifact@20319c5641d495c8a52e688b7dc5fada6c3a9fbc  # v8",
        False,
    ),
    (
        "      uses: dawidd6/action-download-artifact@20319c5641d495c8a52e688b7dc5fada6c3a9fbc # v8",
        True,
    ),
    (
        "      uses: ytanikin/pr-conventional-commits@b628c5a234cc32513014b7bfdd1e47b532124d98  # v1.3.0",
        False,
    ),
]


def selftest() -> int:
    failures = 0
    for line, should_fail in CASES:
        got = check_line(line)
        if bool(got) != should_fail:
            failures += 1
            want = "a problem" if should_fail else "no problem"
            print(
                f"FAIL: expected {want} for:\n  {line}\n  got: {got}", file=sys.stderr
            )
    if failures:
        print(f"{failures} of {len(CASES)} self-test cases failed", file=sys.stderr)
        return 1
    print(f"check_action_pins self-test: {len(CASES)} cases passed")
    return 0


def main(argv: list[str]) -> int:
    if "--test" in argv:
        return selftest()
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent
    problems = scan(root)
    if not problems:
        return 0
    print(
        "GitHub Actions must be pinned to a commit SHA with a version comment:\n",
        file=sys.stderr,
    )
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    print(
        "\nResolve a tag to its commit SHA with:\n"
        "  gh api repos/<owner>/<repo>/tags --jq '.[] | select(.name==\"<tag>\") | .commit.sha'",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
