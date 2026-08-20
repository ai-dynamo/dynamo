#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""suggest_reviewers.py -- the reviewer answer, rendered for a pull request.

``who_owns.py`` already answers "who reviews this?" for anyone who knows it
exists and can run it locally. Most contributors do neither, so the answer
arrives only when GitHub silently requests a team and they wonder who that was.
This renders the same answer as a comment on the pull request itself.

It is deliberately the same resolution. Owners come from ``codeowners_match``,
the module ``emit_codeowners.py`` and ``who_owns.py`` both use, so a comment
cannot disagree with what GitHub actually does. What is added here is the part a
terminal did not need: grouping, so a forty-file pull request is readable;
keeping the co-owner distinction, which a flat union destroys; and a byte
ceiling, because GitHub rejects a comment body over 65536 characters.

Resolution needs no authentication and no dependencies outside the standard
library. CODEOWNERS ships in this repository, so this answer is public and the
comment can go to anyone who opens a pull request -- including an external
contributor with no org membership. Naming *which people are on* an owning team,
or how loaded they are, is a different question with a different answer, and
this does not ask it.

  # from a JSON array of changed paths, as the workflow provides it
  gh api --paginate "repos/$REPO/pulls/$PR/files" --jq '.[].filename' \
    | jq -Rs 'split("\n") | map(select(length > 0))' \
    | python suggest_reviewers.py --codeowners CODEOWNERS --files -
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from codeowners_match import parse_codeowners, resolve_owners  # noqa: E402

#: Hidden marker identifying the bot's own comment, so a second run edits the
#: first one instead of appending. Without it every push adds a comment and
#: notifies every subscriber again.
COMMENT_MARKER = "<!-- dynamo-ops:suggest-reviewers -->"

#: GitHub's hard limit on a comment body. Exceeding it fails the API call, which
#: would turn a large pull request into no answer at all.
MAX_COMMENT_BYTES = 65536

#: Files listed per owner group before the rest are counted rather than named.
#: Evidence for the answer, not the answer, so this is the first thing cut.
MAX_FILES_SHOWN = 10

#: Teams named in the summary before the rest are counted. The summary is the
#: answer, so this is cut last -- but it is cut, because otherwise it is not
#: bounded at all. `areas.yaml` declares two dozen teams today, so this is well
#: past any real pull request; the point is that the byte ceiling holds for
#: every input rather than only for realistic ones.
MAX_TEAMS_SHOWN = 100


@dataclass(frozen=True)
class OwnerGroup:
    """Paths sharing one owner line.

    ``owners`` is a disjunction: CODEOWNERS requests every entry on the matching
    line, and any one of their approvals satisfies the gate. Collapsing groups
    into a single list of teams loses that, and the result reads as though every
    team must approve.
    """

    owners: tuple[str, ...]
    files: tuple[str, ...]


@dataclass(frozen=True)
class Suggestion:
    """The full answer for one pull request."""

    #: Owned paths, grouped by owner line. Ordered by group size descending then
    #: by owners, so the largest surface in the pull request reads first and the
    #: order does not depend on how the file list arrived.
    groups: tuple[OwnerGroup, ...]

    #: Every team or login GitHub will request. Accurate as a union -- all of
    #: them are requested -- but see :class:`OwnerGroup` for why the groups are
    #: what tell a contributor how many approvals they actually need.
    requested_teams: tuple[str, ...]

    #: Paths no rule matched. Reported rather than dropped: "no reviewer" and
    #: "no rule covers this" look identical in a union and mean different
    #: things, the second being a gap the coverage gate should have caught.
    unowned: tuple[str, ...]


def suggest(files: list[str], rules: list[tuple[str, list[str]]]) -> Suggestion:
    """Resolve ``files`` into a grouped reviewer answer."""
    by_owners: dict[tuple[str, ...], list[str]] = {}
    unowned: list[str] = []
    blocking: set[str] = set()

    for path in sorted(set(files)):
        owners = tuple(resolve_owners(rules, path))
        if owners:
            by_owners.setdefault(owners, []).append(path)
            blocking.update(owners)
        else:
            unowned.append(path)

    groups = tuple(
        OwnerGroup(owners=owners, files=tuple(paths))
        for owners, paths in sorted(
            by_owners.items(), key=lambda kv: (-len(kv[1]), kv[0])
        )
    )
    return Suggestion(
        groups=groups,
        requested_teams=tuple(sorted(blocking)),
        unowned=tuple(unowned),
    )


def _render_group(group: OwnerGroup) -> list[str]:
    """Render one owner group as markdown lines."""
    owners = ", ".join(f"`{o}`" for o in group.owners)
    suffix = " -- any one of them satisfies the gate" if len(group.owners) > 1 else ""
    lines = [f"**{owners}**{suffix}"]
    shown = group.files[:MAX_FILES_SHOWN]
    lines.extend(f"- `{path}`" for path in shown)
    hidden = len(group.files) - len(shown)
    if hidden:
        lines.append(f"- ...and {hidden} more file{'s' if hidden != 1 else ''}")
    return lines


def render_comment(suggestion: Suggestion) -> str:
    """Render ``suggestion`` as one pull request comment body.

    Bounded by :data:`MAX_COMMENT_BYTES`, unconditionally. Sections are cut in
    reverse order of value: the per-path evidence first, then the tail of the
    team list, because the team list is the answer and the paths only show why.

    Every section has its own cap, which is what makes the ceiling hold. An
    earlier version capped only the file lists and left the team list unbounded,
    on the reasoning that no real pull request touches many owner groups. That
    is true and it was still wrong: 4000 groups rendered a 100KB body, and
    GitHub rejects the request outright rather than truncating it, so the
    unrealistic input produced no answer at all instead of a clipped one.
    """
    header = [
        COMMENT_MARKER,
        "### Suggested reviewers",
        "",
    ]

    summary: list[str] = []
    if suggestion.requested_teams:
        shown_teams = suggestion.requested_teams[:MAX_TEAMS_SHOWN]
        summary.append("GitHub will request review from:")
        summary.append("")
        summary.extend(f"- `{team}`" for team in shown_teams)
        hidden_teams = len(suggestion.requested_teams) - len(shown_teams)
        if hidden_teams:
            summary.append(f"- ...and {hidden_teams} more")
    else:
        summary.append("No CODEOWNERS rule matches the changed files.")

    footer: list[str] = [""]
    if suggestion.unowned:
        shown = suggestion.unowned[:MAX_FILES_SHOWN]
        footer.append("These paths match no rule, which the coverage gate should flag:")
        footer.append("")
        footer.extend(f"- `{path}`" for path in shown)
        hidden = len(suggestion.unowned) - len(shown)
        if hidden:
            footer.append(f"- ...and {hidden} more")
        footer.append("")
    footer.append(
        "Resolved from the repository's `CODEOWNERS` with "
        "`python .github/codeowners/who_owns.py --codeowners CODEOWNERS "
        "--changed --base main`, which you can run yourself."
    )

    fixed = "\n".join([*header, *summary, *footer])
    # The overflow note's room is reserved before any group is placed. Adding it
    # only if it happens to fit means the one case that needs it -- a body large
    # enough to truncate -- is the case with no room left, so the list comes out
    # clipped and looking complete.
    note_reserve = len(b"...and 000000 more owner groups\n")
    budget = (
        MAX_COMMENT_BYTES
        - len(fixed.encode())
        - len("\n\n#### Where\n\n")
        - note_reserve
    )

    detail: list[str] = []
    rendered_groups = 0
    for group in suggestion.groups:
        block = "\n".join(_render_group(group))
        if len(("\n".join([*detail, block])).encode()) > budget:
            remaining = len(suggestion.groups) - rendered_groups
            detail.append(
                f"...and {remaining} more owner group{'s' if remaining != 1 else ''}"
            )
            break
        detail.append(block)
        detail.append("")
        rendered_groups += 1

    if not detail:
        return fixed
    body = "\n".join([*header, *summary, "", "#### Where", "", *detail, *footer[1:]])
    return body if len(body.encode()) <= MAX_COMMENT_BYTES else fixed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render a reviewer suggestion comment for a pull request."
    )
    ap.add_argument(
        "--codeowners", required=True, type=Path, help="path to the CODEOWNERS file"
    )
    ap.add_argument(
        "--files",
        required=True,
        help="JSON array of changed paths, or - to read it from stdin",
    )
    args = ap.parse_args()

    raw = sys.stdin.read() if args.files == "-" else Path(args.files).read_text()
    files = json.loads(raw)
    if not isinstance(files, list):
        raise SystemExit("--files must be a JSON array of paths")

    rules = parse_codeowners(args.codeowners.read_text())
    print(render_comment(suggest(files, rules)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
