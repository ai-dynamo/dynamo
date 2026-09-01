#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render DECISIONS.md from the append-only decision log.

GOVERNANCE.md promises that every governance action is recorded publicly with
its reasoning, and that a roster pull request links the decision it records.
``decisions.jsonl`` is that record and this script renders it, so the published
table can never drift from the log the way a hand-maintained file would.

The log is JSONL rather than YAML because decisions are appended, often by
different people in the same week: two appended lines merge cleanly where two
edits to one YAML list conflict.

Usage:
    python3 .github/decisions/emit_decisions.py --log .github/decisions/decisions.jsonl --out DECISIONS.md
    python3 .github/decisions/emit_decisions.py --check      # CI: fail if DECISIONS.md is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REQUIRED = ("id", "date", "type", "subject", "threshold", "result", "rationale")
TYPES = {"ratification", "appointment", "removal", "amendment", "emeritus", "sig"}
RESULTS = {"carried", "failed", "withdrawn"}

# The full Apache block, matching every other governance file at the repository
# root. Duplicated rather than imported: it is boilerplate with no behavior, and
# a cross-package import for one string would couple this emitter to the
# codeowners package for nothing.
HEADER = """<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Dynamo Governance Decisions

This record contains every governance action the project has taken, together
with the decision that produced it and the reasoning recorded at the time.
Promotions, removals, amendments, and SIG lifecycle changes are recorded here,
as required by [GOVERNANCE.md](GOVERNANCE.md).

Individual ballots are private and are never published. Where a decision was
determined by a vote, the tally is recorded: the number eligible, the votes in
favor, against, and abstaining, and whether that met the threshold. Decisions
seated at ratification were not put to a vote, as the electorate the document
establishes did not exist beforehand, and their tally is recorded as `n/a`.

Generated from `.github/decisions/decisions.jsonl`. Do not hand-edit. Append to
that file and regenerate.
"""


def load(path: Path) -> list[dict]:
    entries = []
    seen: set[str] = set()
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        for field in REQUIRED:
            if field not in entry:
                raise SystemExit(f"{path}:{lineno}: missing required field {field!r}")
        if entry["id"] in seen:
            raise SystemExit(f"{path}:{lineno}: duplicate decision id {entry['id']!r}")
        seen.add(entry["id"])
        if entry["type"] not in TYPES:
            raise SystemExit(
                f"{path}:{lineno}: unknown type {entry['type']!r}, expected one of {sorted(TYPES)}"
            )
        try:
            date.fromisoformat(str(entry["date"]))
        except ValueError:
            raise SystemExit(
                f"{path}:{lineno}: date {entry['date']!r} is not ISO format (YYYY-MM-DD)"
            ) from None
        if entry["result"] not in RESULTS:
            raise SystemExit(
                f"{path}:{lineno}: unknown result {entry['result']!r}, expected one of {sorted(RESULTS)}"
            )
        entries.append(entry)
    return entries


def tally(entry: dict) -> str:
    """Aggregate counts for a decision, or "n/a" when no vote produced it."""
    if entry.get("for") is None:
        return "n/a"
    parts = [
        f"{entry['for']} for",
        f"{entry.get('against', 0)} against",
        f"{entry.get('abstain', 0)} abstaining",
    ]
    if entry.get("eligible") is not None:
        parts.append(f"of {entry['eligible']} eligible")
    return ", ".join(parts)


def render(entries: list[dict]) -> str:
    lines = [
        HEADER,
        "",
        "| Decision | Date | Type | Subject | Tally | Result |",
        "| :- | :- | :- | :- | :- | :- |",
    ]
    for e in sorted(entries, key=lambda x: (x["date"], x["id"]), reverse=True):
        subject = f"[{e['subject']}]({e['issue']})" if e.get("issue") else e["subject"]
        lines.append(
            f"| `{e['id']}` | {e['date']} | {e['type']} | {subject} | {tally(e)} | {e['result']} |"
        )
    lines += ["", "## Reasoning", ""]
    for e in sorted(entries, key=lambda x: (x["date"], x["id"]), reverse=True):
        lines.append(f"### `{e['id']}` {e['subject']}")
        lines.append("")
        lines.append(e["rationale"])
        lines.append("")
        detail = [f"Threshold: {e['threshold']}."]
        if e.get("issue"):
            detail.append(f"Vote: {e['issue']}.")
        if e.get("roster_sha"):
            detail.append(
                f"Eligibility taken from MAINTAINERS.md at `{e['roster_sha']}`."
            )
        lines.append(" ".join(detail))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=".github/decisions/decisions.jsonl", type=Path)
    ap.add_argument("--out", default="DECISIONS.md", type=Path)
    ap.add_argument("--check", action="store_true", help="fail if the output is stale")
    args = ap.parse_args()

    entries = load(args.log)
    rendered = render(entries)

    if args.check:
        current = args.out.read_text() if args.out.exists() else ""
        if current != rendered:
            print(
                f"{args.out} is stale. Regenerate with:\n"
                f"  python3 {Path(__file__).as_posix()} --log {args.log} --out {args.out}",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out} is current")
        return 0

    args.out.write_text(rendered)
    print(f"wrote {args.out} ({len(entries)} decisions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
