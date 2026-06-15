#!/usr/bin/env python3
"""who_owns.py -- "who reviews this?" from a generated CODEOWNERS (+ advisory).

The CODEOWNERS file is a machine input: GitHub auto-requests the owning team
when a PR opens. This tool answers the human question on demand, so nobody
has to read 300 rules to find a reviewer.

  # owners of specific paths (last-match-wins, exactly as GitHub resolves)
  python who_owns.py --codeowners CODEOWNERS lib/llm/foo.rs components/.../snapshot.py

  # the teams that will be auto-requested on your PR (union over changed files)
  python who_owns.py --codeowners CODEOWNERS --changed --base main

Owners listed on a single line are co-owners (any one's approval satisfies
the gate). Advisory teams are auto-requested too, but never block the merge.

The CODEOWNERS parser and matcher live in ``codeowners_match`` so this tool
resolves a path exactly the same way ``emit_codeowners.py`` routes it -- there
is no second implementation that could drift.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from codeowners_match import (  # noqa: E402
    anchor,
    match,
    parse_codeowners,
    resolve_owners,
)


def load_advisory(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (path_rules, filetype_rules) from an advisory-reviewers.yaml."""
    if not path.exists():
        return [], []
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    return data.get("path_rules", []) or [], data.get("filetype_rules", []) or []


def advisory_for(
    filepath: str, path_rules: list[dict], filetype_rules: list[dict]
) -> set[str]:
    """Non-blocking teams an advisory Action would request for ``filepath``."""
    teams: set[str] = set()
    for r in path_rules:
        pat = r.get("path", "")
        if pat and match(anchor(pat), filepath):
            teams.update(r.get("request_review_from", []))
    for r in filetype_rules:
        pat = r.get("pattern", "")
        if pat and match(pat, filepath):
            teams.update(r.get("request_review_from", []))
    return teams


def changed_files(repo: str, base: str) -> list[str]:
    """Files changed vs ``base`` (merge-base diff), falling back to plain diff."""
    for args in ([f"{base}...HEAD"], [base], []):
        try:
            out = subprocess.check_output(
                ["git", "-C", repo, "diff", "--name-only", *args],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            files = [p for p in out.splitlines() if p.strip()]
            if files:
                return files
        except subprocess.CalledProcessError:
            continue
    return []


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Who reviews a path, per a generated CODEOWNERS."
    )
    ap.add_argument(
        "--codeowners",
        required=True,
        type=Path,
        help="path to the CODEOWNERS file",
    )
    ap.add_argument(
        "--advisory",
        type=Path,
        default=None,
        help="advisory-reviewers.yaml (default: alongside CODEOWNERS)",
    )
    ap.add_argument(
        "--changed",
        action="store_true",
        help="resolve the repo's changed files instead of explicit paths",
    )
    ap.add_argument(
        "--base", default="main", help="base ref for --changed (default: main)"
    )
    ap.add_argument("--repo", default=".", help="repo root for --changed (default: .)")
    ap.add_argument(
        "paths", nargs="*", help="paths to resolve (when not using --changed)"
    )
    args = ap.parse_args()

    rules = parse_codeowners(args.codeowners.read_text())
    adv_path = args.advisory or args.codeowners.parent / "advisory-reviewers.yaml"
    path_rules, filetype_rules = load_advisory(adv_path)

    if args.changed:
        files = changed_files(args.repo, args.base)
        if not files:
            print(f"No changed files vs {args.base}.")
            return 0
    else:
        files = args.paths
        if not files:
            ap.error("pass one or more paths, or use --changed")

    union_owners: set[str] = set()
    union_advisory: set[str] = set()
    for f in files:
        owners = resolve_owners(rules, f)
        adv = advisory_for(f, path_rules, filetype_rules) - set(owners)
        union_owners.update(owners)
        union_advisory.update(adv)
        owners_str = (
            " ".join(owners)
            if owners
            else "(no owner -- falls through; CI coverage gate should block this)"
        )
        line = f"{f}\n    review: {owners_str}"
        if adv:
            line += f"\n    advisory (non-blocking): {' '.join(sorted(adv))}"
        print(line)

    if args.changed:
        union_advisory -= union_owners
        print("\n" + "=" * 60)
        print(f"Teams auto-requested on this PR ({len(union_owners)}):")
        for t in sorted(union_owners):
            print(f"  {t}")
        if union_advisory:
            print(f"Advisory (non-blocking), {len(union_advisory)}:")
            for t in sorted(union_advisory):
                print(f"  {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
