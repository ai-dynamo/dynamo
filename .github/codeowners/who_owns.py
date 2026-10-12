#!/usr/bin/env python3
"""who_owns.py -- "who reviews this?" from a generated CODEOWNERS.

The CODEOWNERS file is a machine input: GitHub auto-requests the owning team
when a PR opens. This tool answers the human question on demand, so nobody
has to read 300 rules to find a reviewer.

  # owners of specific paths (last-match-wins, exactly as GitHub resolves)
  python who_owns.py --codeowners CODEOWNERS lib/llm/foo.rs components/.../snapshot.py

  # the teams that will be auto-requested on your PR (union over changed files)
  python who_owns.py --codeowners CODEOWNERS --changed --base main

Owners listed on a single line are co-owners (any one's approval satisfies
the gate).

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
from codeowners_match import parse_codeowners, resolve_owners  # noqa: E402


def changed_files(repo: str, base: str) -> list[str]:
    """Files changed vs ``base`` (merge-base diff), falling back to plain diff.

    Returns ``[]`` only when a diff actually succeeded and was empty. If every
    fallback fails (not a git checkout, unknown base), the last git error is
    surfaced instead of masquerading as "no changed files".
    """
    last_err: subprocess.CalledProcessError | None = None
    any_ok = False
    for args in ([f"{base}...HEAD"], [base], []):
        try:
            out = subprocess.check_output(
                ["git", "-C", repo, "diff", "--name-only", *args],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as err:
            last_err = err
            continue
        any_ok = True
        files = [p for p in out.splitlines() if p.strip()]
        if files:
            return files
    if not any_ok and last_err is not None:
        raise SystemExit(
            f"git diff failed in {repo!r} (not a checkout, or base "
            f"{base!r} unavailable): {last_err}"
        )
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
    for f in files:
        owners = resolve_owners(rules, f)
        union_owners.update(owners)
        owners_str = (
            " ".join(owners)
            if owners
            else "(no owner -- falls through; CI coverage gate should block this)"
        )
        print(f"{f}\n    review: {owners_str}")

    if args.changed:
        print("\n" + "=" * 60)
        print(f"Teams auto-requested on this PR ({len(union_owners)}):")
        for t in sorted(union_owners):
            print(f"  {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
