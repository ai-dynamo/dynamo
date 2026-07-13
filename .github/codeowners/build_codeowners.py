"""Validate CODEOWNERS coverage against a live tree (repo-agnostic).

Reads an ``areas.yaml`` (each area declares its path globs directly), asks the
pure resolver in ``codeowners_match`` what the emitted CODEOWNERS would cover,
and reports how much of the live tree is EXPLICITLY owned vs. falls to the
catch-all.

This is the ONLY place in the pipeline that reads ``git ls-files``. Emission
is a pure function of the policy YAML; the tree only enters here, in the
``--strict`` gate that asserts every tracked file matches some non-catch-all
rule. The gate and the emitted file share the same resolver, so a file the
gate accepts is a file the emitter has a rule for.

Usage:
  uv run python .github/codeowners/build_codeowners.py \\
      --areas .github/codeowners/areas.yaml --repo . [--strict]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from codeowners_match import compute_resolution, load_tree, match  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--areas", required=True, help="path to areas.yaml (source of truth)"
    )
    ap.add_argument("--repo", required=True, help="path to the target git repo")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any file falls to the catch-all (CI gate)",
    )
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.areas).read_text())
    # Resolution is a pure function of the YAML; the tree only feeds the
    # coverage/drift reports below, never the rule set.
    model = compute_resolution(spec)
    tree = load_tree(Path(args.repo))
    unmatched = model.unmatched_paths(tree)
    # Deletions never fail a gate (coverage counts files, and the drift check
    # forces the CODEOWNERS regeneration), so stale claims would otherwise
    # accumulate silently in areas.yaml. Surface them; never block on them.
    dead = [g for g in model.owned_patterns() if not any(match(g, p) for p in tree)]

    n_tree = len(tree)
    n_owned = n_tree - len(unmatched)
    pct = (100 * n_owned / n_tree) if n_tree else 100.0

    print(f"areas: {len(model.areas)} | tree files: {n_tree}")
    print(
        f"explicitly owned: {n_owned}/{n_tree} ({pct:.2f}%) | catch-all only: {len(unmatched)}"
    )
    if unmatched:
        print("catch-all-only sample (add an explicit glob to cover these):")
        print("   ", unmatched[:15])
    if dead:
        print(
            f"globs matching no files: {len(dead)} "
            "(prune from areas.yaml when the paths are gone; never blocking):"
        )
        for g in dead[:10]:
            print(f"    {g}")
    print("\nper-area glob counts:")
    counts = Counter({a.label: len(a.path_globs) for a in model.areas})
    for lbl, c in counts.most_common():
        print(f"  {lbl:<22} {c}")

    if args.strict and unmatched:
        print(
            f"!! strict: {len(unmatched)} file(s) fall to the catch-all -- cover them in areas.yaml"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
