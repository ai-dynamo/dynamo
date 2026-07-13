"""Validate CODEOWNERS coverage against a live tree (repo-agnostic).

Reads an ``areas.yaml`` (each area declares its path globs directly), runs the
declared path-globs -> auto-classify keyword layer over the tree via the
shared resolution pipeline, then reports how much of the tree is EXPLICITLY
owned vs. falls to the catch-all.

The on-disk handoff to ``emit_codeowners.py`` used to be a near-copy of
``areas.yaml`` written to ``/tmp/areas.resolved.yaml``; that's gone. Both
scripts call ``codeowners_match.compute_resolution`` directly, so the gate
here and the file produced by ``emit_codeowners.py`` cannot disagree.

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
from codeowners_match import compute_resolution, load_tree  # noqa: E402


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
    tree = load_tree(Path(args.repo))
    model = compute_resolution(spec, tree)
    unmatched = model.unmatched_paths(tree)

    n_tree = len(tree)
    n_owned = n_tree - len(unmatched)
    pct = (100 * n_owned / n_tree) if n_tree else 100.0

    print(f"areas: {len(model.areas)} | tree files: {n_tree}")
    print(
        f"explicitly owned: {n_owned}/{n_tree} ({pct:.2f}%) | "
        f"catch-all only: {len(unmatched)}"
    )
    print(f"auto-classified new dirs: {len(model.auto_classified)}")
    for d, lbl in model.auto_classified[:20]:
        print(f"    {d} -> {lbl}")
    print(f"keyword co-owned dirs: {len(model.keyword_coowned)}")
    for s in model.keyword_coowned[:20]:
        print(f"    {s['glob']} -> {' + '.join(s['owners'])}")
    if unmatched:
        print("catch-all-only sample (add an area or classify rule to cover these):")
        print("   ", unmatched[:15])
    print("\nper-area glob counts:")
    counts = Counter({a.label: len(a.path_globs) for a in model.areas})
    for lbl, c in counts.most_common():
        print(f"  {lbl:<22} {c}")

    if args.strict and unmatched:
        print(
            f"!! strict: {len(unmatched)} file(s) fall to the catch-all -- "
            "cover them in areas.yaml"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
