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
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from codeowners_match import (  # noqa: E402
    changed_paths,
    compute_resolution,
    load_tree,
    match,
)


@dataclass
class CoverageGate:
    """Catch-all-only paths split into what blocks the gate vs what only warns."""

    blocking: list[str]
    warnings: list[str]


def split_coverage(unmatched: list[str], changed: list[str] | None) -> CoverageGate:
    """Partition catch-all-only paths into blocking vs. non-blocking.

    Full-tree mode (``changed is None``): every catch-all-only path blocks --
    the whole-tree 100%-coverage assertion a scheduled/maintenance run wants.

    Diff-aware mode (``changed`` given): only catch-all-only paths this change
    added/renamed/modified block; catch-all-only paths inherited unchanged
    from the base branch are reported as warnings, so unrelated churn on the
    base never red-Xes a PR that did not touch it. The PR's OWN surface still
    must be 100% owned -- that is exactly the ``blocking`` set.
    """
    if changed is None:
        return CoverageGate(blocking=list(unmatched), warnings=[])
    changed_set = set(changed)
    blocking = [p for p in unmatched if p in changed_set]
    warnings = [p for p in unmatched if p not in changed_set]
    return CoverageGate(blocking=blocking, warnings=warnings)


def is_policy_change(changed: list[str], areas: str, repo: str) -> bool:
    """True if the PR touches ownership policy -> judge coverage whole-tree.

    A change to the policy directory (``areas.yaml``, the emit/gate scripts,
    ``external_contributors.yaml``) or to any ``CODEOWNERS`` file can re-route
    ANY path, so restricting coverage to the PR's own file surface would let a
    policy edit orphan untouched paths. When the diff includes such a file the
    gate falls back to full-tree strict.
    """
    repo_root = Path(repo).resolve()
    try:
        areas_rel = Path(areas).resolve().relative_to(repo_root).as_posix()
    except ValueError:
        areas_rel = None
    policy_dir = Path(areas_rel).parent.as_posix() if areas_rel else None
    for p in changed:
        if Path(p).name == "CODEOWNERS":
            return True
        if areas_rel is not None and p == areas_rel:
            return True
        if policy_dir not in (None, ".") and (
            p == policy_dir or p.startswith(policy_dir + "/")
        ):
            return True
    return False


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
    ap.add_argument(
        "--changed-only",
        action="store_true",
        help="diff-aware strict: gate only paths this branch adds/changes vs "
        "--base; report inherited-base gaps as a non-fatal warning. Pass on "
        "pull_request events so unrelated base churn never fails the check. A "
        "PR that edits ownership policy is still judged full-tree.",
    )
    ap.add_argument(
        "--base",
        default="main",
        help="base ref for --changed-only (default: main)",
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

    # Diff-aware mode judges only the PR's own surface; full-tree mode (the
    # default) judges every tracked file. A PR that edits ownership policy
    # (areas/scripts/CODEOWNERS) can re-route any path, so it is always judged
    # whole-tree -- otherwise a policy edit could orphan untouched paths.
    changed = None
    if args.changed_only:
        changed = changed_paths(Path(args.repo), args.base)
        if is_policy_change(changed, args.areas, args.repo):
            print(
                "note: PR touches ownership policy (areas/scripts/CODEOWNERS); "
                "evaluating full-tree coverage instead of the changed surface"
            )
            changed = None
    gate = split_coverage(unmatched, changed)
    if gate.warnings:
        print(
            f"warning: {len(gate.warnings)} catch-all-only path(s) inherited from "
            f"{args.base} (not touched by this change; not blocking):"
        )
        print("   ", gate.warnings[:15])

    if args.strict and gate.blocking:
        scope = "changed" if changed is not None else "tree"
        print(
            f"!! strict: {len(gate.blocking)} {scope} file(s) fall to the catch-all "
            "-- cover them in areas.yaml"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
