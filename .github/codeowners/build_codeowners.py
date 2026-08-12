"""Validate CODEOWNERS coverage against a live tree (repo-agnostic).

Reads an ``areas.yaml`` (each area declares its path globs directly), asks the
pure resolver in ``codeowners_match`` what the emitted CODEOWNERS would cover,
and reports how much of the live tree is EXPLICITLY owned vs. falls to the
catch-all. It also rejects stale globs (all of them in full-tree runs; in
diff-aware runs, the ones this branch itself orphaned) and verifies that final
last-match resolution retains every owner promised by required and blocking
file-type declarations.

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
    ResolvedModel,
    anchor,
    changed_paths,
    compute_resolution,
    load_tree,
    match,
    merge_base_tree,
    parse_codeowners,
    resolve_owners,
)
from emit_codeowners import render_codeowners  # noqa: E402


@dataclass
class CoverageGate:
    """Catch-all-only paths split into what blocks the gate vs what only warns."""

    blocking: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class OwnershipContractViolation:
    """A declared owner that final CODEOWNERS precedence removed."""

    glob: str
    path: str
    missing: tuple[str, ...]
    actual: tuple[str, ...]


OwnershipContract = tuple[str, str, list[str]]


def _ownership_contracts(model: ResolvedModel) -> list[OwnershipContract]:
    """Flatten required_owners and blocking file-type contracts for validation.

    ``shared`` entries are additive co-ownership lines -- a more-specific area
    rule may legitimately override them.  Only ``required_owners`` (explicit
    joint-ownership guarantees) and ``filetype_shared`` (blocking file-type
    co-ownership) are enforced as hard contracts here.
    """
    contracts = [
        (anchor(rule["glob"]), rule["glob"], rule["owners"])
        for rule in model.required_owners
    ]
    contracts.extend(
        (rule.glob, rule.glob, rule.owners) for rule in model.filetype_shared
    )
    return contracts


def _contract_violation(
    contract: OwnershipContract,
    path: str,
    rules: list[tuple[str, list[str]]],
    label_to_team: dict[str, str],
) -> OwnershipContractViolation | None:
    """Return the owner loss for one contract/path pair, if any."""
    pattern, declared_glob, declared_owners = contract
    if not match(pattern, path):
        return None
    actual = set(resolve_owners(rules, path))
    required = {label_to_team.get(owner, owner) for owner in declared_owners}
    missing = required - actual
    if not missing:
        return None
    return OwnershipContractViolation(
        glob=declared_glob,
        path=path,
        missing=tuple(sorted(missing)),
        actual=tuple(sorted(actual)),
    )


def ownership_contract_violations(
    model: ResolvedModel, tree: list[str]
) -> list[OwnershipContractViolation]:
    """Find declared owners removed by final last-match routing."""
    lines, _ = render_codeowners(model, group=True, external=[])
    rules = parse_codeowners("\n".join(lines))
    label_to_team = model.label_to_team()
    violations: list[OwnershipContractViolation] = []
    for contract in _ownership_contracts(model):
        for path in tree:
            violation = _contract_violation(contract, path, rules, label_to_team)
            if violation:
                violations.append(violation)
    return violations


@dataclass(frozen=True)
class SharedAdditivityViolation:
    """A ``shared`` row that drops an owner the row it overrides granted."""

    glob: str
    path: str
    missing: tuple[str, ...]
    declared: tuple[str, ...]


def shared_additivity_violations(
    model: ResolvedModel, tree: list[str]
) -> list[SharedAdditivityViolation]:
    """Find ``shared`` rows that drop an owner their enclosing row grants.

    A CODEOWNERS row replaces earlier rows outright, so a ``shared`` entry must
    restate every owner it means to keep. ``inherits`` was removed on the
    promise that this is machine-checked rather than trusted; this is that
    check.

    "Enclosing" is resolved, not tiered. For a shared row at index ``i``, the
    owners it replaces are ``resolve_owners(rules[:i], path)`` -- the final
    last-match answer over exactly the rows this one sits after. That sidesteps
    the question of which ancestor counts as the parent: a shared row under
    ``/lib/llm/`` (runtime) whose path is already overridden by
    ``/lib/llm/src/protocols/`` (frontend) is measured against frontend, the
    owner actually in force, not against runtime several levels up.

    Rows a later rule overrides for a given path are skipped: the shared row is
    not in force there, so any loss belongs to whatever outranks it.

    The catch-all is never an enclosing rule. It is the fallback every explicit
    row exists to replace, so counting it would demand that each shared entry
    restate the catch-all team -- 22 paths under ``deploy/power-agent/`` alone.
    """
    lines, _ = render_codeowners(model, group=True, external=[])
    # ``model.catch_all`` is a team, not a pattern; the row it emits is "*".
    rules = [rule for rule in parse_codeowners("\n".join(lines)) if rule[0] != "*"]
    label_to_team = model.label_to_team()
    violations: list[SharedAdditivityViolation] = []
    for spec in model.shared:
        pattern = anchor(spec["glob"])
        # Shared rows are emitted last, so the final row carrying this pattern
        # is the shared one even when an area declares the same glob.
        indexes = [i for i, (pat, _) in enumerate(rules) if pat == pattern]
        if not indexes:
            continue
        index = indexes[-1]
        declared = {label_to_team.get(o, o) for o in spec["owners"]}
        enclosing_rules = rules[:index]
        for path in tree:
            if not match(pattern, path):
                continue
            # Only judge paths where this row actually decides the owners.
            if set(resolve_owners(rules, path)) != declared:
                continue
            missing = set(resolve_owners(enclosing_rules, path)) - declared
            if not missing:
                continue
            violations.append(
                SharedAdditivityViolation(
                    glob=spec["glob"],
                    path=path,
                    missing=tuple(sorted(missing)),
                    declared=tuple(sorted(declared)),
                )
            )
    return violations


def print_shared_additivity_violations(
    violations: list[SharedAdditivityViolation],
) -> None:
    """Print a bounded shared-additivity report."""
    if not violations:
        return
    print(
        f"shared additivity violations: {len(violations)} "
        "(a shared rule dropped an owner the rule it overrides granted):"
    )
    for violation in violations[:15]:
        print(
            f"    {violation.path} (shared {violation.glob}): "
            f"drops {list(violation.missing)}; declared {list(violation.declared)}"
        )


def print_ownership_violations(
    violations: list[OwnershipContractViolation],
) -> None:
    """Print a bounded ownership-loss report."""
    if not violations:
        return
    print(
        f"ownership contract violations: {len(violations)} "
        "(a later rule removed declared co-owners):"
    )
    for violation in violations[:15]:
        print(
            f"    {violation.path} (declared by {violation.glob}): "
            f"missing {list(violation.missing)}; actual {list(violation.actual)}"
        )


def newly_stale_patterns(
    dead: list[str], base_paths: list[str] | None
) -> list[str] | None:
    """Split THIS change's stale globs from staleness inherited off the base.

    A dead pattern (matches nothing at HEAD) that still matched something at
    the merge-base went stale because this branch deleted or renamed its last
    matching file -- the branch must prune the declaration too, so ``main``
    never inherits a stale glob from a merged deletion PR. Note the pruning
    edit touches areas.yaml, so ``is_policy_change`` then reclassifies the PR
    and judges it FULL-TREE -- deliberate: a policy edit can re-route any
    path, and every routing change pays that same tax (it also means the PR
    needs a green base tree, which the push-to-main full-tree run protects).
    A pattern that was already dead at the merge-base is base staleness this
    PR did not cause; blocking on it would red-X every open PR the moment
    the base goes stale (the cascade diff-aware mode exists to prevent), so
    it stays a warning until a policy PR or full-tree run prunes it.

    Returns ``None`` in full-tree mode (``base_paths is None``), where every
    stale glob blocks and the split is meaningless.
    """
    if base_paths is None:
        return None
    return [p for p in dead if any(match(p, path) for path in base_paths)]


def strict_failure(
    strict: bool,
    gate: CoverageGate,
    changed: list[str] | None,
    ownership_violations: list[OwnershipContractViolation],
    dead: list[str],
    newly_stale: list[str] | None,
    additivity_violations: list[SharedAdditivityViolation] | None = None,
) -> str | None:
    """Return the fail-closed message for the active strict gate.

    Stale-glob blocking is scope-aware. Full-tree runs (``changed is None``:
    policy PRs, push-to-main, scheduled) block on EVERY stale glob -- the
    maintenance assertion. Diff-aware runs block only on ``newly_stale``
    globs, the ones this branch itself orphaned (see
    ``newly_stale_patterns``); staleness inherited from the base branch
    surfaces as a non-fatal report line so base churn cannot red-X
    unrelated PRs.
    """
    if not strict:
        return None
    if gate.blocking:
        scope = "changed" if changed is not None else "tree"
        return (
            f"!! strict: {len(gate.blocking)} {scope} file(s) fall to the "
            "catch-all -- cover them in areas.yaml"
        )
    if dead and changed is None:
        return (
            f"!! strict: {len(dead)} glob(s) match no tracked files -- "
            "remove them from areas.yaml"
        )
    if newly_stale:
        return (
            f"!! strict: {len(newly_stale)} ownership glob(s) match no "
            "tracked files after this change's deletions -- prune them from "
            "areas.yaml and regenerate CODEOWNERS in this PR"
        )
    if ownership_violations:
        return (
            f"!! strict: {len(ownership_violations)} path(s) lost declared "
            "owners after final CODEOWNERS precedence"
        )
    if additivity_violations:
        return (
            f"!! strict: {len(additivity_violations)} path(s) where a shared "
            "rule drops an owner the rule it overrides granted -- restate "
            "every retained owner under that entry's 'owners' in areas.yaml"
        )
    return None


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


def validation_scope(
    repo: Path, base: str, areas: str, changed_only: bool, tree: list[str]
) -> tuple[list[str] | None, list[str]]:
    """Return the coverage selector and matching ownership-contract tree."""
    if not changed_only:
        return None, tree
    changed = changed_paths(repo, base)
    if is_policy_change(changed, areas, str(repo)):
        print(
            "note: PR touches ownership policy (areas/scripts/CODEOWNERS); "
            "evaluating full-tree coverage instead of the changed surface"
        )
        return None, tree
    changed_set = set(changed)
    return changed, [path for path in tree if path in changed_set]


def _dead_patterns(model: ResolvedModel, tree: list[str]) -> list[str]:
    """Return unique blocking ownership patterns that match no tracked files."""
    patterns = [
        *model.owned_patterns(),
        *(anchor(rule["glob"]) for rule in model.required_owners),
        *(anchor(rule["glob"]) for rule in model.advisory),
        *(rule["pattern"] for rule in model.filetype_advisory),
    ]
    return [
        pattern
        for pattern in dict.fromkeys(patterns)
        if not any(match(pattern, path) for path in tree)
    ]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--areas", required=True, help="path to areas.yaml (source of truth)"
    )
    ap.add_argument("--repo", required=True, help="path to the target git repo")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on coverage, stale glob, or ownership contract failures",
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
    return ap.parse_args()


def _print_dead_patterns(dead: list[str], newly_stale: list[str] | None) -> None:
    """Print a bounded stale-pattern report, split by who must prune it.

    ``newly_stale is None`` means full-tree mode (every stale glob blocks).
    In diff-aware mode the globs this change orphaned block THIS PR; the
    rest are inherited base staleness and only warn here.
    """
    if not dead:
        return
    if newly_stale is None:
        print(
            f"globs matching no files: {len(dead)} "
            "(prune from areas.yaml; blocks policy PRs and full-tree runs):"
        )
        for pattern in dead[:10]:
            print(f"    {pattern}")
        return
    inherited = [p for p in dead if p not in set(newly_stale)]
    if newly_stale:
        print(
            f"globs matching no files: {len(newly_stale)} orphaned by this "
            "change (prune from areas.yaml in this PR; blocking):"
        )
        for pattern in newly_stale[:10]:
            print(f"    {pattern}")
    if inherited:
        print(
            f"globs matching no files: {len(inherited)} inherited from the "
            "base branch (not blocking here; a policy PR or full-tree run "
            "must prune them):"
        )
        for pattern in inherited[:10]:
            print(f"    {pattern}")


def _print_summary(
    model: ResolvedModel,
    tree: list[str],
    unmatched: list[str],
    dead: list[str],
    newly_stale: list[str] | None,
    violations: list[OwnershipContractViolation],
) -> None:
    """Print coverage, stale-pattern, contract, and per-area summaries."""
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
    _print_dead_patterns(dead, newly_stale)
    print_ownership_violations(violations)
    print("\nper-area glob counts:")
    counts = Counter({a.label: len(a.path_globs) for a in model.areas})
    for label, count in counts.most_common():
        print(f"  {label:<22} {count}")


def _print_warnings(gate: CoverageGate, base: str) -> None:
    """Print inherited catch-all gaps that do not block diff-aware mode."""
    if not gate.warnings:
        return
    print(
        f"warning: {len(gate.warnings)} catch-all-only path(s) inherited from "
        f"{base} (not touched by this change; not blocking):"
    )
    print("   ", gate.warnings[:15])


def main() -> int:
    args = _parse_args()
    spec = yaml.safe_load(Path(args.areas).read_text())
    model = compute_resolution(spec)
    tree = load_tree(Path(args.repo))
    unmatched = model.unmatched_paths(tree)
    changed, contract_tree = validation_scope(
        Path(args.repo), args.base, args.areas, args.changed_only, tree
    )
    violations = ownership_contract_violations(model, contract_tree)
    # Scoped like the ownership contracts: a diff-aware run judges only the
    # paths this change touches, so base-branch policy cannot red-X a PR.
    additivity = shared_additivity_violations(model, contract_tree)
    dead = _dead_patterns(model, tree)
    if changed is None:
        newly_stale = None
    else:
        # One git call, and only when something is stale to attribute.
        base_paths = merge_base_tree(Path(args.repo), args.base) if dead else []
        newly_stale = newly_stale_patterns(dead, base_paths)
    _print_summary(model, tree, unmatched, dead, newly_stale, violations)
    print_shared_additivity_violations(additivity)
    gate = split_coverage(unmatched, changed)
    _print_warnings(gate, args.base)
    failure = strict_failure(
        args.strict, gate, changed, violations, dead, newly_stale, additivity
    )
    if failure:
        print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
