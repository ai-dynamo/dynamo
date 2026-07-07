"""Emit a GitHub CODEOWNERS file (+ advisory-reviewers.yaml) from areas.yaml.

GitHub CODEOWNERS is last-match-wins. The base tier is emitted as a *minimal
last-match cover* (broad parent globs + carve-out exceptions) computed against
the live tree, grouped per area for readability, with cross-area carve-outs
pulled into a trailing override section. Shared co-ownership and file-type
co-ownership are emitted LAST so they win over the base rules they refine.

Reads ``areas.yaml`` directly via ``codeowners_match.compute_resolution`` --
no more ``/tmp/areas.resolved.yaml`` round-trip with ``build_codeowners.py``.
The shared module is the single source of truth, so the coverage gate and the
file emitted here cannot disagree.

Nobody reads this file to find a reviewer -- GitHub auto-requests the owning
team, and ``who_owns.py`` answers "who reviews this path" on demand. The
grouping + legend just make the generated artifact navigable.

Usage:
  uv run python .github/codeowners/emit_codeowners.py \\
      --areas .github/codeowners/areas.yaml --repo . \\
      --out CODEOWNERS \\
      --advisory-out .github/codeowners/advisory-reviewers.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from codeowners_match import (  # noqa: E402
    ResolvedModel,
    anchor,
    compute_resolution,
    load_tree,
    minimal_cover,
    resolve_owners,
)


def _owners_str(label_to_team: dict[str, str], owners: list[str]) -> str:
    """Render a list of area labels (or raw teams) as a space-joined team list."""
    return " ".join(label_to_team.get(o, o) for o in owners)


def _base_rules(model: ResolvedModel, tree: list[str]) -> list[tuple[str, str]]:
    """Compute the minimal last-match cover for the base (single-owner) tier."""
    label_to_team = model.label_to_team()
    base_lookup = sorted(
        ((anchor(g), [a.github_team]) for a in model.areas for g in a.path_globs),
        key=lambda r: len(r[0]),
    )
    file_team: dict[str, str] = {}
    for f in tree:
        owners = resolve_owners(base_lookup, f)
        file_team[f] = owners[0] if owners else (model.catch_all or "")
    rules = minimal_cover(file_team, model.catch_all)

    # Drop accidental duplicates while preserving order.
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for p, t in rules:
        if (p, t) in seen:
            continue
        seen.add((p, t))
        deduped.append((p, t))
    # Suppress unused; label_to_team is consumed by callers above when needed.
    _ = label_to_team
    return deduped


def _render_codeowners(
    model: ResolvedModel,
    tree: list[str],
    group: bool,
) -> tuple[list[str], dict[str, int]]:
    """Build the CODEOWNERS file body. Returns (lines, stats)."""
    catch_all = model.catch_all
    label_to_team = model.label_to_team()
    team_to_label = {a.github_team: a.label for a in model.areas}
    area_order = [a.label for a in model.areas]

    base_rules = _base_rules(model, tree)

    shared_rules = sorted(
        (
            (anchor(s["glob"]), _owners_str(label_to_team, s["owners"]))
            for s in model.shared
        ),
        key=lambda r: (len(r[0]), r[0]),
    )
    ft_rules = sorted(
        (
            (anchor(fs.glob), _owners_str(label_to_team, fs.owners))
            for fs in model.filetype_shared
        ),
        key=lambda r: (len(r[0]), r[0]),
    )

    teams = sorted(
        {a.github_team for a in model.areas}
        | ({catch_all} if catch_all else set())
        | {label_to_team.get(o, o) for fs in model.filetype_shared for o in fs.owners}
    )

    # Group base rules per area; cross-area carve-outs -> override tail.
    dir_rules = [(p, t) for p, t in base_rules if p.endswith("/")]

    def is_override(path: str, team: str) -> bool:
        return any(
            tp != team and path != pp and path.startswith(pp) for pp, tp in dir_rules
        )

    groups: dict[str, list[tuple[str, str]]] = {}
    overrides: list[tuple[str, str]] = []
    for p, t in base_rules:
        if group and is_override(p, t):
            overrides.append((p, t))
        else:
            groups.setdefault(team_to_label.get(t, t), []).append((p, t))
    for lst in groups.values():
        lst.sort(key=lambda r: (len(r[0]), r[0]))
    overrides.sort(key=lambda r: (len(r[0]), r[0]))

    all_paths = [p for p, _ in base_rules + shared_rules + ft_rules] or ["*"]
    width = max(len(p) for p in all_paths) + 2

    def fmt(path: str, team: str) -> str:
        return f"{path:<{width}}{team}"

    lines = [
        "# CODEOWNERS -- generated from .github/codeowners/areas.yaml.",
        "# Do not hand-edit. Change areas.yaml and regenerate.",
        "#",
        "# GitHub reads this file; engineers don't. To see who reviews a change, run:",
        "#   python .github/codeowners/who_owns.py --codeowners CODEOWNERS --changed  # owners of your PR's files",
        "#   python .github/codeowners/who_owns.py --codeowners CODEOWNERS <path> ...  # owners of specific paths",
        "#",
        "# Area index (base owner per subsystem; the catch-all owns everything else):",
    ]
    idx_w = max((len(a.label) for a in model.areas), default=4) + 2
    for a in sorted(model.areas, key=lambda a: a.label):
        lines.append(f"#   {a.label:<{idx_w}}{a.github_team}")
    if catch_all:
        lines.append(f"#   {'*':<{idx_w}}{catch_all}  (catch-all)")
    lines += [
        "#",
        "# Teams referenced (each must exist in the org before this file validates):",
    ]
    lines += [f"#   {t}" for t in teams]

    if catch_all:
        lines += ["", fmt("*", catch_all)]

    for lbl in area_order:
        rules = groups.get(lbl)
        if not rules:
            continue
        lines += ["", f"# === {lbl}  ({rules[0][1]}) ==="]
        lines += [fmt(p, t) for p, t in rules]
    for lbl, rules in groups.items():
        if lbl not in area_order:
            lines += ["", f"# === {lbl} ==="]
            lines += [fmt(p, t) for p, t in rules]

    if overrides:
        lines += [
            "",
            "# === Path overrides: a subsystem nested inside another area's tree.",
            "# More specific, so they win via last-match over the area globs above. ===",
        ]
        lines += [fmt(p, t) for p, t in overrides]

    if shared_rules:
        lines += [
            "",
            "# --- Shared ownership: multi-team (any one approves; wins via last-match) ---",
        ]
        lines += [fmt(p, t) for p, t in shared_rules]
    if ft_rules:
        lines += [
            "",
            "# --- File-type co-ownership: area + type owner (wins via last-match) ---",
        ]
        lines += [fmt(p, t) for p, t in ft_rules]

    stats = {
        "base": len(base_rules),
        "shared": len(shared_rules),
        "filetype": len(ft_rules),
        "overrides": len(overrides),
        "teams": len(teams),
    }
    return lines, stats


def _render_advisory(model: ResolvedModel) -> dict | None:
    """Build the advisory-reviewers.yaml body, or ``None`` if no advisory rules."""
    if not model.advisory and not model.filetype_advisory:
        return None
    label_to_team = model.label_to_team()
    adv: dict = {
        "_comment": (
            "Non-blocking auto-request reviewers. Consumed by a review-request "
            "Action, NOT by CODEOWNERS. path_rules match path globs; "
            "filetype_rules match file globs by basename across the whole tree."
        ),
        "path_rules": [],
        "filetype_rules": [],
    }
    for s in model.advisory:
        adv["path_rules"].append(
            {
                "path": s["glob"],
                "request_review_from": [label_to_team.get(o, o) for o in s["owners"]],
            }
        )
    for r in model.filetype_advisory:
        pattern = r.get("pattern")
        if not pattern:
            continue
        adv["filetype_rules"].append(
            {
                "pattern": pattern,
                "request_review_from": [label_to_team.get(r["coowner"], r["coowner"])],
            }
        )
    return adv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--areas", required=True, help="path to areas.yaml (source of truth)"
    )
    ap.add_argument(
        "--repo", default=".", help="repo whose tree the cover is built against"
    )
    ap.add_argument("--out", default="CODEOWNERS", help="CODEOWNERS output path")
    ap.add_argument(
        "--advisory-out",
        default=None,
        help="advisory config output (default: alongside --areas)",
    )
    ap.add_argument(
        "--no-group",
        action="store_true",
        help="emit base shortest-path-first instead of per-area groups",
    )
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.areas).read_text())
    tree = load_tree(Path(args.repo))
    model = compute_resolution(spec, tree)

    lines, stats = _render_codeowners(model, tree, group=not args.no_group)
    Path(args.out).write_text("\n".join(lines) + "\n")
    total = (
        stats["base"]
        + stats["shared"]
        + stats["filetype"]
        + (1 if model.catch_all else 0)
    )
    print(
        f"wrote {args.out} | rules: {total} (base {stats['base']} | "
        f"shared {stats['shared']} | file-type {stats['filetype']}) | "
        f"overrides pulled out: {stats['overrides']} | "
        f"teams referenced: {stats['teams']}"
    )

    adv = _render_advisory(model)
    adv_out = (
        Path(args.advisory_out)
        if args.advisory_out
        else Path(args.areas).parent / "advisory-reviewers.yaml"
    )
    if adv is not None:
        adv_out.write_text(yaml.safe_dump(adv, sort_keys=False, width=120))
        print(
            f"wrote {adv_out} ({len(model.advisory)} path + "
            f"{len(model.filetype_advisory)} filetype advisory rules)"
        )
    elif adv_out.exists():
        # The last advisory rule was removed from areas.yaml; a stale file
        # would keep driving obsolete reviewer requests.
        adv_out.unlink()
        print(f"removed {adv_out} (no advisory rules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
