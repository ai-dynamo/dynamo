"""Emit a GitHub CODEOWNERS file (+ advisory-reviewers.yaml) from areas.yaml.

GitHub CODEOWNERS is last-match-wins. The base tier is emitted by rendering
each area's declared ``path_globs`` verbatim as anchored rules, grouped per
area for readability, with cross-area carve-outs (a glob that lexically nests
under another area's dir glob) pulled into a trailing override section.
Shared co-ownership and file-type co-ownership are emitted LAST so they win
over the base rules they refine.

Emission is a PURE FUNCTION of the policy inputs (``areas.yaml`` and
``external_contributors.yaml``): the repository tree is NEVER read at emit
time. That fixes a base-branch race the old generator suffered from -- the
tree-dependent minimal-cover / singleton-file optimization used to rewrite
rules whenever an unrelated commit on ``main`` added, deleted, or moved a
file under an already-owned prefix, breaking the ``codeowners`` CI check on
long-running PRs that had touched nothing structural.

Coverage is still validated by ``build_codeowners.py --strict`` (which does
read ``git ls-files``). Emit and gate share ``codeowners_match``'s pure
resolver, so a tree file that the gate accepts also gets a rule at emit.

Nobody reads this file to find a reviewer -- GitHub auto-requests the owning
team, and ``who_owns.py`` answers "who reviews this path" on demand. The
grouping + legend just make the generated artifact navigable.

External contributors (``external_contributors.yaml``) attach to an area LABEL
and are appended as co-owners on every line that area's team owns, so they
inherit the team's globs without duplicating them. The same file drives
``CONTRIBUTORS.md``.

Usage:
  uv run python .github/codeowners/emit_codeowners.py \\
      --areas .github/codeowners/areas.yaml \\
      --out CODEOWNERS \\
      --advisory-out .github/codeowners/advisory-reviewers.yaml \\
      --external .github/codeowners/external_contributors.yaml \\
      --contributors-out CONTRIBUTORS.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from codeowners_match import ResolvedModel, anchor, compute_resolution  # noqa: E402


def _owners_str(label_to_team: dict[str, str], owners: list[str]) -> str:
    """Render a list of area labels (or raw teams) as a space-joined team list."""
    return " ".join(label_to_team.get(o, o) for o in owners)


def _handle(github: str) -> str:
    """Normalize a bare GitHub username to a CODEOWNERS ``@handle``."""
    return "@" + github.strip().lstrip("@")


def _github(c: dict) -> str:
    """Required GitHub username for a contributor; friendly error if absent."""
    gh = c.get("github")
    if not gh:
        who = c.get("name") or "<unnamed>"
        raise SystemExit(f"external_contributors.yaml: {who!r} is missing 'github'")
    return str(gh)


# Contributor standing, ordered low -> high. The rank drives CONTRIBUTORS.md
# sort order (most senior first). This is metadata about a person's standing;
# it does not change CODEOWNERS routing (co-ownership is by attached area).
CONTRIBUTOR_LEVELS: tuple[str, ...] = (
    "contributor",
    "trusted_contributor",
    "maintainer",
    "core_maintainer",
)
LEVEL_DISPLAY: dict[str, str] = {
    "contributor": "Contributor",
    "trusted_contributor": "Trusted Contributor",
    "maintainer": "Maintainer",
    "core_maintainer": "Core Maintainer",
}
LEVEL_RANK: dict[str, int] = {lvl: i for i, lvl in enumerate(CONTRIBUTOR_LEVELS)}


def contributor_level(c: dict) -> str:
    """Canonical level for a contributor; hard error on missing/invalid.

    Accepts human spellings ("Trusted Contributor", "trusted-contributor") and
    normalizes to the canonical enum token. A typo must not silently demote or
    drop someone, so an unknown value fails the generation.
    """
    who = c.get("name") or c.get("github") or "<unknown>"
    raw = c.get("level")
    if raw is None:
        raise SystemExit(
            f"external_contributors.yaml: {who!r} is missing 'level' "
            f"(one of: {', '.join(CONTRIBUTOR_LEVELS)})"
        )
    key = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in LEVEL_RANK:
        raise SystemExit(
            f"external_contributors.yaml: invalid level {raw!r} for {who!r} "
            f"(one of: {', '.join(CONTRIBUTOR_LEVELS)})"
        )
    return key


def load_external_contributors(path: Path) -> list[dict]:
    """Read ``external_contributors.yaml`` -> list of contributor dicts (or [])."""
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("contributors") or []


def team_externals_map(
    contributors: list[dict], label_to_team: dict[str, str]
) -> dict[str, list[str]]:
    """Map each area team -> the external ``@handles`` co-owning that area.

    A contributor attaches to area LABELS (not globs); each label resolves to
    its GitHub team here, so the team's every CODEOWNERS line can pick up the
    handle at render time. An unknown label is a hard error -- a typo must not
    silently drop an owner.
    """
    mapping: dict[str, list[str]] = {}
    for c in contributors:
        handle = _handle(_github(c))
        for label in c.get("areas", []) or []:
            team = label_to_team.get(label)
            if team is None:
                raise SystemExit(
                    "external_contributors.yaml: unknown area label "
                    f"{label!r} for {c.get('name', handle)!r} "
                    "(must match an area in areas.yaml)"
                )
            bucket = mapping.setdefault(team, [])
            if handle not in bucket:
                bucket.append(handle)
    return mapping


def decorate_owners(owner_str: str, team_externals: dict[str, list[str]]) -> str:
    """Append external co-owner handles for any team present in ``owner_str``.

    Keeps existing owners first (area team wins the line), then appends each
    attached contributor once, so co-ownership reads ``@team @handle``.
    """
    if not team_externals:
        return owner_str
    tokens = owner_str.split()
    out: list[str] = list(tokens)
    for tok in tokens:
        for handle in team_externals.get(tok, []):
            if handle not in out:
                out.append(handle)
    return " ".join(out)


def render_contributors_md(contributors: list[dict]) -> str:
    """Render CONTRIBUTORS.md from the external-contributor source (external only).

    Rows are ordered by standing (most senior first), then by name. ``areas``
    are shown as inline-code chips; the resolved team lives in CODEOWNERS.
    """
    lines = [
        "# Contributors",
        "",
        "External contributors who hold area-scoped **codeownership** in this",
        "repository. Each person below has earned review and approval rights over",
        "one or more subsystem areas, and is added as a co-owner on those areas'",
        "paths alongside the owning NVIDIA team.",
        "",
        "Generated from `.github/codeowners/external_contributors.yaml`. Do not",
        "hand-edit \u2014 update that file and regenerate (see",
        "`.github/codeowners/README.md`).",
        "",
    ]
    if not contributors:
        lines += ["_No external contributors yet._", ""]
        return "\n".join(lines)
    lines += [
        "| Contributor | Level | GitHub | Affiliation | Areas |",
        "| --- | --- | --- | --- | --- |",
    ]
    ordered = sorted(
        contributors,
        key=lambda c: (
            -LEVEL_RANK[contributor_level(c)],
            str(c.get("name", "")).lower(),
        ),
    )
    for c in ordered:
        handle = _handle(_github(c))
        gh_link = f"[{handle}](https://github.com/{handle.lstrip('@')})"
        level = LEVEL_DISPLAY[contributor_level(c)]
        affiliation = c.get("affiliation") or "n/a"
        areas = ", ".join(f"`{label}`" for label in (c.get("areas", []) or [])) or "n/a"
        name = c.get("name", handle)
        lines.append(f"| {name} | {level} | {gh_link} | {affiliation} | {areas} |")
    lines.append("")
    return "\n".join(lines)


def _base_rules(model: ResolvedModel) -> list[tuple[str, str]]:
    """Render every declared ``path_globs`` entry as an anchored ``(pat, team)``.

    Pure function of the resolved model -- no tree access. Duplicates that
    would arise from the same glob showing up in two areas (a configuration
    smell) are collapsed while preserving first-seen order so downstream
    grouping stays stable.
    """
    seen: set[tuple[str, str]] = set()
    rules: list[tuple[str, str]] = []
    for area in model.areas:
        for g in area.path_globs:
            key = (anchor(g), area.github_team)
            if key in seen:
                continue
            seen.add(key)
            rules.append(key)
    return rules


def _render_codeowners(
    model: ResolvedModel,
    group: bool,
    external: list[dict] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Build the CODEOWNERS file body. Returns (lines, stats).

    Pure function of the resolved model plus the external-contributors list:
    no tree, no filesystem. Same inputs -> byte-identical output.
    """
    catch_all = model.catch_all
    label_to_team = model.label_to_team()
    team_to_label = {a.github_team: a.label for a in model.areas}
    area_order = [a.label for a in model.areas]

    # External contributors co-own an area's paths by attaching to its label;
    # their handle is appended to every line that area's team owns (deco()).
    team_externals = team_externals_map(external or [], label_to_team)

    def deco(owners: str) -> str:
        return decorate_owners(owners, team_externals)

    base_rules = _base_rules(model)

    shared_rules = sorted(
        (
            (anchor(s["glob"]), _owners_str(label_to_team, s["owners"]))
            for s in model.shared
        ),
        key=lambda r: (len(r[0]), r[0]),
    )
    # Filetype patterns emit UNANCHORED (bare ``*Dockerfile*`` matches by
    # basename at any depth under GitHub CODEOWNERS rules). Anchoring them
    # would silently narrow ``*Dockerfile*`` to only match root-level files.
    ft_rules = sorted(
        (
            (fs.glob, _owners_str(label_to_team, fs.owners))
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
        lines += ["", fmt("*", deco(catch_all))]

    for lbl in area_order:
        rules = groups.get(lbl)
        if not rules:
            continue
        lines += ["", f"# === {lbl}  ({deco(rules[0][1])}) ==="]
        lines += [fmt(p, deco(t)) for p, t in rules]
    for lbl, rules in groups.items():
        if lbl not in area_order:
            lines += ["", f"# === {lbl} ==="]
            lines += [fmt(p, deco(t)) for p, t in rules]

    if overrides:
        lines += [
            "",
            "# === Path overrides: a subsystem nested inside another area's tree.",
            "# More specific, so they win via last-match over the area globs above. ===",
        ]
        lines += [fmt(p, deco(t)) for p, t in overrides]

    if shared_rules:
        lines += [
            "",
            "# --- Shared ownership: multi-team (any one approves; wins via last-match) ---",
        ]
        lines += [fmt(p, deco(t)) for p, t in shared_rules]
    if ft_rules:
        lines += [
            "",
            "# --- File-type ownership: unanchored patterns (wins via last-match at any depth) ---",
        ]
        lines += [fmt(p, deco(t)) for p, t in ft_rules]

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
        "--repo",
        default=None,
        help=(
            "DEPRECATED: emission no longer reads the tree. Accepted for "
            "backward compatibility with older CI invocations and ignored."
        ),
    )
    ap.add_argument("--out", default="CODEOWNERS", help="CODEOWNERS output path")
    ap.add_argument(
        "--advisory-out",
        default=None,
        help="advisory config output (default: alongside --areas)",
    )
    ap.add_argument(
        "--external",
        default=None,
        help="external_contributors.yaml (default: alongside --areas)",
    )
    ap.add_argument(
        "--contributors-out",
        default="CONTRIBUTORS.md",
        help="CONTRIBUTORS.md output path",
    )
    ap.add_argument(
        "--no-group",
        action="store_true",
        help="emit base shortest-path-first instead of per-area groups",
    )
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.areas).read_text())
    model = compute_resolution(spec)

    external_path = (
        Path(args.external)
        if args.external
        else Path(args.areas).parent / "external_contributors.yaml"
    )
    external = load_external_contributors(external_path)

    lines, stats = _render_codeowners(model, group=not args.no_group, external=external)
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

    contributors_md = render_contributors_md(external)
    Path(args.contributors_out).write_text(contributors_md)
    print(f"wrote {args.contributors_out} ({len(external)} external contributor(s))")

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
