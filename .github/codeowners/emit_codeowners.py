"""Emit a GitHub CODEOWNERS file (+ advisory-reviewers.yaml) from the resolved
area taxonomy produced by `build_codeowners.py`. Repo-agnostic, PyYAML-only.

GitHub CODEOWNERS is last-match-wins. The base tier is emitted as a *minimal
last-match cover* (broad parent globs + carve-out exceptions) computed against
the live tree, grouped per area for readability, with cross-area carve-outs
pulled into a trailing override section. Shared co-ownership and file-type
co-ownership are emitted LAST so they win over the base rules they refine.

Nobody reads this file to find a reviewer -- GitHub auto-requests the owning
team, and `who_owns.py` answers "who reviews this path" on demand. The grouping
+ legend just make the generated artifact navigable.

Usage:
  uv run python scripts/emit_codeowners.py --resolved areas.resolved.yaml \
      --repo . --out CODEOWNERS
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import subprocess
from pathlib import Path

import yaml


def anchor(g: str) -> str:
    """Anchor a glob to repo root for CODEOWNERS (leading slash)."""
    return g if g.startswith("/") else "/" + g


def match(pattern: str, filepath: str) -> bool:
    """True if `filepath` matches `pattern` per GitHub CODEOWNERS rules."""
    if pattern == "*":
        return True
    if pattern.startswith("/"):
        body = pattern[1:]
        if body.endswith("/"):
            return filepath.startswith(body)
        if any(c in body for c in "*?["):
            return fnmatch.fnmatch(filepath, body)
        return filepath == body
    if pattern.endswith("/"):
        return ("/" + pattern) in ("/" + filepath) or filepath.startswith(pattern)
    if "/" not in pattern:
        base = filepath.rsplit("/", 1)[-1]
        return fnmatch.fnmatch(base, pattern) or fnmatch.fnmatch(filepath, pattern)
    return fnmatch.fnmatch(filepath, pattern)


def owner_of(rules: list[tuple[str, str]], filepath: str) -> str | None:
    """Last-match-wins owner (team) of `filepath`, or None if unrouted."""
    owner = None
    for pattern, team in rules:
        if match(pattern, filepath):
            owner = team
    return owner


def minimal_cover(file_team: dict[str, str], catch_all: str) -> list[tuple[str, str]]:
    """Smallest set of base rules reproducing `file_team` under last-match.

    Returns `(anchored_pattern, team)` pairs: directory globs covering whole
    subtrees plus file globs for in-directory exceptions. The catch-all is the
    root default and is NOT returned. Emit shortest-path-first (or grouped, with
    deeper rules after) so a more-specific rule still wins. This is the
    broad-parent-glob + carve-exceptions form; verify by replaying the tree.
    """
    from collections import defaultdict

    children: dict[str, set[str]] = defaultdict(set)
    dir_files: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, team in file_team.items():
        parts = path.split("/")
        for i in range(1, len(parts)):
            children["/".join(parts[: i - 1])].add("/".join(parts[:i]))
        dir_files["/".join(parts[:-1])].append((path, team))

    subtree: dict[str, set[str]] = {}

    def teams_under(d: str) -> set[str]:
        if d not in subtree:
            ts = {t for _, t in dir_files.get(d, ())}
            for c in children.get(d, ()):
                ts |= teams_under(c)
            subtree[d] = ts
        return subtree[d]

    memo: dict[tuple[str, str], int] = {}

    def cost(d: str, inh: str) -> int:
        key = (d, inh)
        if key not in memo:
            best = None
            for c in {inh} | teams_under(d):
                x = 0 if c == inh else 1
                x += sum(1 for _, t in dir_files.get(d, ()) if t != c)
                x += sum(cost(ch, c) for ch in children.get(d, ()))
                best = x if best is None else min(best, x)
            memo[key] = best or 0
        return memo[key]

    def choose(d: str, inh: str) -> str:
        best_c, best_x = inh, None
        for c in [inh, *sorted(teams_under(d) - {inh})]:
            x = 0 if c == inh else 1
            x += sum(1 for _, t in dir_files.get(d, ()) if t != c)
            x += sum(cost(ch, c) for ch in children.get(d, ()))
            if best_x is None or x < best_x:
                best_c, best_x = c, x
        return best_c

    rules: list[tuple[str, str]] = []

    def emit(d: str, inh: str) -> None:
        c = catch_all if d == "" else choose(d, inh)
        if d != "" and c != inh:
            rules.append(("/" + d + "/", c))
        for path, team in dir_files.get(d, ()):
            if team != c:
                rules.append(("/" + path, team))
        for ch in sorted(children.get(d, ())):
            emit(ch, c)

    emit("", catch_all)
    return rules


def load_tree(repo: Path) -> list[str]:
    out = subprocess.check_output(["git", "-C", str(repo), "ls-files"], text=True)
    return [p for p in out.splitlines() if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--resolved", required=True, help="areas.resolved.yaml from build_codeowners.py"
    )
    ap.add_argument(
        "--repo", default=".", help="repo whose tree the cover is built against"
    )
    ap.add_argument("--out", default="CODEOWNERS", help="CODEOWNERS output path")
    ap.add_argument(
        "--advisory-out",
        default=None,
        help="advisory config output (default: alongside --out)",
    )
    ap.add_argument(
        "--no-group",
        action="store_true",
        help="emit base shortest-path-first instead of per-area groups",
    )
    args = ap.parse_args()

    res = yaml.safe_load(Path(args.resolved).read_text())
    catch_all = res.get("meta", {}).get("catch_all")
    shared = res.get("shared", [])
    ft_shared = res.get("filetype_shared", [])
    label2team = {a["label"]: a["github_team"] for a in res["areas"]}
    team2label = {a["github_team"]: a["label"] for a in res["areas"]}
    area_order = [a["label"] for a in res["areas"]]

    def owners_str(owners: list[str]) -> str:
        return " ".join(label2team.get(o, o) for o in owners)

    # ---- base tier: minimal last-match cover over the live tree ----
    tree = load_tree(Path(args.repo))
    base_lookup = sorted(
        ((anchor(g), a["github_team"]) for a in res["areas"] for g in a["path_globs"]),
        key=lambda r: len(r[0]),
    )
    file_team = {f: (owner_of(base_lookup, f) or catch_all) for f in tree}
    base_rules = minimal_cover(file_team, catch_all)
    seen: set[tuple[str, str]] = set()
    base_rules = [
        (p, t) for p, t in base_rules if (p, t) not in seen and not seen.add((p, t))
    ]

    shared_t = sorted(
        ((anchor(s["glob"]), owners_str(s["owners"])) for s in shared),
        key=lambda r: (len(r[0]), r[0]),
    )
    ft_t = sorted(
        ((anchor(s["glob"]), owners_str(s["owners"])) for s in ft_shared),
        key=lambda r: (len(r[0]), r[0]),
    )

    teams = sorted(
        {a["github_team"] for a in res["areas"]}
        | ({catch_all} if catch_all else set())
        | {label2team.get(o, o) for s in ft_shared for o in s["owners"]}
    )

    # ---- group base rules per area; cross-area carve-outs -> override tail ----
    dir_rules = [(p, t) for p, t in base_rules if p.endswith("/")]

    def is_override(path: str, team: str) -> bool:
        return any(
            tp != team and path != pp and path.startswith(pp) for pp, tp in dir_rules
        )

    groups: dict[str, list[tuple[str, str]]] = {}
    overrides: list[tuple[str, str]] = []
    for p, t in base_rules:
        if not args.no_group and is_override(p, t):
            overrides.append((p, t))
        else:
            groups.setdefault(team2label.get(t, t), []).append((p, t))
    for lst in groups.values():
        lst.sort(key=lambda r: (len(r[0]), r[0]))
    overrides.sort(key=lambda r: (len(r[0]), r[0]))

    all_paths = [p for p, _ in base_rules + shared_t + ft_t] or ["*"]
    width = max(len(p) for p in all_paths) + 2

    def fmt(path: str, team: str) -> str:
        return f"{path:<{width}}{team}"

    lines = [
        "# CODEOWNERS -- generated by the codeowners-iac skill from areas.yaml.",
        "# Do not hand-edit. Change areas.yaml and regenerate.",
        f"# Generated {datetime.date.today().isoformat()}.",
        "#",
        "# GitHub reads this file; engineers don't. To see who reviews a change, run:",
        "#   python who_owns.py --codeowners CODEOWNERS --changed     # owners of your PR's files",
        "#   python who_owns.py --codeowners CODEOWNERS <path> ...     # owners of specific paths",
        "#",
        "# Area index (base owner per subsystem; the catch-all owns everything else):",
    ]
    idx_w = max((len(a["label"]) for a in res["areas"]), default=4) + 2
    for a in sorted(res["areas"], key=lambda a: a["label"]):
        name = a.get("name", "")
        suffix = f"  ({name})" if name and name != a["label"] else ""
        lines.append(f"#   {a['label']:<{idx_w}}{a['github_team']}{suffix}")
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

    if shared_t:
        lines += [
            "",
            "# --- Shared ownership: multi-team (any one approves; wins via last-match) ---",
        ]
        lines += [fmt(p, t) for p, t in shared_t]
    if ft_t:
        lines += [
            "",
            "# --- File-type co-ownership: area + type owner (wins via last-match) ---",
        ]
        lines += [fmt(p, t) for p, t in ft_t]

    Path(args.out).write_text("\n".join(lines) + "\n")
    n = len(base_rules) + len(shared_t) + len(ft_t) + (1 if catch_all else 0)
    print(
        f"wrote {args.out} | rules: {n} (base {len(base_rules)} | shared {len(shared_t)} | "
        f"file-type {len(ft_t)}) | overrides pulled out: {len(overrides)} | teams referenced: {len(teams)}"
    )

    # Advisory (NON-BLOCKING) reviewers: NOT in CODEOWNERS. Emit a separate config
    # a lightweight review-request GitHub Action can consume.
    advisory = res.get("advisory", [])
    filetype = res.get("filetype_advisory", [])
    if advisory or filetype:
        adv = {
            "_comment": "Non-blocking auto-request reviewers. Consumed by a review-request "
            "Action, NOT by CODEOWNERS. path_rules match path globs; filetype_rules match "
            "by extension/name across the whole tree.",
            "path_rules": [],
            "filetype_rules": [],
        }
        for s in advisory:
            adv["path_rules"].append(
                {
                    "path": s["glob"],
                    "request_review_from": [label2team.get(o, o) for o in s["owners"]],
                }
            )
        for r in filetype:
            adv["filetype_rules"].append(
                {
                    "match": r["match"],
                    "request_review_from": [label2team.get(r["coowner"], r["coowner"])],
                }
            )
        adv_out = (
            Path(args.advisory_out)
            if args.advisory_out
            else Path(args.out).parent / "advisory-reviewers.yaml"
        )
        adv_out.write_text(yaml.safe_dump(adv, sort_keys=False, width=120))
        print(
            f"wrote {adv_out} ({len(advisory)} path + {len(filetype)} filetype advisory rules)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
