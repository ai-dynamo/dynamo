"""Canonical CODEOWNERS matcher + resolution pipeline (single source of truth).

The CODEOWNERS pipeline used to have three subtly-different path matchers --
one in `build_codeowners.py` for coverage gating, byte-identical copies in
`emit_codeowners.py` and `who_owns.py` for routing. Coverage and routing could
disagree, and nothing cross-checked them. Everything in the pipeline now goes
through `match()` here so the gate and the artifact are forced to agree.

This module exposes:

  - ``match(pattern, path) -> bool`` -- canonical GitHub CODEOWNERS matcher.
  - ``resolve_owners(rules, path) -> list[str]`` -- last-match-wins resolution.
  - ``parse_codeowners(text) -> list[(pattern, owners)]`` -- shared parser.
  - ``anchor(glob) -> str`` -- repo-root anchoring helper.
  - ``compute_resolution(spec, tree) -> ResolvedModel`` -- pure resolution
    function used by both ``build_codeowners.py`` and ``emit_codeowners.py``.
  - ``minimal_cover(file_team, catch_all)`` -- min-cost last-match cover.
  - Typed shapes (``Area``, ``SharedSpec``, ``FiletypeRule``,
    ``ResolvedArea``, ``FiletypeShared``, ``ResolvedModel``).

GitHub CODEOWNERS semantics implemented here:

  * ``*``                  -- catch-all.
  * ``/foo/``              -- anchored directory subtree.
  * ``/foo`` (no wildcards)-- exact anchored path.
  * ``/foo/*.rs``          -- anchored fnmatch glob.
  * ``foo/``               -- unanchored directory; matches any subtree named
    ``foo/`` at any depth.
  * ``*.md`` / ``Dockerfile`` (no slash) -- basename or full-path fnmatch.
  * any with ``/`` and wildcards -- full-path fnmatch.
"""

from __future__ import annotations

import fnmatch
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TypedDict

# ----------------------------------------------------------------------
# Typed shapes (S6)
# ----------------------------------------------------------------------


class Area(TypedDict, total=False):
    """Area entry as declared in ``areas.yaml``."""

    label: str
    github_team: str
    path_globs: list[str]


class SharedSpec(TypedDict):
    """Multi-owner override (``shared:``/``advisory:`` entry)."""

    glob: str
    owners: list[str]


class FiletypeRule(TypedDict, total=False):
    """Filetype-level rule (``classify.filetype_rules`` entry).

    ``pattern`` is the single source of truth for the glob; ``coowner`` names
    the area that joins the enclosing owner; ``advisory: true`` routes the
    rule to the non-blocking advisory file instead of CODEOWNERS.
    """

    pattern: str
    coowner: str
    advisory: bool


@dataclass
class ResolvedArea:
    """An area after auto-classify has expanded its ``path_globs``."""

    label: str
    github_team: str
    path_globs: list[str]


@dataclass
class FiletypeShared:
    """File-type co-ownership row (file glob + ordered owner labels)."""

    glob: str
    owners: list[str]


@dataclass
class ResolvedModel:
    """Resolved taxonomy, ready for emission OR coverage gating.

    Both ``build_codeowners.py`` and ``emit_codeowners.py`` run resolution
    through this dataclass instead of round-tripping a YAML file between the
    two processes.
    """

    catch_all: str
    areas: list[ResolvedArea]
    shared: list[SharedSpec]
    advisory: list[SharedSpec]
    filetype_shared: list[FiletypeShared]
    filetype_advisory: list[FiletypeRule]
    auto_classified: list[tuple[str, str]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def label_to_team(self) -> dict[str, str]:
        return {a.label: a.github_team for a in self.areas}

    def owned_patterns(self) -> list[str]:
        """Every glob that contributes to explicit (non-catch-all) ownership.

        The set used by the coverage gate -- if some pattern here matches a
        path, the path is "owned" and won't be reported as catch-all-only.
        Mirrors what ``emit_codeowners.py`` will route, so gate and artifact
        can't disagree.
        """
        pats: list[str] = []
        for area in self.areas:
            pats.extend(area.path_globs)
        for s in self.shared:
            pats.append(s["glob"])
        for fs in self.filetype_shared:
            pats.append(fs.glob)
        return pats

    def unmatched_paths(self, tree: Iterable[str]) -> list[str]:
        """Paths in ``tree`` that fall through to the catch-all only."""
        patterns = self.owned_patterns()
        return [p for p in tree if not any(match(g, p) for g in patterns)]


# ----------------------------------------------------------------------
# Matching primitives (S1)
# ----------------------------------------------------------------------


def match(pattern: str, filepath: str) -> bool:
    """True if ``filepath`` matches ``pattern`` per GitHub CODEOWNERS rules.

    This is the ONLY matcher in the pipeline. Build coverage, emit routing,
    and who_owns lookups all call this. If you change a case, update the
    tests in ``test_codeowners.py``.
    """
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


def resolve_owners(rules: list[tuple[str, list[str]]], filepath: str) -> list[str]:
    """Last-match-wins owners of ``filepath``. ``[]`` if unrouted."""
    owners: list[str] = []
    for pattern, rule_owners in rules:
        if match(pattern, filepath):
            owners = rule_owners
    return owners


def parse_codeowners(text: str) -> list[tuple[str, list[str]]]:
    """Parse a CODEOWNERS file body into ordered ``(pattern, [owner, ...])``."""
    rules: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        pattern, *owners = stripped.split()
        if owners:
            rules.append((pattern, owners))
    return rules


def anchor(glob: str) -> str:
    """Anchor a glob to repo root for CODEOWNERS output (leading slash)."""
    return glob if glob.startswith("/") else "/" + glob


def load_tree(repo: Path) -> list[str]:
    """Return tracked files under ``repo`` via ``git ls-files``."""
    out = subprocess.check_output(["git", "-C", str(repo), "ls-files"], text=True)
    return [p for p in out.splitlines() if p.strip()]


# ----------------------------------------------------------------------
# Resolution pipeline (S2, S5)
# ----------------------------------------------------------------------


def _is_covered(path: str, patterns: Iterable[str]) -> bool:
    return any(match(g, path) for g in patterns)


def _dir_prefix(path: str, max_segments: int = 3) -> str:
    """Group an unmatched file under a directory prefix (<= ``max_segments``).

    Returns ``""`` for a repo-root file -- those are skipped by auto-classify
    because we never want to invent a glob like ``/Dockerfile/``.
    """
    segs = path.split("/")
    depth = min(max_segments, len(segs) - 1)
    return "/".join(segs[:depth])


def _auto_classify(
    tree: list[str],
    area_globs: dict[str, set[str]],
    keyword_rules: list[dict],
    label_by: dict[str, str],
) -> list[tuple[str, str]]:
    """Promote unmatched dirs into areas via keyword rules.

    Mutates ``area_globs`` in place: new dir globs are appended to the chosen
    area's set. Returns the ``(dir, label)`` audit list. First matching rule
    wins (rules are evaluated in declared order).
    """
    if not keyword_rules:
        return []
    initial_patterns = [g for gs in area_globs.values() for g in gs]
    unmatched = [p for p in tree if not _is_covered(p, initial_patterns)]
    if not unmatched:
        return []

    by_dir = sorted({d for p in unmatched if (d := _dir_prefix(p))})
    audit: list[tuple[str, str]] = []
    for d in by_dir:
        pl = ("/" + d + "/").lower()
        for r in keyword_rules:
            needle = r.get("match", "").lower()
            if needle and needle in pl and r.get("area"):
                lbl = label_by.get(r["area"], r["area"])
                area_globs.setdefault(lbl, set()).add(d.rstrip("/") + "/")
                audit.append((d, lbl))
                break
    return audit


def _filetype_coownership(
    tree: list[str],
    area_globs: dict[str, set[str]],
    filetype_rules: list[FiletypeRule],
) -> list[FiletypeShared]:
    """Compute the file-type co-ownership table (blocking rules only).

    Each match becomes ``[enclosing_area, coowner]`` so the area stays in the
    review loop. Files inside the coowner's own area stay sole-owned.
    """
    # Sort by glob length descending so the most-specific enclosing dir wins.
    enc_pairs = sorted(
        ((g, lbl) for lbl, s in area_globs.items() for g in s),
        key=lambda kv: -len(kv[0]),
    )

    def enclosing(path: str) -> str | None:
        for g, lbl in enc_pairs:
            gg = g.rstrip("/")
            if path == gg or path.startswith(gg + "/"):
                return lbl
        return None

    out: list[FiletypeShared] = []
    for rule in filetype_rules:
        if rule.get("advisory"):
            continue
        pattern = rule.get("pattern")
        coowner = rule["coowner"]
        if not pattern:
            continue
        for path in tree:
            base = path.rsplit("/", 1)[-1]
            if not fnmatch.fnmatch(base, pattern):
                continue
            enc = enclosing(path)
            owners = [enc] if enc and enc != coowner else []
            owners.append(coowner)
            out.append(FiletypeShared(glob=path, owners=owners))
    return out


def compute_resolution(spec: dict, tree: list[str]) -> ResolvedModel:
    """Resolve an ``areas.yaml`` spec against a live tree.

    Pure function: same spec + same tree -> same model. No I/O beyond the
    tree list it's given. Both ``build_codeowners.py`` (coverage gate) and
    ``emit_codeowners.py`` (generator) call this -- the gate and the artifact
    cannot disagree because there is no second resolver.
    """
    catch_all = spec.get("meta", {}).get("catch_all", "")
    raw_areas = spec.get("areas", [])
    classify = spec.get("classify", {}) or {}
    keyword_rules = classify.get("keyword_rules", []) or []
    filetype_rules: list[FiletypeRule] = classify.get("filetype_rules", []) or []

    label_by = {a["label"]: a["label"] for a in raw_areas}
    area_globs: dict[str, set[str]] = {
        a["label"]: set(a.get("path_globs", []) or []) for a in raw_areas
    }

    audit = _auto_classify(tree, area_globs, keyword_rules, label_by)
    filetype_shared = _filetype_coownership(tree, area_globs, filetype_rules)

    areas = [
        ResolvedArea(
            label=a["label"],
            github_team=a["github_team"],
            path_globs=sorted(area_globs[a["label"]]),
        )
        for a in raw_areas
    ]
    filetype_advisory = [r for r in filetype_rules if r.get("advisory")]

    return ResolvedModel(
        catch_all=catch_all,
        areas=areas,
        shared=spec.get("shared", []) or [],
        advisory=spec.get("advisory", []) or [],
        filetype_shared=filetype_shared,
        filetype_advisory=filetype_advisory,
        auto_classified=audit,
        meta=dict(spec.get("meta", {})),
    )


# ----------------------------------------------------------------------
# minimal_cover (S5, S7) -- min-cost last-match cover
# ----------------------------------------------------------------------


def minimal_cover(file_team: dict[str, str], catch_all: str) -> list[tuple[str, str]]:
    """Smallest set of base rules reproducing ``file_team`` under last-match.

    Returns ``(anchored_pattern, team)`` pairs: directory globs covering whole
    subtrees plus file globs for in-directory exceptions. The catch-all is
    the root default and is NOT returned. Emit shortest-path-first (or
    grouped, with deeper rules after) so a more-specific rule still wins.
    """
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
