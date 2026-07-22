"""Canonical CODEOWNERS matcher + resolution pipeline (single source of truth).

The CODEOWNERS pipeline used to have three subtly-different path matchers --
one in `build_codeowners.py` for coverage gating, byte-identical copies in
`emit_codeowners.py` and `who_owns.py` for routing. Coverage and routing could
disagree, and nothing cross-checked them. Everything in the pipeline now goes
through `match()` here so the gate and the artifact are forced to agree.

Emission is a PURE FUNCTION of the policy YAML: ``compute_resolution(spec)``
takes only the parsed ``areas.yaml`` and returns the model that ``emit`` uses,
with no ``git ls-files`` in the loop. The tree is read exactly once, in the
``--strict`` coverage gate of ``build_codeowners.py``, to assert that every
tracked file matches a non-catch-all rule. Same policy in -> byte-identical
CODEOWNERS out, no matter what the checked-out tree looks like.

This module exposes:

  - ``match(pattern, path) -> bool`` -- canonical GitHub CODEOWNERS matcher.
  - ``resolve_owners(rules, path) -> list[str]`` -- last-match-wins resolution.
  - ``parse_codeowners(text) -> list[(pattern, owners)]`` -- shared parser.
  - ``anchor(glob) -> str`` -- repo-root anchoring helper.
  - ``compute_resolution(spec) -> ResolvedModel`` -- pure, tree-independent
    resolution used by both ``build_codeowners.py`` and ``emit_codeowners.py``.
  - ``load_tree(repo)`` -- ``git ls-files`` helper (``--strict`` gate only).
  - ``minimal_cover(file_team, catch_all)`` -- legacy min-cost last-match
    cover, kept for external callers; NOT used by the emitter anymore.
  - Typed shapes (``Area``, ``SharedSpec``, ``FiletypeRule``,
    ``ResolvedArea``, ``FiletypeShared``, ``ResolvedModel``).

GitHub CODEOWNERS semantics implemented here:

  * ``*``                  -- catch-all.
  * ``/foo/``              -- anchored directory subtree.
  * ``/foo`` (no wildcards)-- exact anchored path.
  * ``/foo/*.rs``          -- anchored glob; ``*``/``?`` stop at ``/``,
    ``**`` crosses directories (as GitHub resolves them).
  * ``foo/``               -- unanchored directory; matches any subtree named
    ``foo/`` at any depth.
  * ``*.md`` / ``Dockerfile`` (no slash) -- basename glob at any depth.
  * any with ``/`` and wildcards -- full-path glob, same ``*`` semantics.
"""

from __future__ import annotations

import functools
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

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

    ``pattern`` is the single source of truth for the glob; ``coowner`` is the
    legacy key naming the file-type default owner. ``advisory: true`` routes
    the rule to the non-blocking advisory file instead of CODEOWNERS.
    """

    pattern: str
    coowner: str
    advisory: bool


@dataclass
class ResolvedArea:
    """An area with normalized, explicitly declared ``path_globs``."""

    label: str
    github_team: str
    path_globs: list[str]


@dataclass
class FiletypeShared:
    """File-type ownership default (file glob + ordered owner labels)."""

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
    meta: dict = field(default_factory=dict)

    def label_to_team(self) -> dict[str, str]:
        return {a.label: a.github_team for a in self.areas}

    def owned_patterns(self) -> list[str]:
        """Every glob that contributes to explicit (non-catch-all) ownership.

        The set used by the coverage gate -- if some pattern here matches a
        path, the path is "owned" and won't be reported as catch-all-only.
        Area and shared globs are anchored exactly as ``emit_codeowners.py``
        anchors them at emission, so gate and artifact can't disagree: an
        unanchored ``README.md`` in an area's ``path_globs`` must not
        silently cover ``foo/README.md``. Filetype-rule patterns are NOT
        anchored (a bare ``*Dockerfile*`` matches by basename at any depth
        under GitHub CODEOWNERS rules, and the emitter writes it that way).
        """
        pats: list[str] = []
        for area in self.areas:
            pats.extend(anchor(g) for g in area.path_globs)
        for s in self.shared:
            pats.append(anchor(s["glob"]))
        # Filetype patterns are emitted unanchored (basename-any-depth), so
        # keep them unanchored in the coverage set too.
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


def _glob_to_re(pattern: str) -> str:
    """Translate a CODEOWNERS glob to a regex: ``*``/``?`` stop at ``/``,
    ``**`` crosses directories. fnmatch is wrong here -- its ``*`` greedily
    crosses path separators, which GitHub's resolver does not."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i + 1
            if j < len(pattern) and pattern[j] in "!^":
                j += 1
            if j < len(pattern) and pattern[j] == "]":
                j += 1
            while j < len(pattern) and pattern[j] != "]":
                j += 1
            if j >= len(pattern):
                out.append(re.escape(c))
            else:
                cls = pattern[i + 1 : j]
                if cls.startswith("!"):
                    cls = "^" + cls[1:]
                out.append("[" + cls + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


@functools.cache
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(_glob_to_re(pattern))


def _glob_match(pattern: str, path: str) -> bool:
    return _compiled(pattern).fullmatch(path) is not None


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
            return _glob_match(body, filepath)
        return filepath == body
    if pattern.endswith("/"):
        return ("/" + pattern) in ("/" + filepath) or filepath.startswith(pattern)
    if "/" not in pattern:
        base = filepath.rsplit("/", 1)[-1]
        return _glob_match(pattern, base) or _glob_match(pattern, filepath)
    return _glob_match(pattern, filepath)


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


def changed_paths(repo: Path, base: str) -> list[str]:
    """Paths this branch adds/changes vs ``base`` (``git diff base...HEAD``).

    ``--diff-filter=ACMR`` keeps Added/Copied/Modified/Renamed and drops
    Deletions -- a removed file is not a coverage concern. Three-dot
    ``base...HEAD`` diffs against the merge-base, so a long-running branch is
    judged only on the surface it actually touched, not on unrelated paths
    that landed on ``base`` after it forked. This is the diff-aware input to
    the ``--strict`` gate; it reads the tree, like ``load_tree``, and lives
    only in the coverage tool, never in emission.
    """
    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--name-only",
                "--diff-filter=ACMR",
                f"{base}...HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as err:
        raise SystemExit(
            f"git diff failed in {repo!r} (not a checkout, or base {base!r} unavailable): {err}"
        ) from err
    return [p for p in out.splitlines() if p.strip()]


# ----------------------------------------------------------------------
# Resolution pipeline
# ----------------------------------------------------------------------
#
# ``compute_resolution`` is a PURE FUNCTION of the parsed policy YAML: it
# never touches ``git ls-files``. The base-branch race the old generator
# suffered from -- unrelated tree churn on ``main`` mutating the minimal
# cover, the auto-classified globs, or the filetype co-ownership rows for
# a PR that changed none of them -- comes entirely from resolving against
# a live tree at emit time. Coverage is still checked against the tree, but
# the check lives in ``build_codeowners.py --strict`` and does not feed
# back into the emitted rules.


def compute_resolution(spec: dict, tree: Iterable[str] | None = None) -> ResolvedModel:
    """Resolve an ``areas.yaml`` spec into the model the emitter renders.

    Pure function of ``spec``: no tree, no filesystem, no ``git``. Same YAML
    in -> byte-identical model out, so the emitted CODEOWNERS is a pure
    function of the policy inputs. ``tree`` is accepted for backward
    compatibility (older callers passed it) and ignored.

    Semantics per section:

    * ``areas``       -- ``path_globs`` are emitted verbatim (sorted).
    * ``shared``      -- passed through as declared.
    * ``advisory``    -- passed through as declared.
    * ``classify.filetype_rules`` -- each blocking rule becomes one stable
      row with the coowner as the sole owner (a single ``*Dockerfile*``
      line owns every Dockerfile at any depth unless a later explicit path
      override or shared rule applies). Advisory rules go to the
      advisory-reviewers file.
    * ``classify.keyword_rules`` -- no longer supported. Auto-promotion of
      unmatched dirs into an area, and keyword-level co-ownership, both
      required walking the live tree -- pure poison for a stable output.
      A non-empty legacy block is rejected so dead policy cannot silently
      appear effective; authors use explicit ``path_globs`` / ``shared``
      entries instead.
    """
    if tree is not None:
        # Deprecated argument: accepted for legacy callers, ignored so
        # emission stays a pure function of the policy inputs.
        _ = tree

    catch_all = spec.get("meta", {}).get("catch_all", "")
    raw_areas = spec.get("areas", [])
    classify = spec.get("classify", {}) or {}
    keyword_rules = classify.get("keyword_rules", []) or []
    if keyword_rules:
        raise SystemExit(
            "areas.yaml: classify.keyword_rules is no longer supported; "
            "use explicit area path_globs or shared entries"
        )
    filetype_rules: list[FiletypeRule] = classify.get("filetype_rules", []) or []

    spec_shared: list[SharedSpec] = spec.get("shared", []) or []

    areas = [
        ResolvedArea(
            label=a["label"],
            github_team=a["github_team"],
            path_globs=sorted(set(a.get("path_globs", []) or [])),
        )
        for a in raw_areas
    ]

    # Blocking filetype rule -> one stable coowner-only row (bare pattern
    # matches by basename at any depth per GitHub CODEOWNERS semantics).
    # The old "enclosing area + coowner" behavior required walking the
    # tree; if a specific subtree wants that co-ownership, declare it
    # explicitly in ``shared`` with a path glob.
    filetype_shared: list[FiletypeShared] = []
    for rule in filetype_rules:
        if rule.get("advisory"):
            continue
        pattern = rule.get("pattern")
        coowner = rule.get("coowner")
        if not pattern or not coowner:
            raise SystemExit(
                f"areas.yaml: filetype_rules entry {rule!r} missing "
                "'pattern' or 'coowner'"
            )
        filetype_shared.append(FiletypeShared(glob=pattern, owners=[coowner]))

    filetype_advisory = [r for r in filetype_rules if r.get("advisory")]

    return ResolvedModel(
        catch_all=catch_all,
        areas=areas,
        shared=list(spec_shared),
        advisory=spec.get("advisory", []) or [],
        filetype_shared=filetype_shared,
        filetype_advisory=filetype_advisory,
        meta=dict(spec.get("meta", {})),
    )


# ----------------------------------------------------------------------
# minimal_cover -- LEGACY min-cost last-match cover
# ----------------------------------------------------------------------
#
# Kept exported for external callers that still want a tree-aware minimal
# cover. The CODEOWNERS emitter no longer calls it: whichever rule set the
# cover picks depends on the exact tree, so unrelated file adds/moves on
# ``main`` used to churn the emitted CODEOWNERS for PRs that touched none
# of them. Emission now renders declared ``path_globs`` verbatim.


def minimal_cover(file_team: dict[str, str], catch_all: str) -> list[tuple[str, str]]:
    """Smallest set of base rules reproducing ``file_team`` under last-match.

    Returns ``(anchored_pattern, team)`` pairs: directory globs covering whole
    subtrees plus file globs for in-directory exceptions. The catch-all is
    the root default and is NOT returned. Emit shortest-path-first (or
    grouped, with deeper rules after) so a more-specific rule still wins.

    Legacy: the CODEOWNERS emitter no longer calls this, because the choice
    of cover depends on the tree and thus makes CODEOWNERS non-deterministic
    against unrelated churn on the base branch.
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
