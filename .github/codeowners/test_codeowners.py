"""Unit tests for the shared codeowners matching + resolution module.

These pin down two pieces of the CODEOWNERS pipeline that previously had no
tests and three subtly-different in-tree copies:

  - `match(pattern, path)` -- canonical CODEOWNERS-style matcher used by build
    coverage, emit routing, and who_owns lookups.
  - `minimal_cover(file_team, catch_all)` -- the recursive min-cost cover that
    turns a per-file owner map into the smallest set of last-match base rules.

If either drifts, the tests catch it before the generated CODEOWNERS goes wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `import codeowners_match` when pytest runs from the repo root.
sys.path.insert(0, str(Path(__file__).parent))

from codeowners_match import (  # noqa: E402
    Area,
    ResolvedModel,
    SharedSpec,
    anchor,
    compute_resolution,
    match,
    minimal_cover,
    resolve_owners,
)

# ------------------------------------------------------------------
# match() -- canonical CODEOWNERS path matcher
# ------------------------------------------------------------------


class TestMatchCatchAll:
    def test_star_matches_any_path(self) -> None:
        assert match("*", "foo.py")
        assert match("*", "a/b/c.md")
        assert match("*", "")


class TestMatchAnchoredDir:
    def test_anchored_dir_matches_inside(self) -> None:
        assert match("/lib/llm/", "lib/llm/foo.rs")
        assert match("/lib/llm/", "lib/llm/src/preprocessor.rs")

    def test_anchored_dir_rejects_sibling(self) -> None:
        assert not match("/lib/llm/", "lib/llmx/foo.rs")
        assert not match("/lib/llm/", "lib_other/llm/foo.rs")

    def test_anchored_dir_rejects_unrelated(self) -> None:
        assert not match("/lib/llm/", "tests/foo.py")


class TestMatchAnchoredFile:
    def test_anchored_file_exact_match(self) -> None:
        assert match("/Cargo.toml", "Cargo.toml")
        assert not match("/Cargo.toml", "subdir/Cargo.toml")
        assert not match("/Cargo.toml", "Cargo.toml.bak")

    def test_anchored_file_with_glob(self) -> None:
        assert match("/lib/*.rs", "lib/foo.rs")
        assert match("/lib/*.rs", "lib/bar.rs")
        # GitHub CODEOWNERS `*` stays within one path segment (docs/* matches
        # docs/getting-started.md but NOT docs/build-app/troubleshooting.md).
        # Nested files need a recursive `**` pattern.
        assert not match("/lib/*.rs", "lib/sub/foo.rs")
        assert match("/lib/**.rs", "lib/sub/foo.rs")
        assert match("/lib/**/foo.rs", "lib/a/b/foo.rs")

    def test_question_mark_stays_in_segment(self) -> None:
        assert match("/lib/?.rs", "lib/a.rs")
        assert not match("/lib/?.rs", "lib/ab.rs")
        assert not match("/lib/?.rs", "lib/a/b.rs")


class TestMatchBasenameGlob:
    def test_md_basename_glob_matches_anywhere(self) -> None:
        assert match("*.md", "README.md")
        assert match("*.md", "docs/intro.md")
        assert match("*.md", "a/b/c.md")

    def test_md_basename_glob_rejects_non_md(self) -> None:
        assert not match("*.md", "README.txt")
        assert not match("*.md", "docs/notes.rst")

    def test_bare_name_matches_anywhere(self) -> None:
        assert match("Dockerfile", "Dockerfile")
        assert match("Dockerfile", "container/Dockerfile")
        assert match("Dockerfile", "deploy/operator/Dockerfile")
        assert not match("Dockerfile", "Dockerfile.test")

    def test_wildcard_basename(self) -> None:
        assert match("*Dockerfile*", "container/Dockerfile.test")
        assert match("*Dockerfile*", "deploy/Dockerfile")
        assert not match("*Dockerfile*", "container/run.sh")


class TestMatchUnanchoredDir:
    def test_unanchored_dir_matches_under_root(self) -> None:
        assert match("lib/llm/", "lib/llm/foo.rs")

    def test_unanchored_dir_matches_nested(self) -> None:
        # Bare unanchored dirs (no leading /) match any segment in the path.
        # In areas.yaml all globs are anchored-from-root, so this rarely fires,
        # but the canonical matcher must mirror GitHub's behavior.
        assert match("foo/", "x/foo/y.py")
        assert match("foo/", "foo/bar.py")


class TestMatchPathPattern:
    def test_path_with_slash_no_glob(self) -> None:
        assert match("lib/llm/foo.rs", "lib/llm/foo.rs")
        assert not match("lib/llm/foo.rs", "lib/llm/foo.py")

    def test_path_with_slash_and_glob(self) -> None:
        assert match("lib/llm/*.rs", "lib/llm/foo.rs")


# ------------------------------------------------------------------
# resolve_owners() -- last-match-wins resolution
# ------------------------------------------------------------------


class TestResolveOwners:
    def test_last_match_wins(self) -> None:
        rules = [
            ("*", ["@root"]),
            ("/lib/", ["@runtime"]),
            ("/lib/llm/", ["@frontend"]),
        ]
        assert resolve_owners(rules, "lib/llm/foo.rs") == ["@frontend"]
        assert resolve_owners(rules, "lib/runtime/foo.rs") == ["@runtime"]
        assert resolve_owners(rules, "README.md") == ["@root"]

    def test_unrouted_returns_empty(self) -> None:
        rules = [("/lib/", ["@runtime"])]
        assert resolve_owners(rules, "tests/foo.py") == []

    def test_multi_owner_passthrough(self) -> None:
        rules = [("*", ["@a"]), ("/shared/", ["@b", "@c"])]
        assert resolve_owners(rules, "shared/x") == ["@b", "@c"]


# ------------------------------------------------------------------
# minimal_cover() -- recursive min-cost last-match cover
# ------------------------------------------------------------------


def _resolve_via(rules: list[tuple[str, str]], catch_all: str, path: str) -> str:
    """Replay minimal_cover output against `path`, mirroring GitHub semantics."""
    owner = catch_all
    for pattern, team in rules:
        if match(pattern, path):
            owner = team
    return owner


class TestMinimalCover:
    def test_empty_tree_returns_no_rules(self) -> None:
        assert minimal_cover({}, "@root") == []

    def test_all_catch_all_emits_nothing(self) -> None:
        # Every path is already owned by the catch-all -> no base rule needed.
        file_team = {"a/b.py": "@root", "c/d.py": "@root"}
        assert minimal_cover(file_team, "@root") == []

    def test_single_team_subtree_collapses_to_dir(self) -> None:
        file_team = {
            "lib/llm/a.rs": "@runtime",
            "lib/llm/b.rs": "@runtime",
            "lib/llm/sub/c.rs": "@runtime",
        }
        rules = minimal_cover(file_team, "@root")
        # All three files should resolve to @runtime via at most one dir rule.
        for path in file_team:
            assert _resolve_via(rules, "@root", path) == "@runtime"
        # Smallest cover: a single /lib/ or /lib/llm/ dir rule beats per-file rules.
        assert any(p.endswith("/") for p, _ in rules)

    def test_nested_override(self) -> None:
        # Parent dir owned by @runtime, nested subtree owned by @kvbm.
        file_team = {
            "lib/llm/a.rs": "@runtime",
            "lib/llm/b.rs": "@runtime",
            "lib/llm/kv/x.rs": "@kvbm",
            "lib/llm/kv/y.rs": "@kvbm",
        }
        rules = minimal_cover(file_team, "@root")
        for path, team in file_team.items():
            assert _resolve_via(rules, "@root", path) == team

    def test_single_file_exception(self) -> None:
        # One file in a @runtime subtree goes to a different team.
        file_team = {
            "lib/llm/a.rs": "@runtime",
            "lib/llm/b.rs": "@runtime",
            "lib/llm/special.rs": "@parsers",
        }
        rules = minimal_cover(file_team, "@root")
        for path, team in file_team.items():
            assert _resolve_via(rules, "@root", path) == team

    def test_single_file_exception_back_to_catch_all(self) -> None:
        # An island file that should fall back to the catch-all even though
        # its siblings are all owned.
        file_team = {
            "lib/llm/a.rs": "@runtime",
            "lib/llm/b.rs": "@runtime",
            "lib/llm/exempt.txt": "@root",
        }
        rules = minimal_cover(file_team, "@root")
        for path, team in file_team.items():
            assert _resolve_via(rules, "@root", path) == team

    def test_two_independent_subtrees(self) -> None:
        file_team = {
            "lib/llm/a.rs": "@runtime",
            "tests/foo.py": "@runtime",
            "components/vllm/a.py": "@vllm",
            "components/sglang/a.py": "@sglang",
        }
        rules = minimal_cover(file_team, "@root")
        for path, team in file_team.items():
            assert _resolve_via(rules, "@root", path) == team

    def test_root_level_file_emits_file_rule(self) -> None:
        file_team = {"Cargo.toml": "@ops", "README.md": "@root"}
        rules = minimal_cover(file_team, "@root")
        assert _resolve_via(rules, "@root", "Cargo.toml") == "@ops"
        assert _resolve_via(rules, "@root", "README.md") == "@root"


# ------------------------------------------------------------------
# anchor() -- absolute paths for CODEOWNERS output
# ------------------------------------------------------------------


class TestAnchor:
    def test_anchor_prepends_slash(self) -> None:
        assert anchor("lib/llm/") == "/lib/llm/"
        assert anchor("Cargo.toml") == "/Cargo.toml"

    def test_anchor_preserves_already_anchored(self) -> None:
        assert anchor("/lib/llm/") == "/lib/llm/"


# ------------------------------------------------------------------
# compute_resolution() -- end-to-end on a small synthetic spec + tree
# ------------------------------------------------------------------


class TestComputeResolution:
    def _spec(self) -> dict:
        return {
            "meta": {"catch_all": "@root"},
            "areas": [
                {
                    "label": "runtime",
                    "github_team": "@runtime",
                    "path_globs": ["lib/llm/"],
                },
                {
                    "label": "kvbm",
                    "github_team": "@kvbm",
                    "path_globs": [],
                },
                {
                    "label": "docs",
                    "github_team": "@docs",
                    "path_globs": ["docs/"],
                },
            ],
            "shared": [
                {"glob": "lib/llm/shared/", "owners": ["runtime", "kvbm"]},
            ],
            "advisory": [],
            "classify": {
                "keyword_rules": [{"match": "kvbm", "area": "kvbm"}],
                "filetype_rules": [
                    {"pattern": "*.md", "coowner": "docs", "advisory": True},
                ],
            },
        }

    def _tree(self) -> list[str]:
        return [
            "lib/llm/a.rs",
            "lib/llm/b.rs",
            "lib/llm/shared/x.rs",
            "lib/kvbm/foo.rs",  # auto-classify via keyword
            "docs/intro.md",
            "README.md",  # filetype_rule advisory only, falls to catch-all
        ]

    def test_explicit_paths_resolved(self) -> None:
        model = compute_resolution(self._spec(), self._tree())
        assert isinstance(model, ResolvedModel)
        # docs area unchanged
        docs = next(a for a in model.areas if a.label == "docs")
        assert "docs/" in docs.path_globs

    def test_auto_classify_promotes_new_dir(self) -> None:
        model = compute_resolution(self._spec(), self._tree())
        kvbm = next(a for a in model.areas if a.label == "kvbm")
        # lib/kvbm/foo.rs has no explicit area glob, but the kvbm keyword
        # rule should add a dir-prefix glob to the kvbm area.
        assert any("kvbm" in g for g in kvbm.path_globs)

    def test_catch_all_only_uncovered(self) -> None:
        model = compute_resolution(self._spec(), self._tree())
        # README.md only gets an advisory rule (non-blocking); should not
        # count as explicitly owned for the coverage gate.
        unmatched = model.unmatched_paths(self._tree())
        assert "README.md" in unmatched

    def test_shared_multi_owner_recorded(self) -> None:
        model = compute_resolution(self._spec(), self._tree())
        sh = [s for s in model.shared if s["glob"] == "lib/llm/shared/"]
        assert sh and sh[0]["owners"] == ["runtime", "kvbm"]

    def test_coverage_is_anchored_like_the_generator(self) -> None:
        # An area glob `README.md` is emitted anchored (`/README.md`), so the
        # coverage gate must not let a nested `foo/README.md` ride on it.
        spec = self._spec()
        spec["areas"][2]["path_globs"] = ["docs/", "README.md"]
        tree = self._tree() + ["foo/README.md"]
        model = compute_resolution(spec, tree)
        unmatched = model.unmatched_paths(tree)
        assert "README.md" not in unmatched
        assert "foo/README.md" in unmatched

    def test_filetype_rule_skips_files_outside_any_area(self) -> None:
        # A blocking filetype rule must not count as coverage for a tree
        # nobody owns -- the file stays catch-all-only so --strict flags it.
        spec = self._spec()
        spec["classify"]["filetype_rules"] = [
            {"pattern": "Dockerfile", "coowner": "docs", "advisory": False},
        ]
        tree = self._tree() + ["lib/llm/Dockerfile", "stray/Dockerfile"]
        model = compute_resolution(spec, tree)
        globs = {fs.glob for fs in model.filetype_shared}
        assert "lib/llm/Dockerfile" in globs  # enclosed by runtime: co-owned
        assert "stray/Dockerfile" not in globs  # no enclosing area: skipped
        assert "stray/Dockerfile" in model.unmatched_paths(tree)

    def test_keyword_coowner_rule_emits_shared_dir(self) -> None:
        # A keyword rule with only `coowner` co-owns matching dirs with the
        # enclosing area instead of being silently inert.
        spec = self._spec()
        spec["classify"]["keyword_rules"].append(
            {"match": "metrics", "coowner": "docs"}
        )
        tree = self._tree() + ["lib/llm/metrics/gauge.rs"]
        model = compute_resolution(spec, tree)
        assert {"glob": "lib/llm/metrics/", "owners": ["runtime", "docs"]} in (
            model.keyword_coowned
        )
        assert model.keyword_coowned[-1] in model.shared

    def test_keyword_coowner_defers_to_explicit_shared(self) -> None:
        # A hand-declared shared: entry for the same dir wins -- the keyword
        # rule must not emit a duplicate (or conflicting) row.
        spec = self._spec()
        spec["classify"]["keyword_rules"].append(
            {"match": "metrics", "coowner": "docs"}
        )
        spec["shared"].append(
            {"glob": "lib/llm/metrics/", "owners": ["runtime", "docs"]}
        )
        tree = self._tree() + ["lib/llm/metrics/gauge.rs"]
        model = compute_resolution(spec, tree)
        assert model.keyword_coowned == []
        rows = [s for s in model.shared if s["glob"] == "lib/llm/metrics/"]
        assert len(rows) == 1

    def test_keyword_coowner_skips_unowned_dirs(self) -> None:
        # No enclosing area -> no co-ownership row; the gate flags the dir.
        spec = self._spec()
        spec["classify"]["keyword_rules"].append(
            {"match": "metrics", "coowner": "docs"}
        )
        tree = self._tree() + ["orphan/metrics/gauge.rs"]
        model = compute_resolution(spec, tree)
        assert not any("orphan" in s["glob"] for s in model.keyword_coowned)
        assert "orphan/metrics/gauge.rs" in model.unmatched_paths(tree)


# ------------------------------------------------------------------
# TypedDict / dataclass surface
# ------------------------------------------------------------------


class TestTypedShapes:
    def test_area_typeddict_keys(self) -> None:
        a: Area = {
            "label": "x",
            "github_team": "@x",
            "path_globs": ["x/"],
        }
        assert a["label"] == "x"

    def test_shared_spec_keys(self) -> None:
        s: SharedSpec = {"glob": "x/", "owners": ["a", "b"]}
        assert s["glob"] == "x/"
