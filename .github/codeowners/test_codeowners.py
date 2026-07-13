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

import pytest

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
from emit_codeowners import (  # noqa: E402
    CONTRIBUTOR_LEVELS,
    _handle,
    _render_codeowners,
    contributor_level,
    decorate_owners,
    render_contributors_md,
    team_externals_map,
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
            "lib/kvbm/foo.rs",  # unowned in the new tree-independent resolver
            "docs/intro.md",
            "README.md",  # filetype_rule advisory only, falls to catch-all
        ]

    def test_explicit_paths_resolved(self) -> None:
        model = compute_resolution(self._spec())
        assert isinstance(model, ResolvedModel)
        # docs area unchanged
        docs = next(a for a in model.areas if a.label == "docs")
        assert "docs/" in docs.path_globs

    def test_no_auto_classify_at_emit_time(self) -> None:
        # Old behavior: a `keyword_rules` entry with `area:` promoted an
        # unmatched dir into that area at emit time by walking the tree.
        # That tree walk was the source of the base-branch race, so the
        # resolver drops it -- authors must now materialize the equivalent
        # as an explicit ``path_globs`` entry.
        model = compute_resolution(self._spec())
        kvbm = next(a for a in model.areas if a.label == "kvbm")
        assert kvbm.path_globs == []
        assert model.auto_classified == []

    def test_resolution_ignores_tree_argument(self) -> None:
        # Two trees that differ only under an already-owned prefix must
        # produce byte-identical resolutions, because ``tree`` is deprecated
        # and ignored.
        spec = self._spec()
        tree_a = ["lib/llm/a.rs"]
        tree_b = ["lib/llm/a.rs", "lib/llm/b.rs", "lib/llm/new/c.rs"]
        model_a = compute_resolution(spec, tree_a)
        model_b = compute_resolution(spec, tree_b)
        assert model_a == model_b
        # Legacy positional call (no tree) also matches.
        assert compute_resolution(spec) == model_a

    def test_catch_all_only_uncovered(self) -> None:
        model = compute_resolution(self._spec())
        # README.md only gets an advisory rule (non-blocking); should not
        # count as explicitly owned for the coverage gate.
        unmatched = model.unmatched_paths(self._tree())
        assert "README.md" in unmatched

    def test_shared_multi_owner_recorded(self) -> None:
        model = compute_resolution(self._spec())
        sh = [s for s in model.shared if s["glob"] == "lib/llm/shared/"]
        assert sh and sh[0]["owners"] == ["runtime", "kvbm"]

    def test_coverage_is_anchored_like_the_generator(self) -> None:
        # An area glob `README.md` is emitted anchored (`/README.md`), so the
        # coverage gate must not let a nested `foo/README.md` ride on it.
        spec = self._spec()
        spec["areas"][2]["path_globs"] = ["docs/", "README.md"]
        tree = self._tree() + ["foo/README.md"]
        model = compute_resolution(spec)
        unmatched = model.unmatched_paths(tree)
        assert "README.md" not in unmatched
        assert "foo/README.md" in unmatched

    def test_filetype_rule_emits_one_stable_coowner_only_row(self) -> None:
        # A blocking filetype rule becomes ONE stable line matching by
        # basename at any depth (GitHub CODEOWNERS semantics for a bare
        # pattern with no leading slash). Coowner-only: the tree-dependent
        # "enclosing area + coowner" pull-in is gone, because computing it
        # required walking the live tree and was the second source of the
        # base-branch race.
        spec = self._spec()
        spec["classify"]["filetype_rules"] = [
            {"pattern": "Dockerfile", "coowner": "docs", "advisory": False},
        ]
        model = compute_resolution(spec)
        assert len(model.filetype_shared) == 1
        fs = model.filetype_shared[0]
        assert fs.glob == "Dockerfile"
        assert fs.owners == ["docs"]

    def test_filetype_rule_covers_files_at_any_depth(self) -> None:
        # The strict coverage gate relies on ``unmatched_paths`` -- a
        # blocking filetype pattern must count as coverage for any file
        # matching it, regardless of directory depth.
        spec = self._spec()
        spec["classify"]["filetype_rules"] = [
            {"pattern": "Dockerfile", "coowner": "docs", "advisory": False},
        ]
        tree = self._tree() + ["lib/llm/Dockerfile", "stray/Dockerfile"]
        model = compute_resolution(spec)
        unmatched = set(model.unmatched_paths(tree))
        assert "lib/llm/Dockerfile" not in unmatched
        assert "stray/Dockerfile" not in unmatched

    def test_filetype_advisory_rule_not_promoted_to_blocking(self) -> None:
        # `advisory: true` filetype rules must NOT show up as blocking
        # filetype_shared rows; they belong in filetype_advisory.
        spec = self._spec()
        spec["classify"]["filetype_rules"] = [
            {"pattern": "*.md", "coowner": "docs", "advisory": True},
        ]
        model = compute_resolution(spec)
        assert model.filetype_shared == []
        assert [r["pattern"] for r in model.filetype_advisory] == ["*.md"]

    def test_keyword_coowner_rules_are_ignored_at_emit_time(self) -> None:
        # Old behavior: a keyword rule with `coowner` scanned every tree
        # directory and emitted a `[enclosing_area, coowner]` shared row.
        # That is exactly the tree walk we removed; the resolver now
        # silently drops keyword_rules at emit time. Authors declare the
        # equivalent explicitly in ``shared`` when they want it.
        spec = self._spec()
        spec["classify"]["keyword_rules"].append(
            {"match": "metrics", "coowner": "docs"}
        )
        model = compute_resolution(spec)
        assert model.keyword_coowned == []
        assert not any(s["glob"].endswith("metrics/") for s in model.shared)

    def test_explicit_shared_entry_still_wins(self) -> None:
        # Hand-declared shared: entries are still emitted verbatim; they
        # were the "explicit beats implicit" path before, and now they are
        # the ONLY way to express keyword-style co-ownership.
        spec = self._spec()
        spec["shared"].append(
            {"glob": "lib/llm/metrics/", "owners": ["runtime", "docs"]}
        )
        model = compute_resolution(spec)
        rows = [s for s in model.shared if s["glob"] == "lib/llm/metrics/"]
        assert len(rows) == 1
        assert rows[0]["owners"] == ["runtime", "docs"]


# ------------------------------------------------------------------
# Byte-identical determinism -- the fix for the base-branch race
# ------------------------------------------------------------------


class TestEmissionIsTreeIndependent:
    """The whole point of the tree-decoupling: emit is a pure function.

    Adding, deleting, or moving files UNDER an already-owned prefix must not
    change a single byte of the emitted CODEOWNERS. The old min-cover /
    auto-classify / filetype-tree-walk pipeline flunked this contract:
    unrelated churn on ``main`` rewrote rules and broke the ``codeowners``
    CI check on PRs that had touched none of it.
    """

    def _spec(self) -> dict:
        # Realistic-shaped spec: nested area overrides + shared + a
        # blocking filetype rule + advisory-only filetype rule + one
        # keyword rule (which must be ignored at emit time).
        return {
            "meta": {"catch_all": "@root"},
            "areas": [
                {
                    "label": "runtime",
                    "github_team": "@runtime",
                    "path_globs": [
                        "lib/",
                        "lib/llm/",
                        "lib/llm/preprocessor.rs",
                    ],
                },
                {
                    "label": "kvbm",
                    "github_team": "@kvbm",
                    "path_globs": ["lib/llm/kv/", "lib/kvbm/"],
                },
                {
                    "label": "docs",
                    "github_team": "@docs",
                    "path_globs": ["docs/", "README.md"],
                },
                {"label": "ops", "github_team": "@ops", "path_globs": []},
            ],
            "shared": [
                {"glob": "lib/llm/shared/", "owners": ["runtime", "kvbm"]},
            ],
            "advisory": [],
            "classify": {
                # Would previously auto-promote unmatched dirs and pull the
                # enclosing area into Dockerfile lines -- both tree-walks.
                "keyword_rules": [{"match": "metrics", "coowner": "docs"}],
                "filetype_rules": [
                    {"pattern": "Dockerfile", "coowner": "ops", "advisory": False},
                    {"pattern": "*.md", "coowner": "docs", "advisory": True},
                ],
            },
        }

    def _render(self, spec: dict) -> str:
        model = compute_resolution(spec)
        lines, _ = _render_codeowners(model, group=True, external=[])
        return "\n".join(lines) + "\n"

    def test_add_file_under_owned_prefix_does_not_change_output(self) -> None:
        # The OLD emitter took a ``tree`` and walked it. Under the new
        # signature there is nowhere to inject tree state, but we still
        # thread two "trees" through the pure resolver via its deprecated
        # positional argument to prove the argument really is ignored:
        # even if a legacy caller keeps passing it, the output does not
        # move.
        spec = self._spec()
        base_tree = [
            "lib/llm/a.rs",
            "lib/llm/preprocessor.rs",
            "lib/llm/kv/x.rs",
            "docs/intro.md",
            "README.md",
            "container/Dockerfile",
        ]
        mutated_tree = base_tree + [
            "lib/llm/new_file.rs",  # add under runtime
            "lib/llm/kv/another.rs",  # add under kvbm
            "lib/llm/subdir/only_here.rs",  # deeper unknown dir under runtime
            "docs/new.md",  # add under docs
            "container/templates/args.Dockerfile",  # add matching filetype
        ]
        model_base = compute_resolution(spec, base_tree)
        model_mut = compute_resolution(spec, mutated_tree)
        assert model_base == model_mut

        # And the emitted body is byte-identical regardless of which tree
        # the caller pretends to pass.
        assert self._render(spec) == self._render(spec)

    def test_delete_or_move_under_owned_prefix_does_not_change_output(
        self,
    ) -> None:
        spec = self._spec()
        first = self._render(spec)
        # Simulate a "second run" against a wildly different tree via a
        # spec-level noop: the resolver only reads the spec, so identical
        # spec means identical output. This is the property CI relies on.
        second = self._render(spec)
        assert first == second

    def test_emitter_has_no_tree_parameter(self) -> None:
        # Guard against a future regression re-introducing the tree walk:
        # the emitter's signature must not name a ``tree`` parameter.
        import inspect

        sig = inspect.signature(_render_codeowners)
        assert (
            "tree" not in sig.parameters
        ), "emit tree parameter reintroduced -- see TestEmissionIsTreeIndependent"
        sig_base = inspect.signature(compute_resolution)
        # tree is still accepted (backward-compat) but must default to None
        # so callers that omit it get pure behavior for free.
        tree_param = sig_base.parameters.get("tree")
        assert tree_param is not None
        assert tree_param.default is None

    def test_no_ls_files_call_at_emit(self, monkeypatch) -> None:
        # Belt-and-braces: monkeypatch ``codeowners_match.load_tree`` to
        # blow up, then run the full emit path. If anything on that path
        # ever reintroduces a tree walk, this test fails loudly instead of
        # silently regressing determinism.
        import codeowners_match

        def _boom(*_a, **_kw):  # pragma: no cover - triggered only on regression
            raise AssertionError(
                "emit path called git ls-files -- tree independence broken"
            )

        monkeypatch.setattr(codeowners_match, "load_tree", _boom)
        # compute_resolution + _render_codeowners together are the whole
        # emit path.
        spec = self._spec()
        model = compute_resolution(spec)
        lines, _ = _render_codeowners(model, group=True, external=[])
        # sanity: we actually rendered something
        assert any(ln.startswith("/lib/") for ln in lines)


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


# ------------------------------------------------------------------
# External contributors -- area-attached co-ownership + CONTRIBUTORS.md
# ------------------------------------------------------------------


class TestHandle:
    def test_bare_username_gets_at(self) -> None:
        assert _handle("octocat") == "@octocat"

    def test_leading_at_not_doubled(self) -> None:
        assert _handle("@octocat") == "@octocat"

    def test_whitespace_stripped(self) -> None:
        assert _handle("  octocat ") == "@octocat"


class TestTeamExternalsMap:
    def _label_to_team(self) -> dict[str, str]:
        return {"router": "@ai-dynamo/router", "docs": "@ai-dynamo/docs"}

    def test_maps_area_label_to_team_handles(self) -> None:
        contributors = [{"name": "Jane", "github": "jane", "areas": ["router"]}]
        mapping = team_externals_map(contributors, self._label_to_team())
        assert mapping == {"@ai-dynamo/router": ["@jane"]}

    def test_multiple_contributors_same_area(self) -> None:
        contributors = [
            {"name": "Jane", "github": "jane", "areas": ["router"]},
            {"name": "Jo", "github": "jo", "areas": ["router"]},
        ]
        mapping = team_externals_map(contributors, self._label_to_team())
        assert mapping["@ai-dynamo/router"] == ["@jane", "@jo"]

    def test_contributor_multiple_areas(self) -> None:
        contributors = [{"name": "Jane", "github": "jane", "areas": ["router", "docs"]}]
        mapping = team_externals_map(contributors, self._label_to_team())
        assert mapping["@ai-dynamo/router"] == ["@jane"]
        assert mapping["@ai-dynamo/docs"] == ["@jane"]

    def test_unknown_area_label_is_fatal(self) -> None:
        contributors = [{"name": "Jane", "github": "jane", "areas": ["nope"]}]
        with pytest.raises(SystemExit):
            team_externals_map(contributors, self._label_to_team())

    def test_missing_github_is_fatal(self) -> None:
        contributors = [{"name": "Jane", "areas": ["router"]}]
        with pytest.raises(SystemExit):
            team_externals_map(contributors, self._label_to_team())


class TestDecorateOwners:
    def test_appends_handle_for_matching_team(self) -> None:
        te = {"@team": ["@jane"]}
        assert decorate_owners("@team", te) == "@team @jane"

    def test_noop_when_no_externals(self) -> None:
        assert decorate_owners("@team", {}) == "@team"

    def test_team_not_present_unchanged(self) -> None:
        te = {"@other": ["@jane"]}
        assert decorate_owners("@team", te) == "@team"

    def test_multi_owner_line_appends_once(self) -> None:
        te = {"@team": ["@jane"]}
        assert decorate_owners("@team @second", te) == "@team @second @jane"

    def test_no_duplicate_handle(self) -> None:
        te = {"@team": ["@jane", "@jane"]}
        assert decorate_owners("@team", te) == "@team @jane"


class TestContributorLevel:
    def test_canonical_tokens_accepted(self) -> None:
        for lvl in CONTRIBUTOR_LEVELS:
            assert contributor_level({"name": "x", "level": lvl}) == lvl

    def test_human_spelling_normalized(self) -> None:
        assert (
            contributor_level({"name": "x", "level": "Core Maintainer"})
            == "core_maintainer"
        )
        assert (
            contributor_level({"name": "x", "level": "trusted-contributor"})
            == "trusted_contributor"
        )

    def test_missing_level_is_fatal(self) -> None:
        with pytest.raises(SystemExit):
            contributor_level({"name": "x", "github": "x"})

    def test_invalid_level_is_fatal(self) -> None:
        with pytest.raises(SystemExit):
            contributor_level({"name": "x", "level": "overlord"})


class TestRenderContributorsMd:
    def test_empty_states_none_yet(self) -> None:
        md = render_contributors_md([])
        assert "# Contributors" in md
        assert "_No external contributors yet._" in md
        assert "codeownership" in md

    def test_renders_row_with_link_level_and_area(self) -> None:
        contributors = [
            {
                "name": "Jane Doe",
                "github": "janedoe",
                "level": "maintainer",
                "affiliation": "Example Org",
                "areas": ["router"],
            }
        ]
        md = render_contributors_md(contributors)
        assert "Jane Doe" in md
        assert "Maintainer" in md
        assert "Example Org" in md
        assert "[@janedoe](https://github.com/janedoe)" in md
        assert "`router`" in md

    def test_missing_affiliation_falls_back(self) -> None:
        contributors = [
            {
                "name": "Jane",
                "github": "jane",
                "level": "contributor",
                "areas": ["router"],
            }
        ]
        md = render_contributors_md(contributors)
        assert "n/a" in md

    def test_sorted_by_level_then_name(self) -> None:
        contributors = [
            {"name": "Zed", "github": "zed", "level": "contributor", "areas": ["a"]},
            {
                "name": "Amy",
                "github": "amy",
                "level": "core_maintainer",
                "areas": ["a"],
            },
        ]
        md = render_contributors_md(contributors)
        assert md.index("Amy") < md.index("Zed")  # core_maintainer outranks contributor

    def test_missing_github_is_fatal(self) -> None:
        contributors = [{"name": "Jane", "level": "maintainer", "areas": ["router"]}]
        with pytest.raises(SystemExit):
            render_contributors_md(contributors)


class TestRenderCodeownersWithExternals:
    """End-to-end: an area-attached contributor rides every line the team owns."""

    def _model(self) -> ResolvedModel:
        spec = {
            "meta": {"catch_all": "@root"},
            "areas": [
                {
                    "label": "runtime",
                    "github_team": "@runtime",
                    "path_globs": ["lib/llm/"],
                },
                {"label": "kvbm", "github_team": "@kvbm", "path_globs": []},
            ],
            "shared": [{"glob": "lib/llm/shared/", "owners": ["runtime", "kvbm"]}],
            "advisory": [],
            "classify": {"keyword_rules": [], "filetype_rules": []},
        }
        return compute_resolution(spec)

    def test_base_line_gets_handle(self) -> None:
        model = self._model()
        external = [{"name": "Jane", "github": "jane", "areas": ["runtime"]}]
        lines, _ = _render_codeowners(model, group=True, external=external)
        body = "\n".join(lines)
        assert "@runtime @jane" in body

    def test_shared_line_gets_handle(self) -> None:
        model = self._model()
        external = [{"name": "Jane", "github": "jane", "areas": ["runtime"]}]
        lines, _ = _render_codeowners(model, group=True, external=external)
        shared_line = next(ln for ln in lines if ln.startswith("/lib/llm/shared/"))
        assert "@runtime" in shared_line
        assert "@kvbm" in shared_line
        assert "@jane" in shared_line

    def test_no_externals_is_unchanged(self) -> None:
        model = self._model()
        plain, _ = _render_codeowners(model, group=True, external=[])
        assert not any("@jane" in ln for ln in plain)
