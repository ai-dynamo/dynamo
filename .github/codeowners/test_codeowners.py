"""Unit tests for the shared codeowners matching + resolution module.

These pin down the CODEOWNERS pipeline's shared semantics and deterministic
policy-only emission:

  - `match(pattern, path)` -- canonical CODEOWNERS-style matcher used by build
    coverage, emit routing, and who_owns lookups.
  - `minimal_cover(file_team, catch_all)` -- the recursive min-cost cover that
    turns a per-file owner map into the smallest set of last-match base rules
    for legacy callers (the emitter no longer uses it).
  - `compute_resolution(spec)` + `_render_codeowners(...)` -- pure policy
    resolution, explicit precedence, and byte-identical output across trees.

If either drifts, the tests catch it before the generated CODEOWNERS goes wrong.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# Allow `import codeowners_match` when pytest runs from the repo root.
sys.path.insert(0, str(Path(__file__).parent))

from build_codeowners import (  # noqa: E402
    CoverageGate,
    _dead_patterns,
    is_policy_change,
    ownership_contract_violations,
    shared_additivity_violations,
    split_coverage,
    strict_failure,
)
from codeowners_match import (  # noqa: E402
    Area,
    ResolvedModel,
    SharedSpec,
    anchor,
    changed_paths,
    compute_resolution,
    match,
    minimal_cover,
    parse_codeowners,
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

    def test_double_star_slash_matches_zero_or_more_directories(self) -> None:
        pattern = "/recipes/**/vllm/**"
        assert match(pattern, "recipes/vllm/deploy.yaml")
        assert match(pattern, "recipes/nested/vllm/deploy.yaml")
        assert match(pattern, "recipes/a/b/vllm/deploy.yaml")
        assert not match(pattern, "recipes/sglang/deploy.yaml")

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
            "classify": {
                "keyword_rules": [],
                "filetype_rules": [],
            },
        }

    def _tree(self) -> list[str]:
        return [
            "lib/llm/a.rs",
            "lib/llm/b.rs",
            "lib/llm/shared/x.rs",
            "lib/kvbm/foo.rs",  # unowned in the new tree-independent resolver
            "docs/intro.md",
            "README.md",  # no filetype rule covers it; falls to catch-all
        ]

    def test_explicit_paths_resolved(self) -> None:
        model = compute_resolution(self._spec())
        assert isinstance(model, ResolvedModel)
        # docs area unchanged
        docs = next(a for a in model.areas if a.label == "docs")
        assert "docs/" in docs.path_globs

    @pytest.mark.parametrize(
        "rule",
        [
            {"match": "kvbm", "area": "kvbm"},
            {"match": "metrics", "coowner": "docs"},
        ],
    )
    def test_legacy_keyword_rules_are_rejected(self, rule: dict) -> None:
        # Keyword auto-classification/co-ownership required a live tree.
        # Reject stale configuration instead of silently pretending it still
        # affects the pure policy resolver.
        spec = self._spec()
        spec["classify"]["keyword_rules"] = [rule]
        with pytest.raises(SystemExit, match="keyword_rules is no longer supported"):
            compute_resolution(spec)

    @pytest.mark.parametrize(
        "value", [[{"glob": "docs/", "owners": ["docs"]}], [], None, False, {}]
    )
    def test_legacy_advisory_block_is_rejected(self, value: object) -> None:
        # Advisory routing is gone. A silently ignored block would read as
        # non-blocking routing that is in fact doing nothing at all.
        # Keyed on presence, not truthiness: `advisory: []` is the shape the
        # fixtures carried and `advisory: false` the shape areas.yaml did, so
        # a truthiness check would wave through exactly the leftovers most
        # likely to exist.
        spec = self._spec()
        spec["advisory"] = value
        with pytest.raises(SystemExit, match="advisory is no longer supported"):
            compute_resolution(spec)

    def test_legacy_advisory_on_shared_entry_is_rejected(self) -> None:
        # shared: is where the migration message points, and it took the same
        # {glob, owners} shape advisory did -- so pasting a block across is the
        # natural move. Unguarded, the entry its author marked non-blocking
        # becomes a blocking required approver.
        spec = self._spec()
        spec["shared"] = [
            {"glob": "docs/design/", "owners": ["docs"], "advisory": True}
        ]
        with pytest.raises(SystemExit, match="stale 'advisory' key"):
            compute_resolution(spec)

    @pytest.mark.parametrize("flag", [True, False])
    def test_legacy_advisory_filetype_key_is_rejected(self, flag: bool) -> None:
        # Worse than ignored: dropping the key would promote the rule to a
        # *blocking* owner, the opposite of what its author asked for. Both
        # values are rejected, since 'advisory: false' is equally stale.
        spec = self._spec()
        spec["classify"]["filetype_rules"] = [
            {"pattern": "*.md", "coowner": "docs", "advisory": flag}
        ]
        with pytest.raises(SystemExit, match="no longer supports"):
            compute_resolution(spec)

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
        # No rule covers README.md, so it must not count as explicitly
        # owned for the coverage gate.
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
            {"pattern": "Dockerfile", "coowner": "docs"},
        ]
        model = compute_resolution(spec)
        assert len(model.filetype_shared) == 1
        fs = model.filetype_shared[0]
        assert fs.glob == "Dockerfile"
        assert fs.owners == ["docs"]

    @pytest.mark.parametrize(
        "rule",
        [
            {"coowner": "docs"},
            {"pattern": "Dockerfile"},
            {"pattern": "Dockerfile", "coowner": "typoed-owner"},
            {"pattern": "Docker file", "coowner": "docs"},
            {"pattern": "Dockerfile", "coowner": "@org/team extra"},
            {"pattern": "Dockerfile", "coowner": "owner @example.com"},
            ["not", "a", "mapping"],
        ],
    )
    def test_blocking_filetype_rule_requires_pattern_and_coowner(
        self, rule: object
    ) -> None:
        spec = self._spec()
        spec["classify"]["filetype_rules"] = [rule]
        with pytest.raises(SystemExit, match="filetype_rules entry"):
            compute_resolution(spec)

    def test_filetype_rules_accept_explicit_raw_principals(self) -> None:
        spec = self._spec()
        spec["classify"]["filetype_rules"] = [
            {"pattern": "*.owned", "coowner": "@org/team"},
            {"pattern": "*.reviewed", "coowner": "owner@example.com"},
        ]
        model = compute_resolution(spec)
        assert model.filetype_shared[0].owners == ["@org/team"]
        assert model.filetype_shared[1].owners == ["owner@example.com"]

    def test_filetype_rule_covers_files_at_any_depth(self) -> None:
        # The strict coverage gate relies on ``unmatched_paths`` -- a
        # blocking filetype pattern must count as coverage for any file
        # matching it, regardless of directory depth.
        spec = self._spec()
        spec["classify"]["filetype_rules"] = [
            {"pattern": "Dockerfile", "coowner": "docs"},
        ]
        tree = self._tree() + ["lib/llm/Dockerfile", "stray/Dockerfile"]
        model = compute_resolution(spec)
        unmatched = set(model.unmatched_paths(tree))
        assert "lib/llm/Dockerfile" not in unmatched
        assert "stray/Dockerfile" not in unmatched

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

    def test_shared_owner_lists_are_stably_deduplicated(self) -> None:
        spec = self._spec()
        spec["shared"].append(
            {
                "glob": "lib/llm/metrics/",
                "owners": ["runtime", "docs", "kvbm", "runtime"],
            }
        )
        model = compute_resolution(spec)
        row = next(s for s in model.shared if s["glob"] == "lib/llm/metrics/")
        assert row == {
            "glob": "lib/llm/metrics/",
            "owners": ["runtime", "docs", "kvbm"],
        }

    @pytest.mark.parametrize("section", ["shared", "required_owners"])
    def test_removed_inherits_key_is_rejected_with_guidance(self, section: str) -> None:
        # The inherits/owners split is gone: an owner rule declares its
        # complete owner set under 'owners'. Stale authoring
        # muscle-memory gets a pointed error.
        spec = self._spec()
        spec[section] = [
            {
                "glob": "lib/llm/metrics/",
                "inherits": ["runtime"],
                "owners": ["kvbm"],
            }
        ]
        with pytest.raises(SystemExit, match="removed key 'inherits'"):
            compute_resolution(spec)

    @pytest.mark.parametrize(
        "rule",
        [
            {"owners": ["runtime"]},
            {"glob": "lib/llm/metrics/", "owners": "runtime"},
            {"glob": "lib/llm/metrics/", "owners": []},
            {"glob": "lib/llm/metrics/", "owners": ["typoed-owner"]},
            {"glob": "lib/llm/ metrics/", "owners": ["runtime"]},
            {"glob": "lib/llm/metrics/", "owners": ["@org/team extra"]},
            {"glob": "lib/llm/metrics/", "owners": ["owner @example.com"]},
            {"glob": "lib/llm/metrics/", "owners": ["runtime docs"]},
            {"glob": "lib/llm/metrics/", "owners": [["not-hashable"]]},
            ["not", "a", "mapping"],
        ],
    )
    def test_shared_entries_require_a_glob_and_effective_owner_list(
        self, rule: object
    ) -> None:
        spec = self._spec()
        spec["shared"] = [rule]
        with pytest.raises(SystemExit, match="shared entry"):
            compute_resolution(spec)

    def test_shared_entries_accept_explicit_raw_principals(self) -> None:
        spec = self._spec()
        spec["shared"] = [
            {
                "glob": "lib/llm/metrics/",
                "owners": ["runtime", "@org/team", "owner@example.com"],
            }
        ]
        model = compute_resolution(spec)
        assert model.shared[0]["owners"] == [
            "runtime",
            "@org/team",
            "owner@example.com",
        ]

    @pytest.mark.parametrize("section", ["shared", "required_owners"])
    @pytest.mark.parametrize(
        "rule",
        [
            {"glob": "lib/llm/ metrics/", "owners": ["runtime"]},
            {"glob": "lib/llm/metrics/", "owners": ["@org/team extra"]},
            {"glob": "lib/llm/metrics/", "owners": ["owner @example.com"]},
            {"glob": "lib/llm/metrics/", "owners": ["runtime docs"]},
        ],
    )
    def test_owner_rule_sections_reject_whitespace_tokens(
        self, section: str, rule: dict
    ) -> None:
        spec = self._spec()
        spec[section] = [rule]
        with pytest.raises(SystemExit, match=f"{section} entry"):
            compute_resolution(spec)


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
        # blocking filetype rule.
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
            "classify": {
                "keyword_rules": [],
                "filetype_rules": [
                    {"pattern": "Dockerfile", "coowner": "ops"},
                ],
            },
        }

    def _render(self, spec: dict, tree: list[str] | None = None) -> str:
        model = compute_resolution(spec, tree)
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
        assert compute_resolution(spec, base_tree) == compute_resolution(
            spec, mutated_tree
        )
        assert self._render(spec, base_tree) == self._render(spec, mutated_tree)

    def test_delete_or_move_under_owned_prefix_does_not_change_output(
        self,
    ) -> None:
        # The delete + move half of the pure-emit contract. Prove that
        # (a) removing tracked files from under an owned prefix and
        # (b) reshuffling their paths do not change the resolved model or
        # the rendered body -- both are pure functions of the spec.
        spec = self._spec()
        base_tree = [
            "lib/llm/a.rs",
            "lib/llm/preprocessor.rs",
            "lib/llm/kv/x.rs",
            "lib/llm/kv/y.rs",
            "lib/llm/shared/z.rs",
            "docs/intro.md",
            "docs/api/ref.md",
            "README.md",
            "container/Dockerfile",
            "container/templates/args.Dockerfile",
        ]
        deleted_tree = [
            # dropped: lib/llm/a.rs, lib/llm/kv/y.rs, docs/api/ref.md,
            # container/templates/args.Dockerfile.
            "lib/llm/preprocessor.rs",
            "lib/llm/kv/x.rs",
            "lib/llm/shared/z.rs",
            "docs/intro.md",
            "README.md",
            "container/Dockerfile",
        ]
        moved_tree = [
            "lib/llm/preprocessor.rs",  # unchanged
            "lib/llm/renamed_a.rs",  # moved from lib/llm/a.rs
            "lib/llm/kv/renamed_x.rs",  # moved from lib/llm/kv/x.rs
            "lib/llm/kv/y.rs",
            "lib/llm/shared/moved_z.rs",  # moved within owned prefix
            "docs/intro_renamed.md",  # moved within docs
            "docs/api/ref.md",
            "README.md",
            "deploy/Dockerfile",  # moved from container/Dockerfile
            "deploy/templates/args.Dockerfile",  # moved from container/
        ]
        model_base = compute_resolution(spec, base_tree)
        assert model_base == compute_resolution(spec, deleted_tree)
        assert model_base == compute_resolution(spec, moved_tree)
        # And the emitted body is byte-identical: the render path never
        # reads the tree, so the three "runs" produce the same file even
        # though the underlying trees differ wildly.
        rendered = self._render(spec, base_tree)
        assert rendered == self._render(spec, deleted_tree)
        assert rendered == self._render(spec, moved_tree)

    def test_emitter_has_no_tree_parameter(self) -> None:
        # Guard against a future regression re-introducing the tree walk:
        # the emitter's signature must not name a ``tree`` parameter.
        import inspect

        sig = inspect.signature(_render_codeowners)
        assert "tree" not in sig.parameters
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

    def test_explicit_path_rules_win_over_filetype_defaults(self) -> None:
        spec = self._spec()
        spec["areas"].append(
            {
                "label": "xpu",
                "github_team": "@xpu",
                "path_globs": ["lib/llm/Dockerfile"],
            }
        )
        spec["shared"].append({"glob": "lib/llm/shared/", "owners": ["runtime", "xpu"]})

        rules = parse_codeowners(self._render(spec))
        assert resolve_owners(rules, "other/Dockerfile") == ["@ops"]
        assert resolve_owners(rules, "lib/llm/Dockerfile") == ["@xpu"]
        assert resolve_owners(rules, "lib/llm/shared/Dockerfile") == [
            "@runtime",
            "@xpu",
        ]

    def test_shared_restatement_makes_filetype_override_additive(self) -> None:
        # A shared row lists its COMPLETE owner set: the retained enclosing
        # owners first, then the added ones. The row replaces the file-type
        # default under last-match, so restating is what keeps it additive.
        spec = self._spec()
        spec["shared"].append(
            {
                "glob": "lib/llm/Dockerfile",
                "owners": ["runtime", "ops", "docs"],
            }
        )

        rules = parse_codeowners(self._render(spec))
        assert resolve_owners(rules, "lib/llm/Dockerfile") == [
            "@runtime",
            "@ops",
            "@docs",
        ]

    def test_overlapping_filetype_rules_preserve_declaration_order(self) -> None:
        spec = self._spec()
        spec["classify"]["filetype_rules"] = [
            {"pattern": "*Dockerfile*", "coowner": "ops"},
            {"pattern": "Dockerfile", "coowner": "docs"},
        ]

        rules = parse_codeowners(self._render(spec))
        assert resolve_owners(rules, "nested/Dockerfile") == ["@docs"]


# ------------------------------------------------------------------
# split_coverage() -- diff-aware strict gate partitioning
# ------------------------------------------------------------------


class TestSplitCoverage:
    def test_full_tree_mode_blocks_every_unowned(self) -> None:
        # Default (changed is None): whole-tree strict blocks on ANY unowned
        # path -- the scheduled/maintenance 100%-coverage assertion.
        gate = split_coverage(["a/x", "b/y"], None)
        assert isinstance(gate, CoverageGate)
        assert gate.blocking == ["a/x", "b/y"]
        assert gate.warnings == []

    def test_diff_aware_ignores_inherited_base_gap(self) -> None:
        # A catch-all-only path the PR did NOT touch only warns; it never
        # fails the gate. This is the base-churn race being closed.
        gate = split_coverage(["base_only/x"], changed=["owned/new.py"])
        assert gate.blocking == []
        assert gate.warnings == ["base_only/x"]

    def test_diff_aware_blocks_pr_introduced_gap(self) -> None:
        # A catch-all-only path the PR introduced/touched still blocks: the
        # PR's own surface must be 100% owned.
        gate = split_coverage(["newdir/z"], changed=["newdir/z"])
        assert gate.blocking == ["newdir/z"]
        assert gate.warnings == []

    def test_diff_aware_mixed_surface(self) -> None:
        gate = split_coverage(
            ["base_only/x", "newdir/z"], changed=["newdir/z", "owned/ok.py"]
        )
        assert gate.blocking == ["newdir/z"]
        assert gate.warnings == ["base_only/x"]


class TestOwnershipContracts:
    def _spec(self) -> dict:
        return {
            "meta": {"catch_all": "@root"},
            "areas": [
                {
                    "label": "runtime",
                    "github_team": "@runtime",
                    "path_globs": ["lib/"],
                },
                {
                    "label": "docs",
                    "github_team": "@docs",
                    "path_globs": [],
                },
            ],
            "shared": [
                {"glob": "lib/", "owners": ["runtime", "docs"]},
            ],
        }

    def test_no_violation_when_declared_owners_survive(self) -> None:
        model = compute_resolution(self._spec())
        assert ownership_contract_violations(model, ["lib/a.rs"]) == []

    def test_required_owner_is_validation_only_and_fail_closed(self) -> None:
        spec = self._spec()
        spec["shared"] = []
        spec["required_owners"] = [
            {"glob": "lib/", "owners": ["docs"]},
        ]
        model = compute_resolution(spec)
        lines, _ = _render_codeowners(model, group=True, external=[])
        rules = parse_codeowners("\n".join(lines))

        assert resolve_owners(rules, "lib/a.rs") == ["@runtime"]
        violations = ownership_contract_violations(model, ["lib/a.rs"])
        assert len(violations) == 1
        assert violations[0].glob == "lib/"
        assert violations[0].missing == ("@docs",)

    def test_shared_rule_is_not_a_tree_level_contract(self) -> None:
        # The tree-level contract check enforces only required_owners and
        # blocking file-type declarations; a shared drop is a POLICY-shape
        # problem, not a per-file contract violation.
        spec = self._spec()
        spec["shared"].append({"glob": "lib/private/", "owners": ["runtime"]})
        model = compute_resolution(spec)

        assert ownership_contract_violations(model, ["lib/private/a.rs"]) == []

    def test_required_owner_blocks_override(self) -> None:
        # required_owners are hard contracts; a more-specific rule that drops
        # a declared owner is caught even when shared would not be.
        spec = self._spec()
        spec["required_owners"] = [{"glob": "lib/", "owners": ["docs"]}]
        spec["shared"] = [{"glob": "lib/private/", "owners": ["runtime"]}]
        model = compute_resolution(spec)

        violations = ownership_contract_violations(model, ["lib/private/a.rs"])

        assert len(violations) == 1
        assert violations[0].glob == "lib/"
        assert violations[0].path == "lib/private/a.rs"
        assert violations[0].missing == ("@docs",)
        assert violations[0].actual == ("@runtime",)

    def test_removing_a_required_owner_contract_lifts_the_requirement(self) -> None:
        # "What happens if I remove this section in a PR?" -- the contract is
        # policy, not history: deleting the entry deletes the requirement, on
        # purpose (an un-removable requirement could never be retired). The
        # protection is WHERE the edit happens, not the entry itself:
        # areas.yaml edits are policy changes, judged full-tree and reviewed
        # by the ops team that owns .github/codeowners/, so lifting a
        # contract is a visible, owned policy decision -- never a side
        # effect of an unrelated PR.
        spec = self._spec()
        spec["required_owners"] = [{"glob": "lib/", "owners": ["docs"]}]
        spec["shared"] = [{"glob": "lib/private/", "owners": ["runtime"]}]
        model = compute_resolution(spec)
        assert ownership_contract_violations(model, ["lib/private/a.rs"])

        del spec["required_owners"]
        model = compute_resolution(spec)
        assert ownership_contract_violations(model, ["lib/private/a.rs"]) == []

    def test_filetype_owner_cannot_be_silently_removed(self) -> None:
        spec = self._spec()
        spec["areas"].append({"label": "ops", "github_team": "@ops", "path_globs": []})
        spec["classify"] = {
            "filetype_rules": [{"pattern": "*Dockerfile*", "coowner": "ops"}]
        }
        spec["shared"].append({"glob": "lib/private/", "owners": ["runtime", "docs"]})
        model = compute_resolution(spec)

        violations = ownership_contract_violations(model, ["lib/private/Dockerfile"])

        assert len(violations) == 1
        assert violations[0].glob == "*Dockerfile*"
        assert violations[0].missing == ("@ops",)

    def test_strict_gate_fails_on_ownership_loss(self) -> None:
        # shared alone has no contract: overrides don't trigger the gate
        model = compute_resolution(self._spec())
        violations = ownership_contract_violations(model, ["lib/private/a.rs"])
        message = strict_failure(
            True, CoverageGate(blocking=[], warnings=[]), None, violations, [], None
        )
        assert message is None

        # required_owners + a more-specific override does trigger the gate
        model.required_owners.append({"glob": "lib/", "owners": ["docs"]})
        model.shared.append({"glob": "lib/private/", "owners": ["runtime"]})
        violations = ownership_contract_violations(model, ["lib/private/a.rs"])
        message = strict_failure(
            True, CoverageGate(blocking=[], warnings=[]), None, violations, [], None
        )
        assert message and "lost declared owners" in message


class TestSharedAdditivity:
    """The guard the removed ``inherits`` key promised.

    A shared row replaces the row it sits after, so it must restate every
    owner it means to keep. Enclosure is resolved rather than tiered: the
    comparison is against whoever actually owns the path just before this
    row, however many levels up that was decided.
    """

    def _spec(self) -> dict:
        # runtime owns lib/, frontend takes over lib/protocols/, and a shared
        # row co-owns a directory two levels below the ancestor. This is the
        # ancestor-not-parent shape: the owner in force at the shared path is
        # frontend, not runtime.
        return {
            "meta": {"catch_all": "@root"},
            "areas": [
                {
                    "label": "runtime",
                    "github_team": "@runtime",
                    "path_globs": ["lib/"],
                },
                {
                    "label": "frontend",
                    "github_team": "@frontend",
                    "path_globs": ["lib/protocols/"],
                },
                {
                    "label": "multimodal",
                    "github_team": "@multimodal",
                    "path_globs": [],
                },
            ],
            "shared": [
                {
                    "glob": "lib/protocols/audios/",
                    "owners": ["frontend", "multimodal"],
                },
            ],
        }

    def test_restating_the_effective_owner_passes(self) -> None:
        # Measured against frontend (the intermediate override), not runtime
        # (the ancestor) -- so listing frontend + multimodal is complete.
        model = compute_resolution(self._spec())

        assert shared_additivity_violations(model, ["lib/protocols/audios/a.rs"]) == []

    def test_dropping_the_effective_owner_is_caught(self) -> None:
        spec = self._spec()
        spec["shared"][0]["owners"] = ["multimodal"]
        model = compute_resolution(spec)

        violations = shared_additivity_violations(model, ["lib/protocols/audios/a.rs"])

        assert len(violations) == 1
        assert violations[0].glob == "lib/protocols/audios/"
        assert violations[0].missing == ("@frontend",)
        assert violations[0].declared == ("@multimodal",)

    def test_ancestor_owner_replaced_upstream_is_not_demanded(self) -> None:
        # runtime stopped owning this path at lib/protocols/, by an ordinary
        # area override. The shared row is not what dropped it, so it is not
        # required to restate it.
        model = compute_resolution(self._spec())

        violations = shared_additivity_violations(model, ["lib/protocols/audios/a.rs"])

        assert all("@runtime" not in v.missing for v in violations)

    def test_catch_all_is_not_an_enclosing_owner(self) -> None:
        # A shared row over a tree no area claims sits directly on the
        # catch-all. Demanding it restate the catch-all team would fire on
        # every such rule (22 real paths under deploy/power-agent/).
        spec = self._spec()
        spec["shared"].append({"glob": "unclaimed/", "owners": ["multimodal"]})
        model = compute_resolution(spec)

        assert shared_additivity_violations(model, ["unclaimed/a.py"]) == []

    def test_row_a_later_rule_overrides_is_not_judged(self) -> None:
        # When something outranks the shared row for a path, the shared row is
        # not in force there and any loss belongs to whatever won.
        spec = self._spec()
        spec["areas"][2]["path_globs"] = ["lib/protocols/audios/nested/"]
        spec["shared"][0]["owners"] = ["multimodal"]
        model = compute_resolution(spec)

        violations = shared_additivity_violations(
            model, ["lib/protocols/audios/nested/a.rs"]
        )

        assert violations == []

    def test_strict_gate_blocks_on_an_additivity_violation(self) -> None:
        spec = self._spec()
        spec["shared"][0]["owners"] = ["multimodal"]
        model = compute_resolution(spec)
        violations = shared_additivity_violations(model, ["lib/protocols/audios/a.rs"])

        message = strict_failure(
            True,
            CoverageGate(blocking=[], warnings=[]),
            None,
            [],
            [],
            None,
            violations,
        )

        assert message and "drops an owner" in message

    def test_gate_stays_green_when_shared_rules_are_complete(self) -> None:
        model = compute_resolution(self._spec())
        violations = shared_additivity_violations(model, ["lib/protocols/audios/a.rs"])

        message = strict_failure(
            True,
            CoverageGate(blocking=[], warnings=[]),
            None,
            [],
            [],
            None,
            violations,
        )

        assert message is None


def test_dead_patterns_include_required_owner_globs() -> None:
    # A required_owners contract whose final matching path is deleted is
    # stale policy: the guarantee now protects nothing, so the strict gate
    # must surface it rather than let a dead contract look effective.
    spec = {
        "meta": {"catch_all": "@root"},
        "areas": [
            {"label": "owned", "github_team": "@owned", "path_globs": ["owned/"]},
            {"label": "docs", "github_team": "@docs", "path_globs": []},
        ],
        "required_owners": [{"glob": "missing/", "owners": ["docs"]}],
    }

    model = compute_resolution(spec)

    assert _dead_patterns(model, ["owned/file.txt"]) == ["/missing/"]


# ------------------------------------------------------------------
# is_policy_change() -- policy edits force full-tree strict
# ------------------------------------------------------------------


class TestIsPolicyChange:
    _AREAS = ".github/codeowners/areas.yaml"

    def test_areas_file_is_policy(self) -> None:
        assert is_policy_change([self._AREAS], self._AREAS, ".") is True

    def test_script_in_policy_dir_is_policy(self) -> None:
        assert (
            is_policy_change(
                [".github/codeowners/emit_codeowners.py"], self._AREAS, "."
            )
            is True
        )

    def test_codeowners_output_is_policy(self) -> None:
        assert is_policy_change(["CODEOWNERS"], self._AREAS, ".") is True

    def test_unrelated_change_is_not_policy(self) -> None:
        assert (
            is_policy_change(["src/foo.py", "owned/b.txt"], self._AREAS, ".") is False
        )


# ------------------------------------------------------------------
# changed_paths() + end-to-end diff-aware --strict demo
# ------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL)


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")


def _head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _run_build(repo: Path, areas: Path, *extra: str):
    script = Path(__file__).parent / "build_codeowners.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--areas",
            str(areas),
            "--repo",
            str(repo),
            "--strict",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


class TestChangedPaths:
    def test_acmr_includes_add_modify_excludes_delete(self, tmp_path) -> None:
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "keep.txt").write_text("1")
        (repo / "gone.txt").write_text("1")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        base = _head(repo)
        (repo / "keep.txt").write_text("2")  # modified
        (repo / "added.txt").write_text("1")  # added
        (repo / "gone.txt").unlink()  # deleted -> filtered out
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "change")

        got = changed_paths(repo, base)
        assert "added.txt" in got
        assert "keep.txt" in got
        assert "gone.txt" not in got  # deletions are not a coverage concern


class TestDiffAwareStrictGateE2E:
    """Concrete proof the relocated base-churn race is closed.

    (a) a base-inherited unowned path does NOT fail diff-aware strict,
    (b) a PR-introduced unowned path DOES fail it,
    (c) default full-tree strict still fails on any unowned path.
    """

    def _areas(self, tmp_path: Path) -> Path:
        areas = tmp_path / "areas.yaml"
        areas.write_text(
            'meta:\n  catch_all: "@root"\n'
            'areas:\n  - label: owned\n    github_team: "@org/owned"\n'
            '    path_globs: ["owned/"]\n'
        )
        return areas

    def _repo_with_base(self, tmp_path: Path) -> tuple[Path, str]:
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "owned").mkdir()
        (repo / "owned" / "a.txt").write_text("x")
        (repo / "base_unowned").mkdir()
        (repo / "base_unowned" / "x.txt").write_text("x")  # inherited, unowned
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        return repo, _head(repo)

    def test_base_gap_ignored_but_full_tree_fails(self, tmp_path) -> None:
        areas = self._areas(tmp_path)
        repo, base = self._repo_with_base(tmp_path)
        # PR adds an OWNED path only; it never touches base_unowned/.
        (repo / "owned" / "b.txt").write_text("y")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "pr adds owned only")
        # (a) diff-aware strict PASSES despite the inherited base gap.
        assert _run_build(repo, areas, "--changed-only", "--base", base).returncode == 0
        # (c) full-tree strict still FAILS on that same inherited gap.
        assert _run_build(repo, areas).returncode == 1

    def test_deleting_last_matched_file_blocks_diff_aware(self, tmp_path) -> None:
        # A PR that deletes the last file a glob matches orphaned that glob
        # ITSELF, so the diff-aware gate blocks it: the same PR must prune
        # the declaration, or main inherits a stale glob that fails the next
        # full-tree run. (Staleness inherited from the base branch still
        # only warns -- see test_stale_glob_inherited_from_base_warns_only.)
        areas = self._areas(tmp_path)
        repo, base = self._repo_with_base(tmp_path)
        (repo / "owned" / "a.txt").unlink()
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "delete last owned file")

        diff_aware = _run_build(repo, areas, "--changed-only", "--base", base)
        assert diff_aware.returncode == 1
        assert "orphaned by this change" in diff_aware.stdout
        assert "/owned/" in diff_aware.stdout
        assert "prune them from areas.yaml" in diff_aware.stdout

        # Full-tree blocks as before (here also on the coverage gap the same
        # deletion exposes; the stale glob is surfaced either way).
        full_tree = _run_build(repo, areas)
        assert full_tree.returncode == 1
        assert "globs matching no files" in full_tree.stdout

        # NOTE: in real CI, pruning the glob edits areas.yaml INSIDE the
        # repo, which reclassifies the PR as a policy change judged
        # full-tree -- this fixture keeps areas.yaml outside --repo, so it
        # cannot model that. The prune-then-pass path is covered where the
        # reclassification actually fires:
        # TestPolicyChangeFallback.test_pruning_deletion_pr_is_judged_full_tree.

    def test_deleting_required_owner_target_blocks_diff_aware(self, tmp_path) -> None:
        # required_owners contracts go stale the same way: deleting the
        # contract's final matching file blocks until the PR prunes it.
        areas = self._areas(tmp_path)
        areas.write_text(
            areas.read_text()
            + 'required_owners:\n  - glob: "base_unowned/x.txt"\n'
            + "    owners: [owned]\n"
        )
        repo, base = self._repo_with_base(tmp_path)
        (repo / "base_unowned" / "x.txt").unlink()
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "delete required target")

        diff_aware = _run_build(repo, areas, "--changed-only", "--base", base)
        assert diff_aware.returncode == 1
        assert "orphaned by this change" in diff_aware.stdout
        assert "/base_unowned/x.txt" in diff_aware.stdout

        full_tree = _run_build(repo, areas)
        assert full_tree.returncode == 1
        assert "glob(s) match no tracked files" in full_tree.stdout

    def test_stale_glob_inherited_from_base_warns_only(self, tmp_path) -> None:
        # The anti-cascade guarantee: a glob that was ALREADY dead at the
        # merge-base is base staleness this PR did not cause. Blocking on it
        # would red-X every open PR the moment the base goes stale; it stays
        # a warning for diff-aware runs and blocks only full-tree ones.
        areas = tmp_path / "areas.yaml"
        areas.write_text(
            'meta:\n  catch_all: "@root"\n'
            'areas:\n  - label: owned\n    github_team: "@org/owned"\n'
            '    path_globs: ["owned/", "ghost/"]\n'  # ghost/ never existed
        )
        repo, base = self._repo_with_base(tmp_path)
        (repo / "owned" / "b.txt").write_text("y")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "pr adds owned file only")

        diff_aware = _run_build(repo, areas, "--changed-only", "--base", base)
        assert diff_aware.returncode == 0
        assert "inherited from the base branch" in diff_aware.stdout
        assert "/ghost/" in diff_aware.stdout

        # Full-tree still blocks on it (coverage of base_unowned/ fails
        # first, but the stale-glob report names /ghost/ either way).
        full_tree = _run_build(repo, areas)
        assert full_tree.returncode == 1
        assert "globs matching no files" in full_tree.stdout

    def test_inherited_contract_only_blocks_full_tree(self, tmp_path) -> None:
        areas = tmp_path / "areas.yaml"
        areas.write_text(
            'meta:\n  catch_all: "@root"\n'
            'areas:\n  - label: runtime\n    github_team: "@runtime"\n'
            '    path_globs: ["owned/"]\n'
            '  - label: docs\n    github_team: "@docs"\n    path_globs: []\n'
            'required_owners:\n  - glob: "owned/base.txt"\n    owners: [docs]\n'
        )
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "owned").mkdir()
        (repo / "owned" / "base.txt").write_text("x")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        base = _head(repo)
        (repo / "owned" / "new.txt").write_text("y")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "pr adds unrelated owned file")

        diff_aware = _run_build(repo, areas, "--changed-only", "--base", base)
        assert diff_aware.returncode == 0
        full_tree = _run_build(repo, areas)
        assert full_tree.returncode == 1
        assert "lost declared owners" in full_tree.stdout

    def test_pr_introduced_gap_fails(self, tmp_path) -> None:
        areas = self._areas(tmp_path)
        repo, base = self._repo_with_base(tmp_path)
        # PR adds an UNOWNED path -- its own surface is not 100% owned.
        (repo / "newdir").mkdir()
        (repo / "newdir" / "z.txt").write_text("z")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "pr adds unowned")
        # (b) diff-aware strict FAILS on the PR's own unowned path.
        result = _run_build(repo, areas, "--changed-only", "--base", base)
        assert result.returncode == 1
        assert "newdir/z.txt" in result.stdout


class TestPolicyChangeFallback:
    """A PR that edits ownership policy is judged whole-tree: a policy edit can
    orphan paths the PR never touches, so diff-aware must not let it pass."""

    def test_policy_edit_orphaning_untouched_path_blocks(self, tmp_path) -> None:
        repo = tmp_path / "r"
        _init_repo(repo)
        # areas.yaml lives INSIDE the repo (as in CI) so editing it shows in
        # the diff and marks the PR a policy change.
        areas = repo / ".github" / "codeowners" / "areas.yaml"
        areas.parent.mkdir(parents=True)
        areas.write_text(
            'meta:\n  catch_all: "@root"\n'
            'areas:\n  - label: owned\n    github_team: "@org/owned"\n'
            '    path_globs: ["owned/"]\n'
        )
        (repo / "owned").mkdir()
        (repo / "owned" / "a.txt").write_text("x")
        (repo / "owned" / "b.txt").write_text("x")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        base = _head(repo)
        # Narrow the policy so owned/b.txt is orphaned, WITHOUT touching it.
        areas.write_text(
            'meta:\n  catch_all: "@root"\n'
            'areas:\n  - label: owned\n    github_team: "@org/owned"\n'
            '    path_globs: ["owned/a.txt"]\n'
        )
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "narrow policy, orphan b.txt")
        # Plain diff-aware would miss owned/b.txt (not in the file diff); the
        # policy-change fallback forces full-tree strict, so it BLOCKS.
        result = _run_build(repo, areas, "--changed-only", "--base", base)
        assert result.returncode == 1
        assert "owned/b.txt" in result.stdout

    def test_pruning_deletion_pr_is_judged_full_tree(self, tmp_path) -> None:
        # The remediation for an orphaned glob -- prune it in the same PR --
        # edits areas.yaml, which reclassifies the PR as a policy change and
        # judges it FULL-TREE, exactly like any routing edit (so it also
        # requires a green base tree). This is the path real CI takes for a
        # deletion PR that follows the gate's instruction; the unit fixtures
        # above keep areas.yaml outside --repo and cannot model the
        # reclassification.
        repo = tmp_path / "r"
        _init_repo(repo)
        areas = repo / ".github" / "codeowners" / "areas.yaml"
        areas.parent.mkdir(parents=True)
        policy = (
            'meta:\n  catch_all: "@root"\n'
            "areas:\n"
            '  - label: policy\n    github_team: "@org/policy"\n'
            '    path_globs: [".github/"]\n'
            '  - label: owned\n    github_team: "@org/owned"\n'
            "    path_globs: [{globs}]\n"
        )
        areas.write_text(policy.format(globs='"kept/", "doomed/"'))
        (repo / "kept").mkdir()
        (repo / "kept" / "a.txt").write_text("x")
        (repo / "doomed").mkdir()
        (repo / "doomed" / "last.txt").write_text("x")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        base = _head(repo)

        # The deletion alone runs diff-aware and blocks as newly stale.
        (repo / "doomed" / "last.txt").unlink()
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "delete the last doomed file")
        blocked = _run_build(repo, areas, "--changed-only", "--base", base)
        assert blocked.returncode == 1
        assert "orphaned by this change" in blocked.stdout

        # The same PR prunes the dead glob: areas.yaml enters the diff, the
        # run reclassifies to full-tree, and it passes because the remaining
        # tree is 100% owned.
        areas.write_text(policy.format(globs='"kept/"'))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "prune the dead glob")
        pruned = _run_build(repo, areas, "--changed-only", "--base", base)
        assert pruned.returncode == 0
        assert "evaluating full-tree" in pruned.stdout


# ------------------------------------------------------------------
# Contributor action matrix -- the common ownership-affecting actions,
# end-to-end against the strict gate (requested in PR #11869 review)
# ------------------------------------------------------------------


class TestContributorActionMatrix:
    """One test per common contributor action:

    1. add a file inside a covered folder      -> PASS (nothing to declare)
    2. add a file in an uncovered location     -> BLOCK until an area claims it
    3. delete the last file a glob matches     -> BLOCK until the PR prunes the glob
    4. add shared ownership to a path          -> PASS; both teams routed
    5. remove the added co-owner again         -> PASS (a policy decision)
       remove the ENCLOSING owner instead      -> BLOCK (shared rows are additive)
    6. add an area claiming a new directory    -> PASS; new team routed
    7. remove an area while its files live on  -> BLOCK until paths are reassigned

    Ordinary PRs (1-3) run diff-aware, as CI does on pull_request events.
    Policy edits (4-7) run full-tree, exactly how the gate judges any PR
    that touches areas.yaml (see TestPolicyChangeFallback for the
    reclassification itself).
    """

    BASE_AREAS = (
        'meta:\n  catch_all: "@root"\n'
        "areas:\n"
        '  - label: owned\n    github_team: "@org/owned"\n'
        '    path_globs: ["owned/"]\n'
        '  - label: docs\n    github_team: "@docs"\n    path_globs: []\n'
    )

    def _fixture(self, tmp_path: Path) -> tuple[Path, Path, str]:
        areas = tmp_path / "areas.yaml"
        areas.write_text(self.BASE_AREAS)
        repo = tmp_path / "r"
        _init_repo(repo)
        (repo / "owned").mkdir()
        (repo / "owned" / "a.txt").write_text("x")
        (repo / "owned" / "tool.py").write_text("x")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        return areas, repo, _head(repo)

    def _routing(self, areas: Path, path: str) -> list[str]:
        model = compute_resolution(yaml.safe_load(areas.read_text()))
        lines, _ = _render_codeowners(model, group=True, external=[])
        return resolve_owners(parse_codeowners("\n".join(lines)), path)

    def test_1_add_file_in_covered_folder_passes(self, tmp_path) -> None:
        areas, repo, base = self._fixture(tmp_path)
        (repo / "owned" / "new_feature.py").write_text("y")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "add covered file")
        assert _run_build(repo, areas, "--changed-only", "--base", base).returncode == 0

    def test_2_add_file_in_uncovered_location_blocks(self, tmp_path) -> None:
        areas, repo, base = self._fixture(tmp_path)
        (repo / "rogue").mkdir()
        (repo / "rogue" / "new.py").write_text("y")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "add uncovered file")
        result = _run_build(repo, areas, "--changed-only", "--base", base)
        assert result.returncode == 1
        assert "rogue/new.py" in result.stdout
        assert "cover them in areas.yaml" in result.stdout

    def test_3_delete_last_matched_file_blocks_until_pruned(self, tmp_path) -> None:
        areas, repo, base = self._fixture(tmp_path)
        areas.write_text(
            self.BASE_AREAS
            + 'shared:\n  - glob: "owned/tool.py"\n    owners: [owned, docs]\n'
        )
        (repo / "owned" / "tool.py").unlink()
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "delete the shared rule's only file")

        result = _run_build(repo, areas, "--changed-only", "--base", base)
        assert result.returncode == 1
        assert "orphaned by this change" in result.stdout

        # ...and the PR prunes the rule. In real CI that areas.yaml edit
        # reclassifies the PR as a policy change judged FULL-TREE (see
        # TestPolicyChangeFallback.test_pruning_deletion_pr_is_judged_full_tree
        # for the reclassification itself), so model that judgement here.
        areas.write_text(self.BASE_AREAS)
        assert _run_build(repo, areas).returncode == 0

    def test_4_add_shared_ownership_passes_and_routes_both(self, tmp_path) -> None:
        areas, repo, _ = self._fixture(tmp_path)
        areas.write_text(
            self.BASE_AREAS
            + 'shared:\n  - glob: "owned/tool.py"\n    owners: [owned, docs]\n'
        )
        assert _run_build(repo, areas).returncode == 0
        assert self._routing(areas, "owned/tool.py") == ["@org/owned", "@docs"]

    def test_5_remove_added_coowner_passes(self, tmp_path) -> None:
        areas, repo, _ = self._fixture(tmp_path)
        # docs was granted co-ownership earlier; a later policy PR retires
        # the grant by dropping the whole shared row -- a reviewed decision.
        areas.write_text(self.BASE_AREAS)
        assert _run_build(repo, areas).returncode == 0
        assert self._routing(areas, "owned/tool.py") == ["@org/owned"]

    def test_6_add_area_claiming_new_directory_passes(self, tmp_path) -> None:
        areas, repo, _ = self._fixture(tmp_path)
        (repo / "newdir").mkdir()
        (repo / "newdir" / "x.py").write_text("y")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "new subsystem")
        areas.write_text(
            self.BASE_AREAS
            + '  - label: newarea\n    github_team: "@org/newarea"\n'
            + '    path_globs: ["newdir/"]\n'
        )
        assert _run_build(repo, areas).returncode == 0
        assert self._routing(areas, "newdir/x.py") == ["@org/newarea"]

    def test_7_remove_area_with_live_files_blocks_until_reassigned(
        self, tmp_path
    ) -> None:
        areas, repo, _ = self._fixture(tmp_path)
        # Dropping the owned area orphans owned/* -- coverage blocks.
        areas.write_text(
            'meta:\n  catch_all: "@root"\n'
            'areas:\n  - label: docs\n    github_team: "@docs"\n    path_globs: []\n'
        )
        result = _run_build(repo, areas)
        assert result.returncode == 1
        assert "fall to the catch-all" in result.stdout

        # Reassigning the globs to a surviving area satisfies the gate.
        areas.write_text(
            'meta:\n  catch_all: "@root"\n'
            'areas:\n  - label: docs\n    github_team: "@docs"\n'
            '    path_globs: ["owned/"]\n'
        )
        assert _run_build(repo, areas).returncode == 0
        assert self._routing(areas, "owned/a.txt") == ["@docs"]


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
        resolved: SharedSpec = {"glob": "x/", "owners": ["a", "b"]}
        assert resolved["glob"] == "x/"


@pytest.fixture(scope="module")
def real_policy_rules() -> tuple[list[tuple[str, list[str]]], dict[str, str]]:
    repo = Path(__file__).resolve().parents[2]
    spec = yaml.safe_load((repo / ".github/codeowners/areas.yaml").read_text())
    model = compute_resolution(spec)
    lines, _ = _render_codeowners(model, group=True, external=[])
    return parse_codeowners("\n".join(lines)), model.label_to_team()


class TestRealPolicyRoutingContracts:
    @pytest.mark.parametrize(
        ("path", "labels"),
        [
            (
                "deploy/inference-gateway/epp/Makefile",
                ("epp", "operator", "ops"),
            ),
            (
                "deploy/operator/internal/checkpoint/resolve.go",
                ("operator", "gms"),
            ),
            (
                "container/templates/sglang_xpu_framework.Dockerfile",
                ("ops", "backend-sglang", "xpu"),
            ),
            (
                "tests/router/test_router_e2e_with_vllm_xpu.py",
                ("router", "backend-vllm", "xpu"),
            ),
            # NOTE: the docs/backends/vllm/ case was dropped here when the
            # docs restructure (#10855) moved these pages under docs/fern/.
            # The move did not carry the backend teams' co-ownership across,
            # so docs/fern/backends/* is currently docs-only. That is an open
            # ownership question, not a contract to pin -- deliberately not
            # re-added at the new path until the owning teams decide.
            (
                "recipes/qwen3-32b/vllm/agg-round-robin/deploy.yaml",
                ("performance", "backend-vllm"),
            ),
        ],
    )
    def test_required_teams_are_present(
        self,
        real_policy_rules: tuple[list[tuple[str, list[str]]], dict[str, str]],
        path: str,
        labels: tuple[str, ...],
    ) -> None:
        rules, teams = real_policy_rules
        actual = set(resolve_owners(rules, path))
        assert {teams[label] for label in labels} <= actual

    def test_generic_vllm_handler_excludes_rl(
        self,
        real_policy_rules: tuple[list[tuple[str, list[str]]], dict[str, str]],
    ) -> None:
        rules, teams = real_policy_rules
        assert resolve_owners(rules, "components/src/dynamo/vllm/handlers.py") == [
            teams["backend-vllm"]
        ]

    def test_squeeze_evolve_is_router_owned_only(
        self,
        real_policy_rules: tuple[list[tuple[str, list[str]]], dict[str, str]],
    ) -> None:
        rules, teams = real_policy_rules
        owners = resolve_owners(
            rules, "components/src/dynamo/squeeze_evolve/orchestrator.py"
        )
        assert owners == [teams["router"]]


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
