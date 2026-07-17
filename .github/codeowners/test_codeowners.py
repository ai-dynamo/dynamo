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

# Allow `import codeowners_match` when pytest runs from the repo root.
sys.path.insert(0, str(Path(__file__).parent))

import who_owns  # noqa: E402
from build_codeowners import (  # noqa: E402
    CoverageGate,
    is_policy_change,
    split_coverage,
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
                "keyword_rules": [],
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

    @pytest.mark.parametrize(
        "rule",
        [
            {"coowner": "docs", "advisory": False},
            {"pattern": "Dockerfile", "advisory": False},
        ],
    )
    def test_blocking_filetype_rule_requires_pattern_and_coowner(
        self, rule: dict
    ) -> None:
        spec = self._spec()
        spec["classify"]["filetype_rules"] = [rule]
        with pytest.raises(SystemExit, match="missing 'pattern' or 'coowner'"):
            compute_resolution(spec)

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
        # blocking filetype rule + advisory-only filetype rule.
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
                "keyword_rules": [],
                "filetype_rules": [
                    {"pattern": "Dockerfile", "coowner": "ops", "advisory": False},
                    {"pattern": "*.md", "coowner": "docs", "advisory": True},
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
        assert "tree" not in sig.parameters, (
            "emit tree parameter reintroduced -- see TestEmissionIsTreeIndependent"
        )
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

    def test_overlapping_filetype_rules_preserve_declaration_order(self) -> None:
        spec = self._spec()
        spec["classify"]["filetype_rules"] = [
            {"pattern": "*Dockerfile*", "coowner": "ops", "advisory": False},
            {"pattern": "Dockerfile", "coowner": "docs", "advisory": False},
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


# ------------------------------------------------------------------
# who_owns.team_members() -- roster expansion (--people)
# ------------------------------------------------------------------


class TestTeamMembers:
    def test_expands_team_via_fetcher(self) -> None:
        cache: dict = {}
        fetched = []

        def fake(org: str, slug: str) -> list[str]:
            fetched.append((org, slug))
            return ["zoe", "amy"]

        members = who_owns.team_members("@acme/router", fetch=fake, cache=cache)
        assert members == ["amy", "zoe"]  # sorted
        assert fetched == [("acme", "router")]
        # second lookup served from cache, fetcher not called again
        assert who_owns.team_members("@acme/router", fetch=fake, cache=cache) == [
            "amy",
            "zoe",
        ]
        assert fetched == [("acme", "router")]

    def test_fetch_failure_returns_none_and_caches_negative(self) -> None:
        cache: dict = {}
        calls = []

        def boom(org: str, slug: str) -> list[str]:
            calls.append(slug)
            raise subprocess.CalledProcessError(1, "gh")

        assert who_owns.team_members("@acme/router", fetch=boom, cache=cache) is None
        assert who_owns.team_members("@acme/router", fetch=boom, cache=cache) is None
        assert calls == ["router"]  # failure cached; no retry storm

    def test_individual_handle_passes_through(self) -> None:
        def never(org: str, slug: str) -> list[str]:
            raise AssertionError("fetcher must not be called for @handles")

        assert who_owns.team_members("@octocat", fetch=never, cache={}) is None


class TestChangedFiles:
    def test_includes_untracked_files(self, tmp_path) -> None:
        # Brand-new (unstaged) files are the ones the coverage gate cares
        # about most; `git diff` alone never lists them.
        repo = tmp_path / "r"
        repo.mkdir()

        def git(*args: str) -> None:
            subprocess.check_output(
                ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL
            )

        git("init", "-q")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        (repo / "tracked.txt").write_text("x")
        git("add", "tracked.txt")
        git("commit", "-q", "-m", "init")
        (repo / "tracked.txt").write_text("y")  # modified, unstaged
        (repo / "brand_new.txt").write_text("z")  # untracked

        files = who_owns.changed_files(str(repo), "HEAD")
        assert files == ["brand_new.txt", "tracked.txt"]

    def test_bad_base_with_untracked_raises(self, monkeypatch) -> None:
        # When ALL git-diff fallbacks fail AND ls-files succeeds, the diff
        # failure must still be surfaced (not silently masked by ls-files
        # succeeding). Tested via monkeypatch because the no-arg diff fallback
        # always succeeds in a valid repo, making the scenario unreachable
        # with a real git setup.
        def fake_check_output(cmd, **kwargs):
            if "diff" in cmd:
                raise subprocess.CalledProcessError(128, cmd)
            if "ls-files" in cmd:
                return "untracked.txt\n"
            raise subprocess.CalledProcessError(128, cmd)

        monkeypatch.setattr(subprocess, "check_output", fake_check_output)

        with pytest.raises(SystemExit):
            who_owns.changed_files("/fake/repo", "nonexistent-base-ref")

    def test_untracked_in_subdir_is_repo_root_relative(self, tmp_path) -> None:
        # git ls-files without --full-name returns paths relative to cwd
        # when run from a subdirectory. --full-name ensures repo-root-relative
        # output so CODEOWNERS matching is always correct.
        repo = tmp_path / "r"
        repo.mkdir()

        def git(*args: str) -> None:
            subprocess.check_output(
                ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL
            )

        git("init", "-q")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        (repo / "seed.txt").write_text("x")
        git("add", "seed.txt")
        git("commit", "-q", "-m", "init")
        subdir = repo / "subdir"
        subdir.mkdir()
        (subdir / "new.txt").write_text("z")  # untracked in a subdirectory

        # Pass repo=subdir to simulate running from a subdirectory; the
        # returned path must be repo-root-relative (subdir/new.txt), not
        # the bare filename (new.txt).
        files = who_owns.changed_files(str(subdir), "HEAD")
        assert "subdir/new.txt" in files


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
