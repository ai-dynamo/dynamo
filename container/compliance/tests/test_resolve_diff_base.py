# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the CI diff-baseline resolver.

Run from the repo root with the compliance package on the path:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_resolve_diff_base.py
"""

from __future__ import annotations

import pytest
from compliance import resolve_diff_base
from compliance.resolve_diff_base import (
    BaselineError,
    find_run_artifact,
    is_release_branch,
    parse_release_tuple,
    pick_prior_release_tag,
    resolve,
)

# CPU-only unit tests; markers are required by .ai/pytest-guidelines.md
# (lifecycle / test-type / hardware categories).
pytestmark = [
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


class TestParseReleaseTuple:
    def test_plain_and_prefixed(self):
        assert parse_release_tuple("1.3.1") == (1, 3, 1)
        assert parse_release_tuple("v1.3.1") == (1, 3, 1)

    def test_suffixed(self):
        assert parse_release_tuple("v1.2.3-nemo-3") == (1, 2, 3)
        assert parse_release_tuple("v0.0.0-rc6") == (0, 0, 0)

    def test_unparseable(self):
        assert parse_release_tuple("") is None
        assert parse_release_tuple("vfoo") is None
        assert parse_release_tuple("runtime-1.0.0") is None


class TestPickPriorReleaseTag:
    def test_semver_beats_chronology(self):
        # 1.3.0 was published EARLIER than 1.2.5, but for a 1.3.1 build the
        # highest semver strictly older wins -> 1.3.0.
        tags = [("v1.3.0", 100), ("v1.2.5", 200)]
        assert pick_prior_release_tag("1.3.1", tags) == "v1.3.0"

    def test_same_base_tie_broken_by_later_publish(self):
        # Both share base 1.2.3; current is 1.2.4; pick the later-published tag.
        tags = [("v1.2.3-nemo-3", 100), ("v1.2.3-minimax", 200)]
        assert pick_prior_release_tag("1.2.4", tags) == "v1.2.3-minimax"

    def test_same_base_tie_order_independent(self):
        tags = [("v1.2.3-minimax", 200), ("v1.2.3-nemo-3", 100)]
        assert pick_prior_release_tag("1.2.4", tags) == "v1.2.3-minimax"

    def test_no_older_tag_returns_none(self):
        tags = [("v1.3.1", 100), ("v1.4.0", 200)]
        assert pick_prior_release_tag("1.3.1", tags) is None

    def test_empty_returns_none(self):
        assert pick_prior_release_tag("1.3.1", []) is None

    def test_unparseable_tags_skipped(self):
        tags = [("vfoo", 100), ("garbage", 150), ("v1.2.0", 200)]
        assert pick_prior_release_tag("1.3.0", tags) == "v1.2.0"

    def test_unparseable_current_version(self):
        assert pick_prior_release_tag("not-a-version", [("v1.0.0", 1)]) is None

    def test_equal_base_excluded(self):
        # A tag at the SAME base as current is the current release, not a prior.
        tags = [("v1.3.0", 50), ("v1.2.9", 40)]
        assert pick_prior_release_tag("1.3.0", tags) == "v1.2.9"


def _workflow_run(
    run_id=90,
    sha="a" * 40,
    branch="main",
    event="push",
    workflow=".github/workflows/post-merge-ci.yml",
    conclusion="failure",
):
    return {
        "id": run_id,
        "head_sha": sha,
        "head_branch": branch,
        "event": event,
        "path": workflow,
        "workflow_id": 123,
        "head_repository": {"full_name": "org/repo"},
        "conclusion": conclusion,
        "status": "completed",
    }


def _artifact(run, *, expired=False, artifact_id=7, prefix="vllm-runtime"):
    return {
        "id": artifact_id,
        "name": f"compliance-{run['head_sha']}-{prefix}",
        "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]},
        "size_in_bytes": 100,
        "expired": expired,
    }


class TestNightlyBaseline:
    def test_red_previous_run_is_selected_and_bound_to_its_artifact(self, monkeypatch):
        current = _workflow_run(
            100, event="schedule", workflow=".github/workflows/nightly-ci.yml"
        )
        previous = _workflow_run(
            90, event="schedule", workflow=".github/workflows/nightly-ci.yml"
        )
        manual = _workflow_run(
            95, event="workflow_dispatch", workflow=".github/workflows/nightly-ci.yml"
        )
        calls = []

        def api(endpoint):
            calls.append(endpoint)
            if endpoint.endswith("/runs/100"):
                return current
            if "/workflows/123/runs?" in endpoint:
                assert "status=success" not in endpoint
                return {"workflow_runs": [current, manual, previous]}
            assert "/runs/90/artifacts?" in endpoint
            return {"total_count": 1, "artifacts": [_artifact(previous)]}

        monkeypatch.setattr(resolve_diff_base, "_gh_json", api)
        result = resolve("nightly", "main", "", "", "org/repo", "100", "vllm-runtime")
        assert result[0] == previous["head_sha"]
        assert result[2:] == ("7", "90")
        assert not any("/actions/artifacts?" in path for path in calls)

    def test_skips_missing_artifact_but_never_reuses_another_runs_same_sha(
        self, monkeypatch
    ):
        recent, older = _workflow_run(90), _workflow_run(80)
        monkeypatch.setattr(
            resolve_diff_base, "_fetch_workflow_runs", lambda *a: [recent, older]
        )
        seen = []

        def api(endpoint):
            seen.append(endpoint)
            artifacts = [] if "/90/" in endpoint else [_artifact(older, artifact_id=8)]
            return {"total_count": len(artifacts), "artifacts": artifacts}

        monkeypatch.setattr(resolve_diff_base, "_gh_json", api)
        assert resolve("nightly", "main", "", "", "org/repo", "100", "vllm-runtime")[
            2:
        ] == ("8", "80")
        assert len(seen) == 2

    @pytest.mark.parametrize("expired", [False, True])
    def test_missing_or_expired_baseline_fails(self, monkeypatch, expired):
        run = _workflow_run()
        monkeypatch.setattr(resolve_diff_base, "_fetch_workflow_runs", lambda *a: [run])
        artifacts = [_artifact(run, expired=True)] if expired else []
        monkeypatch.setattr(
            resolve_diff_base,
            "_gh_json",
            lambda *a: {"total_count": len(artifacts), "artifacts": artifacts},
        )
        with pytest.raises(BaselineError, match="no prior scheduled"):
            resolve("nightly", "main", "", "", "org/repo", "100", "vllm-runtime")

    def test_api_failure_is_not_a_bootstrap(self, monkeypatch):
        def fail(*args):
            raise BaselineError("API unavailable")

        monkeypatch.setattr(resolve_diff_base, "_fetch_workflow_runs", fail)
        with pytest.raises(BaselineError, match="API unavailable"):
            resolve(
                "nightly",
                "main",
                "",
                "",
                "org/repo",
                "100",
                "vllm-runtime",
                allow_bootstrap=True,
            )

    def test_bootstrap_is_explicit_and_expiry_cannot_use_it(self, monkeypatch):
        monkeypatch.setattr(resolve_diff_base, "_fetch_workflow_runs", lambda *a: [])
        result = resolve(
            "nightly",
            "main",
            "",
            "",
            "org/repo",
            "100",
            "vllm-runtime",
            allow_bootstrap=True,
        )
        assert result[0] == "" and "explicit bootstrap" in result[1]
        run = _workflow_run()
        monkeypatch.setattr(resolve_diff_base, "_fetch_workflow_runs", lambda *a: [run])
        monkeypatch.setattr(
            resolve_diff_base,
            "_gh_json",
            lambda *a: {"total_count": 1, "artifacts": [_artifact(run, expired=True)]},
        )
        with pytest.raises(BaselineError, match="expired"):
            resolve(
                "nightly",
                "main",
                "",
                "",
                "org/repo",
                "100",
                "vllm-runtime",
                allow_bootstrap=True,
            )


class TestArtifactProvenance:
    def test_wrong_run_is_rejected(self, monkeypatch):
        run = _workflow_run()
        wrong = _artifact(_workflow_run(95))
        monkeypatch.setattr(
            resolve_diff_base,
            "_gh_json",
            lambda *a: {"total_count": 1, "artifacts": [wrong]},
        )
        with pytest.raises(BaselineError, match="provenance"):
            find_run_artifact("org/repo", run, "vllm-runtime")

    def test_paginates_and_requires_exact_variant(self, monkeypatch):
        run = _workflow_run()
        calls = []

        def api(endpoint):
            calls.append(endpoint)
            artifacts = (
                [_artifact(run, prefix="vllm-runtime-efa")]
                if endpoint.endswith("&page=1")
                else [_artifact(run)]
            )
            return {"total_count": 101, "artifacts": artifacts}

        monkeypatch.setattr(resolve_diff_base, "_gh_json", api)
        assert find_run_artifact("org/repo", run, "vllm-runtime") == ("7", False)
        assert len(calls) == 2

    def test_main_rejects_pr_and_unrelated_workflow_runs(self, monkeypatch):
        sha = "a" * 40
        allowed = _workflow_run(80)
        rejected = [
            _workflow_run(95, branch="pull-request/7"),
            _workflow_run(94, workflow=".github/workflows/other.yml"),
            _workflow_run(93, event="workflow_dispatch"),
        ]
        seen = []

        def api(endpoint):
            seen.append(endpoint)
            if "/workflows/" in endpoint:
                return {"workflow_runs": rejected + [allowed]}
            assert "/runs/80/artifacts?" in endpoint
            return {"total_count": 1, "artifacts": [_artifact(allowed)]}

        monkeypatch.setattr(resolve_diff_base, "_gh_json", api)
        assert resolve_diff_base._find_commit_artifact(
            "org/repo", sha, "vllm-runtime", "main"
        )[0] == ("7", "80")
        assert len(seen) == 2


class TestResolveDispatch:
    @pytest.mark.parametrize(
        "context,branch,base,version",
        [
            ("push", "feature", "", ""),
            ("push", "release/1.3.0", "", ""),
            ("pr", "pull-request/7", "main", ""),
            ("nightly", "main", "", ""),
        ],
    )
    def test_missing_context_fails_closed(self, context, branch, base, version):
        with pytest.raises(BaselineError):
            resolve(
                context,
                branch,
                base,
                version,
                repo="org/repo",
                artifact_prefix="vllm-runtime",
            )

    def test_pr_keeps_stable_merge_base_and_first_parent_walk(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            resolve_diff_base, "_git", lambda *args: calls.append(args) or "tip\nparent"
        )
        monkeypatch.setattr(
            resolve_diff_base,
            "_find_commit_artifact",
            lambda repo, sha, prefix, branch: (("7", "80"), False)
            if sha == "parent"
            else (None, False),
        )
        result = resolve(
            "pr",
            "pull-request/7",
            "main",
            "",
            "org/repo",
            artifact_prefix="vllm-runtime",
            merge_base_sha="fork-point",
        )
        assert result[0] == "parent"
        assert calls == [("rev-list", "--first-parent", "-n", "25", "fork-point")]

    @pytest.mark.parametrize("prior", ["1.2.1", "1.2.1-vendor-dev.2"])
    def test_release_uses_prior_tag_and_its_own_postmerge_branch(
        self, monkeypatch, prior
    ):
        monkeypatch.setattr(
            resolve_diff_base, "_tag_dates", lambda: [("v" + prior, 1), ("v1.3.0", 2)]
        )
        monkeypatch.setattr(resolve_diff_base, "_git", lambda *a: "a" * 40)
        calls = []
        monkeypatch.setattr(
            resolve_diff_base,
            "_find_commit_artifact",
            lambda *a: calls.append(a) or (("7", "80"), False),
        )
        result = resolve(
            "push",
            "release/1.3.0",
            "",
            "1.3.0",
            "org/repo",
            artifact_prefix="vllm-runtime",
        )
        assert result[2:] == ("7", "80")
        assert calls == [("org/repo", "a" * 40, "vllm-runtime", "release/" + prior)]


def test_is_release_branch():
    assert is_release_branch("release/1.3.0")
    assert is_release_branch("release/1.3.0-cosmos3-dev.1")
    assert not is_release_branch("main")
