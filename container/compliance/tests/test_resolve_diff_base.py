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
    find_artifact,
    is_release_branch,
    parse_release_tuple,
    pick_commit_with_artifact,
    pick_previous_run_sha,
    pick_prior_release_tag,
    resolve,
)

# CPU-only unit tests; markers are required by .ai/pytest-guidelines.md
# (lifecycle / test-type / hardware categories).
pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _run(
    run_id: int,
    head_sha: str,
    conclusion: str | None = "success",
    event: str = "schedule",
) -> dict:
    return {
        "id": run_id,
        "head_sha": head_sha,
        "event": event,
        "conclusion": conclusion,
    }


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


class TestPickPreviousRunSha:
    def test_most_recent_prior_success(self):
        runs = [
            _run(100, "cur", conclusion=None),  # current, in progress
            _run(90, "sha90"),
            _run(80, "sha80"),
        ]
        assert pick_previous_run_sha(100, runs) == "sha90"

    def test_skips_failed_runs(self):
        runs = [_run(90, "sha90", conclusion="failure"), _run(80, "sha80")]
        assert pick_previous_run_sha(100, runs) == "sha80"

    def test_skips_successful_manual_runs(self):
        runs = [
            _run(95, "manual", event="workflow_dispatch"),
            _run(90, "scheduled"),
        ]
        assert pick_previous_run_sha(100, runs) == "scheduled"

    def test_ignores_runs_at_or_after_current(self):
        runs = [_run(100, "cur"), _run(110, "future")]
        assert pick_previous_run_sha(100, runs) is None

    def test_empty(self):
        assert pick_previous_run_sha(100, []) is None


class TestFetchWorkflowRuns:
    def test_requests_only_successful_scheduled_runs(self, monkeypatch):
        responses = iter(
            [
                "1234",
                '[{"id": 90, "head_sha": "scheduled", '
                '"event": "schedule", "conclusion": "success"}]',
            ]
        )
        calls = []

        def fake_gh(*args):
            calls.append(args)
            return next(responses)

        monkeypatch.setattr(resolve_diff_base, "_gh", fake_gh)

        assert resolve_diff_base._fetch_workflow_runs("ai-dynamo/dynamo", 100) == [
            {
                "id": 90,
                "head_sha": "scheduled",
                "event": "schedule",
                "conclusion": "success",
            }
        ]
        assert calls[1][0].endswith("?event=schedule&status=success&per_page=50")
        assert calls[1][2] == ("[.workflow_runs[] | {id, head_sha, event, conclusion}]")


class TestPickCommitWithArtifact:
    def test_returns_first_with_artifact(self):
        # tip (sha_a) has none; walk back to sha_b which does.
        have = {"sha_b"}
        chosen, artifact, saw_error = pick_commit_with_artifact(
            ["sha_a", "sha_b", "sha_c"],
            lambda s: (("artifact", "run"), False) if s in have else (None, False),
        )
        assert chosen == "sha_b"
        assert artifact == ("artifact", "run")
        assert saw_error is False

    def test_tip_has_artifact(self):
        chosen, artifact, saw_error = pick_commit_with_artifact(
            ["sha_a", "sha_b"], lambda s: (("artifact", "run"), False)
        )
        assert chosen == "sha_a"
        assert artifact == ("artifact", "run")
        assert saw_error is False

    def test_none_found(self):
        chosen, artifact, saw_error = pick_commit_with_artifact(
            ["sha_a", "sha_b"], lambda s: (None, False)
        )
        assert chosen is None
        assert artifact is None
        assert saw_error is False

    def test_api_error_then_hit(self):
        results = {
            "sha_a": (None, True),
            "sha_b": (("artifact", "run"), False),
        }
        chosen, artifact, saw_error = pick_commit_with_artifact(
            ["sha_a", "sha_b"], lambda s: results[s]
        )
        assert chosen == "sha_b"
        assert artifact == ("artifact", "run")
        assert saw_error is True

    def test_all_api_error(self):
        chosen, artifact, saw_error = pick_commit_with_artifact(
            ["sha_a", "sha_b"], lambda s: (None, True)
        )
        assert chosen is None
        assert artifact is None
        assert saw_error is True

    def test_empty(self):
        assert pick_commit_with_artifact([], lambda s: (None, False)) == (
            None,
            None,
            False,
        )


class TestFindArtifact:
    def test_returns_exact_artifact_and_run_ids(self, monkeypatch):
        monkeypatch.setattr(resolve_diff_base, "_gh", lambda *args: "123\t456")
        assert find_artifact("ai-dynamo/dynamo", "sha", "vllm-runtime") == (
            ("123", "456"),
            False,
        )

    def test_distinguishes_absent_from_api_error(self, monkeypatch):
        monkeypatch.setattr(resolve_diff_base, "_gh", lambda *args: "absent")
        assert find_artifact("repo", "sha", "prefix") == (None, False)
        monkeypatch.setattr(resolve_diff_base, "_gh", lambda *args: None)
        assert find_artifact("repo", "sha", "prefix") == (None, True)


class TestIsReleaseBranch:
    def test_matches(self):
        assert is_release_branch("release/1.3.0")
        assert is_release_branch("release/1.3.0-cosmos3-dev.1")

    def test_non_matches(self):
        assert not is_release_branch("main")
        assert not is_release_branch("pull-request/42")
        assert not is_release_branch("")


class TestResolveDispatch:
    """Rule dispatch without touching git: assert which branch of resolve() fires
    by inspecting the label. Rules 1/2/3 hit git; the fallback and the
    unparseable-version paths do not."""

    def test_fallback_no_baseline(self):
        sha, label, artifact_id, run_id = resolve("push", "some-feature-branch", "", "")
        assert sha == ""
        assert "unrecognized" in label
        assert artifact_id == run_id == ""

    def test_release_push_without_version(self):
        # Release context but no parseable version -> no git rev-list, empty sha.
        sha, label, _, _ = resolve("push", "release/1.3.0", "", "")
        assert sha == ""
        assert "no parseable current version" in label

    def test_pr_to_release_without_version(self):
        sha, label, _, _ = resolve("pr", "pull-request/7", "release/1.3.0", "")
        assert sha == ""
        assert "no parseable current version" in label

    def test_nightly_without_run_id(self):
        # No run id -> no API call, empty sha, descriptive label.
        sha, label, _, _ = resolve("nightly", "main", "", "", repo="ai-dynamo/dynamo")
        assert sha == ""
        assert "no current run id" in label
