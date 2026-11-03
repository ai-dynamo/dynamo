# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import pr_issue_link  # noqa: E402
import recheck_at_cut  # noqa: E402

REPO = "ai-dynamo/dynamo"


def pr(number: int, body: str = "", draft: bool = False, sha: str = "") -> dict:
    return {
        "number": number,
        "title": f"fix: change {number}",
        "body": body,
        "draft": draft,
        "author_association": "MEMBER",
        "user": {"login": "someone"},
        "head": {
            "ref": "someone/no-link",
            "sha": sha or f"{number:040x}",
            "repo": {"full_name": REPO},
        },
    }


class FakeGitHub:
    """Serves the pulls list and records check runs; any other call fails."""

    def __init__(
        self, prs: list[dict], publish_status: int = 201, issues: set[str] = frozenset()
    ) -> None:
        self.pages = [prs[i : i + 100] for i in range(0, len(prs), 100)] + [[]]
        self.publish_status = publish_status
        self.issues = issues
        self.published: list[dict] = []
        self.publish_headers: list[dict] = []
        self.other_calls: list[str] = []

    def http_json(
        self, url: str, payload: dict | None = None, headers: dict | None = None
    ) -> tuple[int, object]:
        if "/pulls?" in url:
            page = int(url.rsplit("page=", 1)[1])
            return 200, self.pages[page - 1]
        if url.endswith("/check-runs"):
            assert payload is not None
            self.published.append(payload)
            self.publish_headers.append(headers or {})
            return self.publish_status, {}
        self.other_calls.append(url)
        return 500, {}

    def verify_github_issue(
        self, repo: str, number: str, token: str
    ) -> tuple[bool, bool, bool, str]:
        return f"{repo}#{number}" in self.issues, True, False, ""


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch):
    def install(fake: FakeGitHub) -> FakeGitHub:
        monkeypatch.setattr(pr_issue_link, "http_json", fake.http_json)
        monkeypatch.setattr(
            pr_issue_link, "verify_github_issue", fake.verify_github_issue
        )
        monkeypatch.setattr(
            pr_issue_link, "verify_linear_issue", lambda *a, **k: (False, True)
        )
        monkeypatch.setattr(
            pr_issue_link, "release_branch_cut", lambda *a, **k: (True, True)
        )
        monkeypatch.setenv("REPO", REPO)
        monkeypatch.setenv("GITHUB_TOKEN", "token")
        monkeypatch.setenv("BLOCKING_BRANCH", "release/1.7.0")
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        monkeypatch.delenv("PR_TITLE", raising=False)
        return fake

    return install


def by_sha(fake: FakeGitHub) -> dict[str, str]:
    return {run["head_sha"]: run["conclusion"] for run in fake.published}


def test_linkless_pr_with_a_45_day_old_green_check_gets_a_fresh_failure(
    github, capsys: pytest.CaptureFixture[str]
) -> None:
    # The last green run predates GitHub's 30-day re-run limit. The cut must
    # not depend on it: it evaluates the PR as it stands and publishes anew.
    stale = pr(7)
    stale["last_green_check_at"] = (
        datetime.now(timezone.utc) - timedelta(days=45)
    ).isoformat()
    fake = github(FakeGitHub([stale]))

    assert recheck_at_cut.main(sleep=lambda _: None) == 0

    assert by_sha(fake) == {stale["head"]["sha"]: "failure"}
    run = fake.published[0]
    assert run["name"] == "Require an issue link"
    assert run["status"] == "completed"
    assert "missing" in run["output"]["summary"]
    assert fake.publish_headers[0]["Content-Type"] == "application/json"
    assert not any("actions/runs" in url for url in fake.other_calls)


def test_linked_pr_passes_and_draft_is_skipped(github) -> None:
    linked = pr(8, body="Fixes #123")
    draft = pr(9, draft=True)
    fake = github(FakeGitHub([linked, draft], issues={f"{REPO}#123"}))

    assert recheck_at_cut.main(sleep=lambda _: None) == 0

    assert by_sha(fake) == {linked["head"]["sha"]: "success"}


def test_every_page_of_open_prs_is_checked(github) -> None:
    prs = [pr(n) for n in range(1, 251)]
    fake = github(FakeGitHub(prs))

    assert recheck_at_cut.main(sleep=lambda _: None) == 0

    assert len(fake.published) == 250
    assert set(by_sha(fake).values()) == {"failure"}


def test_failure_to_publish_fails_the_job_and_names_the_pr(
    github, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = github(FakeGitHub([pr(10)], publish_status=403))

    assert recheck_at_cut.main(sleep=lambda _: None) == 1

    assert len(fake.published) == 3
    out = capsys.readouterr().out
    assert "::error::Could not publish the check for 1 pull requests (#10)" in out


def test_listing_failure_raises(github, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = github(FakeGitHub([pr(11)]))
    monkeypatch.setattr(pr_issue_link, "http_json", lambda *a, **k: (502, {}))

    with pytest.raises(RuntimeError, match="HTTP 502"):
        recheck_at_cut.main(sleep=lambda _: None)
    assert fake.published == []


def test_evaluation_restores_the_environment(
    github, monkeypatch: pytest.MonkeyPatch
) -> None:
    github(FakeGitHub([]))
    monkeypatch.setenv("PR_TITLE", "outer")

    recheck_at_cut.evaluate(pr(12))

    assert os.environ["PR_TITLE"] == "outer"
    assert "GITHUB_STEP_SUMMARY" not in os.environ
