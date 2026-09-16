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

"""Unit tests for the PR issue-link check.

Two properties carry the check and neither is visible from reading one branch:

  - What becomes a candidate. PR text is untrusted, and a reference that
    cannot survive URL construction must never reach a lookup, because a
    failed lookup is reported as an outage and an outage passes the check.
  - When the check may fail open. Fail-open exists for upstream outages. A
    lookup budget is not an outage, so a pull request whose checked
    references all came back definitively not-an-issue must still fail.

Every test drives `main()` with the three verification calls replaced, so
nothing here touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow `import pr_issue_link` when pytest runs from the repo root.
sys.path.insert(0, str(Path(__file__).parent))

import pr_issue_link  # noqa: E402

REPO = "ai-dynamo/dynamo"

ENV_DEFAULTS = {
    "PR_TITLE": "",
    "PR_BODY": "",
    "PR_HEAD_REF": "",
    "PR_AUTHOR": "someone",
    "PR_AUTHOR_ASSOCIATION": "MEMBER",
    "PR_HEAD_REPO": REPO,
    "REPO": REPO,
    "GITHUB_TOKEN": "gh-token",
    "LINEAR_API_KEY": "linear-key",
}


class FakeApi:
    """Programmed stand-in for the three verification calls.

    Each map is keyed the way the check asks: `owner/repo#number` for GitHub
    issues, `owner/repo` for repository visibility, and the identifier for
    Linear. Anything absent falls to the default, so a test states only the
    references it cares about. Every call is recorded, which is how the
    ordering tests assert what the lookup budget was spent on.
    """

    def __init__(
        self,
        github: dict[str, tuple[bool, bool]] | None = None,
        repos: dict[str, tuple[bool, bool]] | None = None,
        linear: dict[str, tuple[bool, bool]] | None = None,
        default_github: tuple[bool, bool] = (False, True),
        default_linear: tuple[bool, bool] = (False, True),
    ) -> None:
        self.github = github or {}
        self.repos = repos or {}
        self.linear = linear or {}
        self.default_github = default_github
        self.default_linear = default_linear
        self.github_calls: list[str] = []
        self.repo_calls: list[str] = []
        self.linear_calls: list[str] = []

    def verify_github_issue(
        self, repo: str, number: str, token: str
    ) -> tuple[bool, bool]:
        key = f"{repo}#{number}"
        self.github_calls.append(key)
        return self.github.get(key, self.default_github)

    def repo_visible(self, repo: str, token: str) -> tuple[bool, bool]:
        self.repo_calls.append(repo)
        return self.repos.get(repo, (True, True))

    def verify_linear_issue(self, identifier: str, api_key: str) -> tuple[bool, bool]:
        self.linear_calls.append(identifier)
        return self.linear.get(identifier, self.default_linear)


def run(
    monkeypatch: pytest.MonkeyPatch, api: FakeApi | None = None, **env: str
) -> tuple[int, FakeApi]:
    """Run `main()` against a fake API and return its exit code."""
    api = api or FakeApi()
    monkeypatch.setattr(pr_issue_link, "verify_github_issue", api.verify_github_issue)
    monkeypatch.setattr(pr_issue_link, "repo_visible", api.repo_visible)
    monkeypatch.setattr(pr_issue_link, "verify_linear_issue", api.verify_linear_issue)
    # The runner sets GITHUB_STEP_SUMMARY; leaving it set would make these
    # tests append to the real job summary.
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("BLOCKING_DATE", raising=False)
    for key, value in {**ENV_DEFAULTS, **env}.items():
        monkeypatch.setenv(key, value)
    return pr_issue_link.main(), api


# ------------------------------------------------------------------
# What becomes a candidate
# ------------------------------------------------------------------


def test_same_repo_issue_reference_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeApi(github={f"{REPO}#123": (True, True)})
    code, api = run(monkeypatch, api, PR_BODY="Fixes #123")
    assert code == 0
    assert api.github_calls == [f"{REPO}#123"]


def test_cross_repo_reference_inside_the_org_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(github={"ai-dynamo/enhancements#12": (True, True)})
    code, api = run(monkeypatch, api, PR_BODY="Part of ai-dynamo/enhancements#12")
    assert code == 0
    assert api.github_calls == ["ai-dynamo/enhancements#12"]


def test_issue_url_form_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeApi(github={"ai-dynamo/enhancements#12": (True, True)})
    code, api = run(
        monkeypatch,
        api,
        PR_BODY="Closes https://github.com/ai-dynamo/enhancements/issues/12",
    )
    assert code == 0


def test_cross_repo_reference_outside_the_org_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, api = run(monkeypatch, PR_BODY="See other-org/thing#5")
    assert code == 1
    assert api.github_calls == []


def test_non_ascii_repository_segment_never_reaches_a_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment urllib cannot encode must not become a candidate.

    The lookup would raise UnicodeEncodeError, which `http_json` reports as
    status 0, which the check reads as an outage and passes on.
    """
    code, api = run(monkeypatch, PR_BODY="See ai-dynamo/日本語#1")
    assert code == 1
    assert api.github_calls == []


def test_overlong_repository_segment_never_reaches_a_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment past GitHub's 100-character limit must not become a candidate.

    The lookup would come back 414, which is neither 200 nor 404/410, so the
    check reads it as an outage and passes on.
    """
    code, api = run(monkeypatch, PR_BODY=f"See ai-dynamo/{'a' * 9000}#1")
    assert code == 1
    assert api.github_calls == []


def test_html_comment_reference_does_not_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, api = run(monkeypatch, PR_BODY="<!-- Closes #5 -->")
    assert code == 1
    assert api.github_calls == []


def test_linear_identifier_in_the_branch_name_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(linear={"DYN-1234": (True, True)})
    code, api = run(monkeypatch, api, PR_HEAD_REF="user/dyn-1234-short-description")
    assert code == 0
    assert api.linear_calls == ["DYN-1234"]


def test_bot_author_skips_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    code, api = run(monkeypatch, PR_AUTHOR="dependabot[bot]")
    assert code == 0
    assert api.github_calls == []
    assert api.linear_calls == []


# ------------------------------------------------------------------
# When the check may fail open
# ------------------------------------------------------------------


def test_api_outage_on_the_only_reference_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(default_github=(False, False))
    code, api = run(monkeypatch, api, PR_BODY="Fixes #5")
    assert code == 0


def test_definitive_missing_reference_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    code, api = run(monkeypatch, PR_BODY="Fixes #5")
    assert code == 1
    assert api.github_calls == [f"{REPO}#5"]


def test_overflow_with_definitive_answers_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crossing the lookup budget must not convert a fail into a pass.

    Eleven references that every checked lookup calls not-an-issue is a pull
    request with no linked issue, not an upstream outage.
    """
    body = " ".join(f"#{n}" for n in range(101, 112))
    code, api = run(monkeypatch, PR_BODY=body)
    assert code == 1
    assert len(api.github_calls) == pr_issue_link.MAX_CANDIDATES


def test_overflow_with_an_api_outage_still_fails_open(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeApi(default_github=(False, False))
    body = " ".join(f"#{n}" for n in range(101, 112))
    code, api = run(monkeypatch, api, PR_BODY=body)
    assert code == 0
    assert "beyond the 10-lookup bound" in capsys.readouterr().out


def test_overflow_spends_the_budget_on_the_magic_word_reference_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent ordering keeps the budget from burning on incidental references.

    `#999` sorts last by number, so without ordering an aggregation pull
    request's real reference is the one that never gets checked.
    """
    api = FakeApi(github={f"{REPO}#999": (True, True)})
    body = " ".join(f"#{n}" for n in range(101, 112)) + "\n\nCloses #999"
    code, api = run(monkeypatch, api, PR_BODY=body)
    assert code == 0
    assert api.github_calls == [f"{REPO}#999"]


def test_linear_overflow_with_definitive_answers_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = " ".join(f"AAA-{n}" for n in range(1, 12))
    code, api = run(monkeypatch, PR_BODY=body)
    assert code == 1
    assert len(api.linear_calls) == pr_issue_link.MAX_CANDIDATES


def test_linear_overflow_with_an_api_outage_still_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(default_linear=(False, False))
    body = " ".join(f"AAA-{n}" for n in range(1, 12))
    code, api = run(monkeypatch, api, PR_BODY=body)
    assert code == 0


def test_linear_overflow_spends_the_budget_on_the_branch_identifier_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(linear={"DYN-4242": (True, True)})
    body = " ".join(f"AAA-{n}" for n in range(1, 12))
    code, api = run(monkeypatch, api, PR_BODY=body, PR_HEAD_REF="user/dyn-4242-thing")
    assert code == 0
    assert api.linear_calls == ["DYN-4242"]


# ------------------------------------------------------------------
# Fork gating and reporting
# ------------------------------------------------------------------


def test_untrusted_fork_does_not_verify_linear_identifiers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, api = run(
        monkeypatch,
        PR_BODY="Closes DYN-1234",
        PR_HEAD_REPO="contributor/dynamo",
        PR_AUTHOR_ASSOCIATION="CONTRIBUTOR",
    )
    assert code == 1
    assert api.linear_calls == []
    assert "cannot be" in capsys.readouterr().out


def test_org_author_on_a_fork_still_verifies_linear_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(linear={"DYN-1234": (True, True)})
    code, api = run(
        monkeypatch,
        api,
        PR_BODY="Closes DYN-1234",
        PR_HEAD_REPO="contributor/dynamo",
        PR_AUTHOR_ASSOCIATION="MEMBER",
    )
    assert code == 0
    assert api.linear_calls == ["DYN-1234"]


def test_invisible_cross_repo_reference_is_reported_not_passed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeApi(repos={"ai-dynamo/private-thing": (False, True)})
    code, api = run(monkeypatch, api, PR_BODY="Part of ai-dynamo/private-thing#7")
    assert code == 1
    out = capsys.readouterr().out
    assert "ai-dynamo/private-thing#7" in out
    assert "cannot see" in out


def test_missing_message_names_the_blocking_date(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = run(monkeypatch, BLOCKING_DATE="2026-10-15")
    assert code == 1
    assert "becomes required on 2026-10-15" in capsys.readouterr().out


def test_summary_is_appended_to_the_step_summary_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary = tmp_path / "summary.md"
    api = FakeApi(github={f"{REPO}#123": (True, True)})
    monkeypatch.setattr(pr_issue_link, "verify_github_issue", api.verify_github_issue)
    monkeypatch.setattr(pr_issue_link, "repo_visible", api.repo_visible)
    monkeypatch.setattr(pr_issue_link, "verify_linear_issue", api.verify_linear_issue)
    for key, value in {**ENV_DEFAULTS, "PR_BODY": "Fixes #123"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert pr_issue_link.main() == 0
    assert "GitHub issue #123" in summary.read_text()
