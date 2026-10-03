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

import re
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
    Linear. `deps` holds the GitHub keys that carry a `dep:` lifecycle label. Anything absent falls to the default, so a test states only the
    references it cares about. Every call is recorded, which is how the
    ordering tests assert what the lookup budget was spent on.
    """

    def __init__(
        self,
        github: dict[str, tuple[bool, bool]] | None = None,
        repos: dict[str, tuple[bool, bool]] | None = None,
        linear: dict[str, tuple[bool, bool]] | None = None,
        deps: set[str] | None = None,
        default_github: tuple[bool, bool] = (False, True),
        default_linear: tuple[bool, bool] = (False, True),
    ) -> None:
        self.github = github or {}
        self.deps = set(deps or ())
        self.repos = repos or {}
        self.linear = linear or {}
        self.default_github = default_github
        self.default_linear = default_linear
        self.github_calls: list[str] = []
        self.repo_calls: list[str] = []
        self.linear_calls: list[str] = []

    def verify_github_issue(
        self, repo: str, number: str, token: str
    ) -> tuple[bool, bool, bool]:
        key = f"{repo}#{number}"
        self.github_calls.append(key)
        exists, api_ok = self.github.get(key, self.default_github)
        return exists, api_ok, key in self.deps

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


# ------------------------------------------------------------------
# The paths that decline to verify
# ------------------------------------------------------------------


@pytest.mark.parametrize("count", [10, 11])
def test_invisible_cross_repo_references_fail_on_both_sides_of_the_bound(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    """Declining to look is a decision, so the bound must not rescue it.

    An invented repository name answers 404 on both the issue and the
    repository, which lands the reference in `invisible_repo_refs`. Nothing has
    to exist for a body to carry eleven of them.
    """
    api = FakeApi(repos={f"ai-dynamo/absent-{n}": (False, True) for n in range(count)})
    body = " ".join(f"ai-dynamo/absent-{n}#7" for n in range(count))
    code, api = run(monkeypatch, api, PR_BODY=body)
    assert code == 1
    assert len(api.repo_calls) == min(count, pr_issue_link.MAX_CANDIDATES)


def test_invisible_cross_repo_overflow_with_a_repository_outage_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The repository lookup failing is an outage, and outages still pass."""
    api = FakeApi(repos={f"ai-dynamo/absent-{n}": (False, False) for n in range(11)})
    body = " ".join(f"ai-dynamo/absent-{n}#7" for n in range(11))
    code, api = run(monkeypatch, api, PR_BODY=body)
    assert code == 0


@pytest.mark.parametrize("count", [10, 11])
def test_fork_linear_identifiers_fail_on_both_sides_of_the_bound(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    """A fork's Linear identifiers are never looked up, so they never pass.

    The fork gate skips verification before any request, so without marking
    the candidate decided the bound turns these into a pass having made no API
    call at all.
    """
    body = " ".join(f"AAA-{n}" for n in range(1, count + 1))
    code, api = run(
        monkeypatch,
        PR_BODY=body,
        PR_HEAD_REPO="contributor/dynamo",
        PR_AUTHOR_ASSOCIATION="CONTRIBUTOR",
    )
    assert code == 1
    assert api.linear_calls == []


def test_fork_linear_overflow_does_not_mask_a_github_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fork PR with a real GitHub outage still fails open."""
    api = FakeApi(default_github=(False, False))
    body = " ".join(f"AAA-{n}" for n in range(1, 12)) + "\n\nFixes #5"
    code, api = run(
        monkeypatch,
        api,
        PR_BODY=body,
        PR_HEAD_REPO="contributor/dynamo",
        PR_AUTHOR_ASSOCIATION="CONTRIBUTOR",
    )
    assert code == 0


# ------------------------------------------------------------------
# Which pull requests the check runs on at all
# ------------------------------------------------------------------


def test_the_check_runs_only_against_main() -> None:
    """A branch cut for a release must not be subject to this check.

    It takes cherry-picks, and a cherry-pick cites the original pull request
    rather than an issue. The issue link belongs on that original change,
    which was already checked on its way into `main`. Nothing in the
    script enforces this: the base-branch filter on the trigger is the whole
    mechanism, and dropping it would put a red check on every cherry-pick.
    """
    workflow = (Path(__file__).parent / "pr-issue-link.yml").read_text()
    block = re.search(r"\n  pull_request_target:\n((?:    [^\n]*\n|\n)*)", workflow)
    assert block, "the workflow does not trigger on pull_request_target"
    assert "branches: [main]" in block.group(1)


# ------------------------------------------------------------------
# The blocking date, which lives in three files a human edits
# ------------------------------------------------------------------


def test_the_template_and_the_workflow_name_the_same_blocking_date() -> None:
    """Nothing else keeps the three copies in step.

    `BLOCKING_DATE` drives the message a contributor sees when the check
    fails. The pull request template carries the same date so they read it
    before it fails, and the script repeats it as a default for any run that
    does not set the variable. Bumping one without the others leaves the
    template telling people the old date, or the script announcing one the
    workflow never chose.
    """
    workflows = Path(__file__).parent
    workflow = (workflows / "pr-issue-link.yml").read_text()
    template = (workflows.parent / "pull_request_template.md").read_text()
    script = (workflows / "pr_issue_link.py").read_text()
    match = re.search(r'BLOCKING_DATE:\s*"(\d{4}-\d{2}-\d{2})"', workflow)
    assert match, "the workflow does not set BLOCKING_DATE"
    date = match.group(1)
    assert f"becomes required on {date}" in template
    default = re.search(
        r'os\.environ\.get\(\s*"BLOCKING_DATE",\s*"(\d{4}-\d{2}-\d{2})"', script
    )
    assert default, "the script does not default BLOCKING_DATE"
    assert default.group(1) == date


# ------------------------------------------------------------------
# Proposal umbrellas
# ------------------------------------------------------------------

DEP = f"{REPO}#14897"


def test_a_proposal_alone_does_not_satisfy_the_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A DEP is a real issue, and on its own it is not a linked unit of work."""
    api = FakeApi(github={DEP: (True, True)}, deps={DEP})
    code, api = run(monkeypatch, api, PR_BODY="Part of #14897")
    assert code == 1
    out = capsys.readouterr().out
    assert "proposal only" in out
    assert "#14897" in out


def test_a_work_issue_alongside_a_proposal_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeApi(github={DEP: (True, True), f"{REPO}#123": (True, True)}, deps={DEP})
    code, api = run(monkeypatch, api, PR_BODY="Closes #123\n\nPart of #14897")
    assert code == 0
    assert "proposal only" not in capsys.readouterr().out


def test_a_proposal_does_not_stop_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """The proposal is checked first, and the work issue behind it still passes.

    The work issue carries no keyword here, so it sorts behind the proposal
    and the loop has to continue past a decided candidate to reach it.
    """
    dep = f"{REPO}#5"
    api = FakeApi(github={dep: (True, True), f"{REPO}#900": (True, True)}, deps={dep})
    code, api = run(monkeypatch, api, PR_BODY="Part of #5, see also #900")
    assert code == 0
    assert api.github_calls == [dep, f"{REPO}#900"]


def test_proposals_past_the_bound_are_not_rescued_by_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proposal is a decided candidate, so the bound must not pass it."""
    keys = {f"{REPO}#{n}": (True, True) for n in range(101, 112)}
    api = FakeApi(github=keys, deps=set(keys))
    body = " ".join(f"#{n}" for n in range(101, 112))
    code, api = run(monkeypatch, api, PR_BODY=body)
    assert code == 1


def test_a_proposal_with_an_outage_fails_open_and_still_names_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeApi(github={DEP: (True, True), f"{REPO}#7": (False, False)}, deps={DEP})
    code, api = run(monkeypatch, api, PR_BODY="Part of #14897 and Fixes #7")
    assert code == 0
    out = capsys.readouterr().out
    assert "#14897 is a Dynamo Enhancement Proposal" in out


def test_a_closing_reference_is_checked_before_a_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Closes` says unit of work; `Part of` says umbrella. Rank accordingly."""
    dep = f"{REPO}#5"
    api = FakeApi(github={dep: (True, True), f"{REPO}#900": (True, True)}, deps={dep})
    code, api = run(monkeypatch, api, PR_BODY="Part of #5 and Closes #900")
    assert code == 0
    assert api.github_calls == [f"{REPO}#900"]


def test_ten_proposals_do_not_hide_the_work_issue_behind_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decided candidate spends a lookup, so ten of them could bury the work.

    Without closing-first ordering the work issue falls outside the budget and
    a correctly linked pull request is reported as proposal only.
    """
    deps = {f"{REPO}#{n}": (True, True) for n in range(1, 11)}
    api = FakeApi(github={**deps, f"{REPO}#900": (True, True)}, deps=set(deps))
    body = " ".join(f"Part of #{n}" for n in range(1, 11)) + "\n\nCloses #900"
    code, api = run(monkeypatch, api, PR_BODY=body)
    assert code == 0
    assert api.github_calls == [f"{REPO}#900"]


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (11, "1 further reference went unchecked"),
        (13, "3 further references went unchecked"),
    ],
)
def test_the_failure_says_when_candidates_went_unchecked(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    count: int,
    expected: str,
) -> None:
    """Closing-form proposals exhaust the budget and the rest go unlooked-at.

    Both halves of the sentence are pinned. The subject agreed in number from
    the start and the pronoun did not, which is the shape of fault that came
    back once already because nothing held it.
    """
    keys = {f"{REPO}#{n}": (True, True) for n in range(101, 101 + count)}
    api = FakeApi(github=keys, deps=set(keys))
    body = " ".join(f"Closes #{n}" for n in range(101, 101 + count))
    code, api = run(monkeypatch, api, PR_BODY=body)
    assert code == 1
    out = capsys.readouterr().out
    assert expected in out
    assert "before reaching the rest." in out
    assert "are Dynamo Enhancement Proposals" in out
