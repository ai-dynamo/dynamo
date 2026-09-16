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

"""Verify that a pull request references at least one existing issue.

Accepted references, matching how the Linear GitHub integration links work:

- A Linear issue ID (for example ``DYN-1234``) in the PR title or description,
  optionally behind a magic word (``Closes DYN-1234``), or embedded in the
  branch name (``user/dyn-1234-short-description``).
- A GitHub issue reference in the PR title or description: ``#123``, a
  closing keyword form (``Fixes #123``), an org-scoped cross-repo reference
  (``ai-dynamo/enhancements#12`` - contribution requests and the older DEPs
  live in sibling repositories), or a full issue URL.

A Dynamo Enhancement Proposal does not satisfy the check on its own. A DEP is
a proposal umbrella that stays open across many pull requests, so it says what
the work is part of and not what any one change is. Keep the ``Part of <dep>``
reference and link the work issue as well.

Every candidate is verified against the corresponding API; a reference to an
issue that does not exist does not count. Verification failures caused by API
outages are treated as unverified-but-present so that an upstream outage never
fails anyone's PR (fail open).
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

# An owner or repository name is ASCII, starts alphanumeric, and runs to at
# most 100 characters. Matching the segment on `\w` instead let PR text name a
# repository that no lookup URL can carry. A non-ASCII segment made urllib
# raise UnicodeEncodeError, a ValueError, which `http_json` reports as status
# 0 and the check then reads as an outage and passes; a 9000-character segment
# did the same by way of a 414. Bounding the pattern keeps both out of the
# candidate set, which is also the only place the reference text can be
# bounded before it is echoed into the step summary.
REPO_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}"
REPO_PAT = rf"{REPO_SEGMENT}/{REPO_SEGMENT}"
LINEAR_ID = r"[A-Z][A-Z0-9]{1,9}-\d{1,6}"
LINEAR_TEXT_RE = re.compile(rf"\b({LINEAR_ID})\b")
LINEAR_BRANCH_RE = re.compile(r"(?:^|[/_-])([a-z][a-z0-9]{1,9}-\d{1,6})(?:$|[/_-])")
GITHUB_REF_RE = re.compile(r"(?:^|[^\w&])#(\d{1,7})\b")
CROSS_REPO_RE = re.compile(rf"\b({REPO_PAT})#(\d{{1,7}})\b")
ISSUE_URL_RE = re.compile(rf"github\.com/({REPO_PAT})/issues/(\d{{1,7}})\b")
# A magic word marks the reference the author meant as the link. Both forms
# order the lookup budget, so a release pull request carrying dozens of
# references still spends its lookups on the one that matters.
# A closing keyword says the issue is the unit of work this pull request
# completes. A non-closing one says the issue outlives it, which is what a
# proposal umbrella takes. Closing references are checked first, so ten
# `Part of <dep>` lines cannot spend the lookup budget ahead of the
# `Closes #900` that actually answers the check.
CLOSING_WORDS = r"clos(?:e|es|ed)|fix(?:es|ed)?|resolv(?:e|es|ed)"
REFERENCE_WORDS = r"part of|refs?|relates to"
MAGIC_WORDS = rf"{CLOSING_WORDS}|{REFERENCE_WORDS}"
_GITHUB_REF_TAIL = (
    rf"\b[\s:]*(?:https?://github\.com/({REPO_PAT})/issues/|({REPO_PAT})?#)"
    rf"(\d{{1,7}})\b"
)
INTENT_GITHUB_RE = re.compile(rf"\b(?:{MAGIC_WORDS}){_GITHUB_REF_TAIL}", re.IGNORECASE)
CLOSING_GITHUB_RE = re.compile(
    rf"\b(?:{CLOSING_WORDS}){_GITHUB_REF_TAIL}", re.IGNORECASE
)
INTENT_LINEAR_RE = re.compile(
    rf"\b(?:{MAGIC_WORDS})\b[\s:]*({LINEAR_ID})\b", re.IGNORECASE
)
HTML_COMMENT_RE = re.compile(r"<!--.*?(?:-->|$)", re.DOTALL)
BOT_AUTHORS = {"dependabot[bot]", "github-actions[bot]", "copy-pr-bot[bot]"}
# Every Dynamo Enhancement Proposal carries a lifecycle label under this
# prefix, set by `.github/ISSUE_TEMPLATE/dep.yml` and moved through review by
# the `dep-update` skill. Matching the label rather than the title keeps an
# issue *about* the DEP process from reading as a DEP.
DEP_LABEL_PREFIX = "dep:"
# PR text is untrusted input; bound the number of authenticated lookups it
# can trigger.
MAX_CANDIDATES = 10


def http_json(
    url: str, payload: dict | None = None, headers: dict | None = None
) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        # Status 0 flows through the api_ok=False paths so the reference is
        # reported as unverified rather than failing the workflow.
        return 0, {}


def verify_github_issue(repo: str, number: str, token: str) -> tuple[bool, bool, bool]:
    """Return (exists_as_issue, api_ok, is_dep).

    The labels ride along on the response the check already makes, so knowing
    a reference is a proposal umbrella rather than a unit of work costs no
    extra call.
    """
    status, body = http_json(
        f"https://api.github.com/repos/{repo}/issues/{number}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    if status == 200:
        names = [(label or {}).get("name") or "" for label in body.get("labels") or []]
        is_dep = any(name.startswith(DEP_LABEL_PREFIX) for name in names)
        return "pull_request" not in body, True, is_dep
    if status in (404, 410):
        return False, True, False
    return False, False, False


def repo_visible(repo: str, token: str) -> tuple[bool, bool]:
    """Return (visible, api_ok)."""
    status, _ = http_json(
        f"https://api.github.com/repos/{repo}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    if status == 200:
        return True, True
    if status in (403, 404, 410):
        return False, True
    return False, False


def verify_linear_issue(identifier: str, api_key: str) -> tuple[bool, bool]:
    """Return (exists, api_ok)."""
    if not api_key:
        return False, False
    status, body = http_json(
        "https://api.linear.app/graphql",
        payload={
            "query": "query($id: String!) { issue(id: $id) { identifier } }",
            "variables": {"id": identifier},
        },
        headers={"Authorization": api_key, "Content-Type": "application/json"},
    )
    if status == 200:
        issue = (body.get("data") or {}).get("issue")
        if issue:
            return True, True
        errors = body.get("errors") or []
        if errors and not all(
            (e.get("extensions") or {}).get("code") == "INPUT_ERROR" for e in errors
        ):
            # A 200 carrying resolver or service errors is an API failure,
            # not a definitive not-found; fail open.
            return False, False
        # Unknown identifiers come back as 200 with INPUT_ERROR: definitive.
        return False, True
    if status in (400,) and body.get("errors"):
        # Linear returns errors for unknown identifiers.
        return False, True
    return False, False


def main() -> int:
    title = os.environ.get("PR_TITLE", "")
    body = os.environ.get("PR_BODY", "") or ""
    branch = os.environ.get("PR_HEAD_REF", "")
    author = os.environ.get("PR_AUTHOR", "")
    repo = os.environ.get("REPO", "")
    head_repo = os.environ.get("PR_HEAD_REPO", repo)
    association = os.environ.get("PR_AUTHOR_ASSOCIATION", "").upper()
    # Org authors keep Linear verification from forks; the fork gating
    # below applies only to authors outside the org.
    trusted_author = association in {"OWNER", "MEMBER", "COLLABORATOR"}
    is_untrusted_fork = head_repo.lower() != repo.lower() and not trusted_author
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    linear_key = os.environ.get("LINEAR_API_KEY", "")
    blocking_date = os.environ.get("BLOCKING_DATE", "2026-10-15")

    if author in BOT_AUTHORS:
        print(f"Author {author} is a bot; skipping the issue-link check.")
        return 0

    # The PR template carries example references inside HTML comments;
    # commented-out text must not satisfy the check.
    text = HTML_COMMENT_RE.sub(" ", f"{title}\n{body}")
    linear_ids = set(LINEAR_TEXT_RE.findall(text))
    branch_ids = {m.upper() for m in LINEAR_BRANCH_RE.findall(branch)}
    linear_ids.update(branch_ids)
    github_refs = {(repo, n) for n in GITHUB_REF_RE.findall(text)}
    org = repo.split("/")[0]
    for other_repo, number in CROSS_REPO_RE.findall(text) + ISSUE_URL_RE.findall(text):
        # Cross-repo references count when they stay inside the same org.
        # Contribution requests and the DEPs written before August live in
        # sibling repositories. A DEP filed today is an issue in this
        # repository, from `.github/ISSUE_TEMPLATE/dep.yml`, so it needs no
        # cross-repo form at all.
        if other_repo.split("/")[0].lower() == org.lower():
            github_refs.add((other_repo, number))

    # A reference behind a magic word, or the identifier in the branch name
    # the author chose, is the one they meant. Ordering the candidates by that
    # spends the lookup budget on it first.
    intent_github = {
        (url_repo or inline_repo or repo, number)
        for url_repo, inline_repo, number in INTENT_GITHUB_RE.findall(text)
    }
    closing_github = {
        (url_repo or inline_repo or repo, number)
        for url_repo, inline_repo, number in CLOSING_GITHUB_RE.findall(text)
    }
    intent_linear = {m.upper() for m in INTENT_LINEAR_RE.findall(text)} | branch_ids

    verified: list[str] = []
    # Proposal umbrellas found along the way. They are real tracked issues, so
    # the lookup is decisive and the bound must not rescue them, but they do
    # not satisfy the check and the search continues past them.
    dep_refs: list[str] = []
    unverified: list[str] = []
    invisible_repo_refs: list[str] = []
    repo_visibility: dict[str, tuple[bool, bool]] = {}

    all_refs = sorted(
        github_refs,
        key=lambda r: (
            r not in closing_github,
            r not in intent_github,
            r[0],
            int(r[1]),
        ),
    )
    ordered_refs = all_refs[:MAX_CANDIDATES]
    # True once any candidate has been decided: the API answered about it, or
    # the check deliberately declined to look. Both are decisions. Only an
    # outage leaves a candidate undecided, and only an outage earns the
    # fail-open pass on the candidates past the bound.
    decided_github = False
    for ref_repo, number in ordered_refs:
        exists, api_ok, is_dep = verify_github_issue(ref_repo, number, gh_token)
        label = f"#{number}" if ref_repo == repo else f"{ref_repo}#{number}"
        if exists and is_dep:
            dep_refs.append(label)
            decided_github = True
        elif exists:
            verified.append(f"GitHub issue {label}")
        elif not api_ok:
            unverified.append(f"GitHub reference {label} (API unavailable)")
        elif ref_repo != repo:
            # A 404 on a cross-repo issue can mean the workflow token cannot
            # see that repository (internal visibility) rather than that the
            # issue does not exist. Disambiguate against the repository
            # itself: a visible repository makes the 404 a definitive
            # missing issue, while an invisible repository is reported but
            # does not by itself pass the check.
            if ref_repo not in repo_visibility:
                repo_visibility[ref_repo] = repo_visible(ref_repo, gh_token)
            visible, repo_api_ok = repo_visibility[ref_repo]
            if not repo_api_ok:
                unverified.append(f"GitHub reference {label} (API unavailable)")
            else:
                # An invisible repository is an answer, not an outage: the
                # repository lookup came back cleanly. Eleven invented
                # repository names in a description would otherwise cross the
                # bound below and pass with nothing needing to exist.
                if not visible:
                    invisible_repo_refs.append(label)
                decided_github = True
        else:
            decided_github = True
        if verified:
            break

    if not verified and len(all_refs) > MAX_CANDIDATES and not decided_github:
        # Aggregation and release pull requests can carry more references than
        # the lookup budget, and a candidate nobody checked must not hard-fail
        # a pull request. Once a candidate has been decided, though, the cap is
        # a cap and not an outage. Reporting the remainder as unverified there
        # passed a pull request whose every reference resolved to another pull
        # request, and one whose every reference named a repository that does
        # not exist.
        unverified.append(
            f"{len(all_refs) - MAX_CANDIDATES} further GitHub references "
            f"beyond the {MAX_CANDIDATES}-lookup bound (not verified)"
        )

    all_linear_ids = sorted(linear_ids, key=lambda i: (i not in intent_linear, i))
    fork_linear_ids: list[str] = []
    decided_linear = False
    for identifier in all_linear_ids[:MAX_CANDIDATES]:
        if verified:
            break
        if is_untrusted_fork:
            # Do not turn the CI into an existence oracle for Linear IDs
            # guessed from outside-org fork PRs, and do not let an
            # unverifiable ID pass the check; community contributors
            # reference GitHub issues.
            fork_linear_ids.append(identifier)
            # Declining to look is a decision, not an outage. Without this the
            # bound below turns eleven Linear-shaped tokens from a fork into a
            # pass, having made no API call at all.
            decided_linear = True
            continue
        exists, api_ok = verify_linear_issue(identifier, linear_key)
        if exists:
            verified.append(f"Linear issue {identifier}")
        elif not api_ok:
            unverified.append(f"Linear reference {identifier} (not verified)")
        else:
            decided_linear = True

    if not verified and len(all_linear_ids) > MAX_CANDIDATES and not decided_linear:
        unverified.append(
            f"{len(all_linear_ids) - MAX_CANDIDATES} further Linear references "
            f"beyond the {MAX_CANDIDATES}-lookup bound (not verified)"
        )

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")

    def summarize(lines: list[str]) -> None:
        text_out = "\n".join(lines)
        print(text_out)
        if summary_path:
            with open(summary_path, "a") as f:
                f.write(text_out + "\n")

    if verified:
        summarize(["### PR issue link: found", ""] + [f"- {v}" for v in verified])
        return 0

    if unverified:
        # References are present but an API kept us from verifying them.
        # Fail open: an upstream outage should never fail anyone's PR. Any
        # proposal found along the way is still named, so the author is not
        # left thinking it was what passed.
        summarize(
            ["### PR issue link: present but unverified", ""]
            + [f"- {u}" for u in unverified]
            + [
                f"- {d} is a Dynamo Enhancement Proposal and does not satisfy "
                "the check on its own"
                for d in dep_refs
            ]
        )
        return 0

    dep_lead = (
        [
            f"Found {', '.join(dep_refs)}, which "
            + (
                "is a Dynamo Enhancement Proposal."
                if len(dep_refs) == 1
                else "are Dynamo Enhancement Proposals."
            ),
            "A DEP is a proposal umbrella: it stays open across many pull requests, so",
            "it records what the work is part of and not what this change is. Keep the",
            "reference and link the work as well.",
            "",
        ]
        + (
            [
                f"{(over := len(all_refs) - MAX_CANDIDATES)} further "
                f"reference{'' if over == 1 else 's'} went unchecked: the "
                f"{MAX_CANDIDATES}-lookup bound was spent before reaching "
                "the rest. A closing form (`Closes #123`) is checked first.",
                "",
            ]
            if len(all_refs) > MAX_CANDIDATES
            else []
        )
        if dep_refs
        else []
    )

    summarize(
        [
            (
                "### PR issue link: proposal only"
                if dep_refs
                else "### PR issue link: missing"
            ),
            "",
        ]
        + dep_lead
        + [
            "Every PR needs a linked issue so the work is traceable to a tracked task.",
            "Link one of the following and re-run the check (editing the PR description re-triggers it):",
            "",
            "- A Linear issue, for example `Closes DYN-1234` in the description, or the",
            "  issue ID in the branch name (`user/dyn-1234-description`).",
            "- A GitHub issue: `Fixes #123` to close it, or a non-closing form like",
            "  `Part of #123` for an issue that outlives the PR. Issues in sibling",
            "  repositories in the org count too, for example",
            "  `ai-dynamo/enhancements#12`.",
            "",
            "If no issue exists yet, create one first and start the work from it.",
            f"This check is advisory today and becomes required on {blocking_date}.",
        ]
        + (
            [
                "",
                f"Linear references found ({', '.join(fork_linear_ids)}) cannot be",
                "verified for fork PRs from outside the org; reference a GitHub",
                "issue instead.",
            ]
            if fork_linear_ids
            else []
        )
        + (
            [
                "",
                f"References found ({', '.join(invisible_repo_refs)}) point at a",
                "repository the workflow token cannot see; reference an issue the",
                "workflow can verify instead.",
            ]
            if invisible_repo_refs
            else []
        )
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
