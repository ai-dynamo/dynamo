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
  (``ai-dynamo/enhancements#12`` - DEP and contribution-request issues live in
  sibling repositories), or a full issue URL.

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

LINEAR_TEXT_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9}-\d{1,6})\b")
LINEAR_BRANCH_RE = re.compile(r"(?:^|[/_-])([a-z][a-z0-9]{1,9}-\d{1,6})(?:$|[/_-])")
GITHUB_REF_RE = re.compile(r"(?:^|[^\w&])#(\d{1,7})\b")
CROSS_REPO_RE = re.compile(r"\b([\w.-]+/[\w.-]+)#(\d{1,7})\b")
ISSUE_URL_RE = re.compile(r"github\.com/([\w.-]+/[\w.-]+)/issues/(\d{1,7})\b")
BOT_AUTHORS = {"dependabot[bot]", "github-actions[bot]", "copy-pr-bot[bot]"}
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


def verify_github_issue(repo: str, number: str, token: str) -> tuple[bool, bool]:
    """Return (exists_as_issue, api_ok)."""
    status, body = http_json(
        f"https://api.github.com/repos/{repo}/issues/{number}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    if status == 200:
        return "pull_request" not in body, True
    if status in (404, 410):
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
    is_fork = head_repo.lower() != repo.lower()
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    linear_key = os.environ.get("LINEAR_API_KEY", "")
    blocking_date = os.environ.get("BLOCKING_DATE", "2026-10-15")

    if author in BOT_AUTHORS:
        print(f"Author {author} is a bot; skipping the issue-link check.")
        return 0

    text = f"{title}\n{body}"
    linear_ids = set(LINEAR_TEXT_RE.findall(text))
    linear_ids.update(m.upper() for m in LINEAR_BRANCH_RE.findall(branch))
    github_refs = {(repo, n) for n in GITHUB_REF_RE.findall(text)}
    org = repo.split("/")[0]
    for other_repo, number in CROSS_REPO_RE.findall(text) + ISSUE_URL_RE.findall(text):
        # Cross-repo references count when they stay inside the same org -
        # DEP issues (ai-dynamo/enhancements) and contribution requests live
        # in sibling repositories.
        if other_repo.split("/")[0].lower() == org.lower():
            github_refs.add((other_repo, number))

    verified: list[str] = []
    unverified: list[str] = []

    ordered_refs = sorted(github_refs, key=lambda r: (r[0], int(r[1])))[:MAX_CANDIDATES]
    for ref_repo, number in ordered_refs:
        exists, api_ok = verify_github_issue(ref_repo, number, gh_token)
        label = f"#{number}" if ref_repo == repo else f"{ref_repo}#{number}"
        if exists:
            verified.append(f"GitHub issue {label}")
        elif not api_ok:
            unverified.append(f"GitHub reference {label} (API unavailable)")
        elif ref_repo != repo:
            # A 404 on a cross-repo reference can mean the workflow token
            # cannot see that repository (internal visibility) rather than
            # that the issue does not exist. Fail open.
            unverified.append(
                f"GitHub reference {label} (not visible to the workflow token)"
            )
        if verified:
            # One verified issue satisfies the check; stop spending lookups.
            break

    for identifier in sorted(linear_ids)[:MAX_CANDIDATES]:
        if verified:
            break
        if is_fork:
            # Do not turn the CI into an existence oracle for Linear IDs
            # guessed from fork PRs; fork contributors use GitHub issues.
            unverified.append(
                f"Linear reference {identifier} (not verified for fork PRs)"
            )
            continue
        exists, api_ok = verify_linear_issue(identifier, linear_key)
        if exists:
            verified.append(f"Linear issue {identifier}")
        elif not api_ok:
            unverified.append(f"Linear reference {identifier} (not verified)")

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
        # Fail open: an upstream outage should never fail anyone's PR.
        summarize(
            ["### PR issue link: present but unverified", ""]
            + [f"- {u}" for u in unverified]
        )
        return 0

    summarize(
        [
            "### PR issue link: missing",
            "",
            "Every PR needs a linked issue so the work is traceable to a tracked task.",
            "Link one of the following and re-run the check (editing the PR description re-triggers it):",
            "",
            "- A Linear issue, for example `Closes DYN-1234` in the description, or the",
            "  issue ID in the branch name (`user/dyn-1234-description`).",
            "- A GitHub issue: `Fixes #123` to close it, or a non-closing form like",
            "  `Part of #123` for long-running tracking issues (DEPs). Org repos count",
            "  too, for example `ai-dynamo/enhancements#12`.",
            "",
            "If no issue exists yet, create one first and start the work from it.",
            f"This check is advisory today and becomes required on {blocking_date}.",
        ]
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
