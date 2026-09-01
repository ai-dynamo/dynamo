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
  closing keyword form (``Fixes #123``), or a full issue URL.

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
BOT_AUTHORS = {"dependabot[bot]", "github-actions[bot]", "copy-pr-bot[bot]"}


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
        return bool(issue), True
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
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    linear_key = os.environ.get("LINEAR_API_KEY", "")
    blocking_date = os.environ.get("BLOCKING_DATE", "2026-10-15")

    if author in BOT_AUTHORS:
        print(f"Author {author} is a bot; skipping the issue-link check.")
        return 0

    text = f"{title}\n{body}"
    linear_ids = set(LINEAR_TEXT_RE.findall(text))
    linear_ids.update(m.upper() for m in LINEAR_BRANCH_RE.findall(branch))
    github_ids = set(GITHUB_REF_RE.findall(text))

    verified: list[str] = []
    unverified: list[str] = []

    for number in sorted(github_ids, key=int):
        exists, api_ok = verify_github_issue(repo, number, gh_token)
        if exists:
            verified.append(f"GitHub issue #{number}")
        elif not api_ok:
            unverified.append(f"GitHub reference #{number} (API unavailable)")

    for identifier in sorted(linear_ids):
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
            "- A GitHub issue, for example `Fixes #123` or the full issue URL.",
            "",
            "If no issue exists yet, create one first and start the work from it.",
            f"This check is advisory today and becomes required on {blocking_date}.",
        ]
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
