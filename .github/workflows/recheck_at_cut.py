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

"""Evaluate every open pull request into main afresh when the cut happens.

A green issue-link result recorded before the cut stays on its commit, and
creating a branch fires no pull request event. Re-running the old workflow runs
does not close that gap: GitHub refuses to re-run a run older than 30 days, and
a pull request may have no run at all. This script runs the existing check on
each pull request's current title, description, branch and author, and
publishes the result as a new `Require an issue link` check run on its head
commit, which supersedes the older result of the same name.
"""

import contextlib
import io
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator

import pr_issue_link

CHECK_NAME = "Require an issue link"
PUBLISH_DELAYS = (2.0, 8.0)
SUMMARY_LIMIT = 60000
PR_ENV = {
    "PR_TITLE": lambda pr: pr.get("title") or "",
    "PR_BODY": lambda pr: pr.get("body") or "",
    "PR_HEAD_REF": lambda pr: pr["head"]["ref"],
    "PR_AUTHOR": lambda pr: pr["user"]["login"],
    "PR_AUTHOR_ASSOCIATION": lambda pr: pr.get("author_association") or "",
    "PR_HEAD_REPO": lambda pr: (pr["head"].get("repo") or {}).get("full_name", ""),
    "PR_NUMBER": lambda pr: str(pr["number"]),
}


def headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def open_pull_requests(repo: str, token: str) -> Iterator[dict]:
    """Yield every open, non-draft pull request into main, page by page."""
    page = 1
    while True:
        status, body = pr_issue_link.http_json(
            f"https://api.github.com/repos/{repo}/pulls"
            f"?state=open&base=main&per_page=100&page={page}",
            headers=headers(token),
        )
        if status != 200 or not isinstance(body, list):
            raise RuntimeError(f"listing open pull requests failed (HTTP {status})")
        if not body:
            return
        yield from (pr for pr in body if not pr.get("draft"))
        page += 1


def evaluate(pr: dict) -> tuple[int, str]:
    """Run the existing check for one pull request; return (exit code, summary)."""
    saved = {key: os.environ.get(key) for key in [*PR_ENV, "GITHUB_STEP_SUMMARY"]}
    with tempfile.NamedTemporaryFile("r", suffix=".md") as summary:
        try:
            for key, field in PR_ENV.items():
                os.environ[key] = field(pr)
            os.environ["GITHUB_STEP_SUMMARY"] = summary.name
            with contextlib.redirect_stdout(io.StringIO()):
                code = pr_issue_link.main()
            text = summary.read()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return code, text


def publish(
    repo: str,
    sha: str,
    code: int,
    summary: str,
    token: str,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Publish one completed check run; return whether GitHub accepted it."""
    payload = {
        "name": CHECK_NAME,
        "head_sha": sha,
        "status": "completed",
        "conclusion": "success" if code == 0 else "failure",
        "output": {
            "title": "Re-checked at the release cut",
            "summary": summary[:SUMMARY_LIMIT] or "No summary.",
        },
    }
    for delay in (*PUBLISH_DELAYS, None):
        status, _ = pr_issue_link.http_json(
            f"https://api.github.com/repos/{repo}/check-runs",
            payload=payload,
            headers={**headers(token), "Content-Type": "application/json"},
        )
        if status == 201:
            return True
        if delay is not None:
            sleep(delay)
    return False


def main(sleep: Callable[[float], None] = time.sleep) -> int:
    repo = os.environ["REPO"]
    token = os.environ["GITHUB_TOKEN"]
    checked, failing, unpublished = 0, 0, []
    for pr in open_pull_requests(repo, token):
        code, summary = evaluate(pr)
        checked += 1
        failing += code != 0
        if not publish(repo, pr["head"]["sha"], code, summary, token, sleep):
            unpublished.append(f"#{pr['number']}")
        sleep(0.5)
    print(f"Re-checked {checked} open pull requests; {failing} have no linked issue.")
    if unpublished:
        print(
            f"::error::Could not publish the check for {len(unpublished)} pull "
            f"requests ({', '.join(unpublished)}). Re-run this job."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
