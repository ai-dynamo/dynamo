# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify PR issue references using only the Python standard library."""

import html
import json
import os
import re
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener


@dataclass(frozen=True)
class Reference:
    kind: str
    ref: str
    owner: str = ""
    repo: str = ""
    number: int = 0


@dataclass(frozen=True)
class Result:
    title: str = ""
    error: str = ""


def references(text: str, repository: str) -> list[Reference]:
    """Extract unique visible GitHub and Linear references from PR text."""
    text = re.sub(r"<!--[\s\S]*?(?:-->|$)", "", text)
    found = {}

    def github_reference(owner: str, repo: str, number: str) -> None:
        ref = f"{owner}/{repo}#{int(number)}"
        found[ref.lower()] = Reference("github", ref, owner, repo, int(number))

    def github_match(match: re.Match) -> str:
        github_reference(*match.groups())
        return " "

    text = re.sub(
        r"https://github\.com/([\w.-]+)/([\w.-]+)/issues/([1-9]\d*)\b[^\s<>)]*",
        github_match,
        text,
        flags=re.IGNORECASE | re.ASCII,
    )
    text = re.sub(
        r"https://linear\.app/[\w-]+/issue/([A-Z][A-Z0-9]*-[1-9]\d*)\b[^\s<>)]*",
        lambda match: f" {match[1].upper()} ",
        text,
        flags=re.IGNORECASE | re.ASCII,
    )
    # Other URLs' anchors and pull links cannot satisfy the requirement.
    text = re.sub(r"https?://[^\s<>)]*", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?<![\w/.-])([\w.-]+)/([\w.-]+)#([1-9]\d*)\b",
        github_match,
        text,
        flags=re.ASCII,
    )
    owner, repo = repository.split("/", 1)
    for match in re.finditer(r"(?<![\w/])(?:#|GH-)([1-9]\d*)\b", text, re.ASCII):
        github_reference(owner, repo, match[1])
    for match in re.finditer(r"\b([A-Z][A-Z0-9]*-[1-9]\d*)\b", text, re.ASCII):
        ticket = match[1]
        if not ticket.startswith("GH-"):
            found[ticket] = Reference("linear", ticket)
    return list(found.values())


class NoRedirect(HTTPRedirectHandler):
    """Prevent forwarding API credentials to redirect destinations."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_json(request: Request) -> dict:
    """Fetch a JSON API response with a bounded timeout and no redirects."""
    with build_opener(NoRedirect()).open(request, timeout=15) as response:
        return json.load(response)


def lookup_github(ref: Reference, token: str) -> Result:
    """Look up a GitHub issue, rejecting PRs returned by the issues endpoint."""
    # https://docs.github.com/en/rest/issues/issues#get-an-issue
    url = (
        f"https://api.github.com/repos/{quote(ref.owner, safe='')}/"
        f"{quote(ref.repo, safe='')}/issues/{ref.number}"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "dynamo-pr-issue-reference",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        issue = request_json(Request(url, headers=headers))
    except HTTPError as error:
        error.close()
        if error.code == 404:
            return Result(error="does not exist or is not accessible")
        return Result(error=f"could not be verified: GitHub API HTTP {error.code}")
    except (URLError, TimeoutError, HTTPException, json.JSONDecodeError, UnicodeError):
        # Never print raw API exceptions: they can contain credentials.
        return Result(error="could not be verified: GitHub request failed or timed out")
    if "pull_request" in issue:
        return Result(error="is a pull request, not an issue")
    if not isinstance(issue.get("title"), str):
        return Result(error="could not be verified: unexpected GitHub API response")
    return Result(title=issue["title"])


def lookup_linear(ref: Reference, key: str) -> Result:
    """Look up a Linear issue by its human-readable identifier."""
    # https://linear.app/developers/graphql
    if not key:
        return Result(
            error="could not be verified: LINEAR_ACCESS_KEY secret is missing"
        )
    request = Request(
        "https://api.linear.app/graphql",
        data=json.dumps(
            {
                "query": "query IssueReference($id: String!) { issue(id: $id) { identifier title } }",
                "variables": {"id": ref.ref},
            }
        ).encode("utf-8"),
        headers={"Authorization": key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        payload = request_json(request)
    except HTTPError as error:
        error.close()
        return Result(error=f"could not be verified: Linear API HTTP {error.code}")
    except (URLError, TimeoutError, HTTPException, json.JSONDecodeError, UnicodeError):
        return Result(error="could not be verified: Linear request failed or timed out")
    if payload.get("errors"):
        if all(
            "entity not found" in error.get("message", "").lower()
            for error in payload["errors"]
        ):
            return Result(error="does not exist or is not accessible")
        return Result(
            error="could not be verified: Linear GraphQL error (check key access and API availability)"
        )
    data = payload.get("data") or {}
    if "issue" in data and data["issue"] is None:
        return Result(error="does not exist or is not accessible")
    issue = data.get("issue")
    if (
        not isinstance(issue, dict)
        or not isinstance(issue.get("title"), str)
        or not issue.get("identifier")
    ):
        return Result(error="could not be verified: unexpected Linear API response")
    return Result(title=issue["title"])


def annotation(message: str) -> None:
    """Emit an Actions error without interpreting untrusted workflow commands."""
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error::{escaped}")


def write_summary(path: Path, message: str, rows: list[str]) -> None:
    """Append an escaped result summary to the Actions job summary."""
    summary = f"## PR issue references\n\n<p>{html.escape(message)}</p>\n"
    if rows:
        summary += (
            "<table><tr><th>Reference</th><th>Result</th><th>Title / reason</th></tr>"
            + "".join(rows)
            + "</table>\n"
        )
    with path.open("a", encoding="utf-8") as output:
        output.write(summary)


def verify(
    pr: dict, repository: str, github_token: str, linear_key: str, summary_path: Path
) -> int:
    """Verify all listed references, write results, and return the step exit code."""
    refs = references(f"{pr.get('title') or ''}\n{pr.get('body') or ''}", repository)
    if not refs or len(refs) > 50:
        message = (
            "Failed: no issue listed. Add a GitHub issue (#123, owner/repo#123, or an issue URL) "
            "or Linear ticket (TEAM-123 or a Linear issue URL) to the PR title or description."
            if not refs
            else "Failed: more than 50 issue references listed; reduce the number of references."
        )
        annotation(message)
        write_summary(summary_path, message, [])
        return 1
    failures = 0
    rows = []
    for ref in refs:
        result = (
            lookup_github(ref, github_token)
            if ref.kind == "github"
            else lookup_linear(ref, linear_key)
        )
        if result.error:
            failures += 1
            annotation(f"Failed: issue listed {ref.ref} {result.error}")
        else:
            title = result.title.replace("\r", " ").replace("\n", " ")
            print(f"Found: {ref.ref} — {title}")
        status = "Failed" if result.error else "Found"
        rows.append(
            f"<tr><td>{html.escape(ref.ref)}</td><td>{status}</td>"
            f"<td>{html.escape(result.error or result.title)}</td></tr>"
        )
    status = "Failed" if failures else "Passed"
    message = (
        f"{status}: {len(refs) - failures} of {len(refs)} referenced issues verified."
    )
    write_summary(summary_path, message, rows)
    return 1 if failures else 0


def main() -> int:
    """Read trusted runner paths and credentials, treating the event as data."""
    with Path(os.environ["GITHUB_EVENT_PATH"]).open(encoding="utf-8") as event_file:
        event = json.load(event_file)
    return verify(
        event["pull_request"],
        os.environ["GITHUB_REPOSITORY"],
        os.environ.get("GITHUB_TOKEN", ""),
        os.environ.get("LINEAR_ACCESS_KEY", ""),
        Path(os.environ["GITHUB_STEP_SUMMARY"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
