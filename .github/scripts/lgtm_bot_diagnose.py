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

"""LGTM Bot diagnostic script.

Analyzes a PR's merge readiness and posts a detailed diagnostic comment.
Uses the GitHub API for PR data and optionally the NVIDIA Inference API
for LLM-powered CI failure diagnosis.

Environment variables:
    GITHUB_TOKEN: GitHub API token (required)
    PR_NUMBER: Pull request number to diagnose (required)
    GITHUB_REPOSITORY: owner/repo (set automatically by GitHub Actions)
    NVIDIA_INFERENCE_API_KEY: NVIDIA API key for LLM features (optional)
"""

import base64
import fnmatch
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

# --- Configuration ---

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "ai-dynamo/dynamo")
PR_NUMBER = int(os.environ["PR_NUMBER"])
NVIDIA_API_KEY = os.environ.get("NVIDIA_INFERENCE_API_KEY", "")

OWNER, REPO = GITHUB_REPOSITORY.split("/")
GITHUB_API = "https://api.github.com"
NVIDIA_API_URL = "https://inference-api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "us/aws/anthropic/bedrock-claude-opus-4-6"

DIAGNOSIS_MARKER = "<!-- lgtm-bot-diagnosis -->"

REQUIRED_CHECKS = [
    "pre-merge-status-check",
    "copyright-checks",
    "DCO",
    "Validate PR title and add label",
]

CHECK_DISPLAY = {
    "pre-merge-status-check": "Pre-merge CI",
    "copyright-checks": "Copyright headers",
    "DCO": "DCO sign-off",
    "Validate PR title and add label": "PR title format",
}


# --- GitHub API helpers ---


def github_request(
    path: str,
    method: str = "GET",
    data: dict | None = None,
) -> Any:
    """Make a GitHub API request."""
    url = f"{GITHUB_API}{path}" if path.startswith("/") else path
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode()
        except Exception:
            pass
        print(f"GitHub API error: {e.code} {e.reason} {error_body}", file=sys.stderr)
        raise


def github_graphql(query: str, variables: dict | None = None) -> Any:
    """Make a GitHub GraphQL API request."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    url = "https://api.github.com/graphql"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json",
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


# --- CODEOWNERS parsing ---


def parse_codeowners(content: str) -> list[tuple[str, list[str]]]:
    """Parse CODEOWNERS file into list of (pattern, teams) tuples."""
    rules = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pattern = parts[0]
        teams = [p for p in parts[1:] if p.startswith("@")]
        if teams:
            rules.append((pattern, teams))
    return rules


def match_codeowners(filepath: str, rules: list[tuple[str, list[str]]]) -> list[str]:
    """Find the CODEOWNERS teams for a file (last matching rule wins).

    Follows GitHub CODEOWNERS pattern matching rules:
    - Anchored patterns (start with /) match from repo root
    - Unanchored patterns without / match filename at any depth
    - Unanchored patterns with / match against full path and all suffixes
    - Directory patterns (end with /) match any file under that directory
    - Last matching rule wins
    """
    matched_teams: list[str] = []
    for pattern, teams in rules:
        anchored = pattern.startswith("/")
        clean = pattern.lstrip("/")

        if clean.endswith("/"):
            # Directory pattern: match any file under it
            directory = clean
            if anchored:
                if filepath.startswith(directory):
                    matched_teams = teams
            else:
                # Unanchored directory: match anywhere in path
                if filepath.startswith(directory) or f"/{directory}" in f"/{filepath}":
                    matched_teams = teams
        elif anchored:
            # Anchored file/glob pattern — match from repo root
            if fnmatch.fnmatch(filepath, clean):
                matched_teams = teams
        else:
            # Unanchored pattern
            if "/" in clean:
                # Pattern has path separator: match full path and all suffixes
                if fnmatch.fnmatch(filepath, clean):
                    matched_teams = teams
                else:
                    parts = filepath.split("/")
                    for i in range(len(parts)):
                        suffix = "/".join(parts[i:])
                        if fnmatch.fnmatch(suffix, clean):
                            matched_teams = teams
                            break
            else:
                # Simple pattern without /: match filename at any depth
                filename = filepath.rsplit("/", 1)[-1]
                if fnmatch.fnmatch(filename, clean):
                    matched_teams = teams

    return matched_teams


# --- Diagnostic functions ---


def diagnose_ci(head_sha: str) -> tuple[dict, list[str]]:
    """Check CI status for all required checks."""
    # Fetch check runs (paginated)
    all_checks: list[dict] = []
    page = 1
    while True:
        data = github_request(
            f"/repos/{OWNER}/{REPO}/commits/{head_sha}/check-runs"
            f"?per_page=100&page={page}"
        )
        all_checks.extend(data["check_runs"])
        if len(all_checks) >= data["total_count"]:
            break
        page += 1

    # Fetch commit statuses (DCO may use legacy status API)
    combined = github_request(f"/repos/{OWNER}/{REPO}/commits/{head_sha}/status")

    results: dict[str, dict] = {}
    failing: list[str] = []

    for name in REQUIRED_CHECKS:
        matching = [cr for cr in all_checks if cr["name"] == name]

        if matching:
            latest = sorted(
                matching,
                key=lambda cr: cr.get("started_at", ""),
                reverse=True,
            )[0]

            if latest["status"] != "completed":
                results[name] = {
                    "status": "pending",
                    "url": latest.get("html_url", ""),
                }
            elif latest["conclusion"] in ("success", "neutral", "skipped"):
                results[name] = {
                    "status": "pass",
                    "url": latest.get("html_url", ""),
                }
            else:
                results[name] = {
                    "status": "fail",
                    "conclusion": latest["conclusion"],
                    "url": latest.get("html_url", ""),
                    "id": latest["id"],
                }
                failing.append(name)
            continue

        # Fallback: check commit statuses
        status_match = next(
            (
                s
                for s in combined.get("statuses", [])
                if s["context"] == name or s["context"].lower() == name.lower()
            ),
            None,
        )

        if status_match:
            if status_match["state"] == "success":
                results[name] = {
                    "status": "pass",
                    "url": status_match.get("target_url", ""),
                }
            elif status_match["state"] == "pending":
                results[name] = {
                    "status": "pending",
                    "url": status_match.get("target_url", ""),
                }
            else:
                results[name] = {
                    "status": "fail",
                    "conclusion": status_match["state"],
                    "url": status_match.get("target_url", ""),
                }
                failing.append(name)
            continue

        results[name] = {"status": "pending", "url": ""}

    return results, failing


def fetch_failed_logs(check_run_id: int) -> str:
    """Fetch annotations for a failed check run."""
    try:
        annotations = github_request(
            f"/repos/{OWNER}/{REPO}/check-runs/{check_run_id}/annotations"
        )
        if annotations:
            log_lines = []
            for ann in annotations[:10]:
                level = ann.get("annotation_level", "error")
                path = ann.get("path", "")
                start = ann.get("start_line", "")
                msg = ann.get("message", "")
                log_lines.append(f"[{level}] {path}:{start}")
                log_lines.append(f"  {msg}")
            return "\n".join(log_lines)[:8000]
        return ""
    except Exception as e:
        print(f"Failed to fetch logs for check run {check_run_id}: {e}")
        return ""


def call_llm(failing_checks: list[str], logs: str) -> str | None:
    """Call NVIDIA Inference API for CI failure diagnosis."""
    if not NVIDIA_API_KEY:
        return None

    prompt = (
        "You are a CI debugging assistant for the ai-dynamo/dynamo project "
        "(a distributed LLM inference platform written in Rust and Python).\n\n"
        "A pull request has the following CI failures. Analyze the logs and "
        "provide concise, actionable fix suggestions.\n\n"
        f"Failing checks: {', '.join(failing_checks)}\n\n"
        f"Log excerpts:\n{logs}\n\n"
        "Provide a brief diagnosis and suggested fix for each failure. "
        "Be specific and actionable. Format as markdown."
    )

    data = {
        "model": NVIDIA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful CI debugging assistant. Be concise "
                    "and actionable. Focus on the most likely root cause."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
    }

    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        }
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            NVIDIA_API_URL, data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM call failed: {e}", file=sys.stderr)
        return None


def diagnose_reviews(pr_number: int, pr_author: str) -> tuple[list[str], list[str]]:
    """Check review status."""
    reviews = github_request(f"/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews")

    latest_by_user: dict[str, dict] = {}
    for review in reviews:
        user = review["user"]["login"]
        if user == pr_author:
            continue
        if review["state"] not in ("APPROVED", "CHANGES_REQUESTED"):
            continue
        existing = latest_by_user.get(user)
        if not existing or review["submitted_at"] > existing["submitted_at"]:
            latest_by_user[user] = review

    approvals = [u for u, r in latest_by_user.items() if r["state"] == "APPROVED"]
    changes_requested = [
        u for u, r in latest_by_user.items() if r["state"] == "CHANGES_REQUESTED"
    ]

    return approvals, changes_requested


def diagnose_unresolved_conversations(pr_number: int) -> list[dict]:
    """Use GraphQL to find unresolved review threads."""
    query = """
    query($owner: String!, $repo: String!, $pr: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $pr) {
          reviewThreads(first: 100) {
            nodes {
              isResolved
              comments(first: 1) {
                nodes {
                  author { login }
                  body
                  url
                  path
                  line
                }
              }
            }
          }
        }
      }
    }
    """

    try:
        result = github_graphql(query, {"owner": OWNER, "repo": REPO, "pr": pr_number})

        threads = (
            result.get("data", {})
            .get("repository", {})
            .get("pullRequest", {})
            .get("reviewThreads", {})
            .get("nodes", [])
        )

        unresolved = []
        for thread in threads:
            if not thread["isResolved"]:
                nodes = thread.get("comments", {}).get("nodes", [])
                comment = nodes[0] if nodes else None
                if comment:
                    author = (
                        comment.get("author", {}).get("login", "unknown")
                        if comment.get("author")
                        else "unknown"
                    )
                    unresolved.append(
                        {
                            "author": author,
                            "body": comment.get("body", "")[:100],
                            "url": comment.get("url", ""),
                            "path": comment.get("path", ""),
                            "line": comment.get("line", ""),
                        }
                    )
        return unresolved
    except Exception as e:
        print(f"GraphQL query failed: {e}", file=sys.stderr)
        return []


def diagnose_codeowners(pr_number: int, base_ref: str) -> dict[str, list[str]]:
    """Check which CODEOWNERS teams are required."""
    # Fetch CODEOWNERS file
    try:
        codeowners_data = github_request(
            f"/repos/{OWNER}/{REPO}/contents/CODEOWNERS?ref={base_ref}"
        )
        content = base64.b64decode(codeowners_data["content"]).decode()
        rules = parse_codeowners(content)
    except Exception:
        return {}

    # Fetch changed files
    changed_files: list[str] = []
    page = 1
    while True:
        files = github_request(
            f"/repos/{OWNER}/{REPO}/pulls/{pr_number}/files"
            f"?per_page=100&page={page}"
        )
        changed_files.extend(f["filename"] for f in files)
        if len(files) < 100:
            break
        page += 1

    # Match files to teams
    team_files: dict[str, list[str]] = {}
    for filepath in changed_files:
        teams = match_codeowners(filepath, rules)
        for team in teams:
            if team not in team_files:
                team_files[team] = []
            team_files[team].append(filepath)

    return team_files


# --- Comment builder ---


def build_diagnostic_comment(
    pr: dict,
    ci_results: dict,
    failing_checks: list[str],
    approvals: list[str],
    changes_requested: list[str],
    unresolved: list[dict],
    team_files: dict[str, list[str]],
    is_fork: bool,
    head_sha: str,
    llm_response: str | None,
) -> str:
    """Build the diagnostic markdown comment."""
    lines = [DIAGNOSIS_MARKER, "## 🤖 LGTM Bot — Diagnosis", ""]

    # Count blockers
    blockers: list[str] = []
    if failing_checks:
        blockers.append(f"CI failures ({len(failing_checks)})")
    pending = [n for n, r in ci_results.items() if r["status"] == "pending"]
    if pending:
        blockers.append(f"CI checks pending ({len(pending)})")
    if not approvals:
        blockers.append("no approvals")
    if changes_requested:
        blockers.append(f"changes requested ({len(changes_requested)})")
    if unresolved:
        blockers.append(f"unresolved conversations ({len(unresolved)})")
    if pr.get("mergeable_state") == "dirty":
        blockers.append("merge conflict")

    # CI trigger detection (applies to all PRs — /ok to test is needed
    # for both fork PRs and internal branches without GPG signing)
    ci_not_triggered = False
    pm = ci_results.get("pre-merge-status-check", {})
    if pm.get("status") == "pending" and not pm.get("url"):
        ci_not_triggered = True
        blockers.append("full CI not triggered")

    if blockers:
        count = len(blockers)
        word = "blocker" if count == 1 else "blockers"
        lines.append(f"**Status**: Not ready to merge ({count} {word})")
    else:
        lines.append("**Status**: Ready to merge :white_check_mark:")
    lines.append("")

    # --- CI section ---
    lines.append("### CI Checks")
    lines.append("")
    lines.append("| Check | Status | Link |")
    lines.append("|-------|--------|------|")
    for name in REQUIRED_CHECKS:
        result = ci_results.get(name, {"status": "pending", "url": ""})
        display = CHECK_DISPLAY.get(name, name)
        status_text = {
            "pass": "pass",
            "fail": "**FAIL**",
            "pending": "pending",
        }.get(result["status"], result["status"])
        link = f"[View]({result['url']})" if result.get("url") else "—"
        lines.append(f"| {display} | {status_text} | {link} |")
    lines.append("")

    # LLM diagnosis for failures
    if failing_checks:
        if llm_response:
            lines.append("#### Suggested Fixes")
            lines.append("")
            lines.append(llm_response)
            lines.append("")
        else:
            lines.append("> **Tip**: Check the CI logs linked above for error details.")
            lines.append("")

    # CI trigger detection
    if ci_not_triggered:
        lines.append("### Full CI")
        lines.append("")
        lines.append(
            "Full CI has not been triggered for this PR." " A maintainer must comment:"
        )
        lines.append(f"```\n/ok to test {head_sha[:7]}\n```")
        lines.append("")
        lines.append(
            "> **Note**: `/ok to test` must be re-issued after every "
            "new push — the approval is per-commit, not per-PR."
        )
        lines.append("")

    # --- Reviews section ---
    lines.append("### Reviews")
    lines.append("")
    if approvals:
        names = ", ".join(f"@{u}" for u in approvals)
        lines.append(f"- Approved by: {names}")
    else:
        lines.append(
            f"- No approved reviews — "
            f"[request a review]"
            f"(https://github.com/{OWNER}/{REPO}/pull/{PR_NUMBER}"
            f"/reviewers)"
        )
    if changes_requested:
        names = ", ".join(f"@{u}" for u in changes_requested)
        lines.append(f"- **Changes requested** by: {names}")

    # CODEOWNERS
    if team_files:
        lines.append("")
        lines.append("**Required CODEOWNERS teams:**")
        lines.append("")
        lines.append("| Team | Files |")
        lines.append("|------|-------|")
        for team, files in team_files.items():
            file_list = ", ".join(f"`{f}`" for f in files[:3])
            if len(files) > 3:
                file_list += f" (+{len(files) - 3} more)"
            lines.append(f"| {team} | {file_list} |")
    lines.append("")

    # --- Unresolved conversations ---
    if unresolved:
        lines.append(f"### Unresolved Conversations ({len(unresolved)})")
        lines.append("")
        for thread in unresolved[:10]:
            path_info = f"`{thread['path']}"
            if thread.get("line"):
                path_info += f":{thread['line']}"
            path_info += "`"
            lines.append(f"- @{thread['author']} on " f"[{path_info}]({thread['url']})")
        if len(unresolved) > 10:
            lines.append(f"- ...and {len(unresolved) - 10} more")
        lines.append("")

    # --- Other checks ---
    lines.append("### Other")
    lines.append("")

    # DCO
    dco = ci_results.get("DCO", {})
    if dco.get("status") == "pass":
        lines.append("- **DCO**: passing")
    elif dco.get("status") == "fail":
        lines.append(
            f"- **DCO**: failing — "
            f"[see DCO troubleshooting guide]"
            f"(https://github.com/{OWNER}/{REPO}/blob/main/DCO.md)"
        )
    else:
        lines.append("- **DCO**: pending")

    # PR title
    title_check = ci_results.get("Validate PR title and add label", {})
    if title_check.get("status") == "pass":
        lines.append("- **PR title**: passing")
    elif title_check.get("status") == "fail":
        lines.append(
            f"- **PR title**: failing — must follow "
            f"[conventional commits format]"
            f"(https://github.com/{OWNER}/{REPO}/blob/main/"
            f"CONTRIBUTING.md)"
        )
    else:
        lines.append("- **PR title**: pending")

    # Merge conflicts
    if pr.get("mergeable_state") == "dirty":
        lines.append(
            "- **Merge conflicts**: **yes** — rebase or merge "
            "the base branch to resolve"
        )
    elif pr.get("mergeable") is False:
        lines.append("- **Merge conflicts**: checking...")
    else:
        lines.append("- **Merge conflicts**: none")

    lines.append("")
    lines.append("---")
    lines.append("*Run `/lgtm-bot diagnose` again to refresh*")

    return "\n".join(lines)


def post_diagnostic_comment(pr_number: int, body: str) -> None:
    """Post or update the diagnostic comment."""
    # Search from newest first — bot comments are typically recent
    existing = None
    page = 1
    while existing is None:
        comments = github_request(
            f"/repos/{OWNER}/{REPO}/issues/{pr_number}/comments"
            f"?per_page=100&page={page}&direction=desc"
        )
        if not comments:
            break
        existing = next(
            (c for c in comments if DIAGNOSIS_MARKER in (c.get("body") or "")),
            None,
        )
        page += 1

    if existing:
        github_request(
            f"/repos/{OWNER}/{REPO}/issues/comments/{existing['id']}",
            method="PATCH",
            data={"body": body},
        )
        print(f"Updated diagnostic comment {existing['id']}")
    else:
        github_request(
            f"/repos/{OWNER}/{REPO}/issues/{pr_number}/comments",
            method="POST",
            data={"body": body},
        )
        print("Created new diagnostic comment")


# --- Main ---


def main() -> None:
    """Run the full diagnostic pipeline."""
    print(f"Diagnosing PR #{PR_NUMBER}...")

    # Get PR details
    pr = github_request(f"/repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}")
    head_sha = pr["head"]["sha"]
    base_ref = pr["base"]["ref"]
    pr_author = pr["user"]["login"]
    head_repo = pr["head"].get("repo")
    is_fork = (not head_repo) or head_repo["full_name"] != f"{OWNER}/{REPO}"

    # Run all diagnostics
    ci_results, failing_checks = diagnose_ci(head_sha)
    approvals, changes_requested = diagnose_reviews(PR_NUMBER, pr_author)
    unresolved = diagnose_unresolved_conversations(PR_NUMBER)
    team_files = diagnose_codeowners(PR_NUMBER, base_ref)

    # LLM diagnosis for CI failures
    llm_response = None
    if failing_checks:
        all_logs = []
        for name in failing_checks:
            check = ci_results[name]
            if "id" in check:
                logs = fetch_failed_logs(check["id"])
                if logs:
                    all_logs.append(f"### {name}\n{logs}")
        if all_logs:
            combined_logs = "\n\n".join(all_logs)[:8000]
            llm_response = call_llm(failing_checks, combined_logs)

    # Build and post comment
    comment_body = build_diagnostic_comment(
        pr,
        ci_results,
        failing_checks,
        approvals,
        changes_requested,
        unresolved,
        team_files,
        is_fork,
        head_sha,
        llm_response,
    )
    post_diagnostic_comment(PR_NUMBER, comment_body)
    print("Diagnosis complete!")


if __name__ == "__main__":
    main()
