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
Uses the GitHub API for PR data and delegates CI failure analysis to
classify_workflow_errors.py as a subprocess.

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
import subprocess
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
        with urllib.request.urlopen(req, timeout=30) as resp:
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
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"GraphQL error: {e.code} {body[:200]}", file=sys.stderr)
        raise


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


def get_workflow_run_id_for_commit(head_sha: str) -> str | None:
    """Return the most recent workflow run id for the given commit SHA."""
    try:
        data = github_request(
            f"/repos/{OWNER}/{REPO}/actions/runs"
            f"?head_sha={head_sha}&per_page=5&exclude_pull_requests=false"
        )
        runs = data.get("workflow_runs", [])
        if not runs:
            return None
        return str(runs[0]["id"])
    except Exception as e:
        print(f"Failed to get workflow run for {head_sha[:7]}: {e}", file=sys.stderr)
        return None


def run_error_classifier(run_id: str, repo_root: str) -> str | None:
    """Run the error classifier for the given workflow run; return markdown or None."""
    env = {
        **os.environ,
        "WORKFLOW_RUN_ID": run_id,
        "OUTPUT_ONLY": "true",
        "ENABLE_ERROR_CLASSIFICATION": "true",
        "GITHUB_REPOSITORY": GITHUB_REPOSITORY,
    }
    try:
        result = subprocess.run(
            ["python3", ".github/scripts/classify_workflow_errors.py"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=repo_root,
            env=env,
        )
        if result.returncode != 0:
            print(
                f"Classifier exited {result.returncode}: {result.stderr[:500]}",
                file=sys.stderr,
            )
            return None
        out = (result.stdout or "").strip()
        return out if out else None
    except subprocess.TimeoutExpired:
        print("Classifier timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Classifier failed: {e}", file=sys.stderr)
        return None


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
        print(
            f"Failed to fetch logs for check run {check_run_id}: {e}", file=sys.stderr
        )
        return ""


def diagnose_reviews(pr_number: int, pr_author: str) -> tuple[list[str], list[str]]:
    """Check review status."""
    reviews: list[dict] = []
    page = 1
    while True:
        batch = github_request(
            f"/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews"
            f"?per_page=100&page={page}"
        )
        reviews.extend(batch)
        if len(batch) < 100:
            break
        page += 1

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
    except Exception as e:
        print(f"CODEOWNERS analysis failed: {e}", file=sys.stderr)
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
    head_sha: str,
    classifier_markdown: str | None = None,
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
        lines.append("")
        lines.append(
            "> **Tip**: Run `/diagnose` after pushing fixes "
            "to re-check merge readiness."
        )
    else:
        lines.append(
            "**Status**: Ready to merge :white_check_mark: " "— no diagnosis needed."
        )
    lines.append("")

    # --- CI section (collapsible, open by default) ---
    lines.append("<details open>")
    lines.append("<summary><h3>CI Checks</h3></summary>")
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
    lines.append("</details>")
    lines.append("")

    # CI failure analysis (collapsible, closed by default)
    if failing_checks:
        if classifier_markdown:
            lines.append("<details>")
            lines.append(
                "<summary><h4>CI Failure Analysis (from error classifier)</h4></summary>"
            )
            lines.append("")
            lines.append(classifier_markdown)
            lines.append("")
            lines.append("</details>")
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
            "new push — the approval is per-commit, not per-PR. "
            "Org members with GPG-signed commits trigger CI automatically."
        )
        lines.append("")

    # --- Reviews section (collapsible, closed by default) ---
    lines.append("<details>")
    lines.append("<summary><h3>Reviews</h3></summary>")
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
    lines.append("")
    lines.append("</details>")
    lines.append("")

    # CODEOWNERS (collapsible, closed by default)
    if team_files:
        lines.append("<details>")
        lines.append("<summary><h3>Required CODEOWNERS Teams</h3></summary>")
        lines.append("")
        lines.append("| Team | Files |")
        lines.append("|------|-------|")
        for team, files in team_files.items():
            file_list = ", ".join(f"`{f}`" for f in files[:3])
            if len(files) > 3:
                file_list += f" (+{len(files) - 3} more)"
            lines.append(f"| {team} | {file_list} |")
        lines.append("")
        lines.append("</details>")
    lines.append("")

    # --- Unresolved conversations (collapsible, closed by default) ---
    if unresolved:
        lines.append("<details>")
        lines.append(
            f"<summary><h3>Unresolved Conversations ({len(unresolved)})</h3></summary>"
        )
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
        lines.append("</details>")
        lines.append("")

    # --- Other checks (collapsible, closed by default) ---
    lines.append("<details>")
    lines.append("<summary><h3>Other</h3></summary>")
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
    lines.append("</details>")
    lines.append("")
    lines.append("---")
    lines.append("*Run `/diagnose` again to refresh*")

    return "\n".join(lines)


def post_diagnostic_comment(pr_number: int, body: str) -> None:
    """Post or update the diagnostic comment."""
    # Find existing diagnostic comment by marker
    existing = None
    page = 1
    while existing is None:
        comments = github_request(
            f"/repos/{OWNER}/{REPO}/issues/{pr_number}/comments"
            f"?per_page=100&page={page}"
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
    # Run all diagnostics
    ci_results, failing_checks = diagnose_ci(head_sha)
    approvals, changes_requested = diagnose_reviews(PR_NUMBER, pr_author)
    unresolved = diagnose_unresolved_conversations(PR_NUMBER)
    team_files = diagnose_codeowners(PR_NUMBER, base_ref)

    # Short-circuit: if the PR is already mergeable, skip expensive classifier
    classifier_markdown = None
    is_mergeable = (
        not failing_checks
        and len(approvals) >= 1
        and not changes_requested
        and not unresolved
        and pr.get("mergeable_state") != "dirty"
    )

    if is_mergeable:
        print("PR is mergeable, skipping CI failure analysis.")
    elif failing_checks and NVIDIA_API_KEY:
        # CI failure analysis: run error classifier (if API key and run_id available)
        run_id = get_workflow_run_id_for_commit(head_sha)
        if run_id:
            repo_root = os.getcwd()
            classifier_markdown = run_error_classifier(run_id, repo_root)

    # Build and post comment
    comment_body = build_diagnostic_comment(
        pr,
        ci_results,
        failing_checks,
        approvals,
        changes_requested,
        unresolved,
        team_files,
        head_sha,
        classifier_markdown=classifier_markdown,
    )
    post_diagnostic_comment(PR_NUMBER, comment_body)
    print("Diagnosis complete!")


if __name__ == "__main__":
    main()
