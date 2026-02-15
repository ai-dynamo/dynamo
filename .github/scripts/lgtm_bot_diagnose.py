#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""AI-powered CI diagnostics for pull requests.

Called by ``lgtm-bot-diagnose.yml`` when a user comments ``/diagnose``.
Uses ``gh`` CLI for all GitHub API interaction.

Environment variables:
    GITHUB_REPOSITORY  owner/repo
    PR_NUMBER          Pull request number
    COMMENT_ID         Triggering comment ID (for reactions)
    LLM_API_KEY        API key for LLM inference (optional)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any

COMMENT_MARKER = "<!-- lgtm-bot-diagnose -->"


# ---------------------------------------------------------------------------
# GitHub helpers (all via gh CLI)
# ---------------------------------------------------------------------------


def gh_api(
    path: str,
    method: str = "GET",
    fields: dict[str, str] | None = None,
) -> Any:
    """Call GitHub API via ``gh`` CLI."""
    cmd = ["gh", "api", path]
    if method != "GET":
        cmd.extend(["--method", method])
    for k, v in (fields or {}).items():
        cmd.extend(["-f", f"{k}={v}"])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout) if r.stdout.strip() else {}


def add_reaction(repo: str, comment_id: str, reaction: str) -> None:
    """Add a reaction emoji to a comment."""
    gh_api(
        f"/repos/{repo}/issues/comments/{comment_id}/reactions",
        method="POST",
        fields={"content": reaction},
    )


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def get_check_status(repo: str, sha: str) -> dict:
    """Aggregate check-run status for a commit."""
    data = gh_api(f"/repos/{repo}/commits/{sha}/check-runs?per_page=100")
    if not data:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pending": 0,
            "failures": [],
        }

    runs = data.get("check_runs", [])
    passed = [r for r in runs if r.get("conclusion") == "success"]
    failed = [r for r in runs if r.get("conclusion") == "failure"]
    pending = [r for r in runs if r.get("status") in ("queued", "in_progress")]
    return {
        "total": len(runs),
        "passed": len(passed),
        "failed": len(failed),
        "pending": len(pending),
        "failures": [{"name": r["name"], "url": r.get("html_url", "")} for r in failed],
    }


def get_review_summary(repo: str, pr_number: str) -> dict:
    """Latest review state per reviewer."""
    data = gh_api(f"/repos/{repo}/pulls/{pr_number}/reviews?per_page=100")
    if not data or not isinstance(data, list):
        return {"approved": 0, "changes_requested": 0}

    latest: dict[str, str] = {}
    for r in data:
        user = r.get("user", {}).get("login", "")
        state = r.get("state", "")
        if state in ("APPROVED", "CHANGES_REQUESTED"):
            latest[user] = state

    return {
        "approved": sum(1 for s in latest.values() if s == "APPROVED"),
        "changes_requested": sum(
            1 for s in latest.values() if s == "CHANGES_REQUESTED"
        ),
    }


# ---------------------------------------------------------------------------
# LLM classifier (subprocess)
# ---------------------------------------------------------------------------


def run_classifier(repo: str, sha: str) -> list[dict]:
    """Run the classifier script and capture JSON output."""
    script = os.path.join(os.path.dirname(__file__), "classify_workflow_errors.py")
    env = {
        **os.environ,
        "GITHUB_REPOSITORY": repo,
        "COMMIT_SHA": sha,
        "OUTPUT_ONLY": "true",
    }
    try:
        r = subprocess.run(
            ["python3", script],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"Classifier failed: {exc}", file=sys.stderr)
    return []


# ---------------------------------------------------------------------------
# Comment builder
# ---------------------------------------------------------------------------


def build_comment(
    checks: dict,
    reviews: dict,
    diagnoses: list[dict],
    has_api_key: bool,
) -> str:
    """Build the diagnostic comment markdown."""
    lines = [COMMENT_MARKER, "## CI Diagnostics", ""]

    # Quick summary
    parts: list[str] = []
    if checks["failed"]:
        parts.append(f"{checks['failed']} failing")
    if checks["pending"]:
        parts.append(f"{checks['pending']} pending")
    if checks["passed"]:
        parts.append(f"{checks['passed']} passed")
    parts.append(f"{reviews['approved']} approvals")
    if reviews["changes_requested"]:
        parts.append(f"{reviews['changes_requested']} changes requested")
    lines.append(f"**Summary:** {', '.join(parts)}")
    lines.append("")

    # Failed checks
    if checks["failures"]:
        lines.append("### Failed Checks")
        for f in checks["failures"]:
            lines.append(f"- [{f['name']}]({f['url']})")
        lines.append("")

    # LLM analysis — the unique value-add
    if diagnoses:
        lines.append("### AI Analysis")
        lines.append("")
        for d in diagnoses:
            c = d.get("classification")
            if not c:
                continue
            name = d.get("job_name", "Unknown")
            url = d.get("url", "")
            cat = c.get("category", "unknown").replace("_", " ").title()
            conf = c.get("confidence", 0)
            pct = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)
            lines.append(
                f"<details><summary><b>{name}</b>" f" — {cat} ({pct})</summary>"
            )
            lines.append("")
            lines.append(f"**Root cause:** {c.get('root_cause', 'Unknown')}")
            lines.append("")
            lines.append(c.get("explanation", ""))
            lines.append("")
            lines.append(f"**Suggested fix:** {c.get('suggested_fix', 'N/A')}")
            if url:
                lines.append(f"\n[View logs]({url})")
            lines.append("")
            lines.append("</details>")
            lines.append("")
    elif not has_api_key:
        lines.append("*AI analysis unavailable — no LLM API key configured.*")
        lines.append("")
    elif not checks["failed"]:
        lines.append("*No failed checks to analyze.*")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by LGTM Bot `/diagnose`.*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comment posting
# ---------------------------------------------------------------------------


def post_or_update_comment(repo: str, pr_number: str, body: str) -> None:
    """Find existing diagnose comment and update, or create new."""
    comments = gh_api(f"/repos/{repo}/issues/{pr_number}/comments?per_page=100")
    existing_id = None
    if comments and isinstance(comments, list):
        for c in comments:
            if COMMENT_MARKER in c.get("body", ""):
                existing_id = c["id"]
                break

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(body)
        tmp = f.name

    try:
        if existing_id:
            subprocess.run(
                [
                    "gh",
                    "api",
                    f"/repos/{repo}/issues/comments/{existing_id}",
                    "--method",
                    "PATCH",
                    "-F",
                    f"body=@{tmp}",
                ],
                timeout=30,
            )
        else:
            subprocess.run(
                [
                    "gh",
                    "api",
                    f"/repos/{repo}/issues/{pr_number}/comments",
                    "--method",
                    "POST",
                    "-F",
                    f"body=@{tmp}",
                ],
                timeout=30,
            )
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    pr_number = os.environ.get("PR_NUMBER", "")
    comment_id = os.environ.get("COMMENT_ID", "")

    if not repo or not pr_number:
        print(
            "GITHUB_REPOSITORY and PR_NUMBER required",
            file=sys.stderr,
        )
        sys.exit(1)

    # Signal that we're working
    if comment_id:
        add_reaction(repo, comment_id, "eyes")

    # Get PR HEAD SHA
    pr = gh_api(f"/repos/{repo}/pulls/{pr_number}")
    if not pr:
        print("Could not fetch PR", file=sys.stderr)
        sys.exit(1)
    sha = pr.get("head", {}).get("sha", "")
    if not sha:
        print("Could not determine HEAD SHA", file=sys.stderr)
        sys.exit(1)

    # Gather data
    checks = get_check_status(repo, sha)
    reviews = get_review_summary(repo, pr_number)

    # Run LLM classifier if there are failures and a key is available
    diagnoses: list[dict] = []
    has_api_key = bool(os.environ.get("LLM_API_KEY"))
    if checks["failed"] > 0 and has_api_key:
        diagnoses = run_classifier(repo, sha)

    # Build and post comment
    body = build_comment(checks, reviews, diagnoses, has_api_key)
    post_or_update_comment(repo, pr_number, body)

    # Completion reaction
    if comment_id:
        reaction = "rocket" if checks["failed"] == 0 else "confused"
        add_reaction(repo, comment_id, reaction)

    print(
        f"Diagnosis complete: {checks['failed']} failures analyzed",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
