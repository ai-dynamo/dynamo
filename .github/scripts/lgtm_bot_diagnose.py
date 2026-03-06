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

Consolidated entry point for ``/diagnose``.  Gathers check status from
GitHub Actions and GitLab CI, classifies failures using an LLM, and
posts a structured comment grouped by blocking vs non-blocking checks.

Environment variables:
    GITHUB_REPOSITORY  owner/repo
    PR_NUMBER          Pull request number
    COMMENT_ID         Triggering comment ID (for reactions + reply-link)
    LLM_API_KEY        API key for LLM inference (optional)
    GITLAB_TOKEN       GitLab personal access token for log fetching (optional)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any

import requests

COMMENT_MARKER = "<!-- lgtm-bot-diagnose -->"

# Required checks from repository ruleset (ID 4130136) — these block merge
REQUIRED_CHECKS = (
    "copyright-checks",
    "DCO",
    "backend-status-check",
    "dynamo-status-check",
    "pre-merge-status-check",
)
REQUIRED_SET = frozenset(REQUIRED_CHECKS)

# Jobs where logs are self-explanatory — skip LLM classification
SKIP_CLASSIFICATION = frozenset(
    {"pre-commit", "copyright", "dco", "lychee", "label", "lint", "codespell"}
)

GITLAB_PROJECT_ID = "169905"
GITLAB_API = "https://gitlab-master.nvidia.com/api/v4"

# Map GitHub Actions workflow names to their rollup required status check.
# When a sub-job fails, we nest the diagnosis under the parent required check.
WORKFLOW_TO_REQUIRED = {
    "NVIDIA Dynamo Github Validation": "dynamo-status-check",
    "NVIDIA Test Github Validation": "dynamo-status-check",
    "Pre Merge": "pre-merge-status-check",
    "PR": "backend-status-check",
    "Backend Validation Helper": "backend-status-check",
    "Copyright Checks": "copyright-checks",
}


# ---------------------------------------------------------------------------
# GitHub helpers (all via gh CLI)
# ---------------------------------------------------------------------------


def gh_api(
    path: str,
    method: str = "GET",
    fields: dict[str, str] | None = None,
) -> Any:
    """Call GitHub API via ``gh`` CLI.  Returns parsed JSON or *None*."""
    cmd = ["gh", "api", path]
    if method != "GET":
        cmd.extend(["--method", method])
    for k, v in (fields or {}).items():
        cmd.extend(["-f", f"{k}={v}"])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout) if r.stdout.strip() else {}
    except json.JSONDecodeError:
        return None


def add_reaction(repo: str, comment_id: str, reaction: str) -> None:
    gh_api(
        f"/repos/{repo}/issues/comments/{comment_id}/reactions",
        method="POST",
        fields={"content": reaction},
    )


# ---------------------------------------------------------------------------
# Data gathering — GitHub
# ---------------------------------------------------------------------------


def get_all_checks(repo: str, sha: str) -> list[dict]:
    """Fetch check runs and commit statuses for *sha*.

    Returns a unified list: ``{name, state, url, source}``.
    *state* is one of ``passed``, ``failed``, ``pending``.
    """
    checks: list[dict] = []
    own_jobs = {"evaluate-lgtm", "request-reviews", "diagnose"}

    # 1. Check runs (GitHub Actions)
    data = gh_api(f"/repos/{repo}/commits/{sha}/check-runs?per_page=100")
    for r in (data or {}).get("check_runs", []):
        name = r.get("name", "")
        if name in own_jobs:
            continue
        status = r.get("status")
        conclusion = r.get("conclusion")
        if status != "completed":
            state = "pending"
        elif conclusion in ("success", "skipped", "neutral"):
            state = "passed"
        elif conclusion == "failure":
            state = "failed"
        else:
            state = "passed"
        checks.append(
            {
                "name": name,
                "state": state,
                "url": r.get("html_url", ""),
                "source": "github",
            }
        )

    # 2. Commit statuses (GitLab, CodeRabbit, etc.)
    status_data = gh_api(f"/repos/{repo}/commits/{sha}/status")
    for s in (status_data or {}).get("statuses", []):
        context = s.get("context", "")
        gh_state = s.get("state", "")
        state = {"success": "passed", "failure": "failed", "pending": "pending"}.get(
            gh_state, "pending"
        )
        source = "gitlab" if "gitlab" in context.lower() else "external"
        checks.append(
            {
                "name": context,
                "state": state,
                "url": s.get("target_url", ""),
                "source": source,
            }
        )

    return checks


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


def get_failed_github_jobs(repo: str, sha: str) -> list[dict]:
    """Return failed jobs from completed GitHub Actions workflow runs."""
    runs_data = gh_api(
        f"/repos/{repo}/actions/runs?head_sha={sha}&status=completed&per_page=20"
    )
    if not runs_data or not runs_data.get("workflow_runs"):
        return []
    failed: list[dict] = []
    for run in runs_data["workflow_runs"]:
        if run.get("conclusion") != "failure":
            continue
        jobs = gh_api(f"/repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100")
        if not jobs:
            continue
        for job in jobs.get("jobs", []):
            if job.get("conclusion") == "failure":
                failed.append(
                    {
                        "job_name": job["name"],
                        "job_id": job["id"],
                        "run_name": run["name"],
                        "url": job.get("html_url", ""),
                        "source": "github",
                    }
                )
    return failed


def _get_github_log(repo: str, job_id: int) -> str:
    try:
        r = subprocess.run(
            ["gh", "api", f"/repos/{repo}/actions/jobs/{job_id}/logs"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return r.stdout if r.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        return ""


# ---------------------------------------------------------------------------
# Data gathering — GitLab
# ---------------------------------------------------------------------------


def get_gitlab_failures(checks: list[dict]) -> list[dict]:
    """Fetch failed GitLab CI job details and logs.

    Extracts pipeline IDs from commit-status ``target_url``s, then queries
    the GitLab API for failed jobs and their log traces.
    """
    token = os.environ.get("GITLAB_TOKEN", "")
    if not token:
        return []

    headers = {"PRIVATE-TOKEN": token}
    seen: set[str] = set()
    failed: list[dict] = []

    for check in checks:
        if check["source"] != "gitlab" or check["state"] != "failed":
            continue
        m = re.search(r"/pipelines/(\d+)", check.get("url", ""))
        if not m:
            continue
        pid = m.group(1)
        if pid in seen:
            continue
        seen.add(pid)

        try:
            resp = requests.get(
                f"{GITLAB_API}/projects/{GITLAB_PROJECT_ID}/pipelines/{pid}/jobs",
                headers=headers,
                params={"scope": "failed", "per_page": 50},
                timeout=30,
            )
            if resp.status_code != 200:
                print(f"GitLab jobs API: HTTP {resp.status_code}", file=sys.stderr)
                continue
        except requests.RequestException as exc:
            print(f"GitLab jobs API: {exc}", file=sys.stderr)
            continue

        for job in resp.json():
            log = ""
            try:
                lr = requests.get(
                    f"{GITLAB_API}/projects/{GITLAB_PROJECT_ID}/jobs/{job['id']}/trace",
                    headers=headers,
                    timeout=30,
                )
                if lr.status_code == 200:
                    log = lr.text
            except requests.RequestException:
                pass
            failed.append(
                {
                    "job_name": job["name"],
                    "job_id": job["id"],
                    "url": job.get("web_url", ""),
                    "source": "gitlab",
                    "failure_reason": job.get("failure_reason", ""),
                    "log": log,
                }
            )

    return failed


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------


def classify_failures(
    repo: str,
    github_jobs: list[dict],
    gitlab_jobs: list[dict],
) -> list[dict]:
    """Classify failed jobs using LLM.  Returns list with ``classification`` dicts."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from error_classification.config import Config
    from error_classification.llm_client import LLMClient

    config = Config.from_env()
    client = LLMClient(config) if config.api_key else None
    results: list[dict] = []

    # GitHub Actions failures
    for job in github_jobs:
        if any(s in job["job_name"].lower() for s in SKIP_CLASSIFICATION):
            results.append({**job, "classification": None})
            continue
        log = _get_github_log(repo, job["job_id"])
        if not log or not client:
            results.append({**job, "classification": None})
            continue
        results.append(
            {**job, "classification": client.classify_error(job["job_name"], log)}
        )

    # GitLab failures
    for job in gitlab_jobs:
        # Auto-classify timeouts without LLM
        if job.get("failure_reason") == "job_execution_timeout":
            results.append(
                {
                    **job,
                    "classification": {
                        "category": "timeout",
                        "root_cause": "Job exceeded time limit",
                        "explanation": "Infrastructure timeout -- not caused by PR changes.",
                        "suggested_fix": "No action needed. Wait for re-run.",
                        "confidence": 0.95,
                    },
                }
            )
            continue
        if any(s in job["job_name"].lower() for s in SKIP_CLASSIFICATION):
            results.append({**job, "classification": None})
            continue
        log = job.get("log", "")
        if not log or not client:
            results.append({**job, "classification": None})
            continue
        results.append(
            {**job, "classification": client.classify_error(job["job_name"], log)}
        )

    return results


# ---------------------------------------------------------------------------
# Comment builder
# ---------------------------------------------------------------------------


def _fmt_diagnosis(d: dict) -> str:
    """Format a single diagnosis as a collapsible ``<details>`` block."""
    c = d.get("classification")
    if not c:
        return ""
    name = d.get("job_name", "Unknown")
    cat = c.get("category", "unknown").replace("_", " ").title()
    conf = c.get("confidence", 0)
    if isinstance(conf, (int, float)) and conf > 1:
        conf = conf / 100.0
    pct = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)

    # Condensed: merge root cause + suggested fix
    root = c.get("root_cause", "Unknown")
    fix = c.get("suggested_fix", "")
    body = root
    if fix and fix.lower() not in ("n/a", ""):
        body += f" {fix}"

    url = d.get("url", "")
    log_link = f"\n[View Logs]({url})" if url else ""
    source_note = ""
    if d.get("source") == "gitlab":
        source_note = "\n*Internal GitLab logs fetched automatically.*"

    return (
        f"<details><summary>{name}: {cat} ({pct})</summary>\n\n"
        f"{body}{log_link}{source_note}\n"
        f"</details>"
    )


def build_comment(
    checks: list[dict],
    reviews: dict,
    diagnoses: list[dict],
    has_api_key: bool,
    has_gitlab_token: bool,
) -> str:
    """Build the diagnostic comment markdown."""
    # --- Counts ---
    blocking_failed = sum(
        1 for c in checks if c["name"] in REQUIRED_SET and c["state"] == "failed"
    )
    non_blocking_failed = sum(
        1 for c in checks if c["name"] not in REQUIRED_SET and c["state"] == "failed"
    )
    total_passed = sum(1 for c in checks if c["state"] == "passed")
    total_pending = sum(1 for c in checks if c["state"] == "pending")

    # --- Header ---
    lines = [COMMENT_MARKER, "## CI Diagnostics", ""]

    parts: list[str] = []
    if blocking_failed:
        parts.append(f"**{blocking_failed} Blocking**")
    if non_blocking_failed:
        parts.append(f"{non_blocking_failed} non-blocking")
    count_parts: list[str] = []
    if total_passed:
        count_parts.append(f"{total_passed} passed")
    if total_pending:
        count_parts.append(f"{total_pending} pending")
    summary = ", ".join(parts) if parts else "All Passing"
    if count_parts:
        summary += f" · {', '.join(count_parts)}"
    summary += (
        f" · {reviews['approved']} approval{'s' if reviews['approved'] != 1 else ''}"
    )
    if reviews["changes_requested"]:
        summary += f", {reviews['changes_requested']} changes requested"
    lines.append(summary)
    lines.append("")

    # --- Blocking section ---
    lines.append("### Blocking")
    lines.append("")
    gh_diags = [
        d for d in diagnoses if d.get("source") == "github" and d.get("classification")
    ]

    for name in REQUIRED_CHECKS:
        match = next((c for c in checks if c["name"] == name), None)
        state = match["state"] if match else "unknown"
        emoji = {"passed": "✅", "failed": "❌", "pending": "⏳"}.get(state, "❓")
        lines.append(f"- {emoji} {name}")

        # Nest sub-job diagnoses under their parent required check
        if state == "failed":
            child = [
                d
                for d in gh_diags
                if WORKFLOW_TO_REQUIRED.get(d.get("run_name")) == name
            ]
            for d in child:
                block = _fmt_diagnosis(d)
                if block:
                    lines.append(block)

    lines.append("")
    if blocking_failed and not gh_diags and not has_api_key:
        lines.append("*AI analysis unavailable -- no LLM API key configured.*")
        lines.append("")

    # --- Non-blocking section ---
    gl_diags = [
        d for d in diagnoses if d.get("source") == "gitlab" and d.get("classification")
    ]
    other_failed = [
        c
        for c in checks
        if c["name"] not in REQUIRED_SET
        and c["state"] == "failed"
        and c["source"] not in ("gitlab",)
    ]
    gl_no_token = (
        any(c["source"] == "gitlab" and c["state"] == "failed" for c in checks)
        and not has_gitlab_token
    )

    if gl_diags or other_failed or gl_no_token:
        lines.append("### Non-Blocking")
        lines.append("")
        if gl_diags or gl_no_token:
            lines.append(
                "*GitLab CI runs E2E integration tests that require GPU compute"
                " on internal infrastructure."
                " Logs are not publicly accessible — fetched here via API.*"
            )
            lines.append("")
        for d in gl_diags:
            block = _fmt_diagnosis(d)
            if block:
                lines.append(block)
                lines.append("")
        for c in other_failed:
            name = c["name"]
            url = c.get("url", "")
            lines.append(f"- ❌ [{name}]({url})" if url else f"- ❌ {name}")
        if gl_no_token:
            lines.append("")
            lines.append("*GitLab pipeline failed -- logs unavailable (internal CI).*")
        lines.append("")

    if not blocking_failed and not non_blocking_failed:
        lines.append("*No failures detected.*")
        lines.append("")

    lines.append("---")
    lines.append("*LGTM Bot `/diagnose` · Updates on re-run*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comment posting
# ---------------------------------------------------------------------------


def _write_tmp(body: str) -> str:
    """Write *body* to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(body)
        return f.name


def post_or_update_comment(repo: str, pr_number: str, body: str) -> str | None:
    """Post or update the diagnostic comment.  Returns comment URL or *None*."""
    comments = gh_api(f"/repos/{repo}/issues/{pr_number}/comments?per_page=100")
    existing_id = None
    if comments and isinstance(comments, list):
        for c in comments:
            if COMMENT_MARKER in c.get("body", ""):
                existing_id = c["id"]
                break

    tmp = _write_tmp(body)
    try:
        if existing_id:
            endpoint = f"/repos/{repo}/issues/comments/{existing_id}"
            method = "PATCH"
        else:
            endpoint = f"/repos/{repo}/issues/{pr_number}/comments"
            method = "POST"
        r = subprocess.run(
            ["gh", "api", endpoint, "--method", method, "-F", f"body=@{tmp}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode == 0 and r.stdout.strip():
            try:
                return json.loads(r.stdout).get("html_url")
            except json.JSONDecodeError:
                pass
        elif r.returncode != 0:
            print(f"Comment post failed: {r.stderr[:200]}", file=sys.stderr)
    finally:
        os.unlink(tmp)
    return None


def post_reply_link(repo: str, pr_number: str, diag_url: str) -> None:
    """Post a lightweight reply linking to the diagnostic comment."""
    body = f"Diagnostics updated — [view results]({diag_url})"
    tmp = _write_tmp(body)
    try:
        r = subprocess.run(
            [
                "gh",
                "api",
                f"/repos/{repo}/issues/{pr_number}/comments",
                "--method",
                "POST",
                "-F",
                f"body=@{tmp}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            print(f"Reply-link failed: {r.stderr[:200]}", file=sys.stderr)
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
        print("GITHUB_REPOSITORY and PR_NUMBER required", file=sys.stderr)
        sys.exit(1)

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

    # Gather status
    checks = get_all_checks(repo, sha)
    reviews = get_review_summary(repo, pr_number)
    any_failed = any(c["state"] == "failed" for c in checks)
    has_api_key = bool(os.environ.get("LLM_API_KEY"))
    has_gitlab_token = bool(os.environ.get("GITLAB_TOKEN"))

    # Fetch failed job details (GitHub Actions + GitLab)
    github_jobs: list[dict] = []
    gitlab_jobs: list[dict] = []
    if any_failed:
        github_jobs = get_failed_github_jobs(repo, sha)
        gitlab_jobs = get_gitlab_failures(checks)

    # Classify
    diagnoses: list[dict] = []
    if (github_jobs or gitlab_jobs) and has_api_key:
        diagnoses = classify_failures(repo, github_jobs, gitlab_jobs)

    # Post diagnostic comment
    body = build_comment(checks, reviews, diagnoses, has_api_key, has_gitlab_token)
    diag_url = post_or_update_comment(repo, pr_number, body)

    # Reply-link to the triggering /diagnose comment
    if comment_id and diag_url:
        post_reply_link(repo, pr_number, diag_url)

    # Completion reaction
    if comment_id:
        add_reaction(repo, comment_id, "rocket" if not any_failed else "confused")

    failed_count = sum(1 for c in checks if c["state"] == "failed")
    print(f"Diagnosis complete: {failed_count} failures analyzed", file=sys.stderr)


if __name__ == "__main__":
    main()
