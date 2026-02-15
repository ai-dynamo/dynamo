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

"""Classify CI workflow errors using LLM analysis.

Uses ``gh`` CLI for all GitHub API calls and the error_classification
package for LLM inference.

Environment variables:
    GITHUB_REPOSITORY  owner/repo (required)
    COMMIT_SHA         Commit SHA to analyze (falls back to git HEAD)
    LLM_API_KEY        API key for LLM inference (optional -- degrades gracefully)
    OUTPUT_ONLY        "true" → JSON to stdout only (for subprocess capture)
    PR_NUMBER          PR number for automatic comment posting
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any

COMMENT_MARKER = "<!-- ci-error-classifier -->"

# Jobs where logs already explain exactly what to fix — skip LLM classification
SKIP_CLASSIFICATION = {
    "pre-commit",
    "copyright",
    "dco",
    "lychee",
    "label",
    "lint",
    "codespell",
}


# ---------------------------------------------------------------------------
# GitHub helpers (all via gh CLI)
# ---------------------------------------------------------------------------


def gh_api(path: str) -> Any:
    """Call GitHub API via ``gh`` CLI.  Returns parsed JSON or *None*."""
    r = subprocess.run(
        ["gh", "api", path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        print(f"gh api {path}: {r.stderr[:200]}", file=sys.stderr)
        return None
    try:
        return json.loads(r.stdout) if r.stdout.strip() else None
    except json.JSONDecodeError:
        print(f"gh api {path}: invalid JSON response", file=sys.stderr)
        return None


def get_failed_jobs(repo: str, sha: str) -> list[dict]:
    """Return failed jobs from completed workflow runs for *sha*."""
    runs = gh_api(
        f"/repos/{repo}/actions/runs?head_sha={sha}" "&status=completed&per_page=20"
    )
    if not runs or not runs.get("workflow_runs"):
        return []

    failed: list[dict] = []
    for run in runs["workflow_runs"]:
        if run["conclusion"] != "failure":
            continue
        jobs = gh_api(f"/repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100")
        if not jobs:
            continue
        for job in jobs.get("jobs", []):
            if job["conclusion"] == "failure":
                failed.append(
                    {
                        "run_id": run["id"],
                        "run_name": run["name"],
                        "job_id": job["id"],
                        "job_name": job["name"],
                        "url": job["html_url"],
                    }
                )
    return failed


def get_job_log(repo: str, job_id: int) -> str:
    """Fetch raw job log via ``gh`` CLI."""
    r = subprocess.run(
        ["gh", "api", f"/repos/{repo}/actions/jobs/{job_id}/logs"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return r.stdout if r.returncode == 0 else ""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_all(repo: str, sha: str) -> list[dict]:
    """Classify every failed job for *sha* using the LLM client."""
    # Allow import when called from repo root or .github/scripts/
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from error_classification.config import Config
    from error_classification.llm_client import LLMClient

    client = LLMClient(Config.from_env())
    jobs = get_failed_jobs(repo, sha)
    if not jobs:
        return []

    results = []
    for job in jobs:
        job_lower = job["job_name"].lower()
        if any(skip in job_lower for skip in SKIP_CLASSIFICATION):
            # Self-explanatory checks — logs already tell you what to fix
            results.append({**job, "classification": None})
            continue
        log = get_job_log(repo, job["job_id"])
        if not log:
            # Skip classification when log is empty (network error, etc.)
            results.append({**job, "classification": None})
            continue
        classification = client.classify_error(job["job_name"], log)
        results.append({**job, "classification": classification})
    return results


# ---------------------------------------------------------------------------
# Comment posting (used when called from reusable workflow)
# ---------------------------------------------------------------------------


def _build_comment(results: list[dict]) -> str:
    lines = [COMMENT_MARKER, "## CI Error Analysis", ""]
    for r in results:
        c = r.get("classification")
        if not c:
            continue
        name = r.get("job_name", "Unknown")
        url = r.get("url", "")
        cat = c.get("category", "unknown").replace("_", " ").title()
        conf = c.get("confidence", 0)
        if isinstance(conf, (int, float)) and conf > 1:
            conf = conf / 100.0  # Normalize 0-100 to 0-1
        pct = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)

        lines.append(f"<details><summary><b>{name}</b> — {cat} ({pct})</summary>")
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

    if not any(r.get("classification") for r in results):
        lines.append("*No classifiable errors found.*")
        lines.append("")
    lines.append("---")
    lines.append("*Auto-generated by CI error classifier.*")
    return "\n".join(lines)


def _post_or_update_comment(repo: str, pr_number: str, body: str) -> bool:
    """Find existing classifier comment and update, or create new.

    Returns True if the comment was successfully posted/updated.
    """
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
            r = subprocess.run(
                [
                    "gh",
                    "api",
                    f"/repos/{repo}/issues/comments/{existing_id}",
                    "--method",
                    "PATCH",
                    "-F",
                    f"body=@{tmp}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if r.returncode != 0:
                print(f"Comment update failed: {r.stderr[:200]}", file=sys.stderr)
                return False
        else:
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
                print(f"Comment post failed: {r.stderr[:200]}", file=sys.stderr)
                return False
    finally:
        os.unlink(tmp)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    sha = os.environ.get("COMMIT_SHA", "")
    output_only = os.environ.get("OUTPUT_ONLY", "").lower() == "true"
    pr_number = os.environ.get("PR_NUMBER", "")

    if not repo:
        print("GITHUB_REPOSITORY required", file=sys.stderr)
        sys.exit(1)

    if not sha:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        sha = r.stdout.strip()

    if not sha:
        print("No commit SHA available", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing failures for {repo}@{sha[:8]}...", file=sys.stderr)
    results = classify_all(repo, sha)

    if output_only:
        print(json.dumps(results, indent=2))
        return

    if not results:
        print("No failed jobs found.", file=sys.stderr)
        return

    if pr_number:
        body = _build_comment(results)
        posted = _post_or_update_comment(repo, pr_number, body)
        if posted:
            print(
                f"Posted classification for {len(results)} failures.",
                file=sys.stderr,
            )
        else:
            print("Failed to post classification comment.", file=sys.stderr)
    else:
        for r in results:
            c = r.get("classification") or {}
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(
                f"Job: {r['job_name']} ({r['run_name']})",
                file=sys.stderr,
            )
            print(f"Category: {c.get('category', 'unknown')}", file=sys.stderr)
            print(
                f"Root cause: {c.get('root_cause', 'N/A')}",
                file=sys.stderr,
            )
            print(f"Fix: {c.get('suggested_fix', 'N/A')}", file=sys.stderr)


if __name__ == "__main__":
    main()
