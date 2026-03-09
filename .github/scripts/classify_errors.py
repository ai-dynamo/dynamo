#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CI error classifier. Fetches failed job logs from GitHub Actions,
sends them to an OpenAI-compatible LLM for classification, posts
a PR comment (for PR workflows), and writes database-ready JSON.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests

MAX_LOG_CHARS = 400_000


def main():
    parser = argparse.ArgumentParser(description="Classify CI errors from GitHub Actions")
    parser.add_argument("--run-id", required=True, help="GitHub Actions run ID")
    parser.add_argument("--repo", required=True, help="GitHub repo (owner/name)")
    parser.add_argument(
        "--workflow-type",
        required=True,
        choices=["pr", "nightly", "post-merge", "release"],
    )
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument("--output-json", required=True, help="Path to write JSON output")
    parser.add_argument(
        "--prompt-file",
        default=os.path.join(os.path.dirname(__file__), "classify_errors_prompt.txt"),
    )
    args = parser.parse_args()

    github_token = os.environ.get("GITHUB_TOKEN", "")
    api_key = os.environ.get("API_KEY", "")
    base_url = os.environ.get("API_BASE_URL", "")
    model = os.environ.get("MODEL", "")

    if not github_token:
        print("Error: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    if not api_key or not base_url or not model:
        print("Error: API_KEY, API_BASE_URL, and MODEL must be set", file=sys.stderr)
        sys.exit(1)

    system_prompt = load_prompt(args.prompt_file)

    # Get run metadata
    run_meta = get_run_metadata(args.repo, args.run_id, github_token)

    # Get failed jobs
    failed_jobs = get_failed_jobs(args.repo, args.run_id, github_token)
    if not failed_jobs:
        print("No failed jobs found.")
        write_json_output(
            args.output_json,
            run_id=args.run_id,
            repo=args.repo,
            workflow_name=run_meta.get("name", ""),
            workflow_type=args.workflow_type,
            branch=run_meta.get("head_branch", ""),
            commit_sha=run_meta.get("head_sha", ""),
            pr_number=args.pr_number,
            model=model,
            errors=[],
        )
        return

    print(f"Found {len(failed_jobs)} failed job(s). Classifying...")

    # Classify each failed job
    all_errors = []
    for job in failed_jobs:
        job_id = str(job["id"])
        job_name = job["name"]
        print(f"  Processing: {job_name}")

        log = get_job_log(args.repo, job_id, github_token)
        if not log:
            print(f"    Warning: Could not fetch log for job {job_name}")
            continue

        result = classify_errors(log, job_name, system_prompt, api_key, model, base_url)
        if result and result.get("errors_found"):
            for error in result["errors_found"]:
                all_errors.append(
                    {
                        "job_id": job_id,
                        "job_name": job_name,
                        "step": error.get("step", "unknown"),
                        "category": error.get("primary_category", "code_error"),
                        "confidence": error.get("confidence_score", 0.5),
                        "is_transient": error.get("is_transient", False),
                        "summary": error.get("root_cause_summary", ""),
                        "log_excerpt": error.get("log_excerpt", ""),
                    }
                )

    # Write JSON output
    write_json_output(
        args.output_json,
        run_id=args.run_id,
        repo=args.repo,
        workflow_name=run_meta.get("name", ""),
        workflow_type=args.workflow_type,
        branch=run_meta.get("head_branch", ""),
        commit_sha=run_meta.get("head_sha", ""),
        pr_number=args.pr_number,
        model=model,
        errors=all_errors,
    )
    print(f"Wrote classification results to {args.output_json}")

    # Post PR comment only for PR workflows
    if args.workflow_type == "pr" and args.pr_number:
        run_url = f"https://github.com/{args.repo}/actions/runs/{args.run_id}"
        post_pr_comment(
            args.repo,
            args.pr_number,
            all_errors,
            args.run_id,
            run_url,
            run_meta.get("name", ""),
            github_token,
        )
        print(f"Posted PR comment to #{args.pr_number}")


def load_prompt(path):
    """Read the system prompt from a text file."""
    with open(path) as f:
        return f.read()


def get_run_metadata(repo, run_id, token):
    """Fetch workflow run metadata."""
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def get_failed_jobs(repo, run_id, token):
    """Get all failed jobs for a workflow run."""
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    params = {"per_page": 100, "filter": "latest"}
    all_jobs = []
    while url:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        all_jobs.extend(data.get("jobs", []))
        # Handle pagination
        url = resp.links.get("next", {}).get("url")
        params = None  # params are already in the next URL
    return [j for j in all_jobs if j.get("conclusion") == "failure"]


def get_job_log(repo, job_id, token):
    """Fetch job logs, truncated to the last MAX_LOG_CHARS characters."""
    url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    resp = requests.get(url, headers=headers, allow_redirects=True)
    if resp.status_code != 200:
        return None
    text = resp.text
    if len(text) > MAX_LOG_CHARS:
        text = text[-MAX_LOG_CHARS:]
    return text


def classify_errors(log, job_name, system_prompt, api_key, model, base_url):
    """Send a job log to the LLM and parse the classification result."""
    user_prompt = f"Analyze the following CI job log for job '{job_name}':\n\n{log}"
    response_text = call_llm(system_prompt, user_prompt, api_key, model, base_url)
    if not response_text:
        return None
    return parse_llm_response(response_text)


def call_llm(system_prompt, user_prompt, api_key, model, base_url):
    """Make a chat completion request to an OpenAI-compatible API."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_llm_response(text):
    """Extract JSON object from LLM response text."""
    # Find the first { and last } to extract JSON
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        print(f"Warning: Could not find JSON in LLM response: {text[:200]}", file=sys.stderr)
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse LLM JSON: {e}", file=sys.stderr)
        return None


def post_pr_comment(repo, pr_number, results, run_id, run_url, workflow_name, token):
    """Post or update a PR comment with classification results."""
    marker = "<!-- error-classification -->"
    failed_count = len({r["job_id"] for r in results})

    # Build table rows
    rows = []
    for r in results:
        category_short = r["category"].replace("_error", "")
        transient = "Yes" if r["is_transient"] else "No"
        summary = r["summary"][:80] + "..." if len(r["summary"]) > 80 else r["summary"]
        rows.append(
            f"| {r['job_name']} | {r['step']} | {category_short} | {summary} | {transient} |"
        )

    table = "\n".join(rows) if rows else "| (no errors detected) | | | | |"

    body = f"""{marker}
## CI Error Classification

**Run**: [#{run_id}]({run_url}) | **Failed Jobs**: {failed_count}

| Job | Step | Category | Summary | Transient? |
|-----|------|----------|---------|------------|
{table}
"""

    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    # Search for existing comment to update
    comment_id = find_existing_comment(repo, pr_number, marker, headers)
    if comment_id:
        url = f"https://api.github.com/repos/{repo}/issues/comments/{comment_id}"
        resp = requests.patch(url, headers=headers, json={"body": body})
    else:
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
        resp = requests.post(url, headers=headers, json={"body": body})
    resp.raise_for_status()


def find_existing_comment(repo, pr_number, marker, headers):
    """Find an existing comment containing the marker string."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    params = {"per_page": 100}
    while url:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        for comment in resp.json():
            if marker in comment.get("body", ""):
                return comment["id"]
        url = resp.links.get("next", {}).get("url")
        params = None
    return None


def write_json_output(path, run_id, repo, workflow_name, workflow_type, branch, commit_sha,
                      pr_number, model, errors):
    """Write database-ready JSON output."""
    run_url = f"https://github.com/{repo}/actions/runs/{run_id}"
    output = {
        "run_id": str(run_id),
        "run_url": run_url,
        "repo": repo,
        "workflow_name": workflow_name,
        "workflow_type": workflow_type,
        "branch": branch,
        "commit_sha": commit_sha,
        "pr_number": pr_number,
        "classified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "errors": errors,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
