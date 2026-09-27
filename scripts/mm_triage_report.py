#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Daily multimodality (mm) triage report for ai-dynamo/dynamo.

Pulls open/recently-active issues and PRs labeled `multimodal`, groups them
by category and by who is currently engaging, and renders a Markdown report.

Data source: `gh` CLI (must be authenticated, e.g. via GITHUB_TOKEN in Actions).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

REPO = "ai-dynamo/dynamo"
MM_LABEL = "multimodal"
RECENT_DAYS = 7
STALE_DAYS = 30

LINEAR_PATTERN = re.compile(r"\b([A-Z]{2,10}-\d+)\b")
LINEAR_URL_PATTERN = re.compile(r"https?://linear\.app/\S+")

TRIAGE_LABELS = {"needs-triage", "needs-info", "needs-more-info"}
BLOCKED_LABELS = {"blocked", "blocked:external"}
BUG_LABELS = {"bug", "QA Bug"}
FEATURE_LABELS = {"enhancement", "feat", "Feature", "Improvement"}
DOCS_LABELS = {"docs", "documentation"}
PERF_LABELS = {"perf", "performance", "tokenspeed"}


def run_gh(args: list[str]) -> list[dict]:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def fetch_issues() -> list[dict]:
    return run_gh(
        [
            "issue", "list", "--repo", REPO, "--label", MM_LABEL, "--state", "all",
            "--limit", "500",
            "--json", "number,title,state,labels,assignees,author,createdAt,updatedAt,url,body,comments",
        ]
    )


def fetch_prs() -> list[dict]:
    return run_gh(
        [
            "pr", "list", "--repo", REPO, "--label", MM_LABEL, "--state", "all",
            "--limit", "500",
            "--json", "number,title,state,labels,assignees,author,createdAt,updatedAt,url,body,"
                      "isDraft,mergedAt,comments",
        ]
    )


def extract_linear_refs(text: str) -> list[str]:
    if not text:
        return []
    refs = set(LINEAR_PATTERN.findall(text)) | set(LINEAR_URL_PATTERN.findall(text))
    # Filter out obvious false positives from ordinary version-like tokens
    return sorted(r for r in refs if not r.startswith(("UTC", "GMT")))


def category_for(labels: set[str]) -> str:
    if labels & BLOCKED_LABELS:
        return "Blocked"
    if labels & TRIAGE_LABELS:
        return "Needs Triage"
    if labels & BUG_LABELS:
        return "Bug"
    if labels & PERF_LABELS:
        return "Performance"
    if labels & DOCS_LABELS:
        return "Docs"
    if labels & FEATURE_LABELS:
        return "Feature / Enhancement"
    return "Other"


def days_since(iso_ts: str, now: datetime) -> int:
    ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    return (now - ts).days


def engagers_for(item: dict) -> list[str]:
    people = [a["login"] for a in item.get("assignees", [])]
    if not people and item.get("author"):
        people = [item["author"]["login"]]
    return people or ["(unassigned)"]


def render_report(issues: list[dict], prs: list[dict], now: datetime) -> str:
    lines = []
    lines.append(f"# Multimodality Daily Triage Report — {now.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append(
        f"Source: `{REPO}` issues/PRs labeled `{MM_LABEL}`. "
        f"Generated automatically; see workflow `.github/workflows/mm_daily_triage.yml`."
    )
    lines.append("")

    open_issues = [i for i in issues if i["state"] == "OPEN"]
    open_prs = [p for p in prs if p["state"] == "OPEN"]
    recently_closed_issues = [
        i for i in issues if i["state"] == "CLOSED" and days_since(i["updatedAt"], now) <= RECENT_DAYS
    ]
    recently_merged_prs = [
        p for p in prs if p.get("mergedAt") and days_since(p["mergedAt"], now) <= RECENT_DAYS
    ]
    stale_open_issues = [
        i for i in open_issues if days_since(i["updatedAt"], now) >= STALE_DAYS
    ]

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Open issues: **{len(open_issues)}**")
    lines.append(f"- Open PRs: **{len(open_prs)}**")
    lines.append(f"- Closed issues (last {RECENT_DAYS}d): **{len(recently_closed_issues)}**")
    lines.append(f"- Merged PRs (last {RECENT_DAYS}d): **{len(recently_merged_prs)}**")
    lines.append(f"- Open issues stale (>= {STALE_DAYS}d no activity): **{len(stale_open_issues)}**")
    lines.append("")

    # --- By category ---
    lines.append("## Open Issues by Category")
    lines.append("")
    by_cat = defaultdict(list)
    for i in open_issues:
        labels = {l["name"] for l in i["labels"]}
        by_cat[category_for(labels)].append(i)
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        items = by_cat[cat]
        lines.append(f"### {cat} ({len(items)})")
        for i in sorted(items, key=lambda x: x["updatedAt"], reverse=True):
            age = days_since(i["updatedAt"], now)
            lines.append(f"- [#{i['number']}]({i['url']}) {i['title']} — updated {age}d ago")
        lines.append("")

    lines.append("## Open PRs")
    lines.append("")
    if open_prs:
        for p in sorted(open_prs, key=lambda x: x["updatedAt"], reverse=True):
            age = days_since(p["updatedAt"], now)
            draft = " (draft)" if p.get("isDraft") else ""
            lines.append(f"- [#{p['number']}]({p['url']}) {p['title']}{draft} — updated {age}d ago")
    else:
        lines.append("_None open._")
    lines.append("")

    # --- By engagement ---
    lines.append("## By Engagement (assignee / author, most recent activity first)")
    lines.append("")
    by_person = defaultdict(list)
    for item in open_issues + open_prs:
        kind = "PR" if "isDraft" in item else "Issue"
        for person in engagers_for(item):
            by_person[person].append((item, kind))

    for person in sorted(by_person, key=lambda p: max(
        datetime.fromisoformat(it["updatedAt"].replace("Z", "+00:00")) for it, _ in by_person[p]
    ), reverse=True):
        entries = by_person[person]
        lines.append(f"### {person} ({len(entries)})")
        for item, kind in sorted(entries, key=lambda x: x[0]["updatedAt"], reverse=True):
            age = days_since(item["updatedAt"], now)
            linear_refs = extract_linear_refs(item.get("body", ""))
            linear_str = f" — linked: {', '.join(linear_refs)}" if linear_refs else ""
            lines.append(f"- [{kind} #{item['number']}]({item['url']}) {item['title']} — {age}d ago{linear_str}")
        lines.append("")

    # --- Linear cross-references ---
    lines.append("## Items with Linear Ticket References")
    lines.append("")
    any_linear = False
    for item in issues + prs:
        refs = extract_linear_refs(item.get("body", ""))
        if refs:
            any_linear = True
            kind = "PR" if "isDraft" in item else "Issue"
            lines.append(f"- [{kind} #{item['number']}]({item['url']}) {item['title']} — {', '.join(refs)}")
    if not any_linear:
        lines.append("_No Linear references detected in issue/PR bodies this run._")
    lines.append("")

    # --- Stale ---
    if stale_open_issues:
        lines.append(f"## Stale Open Issues (>= {STALE_DAYS}d, needs a follow-up decision)")
        lines.append("")
        for i in sorted(stale_open_issues, key=lambda x: x["updatedAt"]):
            age = days_since(i["updatedAt"], now)
            people = ", ".join(engagers_for(i))
            lines.append(f"- [#{i['number']}]({i['url']}) {i['title']} — {age}d idle, owner(s): {people}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="-", help="Output file path, or - for stdout")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    issues = fetch_issues()
    prs = fetch_prs()
    report = render_report(issues, prs, now)

    if args.out == "-":
        print(report)
    else:
        with open(args.out, "w") as f:
            f.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
