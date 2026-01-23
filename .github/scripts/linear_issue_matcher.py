#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Linear Issue Matcher - GitHub Action to find related Linear issues for PRs.

This script is designed to run as a GitHub Action. It:
1. Checks if PR author is a Dynamo team member (from config file)
2. Fetches Linear issues assigned to the PR author (using their Linear UUID)
3. Falls back to broader issue search if no author-specific matches
4. Uses two-model approach: fast model for screening, better model for final matching
5. Posts a PR comment with related issues (if high-confidence matches found)

Environment Variables (set by workflow):
    PR_NUMBER       - PR number
    PR_TITLE        - PR title
    PR_BODY         - PR body/description
    PR_BRANCH       - Head branch name
    PR_AUTHOR       - PR author username
    CHANGED_FILES   - Comma-separated list of changed files
    LINEAR_API_KEY  - Linear API key
    NVIDIA_API_KEY  - NVIDIA Inference Hub API key
    GITHUB_TOKEN    - GitHub token for posting comments
    GITHUB_REPOSITORY - Repository in owner/repo format
    GITHUB_WORKSPACE - Repository root path (set by GitHub Actions)
"""

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# API endpoints
LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
NVIDIA_INFERENCE_URL = "https://inference-api.nvidia.com/v1/chat/completions"
GITHUB_API_URL = "https://api.github.com"

# Model settings - two-tier approach
FAST_MODEL = "aws/anthropic/bedrock-claude-sonnet-4-5-v1"  # Fast/cheap for screening
FINAL_MODEL = "aws/anthropic/claude-opus-4-5"  # Better model for final matching

# Thresholds
SCREENING_THRESHOLD = 0.50  # Fast model threshold to proceed to final matching
CONFIDENCE_THRESHOLD = 0.70  # Final model threshold to post comment
MAX_CANDIDATES_FOR_SCREENING = 10  # Max issues for fast model screening
MAX_CANDIDATES_FOR_FINAL = 5  # Max issues for final model matching

# Linear settings
LINEAR_DAYS_LOOKBACK = 60  # Days to search Linear issues
LINEAR_AUTHOR_LIMIT = 50  # Max issues to fetch for author
LINEAR_FALLBACK_LIMIT = 500  # Max issues for fallback search (increased for multi-team)
LINEAR_TEAMS = ["DYN", "DIS", "DEP", "LLM", "OPS", "DGH"]  # All teams to search

# Path to team members file (relative to repo root)
TEAM_MEMBERS_FILE = ".github/config/dynamo-team-members.txt"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TeamMember:
    """Represents a team member with GitHub and Linear identities."""

    github: str
    linear_id: Optional[str] = None


@dataclass
class PRContext:
    """Context extracted from the GitHub PR."""

    number: int
    title: str
    body: str
    branch: str
    author: str
    files_changed: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "PRContext":
        """Create PRContext from environment variables."""
        files_str = os.environ.get("CHANGED_FILES", "")
        files = [f.strip() for f in files_str.split(",") if f.strip()]

        return cls(
            number=int(os.environ.get("PR_NUMBER", "0")),
            title=os.environ.get("PR_TITLE", ""),
            body=os.environ.get("PR_BODY", ""),
            branch=os.environ.get("PR_BRANCH", ""),
            author=os.environ.get("PR_AUTHOR", ""),
            files_changed=files,
        )


@dataclass
class LinearIssue:
    """Represents a Linear issue."""

    identifier: str
    title: str
    description: str
    assignee: Optional[str]
    state: str
    priority: int
    url: str
    labels: list[str] = field(default_factory=list)
    project: Optional[str] = None  # Project name for alignment scoring


@dataclass
class MatchResult:
    """Result of matching a PR to a Linear issue."""

    issue: LinearIssue
    confidence: float
    reasoning: str
    screening_score: float = 0.0  # Score from fast model screening


# =============================================================================
# Category Patterns (for pre-filtering)
# =============================================================================

CATEGORY_PATTERNS = [
    # Core distributed inference
    (
        "Router",
        [
            r"router",
            r"routing",
            r"kv.?aware",
            r"radix",
            r"tree.?size",
            r"prefill.?route",
        ],
    ),
    ("KVBM", [r"kvbm", r"kv.?block", r"kv.?event", r"consolidator", r"cache.?hit"]),
    (
        "Planner",
        [
            r"planner",
            r"profiler",
            r"profiling",
            r"sla",
            r"pareto",
            r"gpu.?cost",
            r"webui",
            r"mocker",
        ],
    ),
    # API and frontend
    (
        "Frontend",
        [
            r"frontend",
            r"openai",
            r"/v1/",
            r"validation",
            r"response_format",
            r"nvext",
            r"completions",
            r"chat",
            r"kserve",
            r"usage.?stats",
        ],
    ),
    (
        "Multimodal",
        [r"multimodal", r"image", r"video", r"audio", r"media", r"mm.?flow"],
    ),
    ("Tool Calling", [r"tool.?call", r"tool.?choice", r"tool.?parser"]),
    # Backends
    ("vLLM", [r"vllm", r"backend::vllm"]),
    ("SGLang", [r"sglang", r"backend::sglang"]),
    ("TRT-LLM", [r"trtllm", r"tensorrt", r"backend::trtllm", r"autodeploy"]),
    ("LoRA", [r"lora"]),
    # Kubernetes and deployment
    (
        "Kubernetes",
        [
            r"operator",
            r"k8s",
            r"k8",
            r"dgd",
            r"dgdr",
            r"crd",
            r"helm",
            r"namespace",
            r"webhook",
            r"scaling",
        ],
    ),
    # Observability
    (
        "Observability",
        [
            r"metric",
            r"prometheus",
            r"grafana",
            r"dashboard",
            r"tracing",
            r"otel",
            r"logging",
        ],
    ),
    # Fault tolerance
    (
        "Fault Tolerance",
        [r"fault", r"health.?check", r"ha\b", r"failover", r"resilience", r"graceful"],
    ),
    # Infrastructure
    ("Infrastructure", [r"nats", r"transport", r"nixl", r"filestore", r"storage"]),
    # Discovery and runtime
    ("Discovery", [r"discovery", r"endpoint", r"registration", r"unregister"]),
    # Build and container
    ("Build", [r"cuda", r"container", r"cublas", r"gaudi", r"abi"]),
    # Model support
    ("Model Support", [r"mistral", r"deepseek", r"qwen", r"model.?card"]),
]

# Component patterns for file path analysis
COMPONENT_PATTERNS = [
    (r"^deploy/", "Kubernetes/Operator"),
    (r"^\.github/", "CI/CD"),
    (r"Dockerfile", "Container"),
    (r"^container/", "Container"),
    (r"Earthfile", "Container"),
    (r"^lib/llm/", "LLM Engine"),
    (r"^lib/runtime/", "Runtime"),
    (r"^lib/async-openai/", "OpenAI Client"),
    (r"^lib/memory/", "Memory"),
    (r"\.rs$", "Rust"),
    (r"^components/src/dynamo/router/", "Router"),
    (r"^components/src/dynamo/planner/", "Planner"),
    (r"^lib/bindings/kvbm/", "KVBM"),
    (r"^components/src/dynamo/frontend/", "Frontend"),
    (r"^components/src/dynamo/vllm/", "vLLM"),
    (r"^components/src/dynamo/sglang/", "SGLang"),
    (r"^components/src/dynamo/trtllm/", "TRT-LLM"),
    (r"^docs/", "Documentation"),
    (r"\.md$", "Documentation"),
]


# =============================================================================
# Team Membership
# =============================================================================


def load_team_members(repo_root: str = ".") -> dict[str, TeamMember]:
    """
    Load Dynamo team members from the config file.

    Returns a dict mapping lowercase GitHub username to TeamMember.
    """
    members_file = os.path.join(repo_root, TEAM_MEMBERS_FILE)

    if not os.path.exists(members_file):
        logger.warning(f"Team members file not found: {members_file}")
        return {}

    members = {}
    try:
        with open(members_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse github:linear_id or just github
                if ":" in line:
                    parts = line.split(":", 1)
                    github = parts[0].strip()
                    linear_id = parts[1].strip()
                    members[github.lower()] = TeamMember(
                        github=github, linear_id=linear_id
                    )
                else:
                    github = line.strip()
                    members[github.lower()] = TeamMember(github=github, linear_id=None)

        logger.info(f"Loaded {len(members)} team members from {TEAM_MEMBERS_FILE}")
        return members

    except OSError as e:
        logger.error(f"Error reading team members file: {e}")
        return {}


def get_team_member(
    username: str, team_members: dict[str, TeamMember]
) -> Optional[TeamMember]:
    """
    Get a team member by GitHub username (case-insensitive).

    Returns TeamMember if found, None otherwise.
    """
    return team_members.get(username.lower())


# =============================================================================
# Utility Functions
# =============================================================================


def extract_linear_issue_ids(text: str) -> list[str]:
    """Extract Linear issue IDs from text (e.g., DYN-1234)."""
    standard_pattern = r"\b([A-Za-z]{2,4})-(\d{1,5})\b"
    no_hyphen_pattern = r"\b([A-Za-z]{2,4})(\d{1,5})\b"
    valid_prefixes = {"DYN", "DIS", "DEP", "LLM", "OPS", "DGH", "DIA"}

    matches = []
    for prefix, num in re.findall(standard_pattern, text):
        if prefix.upper() in valid_prefixes:
            matches.append(f"{prefix.upper()}-{num}")
    for prefix, num in re.findall(no_hyphen_pattern, text):
        if prefix.upper() in valid_prefixes:
            normalized = f"{prefix.upper()}-{num}"
            if normalized not in matches:
                matches.append(normalized)
    return matches


def categorize_by_keywords(title: str, description: str = "") -> list[str]:
    """Categorize text by matching against keyword patterns."""
    text = (title + " " + description).lower()
    categories = []
    for category, patterns in CATEGORY_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text):
                categories.append(category)
                break
    return categories if categories else ["Other"]


def infer_components_from_files(files: list[str]) -> set[str]:
    """Infer components from file paths."""
    components = set()
    for file_path in files:
        for pattern, component in COMPONENT_PATTERNS:
            if re.search(pattern, file_path):
                components.add(component)
                break
    return components


# =============================================================================
# Linear API
# =============================================================================


def fetch_linear_issues_for_assignee(
    token: str,
    assignee_id: str,
    days: int = LINEAR_DAYS_LOOKBACK,
    limit: int = LINEAR_AUTHOR_LIMIT,
) -> list[LinearIssue]:
    """Fetch Linear issues assigned to a specific user by their Linear UUID."""
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

    query = """
    query($filter: IssueFilter, $first: Int!, $after: String) {
        issues(filter: $filter, first: $first, after: $after, orderBy: updatedAt) {
            pageInfo { hasNextPage, endCursor }
            nodes {
                identifier, title, description, priority, url
                state { name }
                assignee { name }
                labels { nodes { name } }
                project { name }
            }
        }
    }
    """

    filter_parts = {
        "createdAt": {"gte": cutoff_date},
        "assignee": {"id": {"eq": assignee_id}},
    }

    return _fetch_linear_issues(token, query, filter_parts, limit)


def fetch_linear_issues_for_team(
    token: str,
    team: str = "DYN",
    days: int = LINEAR_DAYS_LOOKBACK,
    limit: int = LINEAR_FALLBACK_LIMIT,
) -> list[LinearIssue]:
    """Fetch Linear issues for a team (fallback when no author-specific issues)."""
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

    query = """
    query($filter: IssueFilter, $first: Int!, $after: String) {
        issues(filter: $filter, first: $first, after: $after, orderBy: updatedAt) {
            pageInfo { hasNextPage, endCursor }
            nodes {
                identifier, title, description, priority, url
                state { name }
                assignee { name }
                labels { nodes { name } }
                project { name }
            }
        }
    }
    """

    filter_parts = {
        "createdAt": {"gte": cutoff_date},
        "team": {"key": {"eq": team}},
    }

    return _fetch_linear_issues(token, query, filter_parts, limit)


def fetch_linear_issues_for_teams(
    token: str,
    teams: list[str] = None,
    days: int = LINEAR_DAYS_LOOKBACK,
    limit: int = LINEAR_FALLBACK_LIMIT,
) -> list[LinearIssue]:
    """Fetch Linear issues from multiple teams using IN filter."""
    if teams is None:
        teams = LINEAR_TEAMS
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

    query = """
    query($filter: IssueFilter, $first: Int!, $after: String) {
        issues(filter: $filter, first: $first, after: $after, orderBy: updatedAt) {
            pageInfo { hasNextPage, endCursor }
            nodes {
                identifier, title, description, priority, url
                state { name }
                assignee { name }
                labels { nodes { name } }
                project { name }
            }
        }
    }
    """

    filter_parts = {
        "createdAt": {"gte": cutoff_date},
        "team": {"key": {"in": teams}},  # Multi-team query
    }

    return _fetch_linear_issues(token, query, filter_parts, limit)


def _fetch_linear_issues(
    token: str,
    query: str,
    filter_parts: dict,
    limit: int,
) -> list[LinearIssue]:
    """Internal helper to fetch Linear issues."""
    headers = {"Authorization": token, "Content-Type": "application/json"}
    issues = []
    cursor = None

    while len(issues) < limit:
        variables = {
            "filter": filter_parts,
            "first": min(50, limit - len(issues)),
            "after": cursor,
        }

        try:
            response = requests.post(
                LINEAR_GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                logger.error(f"Linear GraphQL errors: {data['errors']}")
                break

            issues_data = data.get("data", {}).get("issues", {})
            nodes = issues_data.get("nodes", [])

            for node in nodes:
                issue = LinearIssue(
                    identifier=node["identifier"],
                    title=node["title"],
                    description=node.get("description") or "",
                    assignee=node.get("assignee", {}).get("name")
                    if node.get("assignee")
                    else None,
                    state=node.get("state", {}).get("name", "Unknown"),
                    priority=node.get("priority", 0),
                    url=node["url"],
                    labels=[
                        label["name"]
                        for label in node.get("labels", {}).get("nodes", [])
                    ],
                    project=node.get("project", {}).get("name")
                    if node.get("project")
                    else None,
                )
                issues.append(issue)

            page_info = issues_data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")

        except requests.RequestException as e:
            logger.error(f"Linear API error: {e}")
            break

    # Filter out cancelled issues
    issues = [i for i in issues if i.state.lower() != "canceled"]
    return issues


# =============================================================================
# Pre-filtering (Fast Heuristics)
# =============================================================================


def prefilter_score_issue(
    pr: PRContext, issue: LinearIssue, author_team: Optional[str] = None
) -> tuple[float, list[str]]:
    """Score an issue using fast heuristics (no LLM). Returns (score, reasons)."""
    score = 0.0
    reasons = []
    pr_text = f"{pr.title} {pr.body} {pr.branch}"
    pr_text_upper = pr_text.upper()
    pr_text_lower = pr_text.lower()

    # 1. Direct issue reference (immediate 100% match)
    if issue.identifier.upper() in pr_text_upper:
        return (1.0, [f"Direct reference to {issue.identifier}"])

    # 2. Category overlap (rebalanced: +0.15 for 1 cat, +0.25 for 2+)
    pr_categories = set(categorize_by_keywords(pr.title, pr.body))
    issue_categories = set(categorize_by_keywords(issue.title, issue.description))
    common_categories = pr_categories & issue_categories
    if common_categories and "Other" not in common_categories:
        if len(common_categories) >= 2:
            score += 0.25
        else:
            score += 0.15
        reasons.append(f"Category: {', '.join(list(common_categories)[:2])}")

    # 3. Title keyword overlap (rebalanced: +0.08/word, max 0.35)
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "feat",
        "fix",
        "add",
        "update",
        "support",
        "when",
        "use",
        "using",
        "into",
        "not",
        "are",
    }
    pr_words = set(
        w.lower()
        for w in re.findall(r"\b\w{3,}\b", pr.title)
        if w.lower() not in stop_words
    )
    issue_words = set(
        w.lower()
        for w in re.findall(r"\b\w{3,}\b", issue.title)
        if w.lower() not in stop_words
    )
    common_words = pr_words & issue_words
    if common_words:
        score += min(len(common_words) * 0.08, 0.35)
        reasons.append(f"Keywords: {', '.join(list(common_words)[:3])}")

    # 4. Description keyword overlap (rebalanced: +0.03/word, max 0.20)
    pr_desc_words = set(
        w.lower()
        for w in re.findall(r"\b\w{4,}\b", pr.body or "")
        if w.lower() not in stop_words
    )
    issue_desc_words = set(
        w.lower()
        for w in re.findall(r"\b\w{4,}\b", issue.description or "")
        if w.lower() not in stop_words
    )
    desc_overlap = (pr_words & issue_desc_words) | (issue_words & pr_desc_words)
    desc_overlap = desc_overlap - common_words  # Don't double count
    if desc_overlap:
        score += min(len(desc_overlap) * 0.03, 0.20)
        if len(desc_overlap) >= 2:
            reasons.append(f"Desc: {', '.join(list(desc_overlap)[:2])}")

    # 5. Component match from files
    if pr.files_changed:
        pr_components = infer_components_from_files(pr.files_changed)
        issue_text_lower = (issue.title + " " + (issue.description or "")).lower()
        for comp in pr_components:
            if comp.lower() in issue_text_lower:
                score += 0.15
                reasons.append(f"Component: {comp}")
                break

    # 6. Team compatibility scoring
    issue_team = issue.identifier.split("-")[0].upper()
    core_teams = {"DYN", "DIS", "DEP", "LLM"}
    if author_team:
        if issue_team == "OPS" and author_team.upper() in core_teams:
            score -= 0.2
            reasons.append("Team mismatch: core author, OPS issue")
        elif issue_team == author_team.upper():
            score += 0.1
            reasons.append(f"Team match: {issue_team}")

    # 7. Project alignment scoring
    if issue.project:
        project_words = [w for w in issue.project.lower().split() if len(w) > 3]
        if any(w in pr_text_lower for w in project_words):
            score += 0.08
            reasons.append(f"Project: {issue.project}")

    # Return score without upper clamp to avoid saturation
    return (max(0.0, score), reasons)


def prefilter_issues(
    pr: PRContext,
    issues: list[LinearIssue],
    max_candidates: int = MAX_CANDIDATES_FOR_SCREENING,
    min_score: float = 0.10,
    author_team: Optional[str] = None,
) -> list[tuple[LinearIssue, float, list[str]]]:
    """Pre-filter issues using fast heuristics. Returns sorted (issue, score, reasons)."""
    scored = []
    for issue in issues:
        score, reasons = prefilter_score_issue(pr, issue, author_team=author_team)
        if score >= min_score:
            scored.append((issue, score, reasons))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_candidates]


# =============================================================================
# LLM Matching (Two-Model Approach)
# =============================================================================


def call_llm(
    messages: list[dict],
    api_key: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """Call NVIDIA Inference Hub for LLM completion."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        response = requests.post(
            NVIDIA_INFERENCE_URL, json=payload, headers=headers, timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logger.error(f"LLM API error: {e}")
        raise


def call_llm_with_retry(
    messages: list[dict],
    api_key: str,
    model: str,
    max_retries: int = 3,
    **kwargs,
) -> str:
    """Call LLM with exponential backoff retry logic for transient failures."""
    for attempt in range(max_retries):
        try:
            return call_llm(messages, api_key, model, **kwargs)
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2**attempt
                logger.warning(f"Retry {attempt + 1}/{max_retries} in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def screen_issue_fast(
    pr: PRContext,
    issue: LinearIssue,
    api_key: str,
    prefilter_reasons: list[str],
) -> tuple[float, str]:
    """
    Fast screening with cheaper model. Returns (score 0-1, brief reason).

    This is a quick yes/no check to filter out obvious non-matches.
    """
    prompt = f"""Quick match check: Is this PR related to this Linear issue?

PR: {pr.title}
Branch: {pr.branch}
Issue: {issue.identifier} - {issue.title}
Signals: {', '.join(prefilter_reasons) if prefilter_reasons else 'weak'}

Reply ONLY with JSON: {{"score": 0.0-1.0, "reason": "10 words max"}}
- 0.7+: Likely related (same feature/bug)
- 0.3-0.7: Maybe related (similar area)
- <0.3: Not related"""

    try:
        response = call_llm_with_retry(
            [{"role": "user", "content": prompt}],
            api_key,
            FAST_MODEL,
            max_tokens=100,
        )
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1].replace("json", "").strip()
        result = json.loads(response)
        return (float(result.get("score", 0.0)), result.get("reason", ""))
    except (json.JSONDecodeError, KeyError, ValueError, requests.RequestException) as e:
        logger.warning(f"Fast screening failed for {issue.identifier}: {e}")
        return (0.0, "screening failed")


def match_issue_final(
    pr: PRContext,
    issue: LinearIssue,
    api_key: str,
    prefilter_reasons: list[str],
    screening_score: float,
) -> tuple[float, str]:
    """
    Final matching with better model. Returns (confidence 0-1, detailed reason).
    """
    # Direct reference is always 100%
    if issue.identifier.upper() in f"{pr.title} {pr.body} {pr.branch}".upper():
        return (1.0, f"PR directly references {issue.identifier}")

    components = (
        infer_components_from_files(pr.files_changed) if pr.files_changed else set()
    )
    component_str = ", ".join(components) if components else "Unknown"

    prompt = f"""Analyze if this GitHub PR implements/fixes this Linear issue.

## PR #{pr.number}
Title: {pr.title}
Branch: {pr.branch}
Components: {component_str}
Description: {pr.body[:600] if pr.body else 'None'}

## Issue {issue.identifier}
Title: {issue.title}
State: {issue.state} | Priority: P{issue.priority}
Description: {issue.description[:600] if issue.description else 'None'}

## Context
Pre-filter signals: {', '.join(prefilter_reasons) if prefilter_reasons else 'None'}
Screening score: {screening_score:.0%}

Reply ONLY with JSON:
{{"confidence": 0.0-1.0, "reasoning": "1-2 sentences explaining match/no-match"}}

Confidence guide:
- 0.85+: Clear match - same feature, bug fix, or component
- 0.70-0.85: Strong match - aligned work
- 0.50-0.70: Possible match - related area
- <0.50: Not a match"""

    try:
        response = call_llm_with_retry(
            [
                {
                    "role": "system",
                    "content": "You are a precise bug tracking analyst. Output only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            api_key,
            FINAL_MODEL,
            max_tokens=256,
        )
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1].replace("json", "").strip()
        result = json.loads(response)
        return (
            float(result.get("confidence", 0.0)),
            result.get("reasoning", "No reasoning"),
        )
    except (json.JSONDecodeError, KeyError, ValueError, requests.RequestException) as e:
        logger.warning(f"Final matching failed for {issue.identifier}: {e}")
        return (0.0, f"matching failed: {e}")


# =============================================================================
# Main Matching Pipeline
# =============================================================================


def find_related_issues(
    pr: PRContext,
    issues: list[LinearIssue],
    nvidia_api_key: str,
    author_team: Optional[str] = None,
) -> list[MatchResult]:
    """
    Find Linear issues related to a PR using three-stage matching:
    1. Direct reference check (no LLM)
    2. Pre-filter + fast model screening
    3. Final model matching on top candidates
    """
    results = []

    # Stage 1: Direct reference check (highest priority, no LLM needed)
    pr_text = f"{pr.title} {pr.body} {pr.branch}"
    mentioned_ids = set(extract_linear_issue_ids(pr_text))

    for issue in issues:
        if issue.identifier.upper() in mentioned_ids:
            results.append(
                MatchResult(
                    issue=issue,
                    confidence=1.0,
                    reasoning=f"PR directly references {issue.identifier}",
                    screening_score=1.0,
                )
            )
            logger.info(f"✓ Direct reference: {issue.identifier}")

    if results:
        return results  # Direct matches found, skip LLM

    # Stage 2: Pre-filter with heuristics
    prefiltered = prefilter_issues(
        pr, issues, max_candidates=MAX_CANDIDATES_FOR_SCREENING, author_team=author_team
    )

    if not prefiltered:
        logger.info("No candidates passed pre-filtering")
        return []

    logger.info(f"Pre-filtered to {len(prefiltered)} candidates")

    # Stage 3: Fast model screening
    screened = []
    for issue, pf_score, pf_reasons in prefiltered:
        logger.info(f"  Screening {issue.identifier} (heuristic: {pf_score:.2f})")
        screen_score, screen_reason = screen_issue_fast(
            pr, issue, nvidia_api_key, pf_reasons
        )

        if screen_score >= SCREENING_THRESHOLD:
            screened.append((issue, screen_score, pf_reasons))
            logger.info(f"    → Passed screening: {screen_score:.0%} - {screen_reason}")
        else:
            logger.info(f"    → Filtered out: {screen_score:.0%}")

    if not screened:
        logger.info("No candidates passed fast screening")
        return []

    # Take top candidates for final matching
    screened = screened[:MAX_CANDIDATES_FOR_FINAL]
    logger.info(f"Proceeding to final matching with {len(screened)} candidate(s)")

    # Stage 4: Final model matching
    for issue, screen_score, pf_reasons in screened:
        logger.info(f"  Final matching {issue.identifier}")
        confidence, reasoning = match_issue_final(
            pr, issue, nvidia_api_key, pf_reasons, screen_score
        )

        if confidence >= CONFIDENCE_THRESHOLD:
            results.append(
                MatchResult(
                    issue=issue,
                    confidence=confidence,
                    reasoning=reasoning,
                    screening_score=screen_score,
                )
            )
            logger.info(f"    ✓ Match: {confidence:.0%} - {reasoning[:50]}...")
        else:
            logger.info(f"    ✗ Below threshold: {confidence:.0%}")

    results.sort(key=lambda x: x.confidence, reverse=True)
    return results


# =============================================================================
# GitHub Comment
# =============================================================================


def format_comment(pr: PRContext, matches: list[MatchResult]) -> str:
    """Format the PR comment with matched issues and linking instructions."""
    if not matches:
        return ""

    # Build the table
    rows = []
    for match in matches[:5]:
        conf_pct = f"{match.confidence:.0%}"
        issue_link = f"[{match.issue.identifier}]({match.issue.url})"
        title = (
            match.issue.title[:50] + "..."
            if len(match.issue.title) > 50
            else match.issue.title
        )
        reason = (
            match.reasoning[:60] + "..."
            if len(match.reasoning) > 60
            else match.reasoning
        )
        rows.append(f"| {issue_link} | {title} | {conf_pct} | {reason} |")

    table = "\n".join(rows)
    top_issue = matches[0].issue.identifier

    comment = f"""<details>
<summary>🔗 Hey! I found {len(matches)} Linear issue(s) that might be related to this PR</summary>

### Suggested Matches

| Issue | Title | Confidence | Reason |
|-------|-------|------------|--------|
{table}

---

### 📎 How to Link This PR to Linear

If one of these issues is correct, here's how to connect them:

**Automatic Methods** (Linear will detect the link)

- **PR Title** — Include the issue ID in your PR title
  ```
  fix: resolve memory leak [{top_issue}]
  ```
- **PR Description** — Mention the issue ID anywhere in the PR body
  ```
  This fixes {top_issue}
  ```
- **Magic Words** — Use keywords + issue ID in commits or PR description
  ```
  Fixes {top_issue}
  Closes {top_issue}
  Resolves {top_issue}
  ```

> 💡 **Tip:** Magic words (`Fixes`, `Closes`, `Resolves`) not only link the PR but can also auto-close the Linear issue when the PR merges!

**Manual Methods**

- **Add Link in Linear** — Open the Linear issue → Click "+" or use `⌘+K` → Add the PR URL as an attachment
- **Paste PR URL** — Paste the GitHub PR URL directly into the issue description or a comment — Linear auto-detects and creates a rich link

---

<sub>🤖 *Matched by [Linear Issue Matcher](.github/workflows/linear-issue-matcher.yml) • Confidence threshold: {CONFIDENCE_THRESHOLD:.0%} • Only shown for Dynamo team members*</sub>

</details>"""

    return comment


def post_pr_comment(
    repo: str, pr_number: int, comment_body: str, github_token: str
) -> bool:
    """Post a comment on the PR."""
    url = f"{GITHUB_API_URL}/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        response = requests.post(
            url, json={"body": comment_body}, headers=headers, timeout=30
        )
        response.raise_for_status()
        logger.info(f"Posted comment to PR #{pr_number}")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to post comment: {e}")
        return False


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for the GitHub Action."""
    import argparse

    parser = argparse.ArgumentParser(description="Linear Issue Matcher")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't post comment, just show what would be posted",
    )
    args = parser.parse_args()

    # Get environment variables
    linear_api_key = os.environ.get("LINEAR_API_KEY")
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPOSITORY", "ai-dynamo/dynamo")
    github_workspace = os.environ.get("GITHUB_WORKSPACE", ".")

    # Validate required environment variables
    if not linear_api_key:
        logger.error("LINEAR_API_KEY not set")
        sys.exit(1)
    if not nvidia_api_key:
        logger.error("NVIDIA_API_KEY not set")
        sys.exit(1)
    if not github_token:
        logger.error("GITHUB_TOKEN not set")
        sys.exit(1)

    # Extract PR context
    pr = PRContext.from_env()
    if not pr.number:
        logger.error("PR_NUMBER not set or invalid")
        sys.exit(1)

    logger.info(f"Processing PR #{pr.number}: {pr.title}")
    logger.info(f"Author: {pr.author}, Branch: {pr.branch}")

    # Load team members and check if author is a team member
    team_members = load_team_members(github_workspace)
    member = get_team_member(pr.author, team_members)

    if not member:
        logger.info(f"Skipping - {pr.author} is not a Dynamo team member")
        print("::notice::Skipping - PR author is not a Dynamo team member")
        sys.exit(0)

    logger.info(f"✓ {pr.author} is a Dynamo team member")

    # Fetch Linear issues - prioritize author's assigned issues
    issues = []
    if member.linear_id:
        logger.info(
            f"Fetching issues assigned to {pr.author} (Linear ID: {member.linear_id[:8]}...)"
        )
        issues = fetch_linear_issues_for_assignee(linear_api_key, member.linear_id)
        logger.info(f"Found {len(issues)} issues assigned to author")

    # Fall back to multi-team search if no author issues or author has no Linear ID
    if not issues:
        logger.info(f"Fetching issues for all teams: {LINEAR_TEAMS} (fallback)")
        issues = fetch_linear_issues_for_teams(linear_api_key, teams=LINEAR_TEAMS)
        logger.info(f"Found {len(issues)} issues across all teams")

    if not issues:
        logger.warning("No Linear issues found")
        print("::notice::No Linear issues found")
        sys.exit(0)

    # Infer author's team from their assigned issues (first issue's team prefix)
    author_team = None
    if member.linear_id:
        author_issues = [i for i in issues if i.assignee]
        if author_issues:
            author_team = author_issues[0].identifier.split("-")[0].upper()
            logger.info(f"Inferred author team: {author_team}")

    # Find related issues
    logger.info("Finding related issues...")
    matches = find_related_issues(pr, issues, nvidia_api_key, author_team=author_team)

    if not matches:
        logger.info(f"No matches found above {CONFIDENCE_THRESHOLD:.0%} threshold")
        print(f"::notice::No matches found above {CONFIDENCE_THRESHOLD:.0%} threshold")
        sys.exit(0)

    logger.info(f"Found {len(matches)} match(es) above threshold")

    # Post comment (or show preview in dry-run mode)
    comment = format_comment(pr, matches)
    if comment:
        if args.dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN - Would post this comment:")
            print("=" * 60)
            print(comment)
            print("=" * 60)
            print(
                f"::notice::Dry run - would post comment with {len(matches)} match(es)"
            )
        else:
            success = post_pr_comment(github_repo, pr.number, comment, github_token)
            if success:
                print(f"::notice::Posted comment with {len(matches)} related issue(s)")
            else:
                print("::error::Failed to post comment")
                sys.exit(1)


if __name__ == "__main__":
    main()
