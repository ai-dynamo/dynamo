"""Read GitHub Actions environment variables into a structured context object."""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GitHubActionsContext:
    """Context gathered from GitHub Actions environment variables.

    All fields have sensible defaults so the object can be constructed
    outside CI (e.g. in tests or dry-run mode) by passing explicit values.
    """

    repo: str = ""
    run_id: str = ""
    run_number: str = ""
    run_attempt: int = 1
    workflow_name: str = ""
    job_name: str = ""
    commit_sha: str = ""
    branch: str = ""
    actor: str = ""
    event_name: str = ""
    runner_name: str = ""
    runner_os: str = ""
    runner_arch: str = ""
    pr_number: Optional[str] = None

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "GitHubActionsContext":
        """Build context from current environment variables."""
        pr_number = cls._extract_pr_number()
        return cls(
            repo=os.getenv("GITHUB_REPOSITORY", ""),
            run_id=os.getenv("GITHUB_RUN_ID", ""),
            run_number=os.getenv("GITHUB_RUN_NUMBER", ""),
            run_attempt=int(os.getenv("GITHUB_RUN_ATTEMPT", "1")),
            workflow_name=os.getenv("GITHUB_WORKFLOW", ""),
            job_name=os.getenv("GITHUB_JOB", ""),
            commit_sha=os.getenv("GITHUB_SHA", ""),
            branch=os.getenv("GITHUB_REF_NAME", ""),
            actor=os.getenv("GITHUB_ACTOR", ""),
            event_name=os.getenv("GITHUB_EVENT_NAME", ""),
            runner_name=os.getenv("RUNNER_NAME", ""),
            runner_os=os.getenv("RUNNER_OS", ""),
            runner_arch=os.getenv("RUNNER_ARCH", ""),
            pr_number=pr_number,
        )

    # ── Conversion ───────────────────────────────────────────────────────

    def to_common_fields(self) -> Dict[str, Any]:
        """Return the ``s_*`` / ``l_*`` prefixed dict that the exporters expect."""
        return {
            "s_repo": self.repo,
            "s_run_id": self.run_id,
            "s_workflow_id": self.run_id,  # alias used by exporters
            "s_workflow_name": self.workflow_name,
            "s_job_name": self.job_name,
            "s_job_id": self.job_name,  # best available in CI context
            "s_commit_sha": self.commit_sha,
            "s_branch": self.branch,
            "s_user_alias": self.actor,
            "s_github_event": self.event_name,
            "s_pr_id": self.pr_number or "N/A",
            "s_runner_name": self.runner_name,
            "l_run_attempt": self.run_attempt,
            "l_retry_count": max(0, self.run_attempt - 1),
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_pr_number() -> Optional[str]:
        """Try to extract the PR number from the CI environment.

        Methods (in priority order):
        1. ``GITHUB_EVENT_PATH`` JSON → ``event.pull_request.number``
        2. ``GITHUB_REF`` pattern ``refs/pull/<N>/merge``
        """
        # Method 1: event payload JSON
        event_path = os.getenv("GITHUB_EVENT_PATH")
        if event_path:
            try:
                with open(event_path) as f:
                    data = json.load(f)
                pr_num = (data.get("pull_request") or data.get("number") or
                          (data.get("pull_request", {}) or {}).get("number"))
                if pr_num:
                    return str(pr_num)
            except Exception:
                pass

        # Method 2: ref pattern
        ref = os.getenv("GITHUB_REF", "")
        m = re.match(r"refs/pull/(\d+)/", ref)
        if m:
            return m.group(1)

        return None
