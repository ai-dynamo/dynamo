"""
GitHub PR Comment integration for error classifications.

Creates PR comments with markdown summaries of workflow errors.
"""
import os
import json
import requests
from typing import Dict, Any, Optional, List


# Category to severity mapping
CATEGORY_SEVERITY = {
    # Infrastructure errors - platform/network issues that often block entire jobs
    "infrastructure_error": ("failure", "🔴"),

    # Code errors - build/test/runtime issues in the code itself
    "code_error": ("warning", "🟠"),
}


class PRCommentator:
    """Creates GitHub PR comments for error classifications."""

    def __init__(self, claude_client: Optional[Any] = None):
        """
        Initialize GitHub PR commenter.

        Args:
            claude_client: Optional ClaudeClient instance for generating formatted summaries
        """
        self.claude_client = claude_client

        # GitHub context from environment
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo = os.getenv("GITHUB_REPOSITORY")  # owner/repo
        self.sha = os.getenv("GITHUB_SHA")
        self.run_id = os.getenv("GITHUB_RUN_ID")
        self.job_name = os.getenv("GITHUB_JOB")

        # GitHub API headers
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "error-classification-system/1.0"
        }

        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"

    def is_available(self) -> bool:
        """Check if GitHub PR comments are available (running in GitHub Actions with token)."""
        return all([
            self.github_token,
            self.repo,
            self.sha,
        ])

    def get_category_icon(self, category: str) -> str:
        """
        Get icon for category.

        Args:
            category: Error category

        Returns:
            Icon emoji
        """
        _, icon = CATEGORY_SEVERITY.get(category, ("warning", "⚠️"))
        return icon

    def create_pr_comment(
        self,
        classifications: List[Any],
        error_contexts: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a PR comment with markdown summary of all errors.

        Args:
            classifications: List of ErrorClassification objects
            error_contexts: Optional dict mapping error_id to ErrorContext

        Returns:
            True if successful
        """
        if not classifications:
            print("⚠️  No classifications to comment on")
            return False

        # Get PR number
        pr_number = self._get_pr_number()
        if not pr_number:
            print("⚠️  Not a PR context, skipping PR comment")
            return False

        # Get workflow context
        workflow_name = os.getenv("GITHUB_WORKFLOW", "Unknown Workflow")
        run_id = os.getenv("WORKFLOW_RUN_ID") or os.getenv("GITHUB_RUN_ID", "unknown")
        repo = self.repo or "unknown/repo"
        run_url = f"https://github.com/{repo}/actions/runs/{run_id}"
        failed_jobs = len(set(c.job_name for c in classifications if c.job_name))

        # Build header manually
        markdown = f"## 🔴 Workflow Failed: {workflow_name}\n\n"
        markdown += f"**Workflow Run**: [#{run_id}]({run_url}) | "
        markdown += f"**Failed Jobs**: {failed_jobs}\n\n"

        # Use Claude to generate the rest (grouping + summary) if available
        if self.claude_client:
            try:
                print("  🤖 Generating formatted summary with Claude...")
                claude_summary = self.claude_client.generate_formatted_summary(
                    classifications=classifications,
                    workflow_name=workflow_name,
                    run_id=run_id,
                    run_url=run_url,
                    failed_jobs=failed_jobs
                )
                markdown += claude_summary
            except Exception as e:
                print(f"  ⚠️  Claude summary generation failed, falling back to basic format: {e}")
                # Fallback to old format if Claude fails
                markdown += self._build_summary_markdown_body(classifications, error_contexts or {})
        else:
            # No Claude client available, use basic format
            print("  ℹ️  No Claude client available, using basic format")
            markdown += self._build_summary_markdown_body(classifications, error_contexts or {})

        # Post comment via GitHub API
        try:
            url = f"https://api.github.com/repos/{self.repo}/issues/{pr_number}/comments"
            response = requests.post(
                url,
                headers=self.headers,
                json={"body": markdown}
            )

            if response.status_code == 201:
                print(f"✅ Created PR comment on PR #{pr_number}")
                return True
            else:
                print(f"⚠️  Failed to create PR comment: HTTP {response.status_code}")
                print(f"    Response: {response.text[:200]}")
                return False

        except Exception as e:
            print(f"⚠️  Error creating PR comment: {e}")
            return False

    def _get_pr_number(self) -> Optional[int]:
        """Extract PR number from environment."""
        print("🔍 Attempting to detect PR number...")

        # Try GITHUB_REF (refs/pull/123/merge or refs/heads/pull-request/123)
        pr_ref = os.getenv("GITHUB_REF")
        print(f"  GITHUB_REF: {pr_ref}")
        if pr_ref and "pull" in pr_ref:
            parts = pr_ref.split("/")

            # Standard PR format: refs/pull/123/merge
            if len(parts) >= 3 and parts[1] == "pull":
                try:
                    pr_num = int(parts[2])
                    print(f"  ✅ Found PR #{pr_num} from GITHUB_REF (standard format)")
                    return pr_num
                except (ValueError, IndexError):
                    pass

            # Branch name format: refs/heads/pull-request/123
            if len(parts) >= 4 and parts[2] == "pull-request":
                try:
                    pr_num = int(parts[3])
                    print(f"  ✅ Found PR #{pr_num} from GITHUB_REF (branch name format)")
                    return pr_num
                except (ValueError, IndexError):
                    pass

        # Try GITHUB_EVENT_PATH
        event_path = os.getenv("GITHUB_EVENT_PATH")
        print(f"  GITHUB_EVENT_PATH: {event_path}")
        if event_path and os.path.exists(event_path):
            try:
                with open(event_path) as f:
                    event = json.load(f)

                event_name = os.getenv("GITHUB_EVENT_NAME")
                print(f"  GITHUB_EVENT_NAME: {event_name}")

                # Direct PR event
                pr_number = event.get("pull_request", {}).get("number")
                if pr_number:
                    print(f"  ✅ Found PR #{pr_number} from pull_request event")
                    return int(pr_number)

                # workflow_run event - check if triggered by a PR
                workflow_run = event.get("workflow_run", {})
                if workflow_run:
                    print(f"  workflow_run event detected")
                    print(f"    head_branch: {workflow_run.get('head_branch')}")
                    print(f"    head_sha: {workflow_run.get('head_sha')}")

                    pull_requests = workflow_run.get("pull_requests", [])
                    print(f"    pull_requests array length: {len(pull_requests)}")

                    if pull_requests and len(pull_requests) > 0:
                        pr_num = int(pull_requests[0].get("number"))
                        print(f"  ✅ Found PR #{pr_num} from workflow_run.pull_requests")
                        return pr_num

                    # workflow_run event - use API to find PR by head_sha
                    head_sha = workflow_run.get("head_sha")
                    if head_sha and self.github_token:
                        print(f"  Trying API lookup for commit {head_sha[:8]}...")
                        pr_num = self._find_pr_by_commit(head_sha)
                        if pr_num:
                            print(f"  ✅ Found PR #{pr_num} from API")
                            return pr_num
                        else:
                            print(f"  ❌ No PR found via API for commit {head_sha[:8]}")
            except Exception as e:
                print(f"⚠️  Error reading event file: {e}")
                import traceback
                traceback.print_exc()

        print("  ❌ No PR number found")
        return None

    def _find_pr_by_commit(self, commit_sha: str) -> Optional[int]:
        """Find PR number associated with a commit using GitHub API."""
        try:
            url = f"https://api.github.com/repos/{self.repo}/commits/{commit_sha}/pulls"
            print(f"    API request: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            print(f"    API response status: {response.status_code}")

            if response.status_code == 200:
                pulls = response.json()
                print(f"    API returned {len(pulls)} PR(s)")
                if pulls and len(pulls) > 0:
                    pr_num = pulls[0].get("number")
                    print(f"    First PR: #{pr_num}")
                    return pr_num
            else:
                print(f"    API error: {response.text[:200]}")
        except Exception as e:
            print(f"⚠️  Error finding PR by commit: {e}")
            import traceback
            traceback.print_exc()

        return None

    def _build_summary_markdown_body(
        self,
        classifications: List[Any],
        error_contexts: Dict[str, Any]
    ) -> str:
        """Build markdown summary table (fallback format without Claude)."""
        # Group by category
        infrastructure, code_errors, _ = self._group_by_severity(classifications)

        # Count unique jobs
        unique_jobs = len(set(c.job_name for c in classifications if c.job_name))

        # Build markdown (without the header, which is added in create_pr_comment)
        md = f"Found **{len(classifications)} unique error(s)** across **{unique_jobs} job(s)** in this workflow\n\n"

        # Infrastructure errors table
        if infrastructure:
            md += "### 🔴 Infrastructure Errors (Platform/Network Issues)\n\n"
            md += "| Job | Step | Error Type | Confidence | Summary |\n"
            md += "|-----|------|------------|------------|---------|"
            for c in infrastructure:
                job_name = self._truncate(c.job_name or "unknown", 30)
                step_name = self._truncate(c.step_name or "unknown", 25)
                error_type = c.primary_category.replace("_", " ").title()
                confidence = f"{int(c.confidence_score * 100)}%"
                summary = self._truncate(c.root_cause_summary or "", 60)

                md += f"\n| {job_name} | {step_name} | {error_type} | {confidence} | {summary} |"
            md += "\n\n"

        # Code errors table
        if code_errors:
            md += "### 🟠 Code Errors (Build/Test/Runtime Issues)\n\n"
            md += "| Job | Step | Error Type | Confidence | Summary |\n"
            md += "|-----|------|------------|------------|---------|"
            for c in code_errors:
                job_name = self._truncate(c.job_name or "unknown", 30)
                step_name = self._truncate(c.step_name or "unknown", 25)
                error_type = c.primary_category.replace("_", " ").title()
                confidence = f"{int(c.confidence_score * 100)}%"
                summary = self._truncate(c.root_cause_summary or "", 60)

                md += f"\n| {job_name} | {step_name} | {error_type} | {confidence} | {summary} |"
            md += "\n\n"

        # Statistics details
        md += "<details><summary>📊 Classification Statistics</summary>\n\n"
        md += f"**Total Errors:** {len(classifications)}\n\n"

        # Average confidence
        avg_confidence = sum(c.confidence_score for c in classifications) / len(classifications)
        md += f"**Average Confidence:** {avg_confidence:.1%}\n\n"

        # Breakdown by type
        category_counts = {}
        for c in classifications:
            cat = c.primary_category
            category_counts[cat] = category_counts.get(cat, 0) + 1

        md += "**Breakdown by Type:**\n"
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            icon = self.get_category_icon(cat)
            cat_name = cat.replace("_", " ").title()
            md += f"- {icon} {cat_name}: {count}\n"

        md += "\n</details>\n\n"
        md += "---\n"
        md += "*Generated by [AI Error Classification System](https://github.com/)*\n"

        return md

    def _group_by_severity(self, classifications: List[Any]) -> tuple:
        """Group classifications by category (infrastructure vs code errors)."""
        infrastructure = [c for c in classifications if c.primary_category == "infrastructure_error"]
        code_errors = [c for c in classifications if c.primary_category == "code_error"]
        return infrastructure, code_errors, []  # Return 3-tuple for backward compat (third is empty)

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
