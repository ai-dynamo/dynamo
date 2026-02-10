"""
GitHub Annotations integration for error classifications.

Creates GitHub check run annotations to provide immediate developer feedback
in the PR UI.
"""
import os
import json
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


# Category to severity mapping
CATEGORY_SEVERITY = {
    # Critical - always needs immediate attention
    "infrastructure_error": ("failure", "🔴"),
    "compilation_error": ("failure", "🔴"),
    "dependency_error": ("failure", "🔴"),

    # Important - should be fixed but not blocking
    "configuration_error": ("warning", "🟠"),
    "resource_exhaustion": ("warning", "🟠"),
    "timeout": ("warning", "🟠"),

    # Informational - useful context
    "network_error": ("notice", "🟡"),
    "runtime_error": ("notice", "🟡"),
    "assertion_failure": ("notice", "🔵"),
    "flaky_test": ("notice", "🟣"),
}


@dataclass
class AnnotationConfig:
    """Configuration for GitHub annotations."""

    enabled: bool = True
    min_confidence: float = 0.7  # Only annotate high-confidence classifications
    max_annotations_per_run: int = 50  # GitHub limit
    include_error_text: bool = False  # Include full error in annotation

    @classmethod
    def from_env(cls) -> "AnnotationConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("ENABLE_GITHUB_ANNOTATIONS", "true").lower() == "true",
            min_confidence=float(os.getenv("ANNOTATION_MIN_CONFIDENCE", "0.7")),
            max_annotations_per_run=int(os.getenv("ANNOTATION_MAX_PER_RUN", "50")),
            include_error_text=os.getenv("ANNOTATION_INCLUDE_ERROR_TEXT", "false").lower() == "true",
        )


class GitHubAnnotator:
    """Creates GitHub annotations for error classifications."""

    def __init__(self, config: Optional[AnnotationConfig] = None):
        """
        Initialize GitHub annotator.

        Args:
            config: Annotation configuration (defaults to env-based config)
        """
        self.config = config or AnnotationConfig.from_env()

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
        """Check if GitHub annotations are available (running in GitHub Actions)."""
        return all([
            self.config.enabled,
            self.github_token,
            self.repo,
            self.sha,
        ])

    def should_annotate(self, classification: Any) -> bool:
        """
        Determine if a classification should create an annotation.

        Args:
            classification: ErrorClassification object

        Returns:
            True if annotation should be created
        """
        if not self.is_available():
            return False

        # Check confidence threshold
        if classification.confidence_score < self.config.min_confidence:
            return False

        return True

    def format_annotation_message(
        self,
        classification: Any,
        error_context: Optional[Any] = None
    ) -> str:
        """
        Format annotation message for GitHub UI.

        Args:
            classification: ErrorClassification object
            error_context: Optional ErrorContext object

        Returns:
            Formatted markdown message
        """
        # Get category display info
        level, icon = CATEGORY_SEVERITY.get(
            classification.primary_category,
            ("warning", "⚠️")
        )

        category_name = classification.primary_category.replace("_", " ").title()
        confidence_pct = int(classification.confidence_score * 100)

        # Build message
        lines = [
            f"**{icon} Error Classification: {category_name}**",
            "",
            f"**Confidence**: {confidence_pct}%",
            "",
            "**Root Cause**:",
            classification.root_cause_summary or "Unable to determine root cause",
            "",
        ]

        # Add error source info
        if classification.error_source:
            lines.append(f"**Error Source**: {classification.error_source}")

        if classification.test_name:
            lines.append(f"**Test**: `{classification.test_name}`")

        if classification.framework:
            lines.append(f"**Framework**: {classification.framework}")

        # Add error text if configured
        if self.config.include_error_text and classification.error_snippet:
            lines.extend([
                "",
                "**Error Text**:",
                "```",
                classification.error_snippet,
                "```",
            ])

        # Add footer
        lines.extend([
            "",
            "---",
            f"*Classified by AI Error Classification System*",
            f"*Hash: `{classification.error_hash[:8]}...`*",
        ])

        return "\n".join(lines)

    def get_annotation_level(self, category: str) -> str:
        """
        Get GitHub annotation level for category.

        Args:
            category: Error category

        Returns:
            Annotation level (failure, warning, or notice)
        """
        level, _ = CATEGORY_SEVERITY.get(category, ("warning", "⚠️"))
        return level

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

    def create_annotation(
        self,
        classification: Any,
        error_context: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single GitHub annotation for a classification.

        Args:
            classification: ErrorClassification object
            error_context: Optional ErrorContext object with file/line info

        Returns:
            Annotation dict if created, None otherwise
        """
        if not self.should_annotate(classification):
            return None

        # Determine file path for annotation
        # Default to workflow file if no specific file available
        file_path = ".github"
        start_line = 1
        end_line = 1

        # Try to get more specific location from error context
        if error_context:
            if hasattr(error_context, 'test_file') and error_context.test_file:
                file_path = error_context.test_file
            elif hasattr(error_context, 'metadata') and isinstance(error_context.metadata, dict):
                file_path = error_context.metadata.get('test_file', file_path)

        # Build annotation
        level = self.get_annotation_level(classification.primary_category)
        icon = self.get_category_icon(classification.primary_category)
        category_name = classification.primary_category.replace("_", " ").title()

        annotation = {
            "path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "annotation_level": level,
            "message": self.format_annotation_message(classification, error_context),
            "title": f"{icon} {category_name}",
        }

        return annotation

    def create_check_run_with_annotations(
        self,
        classifications: List[Any],
        error_contexts: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a GitHub check run with annotations for multiple classifications.

        Args:
            classifications: List of ErrorClassification objects
            error_contexts: Optional dict mapping error_id to ErrorContext

        Returns:
            True if successful
        """
        if not self.is_available():
            print("⚠️  GitHub annotations not available (missing context or disabled)")
            return False

        if not classifications:
            print("⚠️  No classifications to annotate")
            return False

        # Filter by confidence and deduplicate
        filtered = self._filter_and_deduplicate(classifications)

        if not filtered:
            print("⚠️  No high-confidence classifications to annotate")
            return False

        # Build annotations (max 50)
        annotations = []
        error_contexts = error_contexts or {}

        for classification in filtered[:self.config.max_annotations_per_run]:
            error_context = error_contexts.get(classification.error_id)
            annotation = self.create_annotation(classification, error_context)

            if annotation:
                annotations.append(annotation)

        if not annotations:
            print("⚠️  No annotations created after filtering")
            return False

        # Create check run via GitHub API
        try:
            url = f"https://api.github.com/repos/{self.repo}/check-runs"

            # Summary of classifications
            summary_lines = [
                f"Classified {len(filtered)} error(s) using AI:",
                "",
            ]

            # Group by category
            category_counts = {}
            for c in filtered:
                category_counts[c.primary_category] = category_counts.get(c.primary_category, 0) + 1

            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                icon = self.get_category_icon(category)
                category_name = category.replace("_", " ").title()
                summary_lines.append(f"- {icon} **{category_name}**: {count}")

            summary_lines.extend([
                "",
                "See annotations below for details.",
            ])

            payload = {
                "name": f"Error Classification - {self.job_name or 'Job'}",
                "head_sha": self.sha,
                "status": "completed",
                "conclusion": "neutral",  # Don't affect overall status
                "output": {
                    "title": "AI Error Classification Results",
                    "summary": "\n".join(summary_lines),
                    "annotations": annotations,
                }
            }

            response = requests.post(url, headers=self.headers, json=payload)

            if response.status_code == 201:
                print(f"✅ Created {len(annotations)} GitHub annotation(s)")
                return True
            else:
                print(f"⚠️  Failed to create check run: {response.status_code}")
                print(f"    Response: {response.text[:200]}")
                return False

        except Exception as e:
            print(f"⚠️  Error creating GitHub annotations: {e}")
            # Don't fail the workflow
            return False

    def _filter_and_deduplicate(self, classifications: List[Any]) -> List[Any]:
        """
        Filter classifications by confidence and deduplicate.

        Args:
            classifications: List of ErrorClassification objects

        Returns:
            Filtered and deduplicated list
        """
        # Filter by confidence
        high_confidence = [
            c for c in classifications
            if c.confidence_score >= self.config.min_confidence
        ]

        # Deduplicate by error hash (keep first occurrence)
        seen_hashes = set()
        unique = []

        for classification in high_confidence:
            if classification.error_hash not in seen_hashes:
                seen_hashes.add(classification.error_hash)
                unique.append(classification)
            else:
                # Increment occurrence count for existing classification
                for existing in unique:
                    if existing.error_hash == classification.error_hash:
                        if hasattr(existing, 'occurrence_count'):
                            existing.occurrence_count += 1
                        else:
                            existing.occurrence_count = 2
                        break

        return unique

    def create_annotation_for_single_error(
        self,
        classification: Any,
        error_context: Optional[Any] = None
    ) -> bool:
        """
        Create annotation for a single error (convenience method).

        Args:
            classification: ErrorClassification object
            error_context: Optional ErrorContext object

        Returns:
            True if successful
        """
        error_contexts = {}
        if error_context:
            error_contexts[classification.error_id] = error_context

        return self.create_check_run_with_annotations([classification], error_contexts)

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

        # Build markdown summary
        markdown = self._build_summary_markdown(classifications, error_contexts or {})

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
        # Try GITHUB_REF (refs/pull/123/merge)
        pr_ref = os.getenv("GITHUB_REF")
        if pr_ref and "pull" in pr_ref:
            parts = pr_ref.split("/")
            if len(parts) >= 3 and parts[1] == "pull":
                try:
                    return int(parts[2])
                except (ValueError, IndexError):
                    pass

        # Try GITHUB_EVENT_PATH
        event_path = os.getenv("GITHUB_EVENT_PATH")
        if event_path and os.path.exists(event_path):
            try:
                with open(event_path) as f:
                    event = json.load(f)
                    pr_number = event.get("pull_request", {}).get("number")
                    if pr_number:
                        return int(pr_number)
            except Exception:
                pass

        return None

    def _build_summary_markdown(
        self,
        classifications: List[Any],
        error_contexts: Dict[str, Any]
    ) -> str:
        """Build markdown summary table."""
        # Group by severity
        critical, important, informational = self._group_by_severity(classifications)

        # Count unique jobs
        unique_jobs = len(set(c.job_name for c in classifications if c.job_name))

        # Build markdown
        md = "## 🤖 AI Error Classification Summary\n\n"
        md += f"Found **{len(classifications)} unique error(s)** across **{unique_jobs} job(s)** in this workflow\n\n"

        # Critical errors table
        if critical:
            md += "### 🔴 Critical (Immediate attention needed)\n\n"
            md += "| Job | Step | Error Type | Confidence | Summary |\n"
            md += "|-----|------|------------|------------|---------|"
            for c in critical:
                job_name = self._truncate(c.job_name or "unknown", 30)
                step_name = self._truncate(c.step_name or "unknown", 25)
                error_type = c.primary_category.replace("_", " ").title()
                confidence = f"{int(c.confidence_score * 100)}%"
                summary = self._truncate(c.root_cause_summary or "", 60)

                md += f"\n| {job_name} | {step_name} | {error_type} | {confidence} | {summary} |"
            md += "\n\n"

        # Important errors table
        if important:
            md += "### 🟠 Important (Should be fixed)\n\n"
            md += "| Job | Step | Error Type | Confidence | Summary |\n"
            md += "|-----|------|------------|------------|---------|"
            for c in important:
                job_name = self._truncate(c.job_name or "unknown", 30)
                step_name = self._truncate(c.step_name or "unknown", 25)
                error_type = c.primary_category.replace("_", " ").title()
                confidence = f"{int(c.confidence_score * 100)}%"
                summary = self._truncate(c.root_cause_summary or "", 60)

                md += f"\n| {job_name} | {step_name} | {error_type} | {confidence} | {summary} |"
            md += "\n\n"

        # Informational errors table
        if informational:
            md += "### 🔵 Informational\n\n"
            md += "| Job | Step | Error Type | Confidence | Summary |\n"
            md += "|-----|------|------------|------------|---------|"
            for c in informational:
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
        """Group classifications by severity."""
        critical = [c for c in classifications if c.primary_category in [
            "infrastructure_error", "compilation_error", "dependency_error"
        ]]
        important = [c for c in classifications if c.primary_category in [
            "configuration_error", "resource_exhaustion", "timeout", "network_error"
        ]]
        informational = [c for c in classifications if c.primary_category in [
            "runtime_error", "assertion_failure", "flaky_test"
        ]]
        return critical, important, informational

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."


def create_annotations_for_classifications(
    classifications: List[Any],
    error_contexts: Optional[Dict[str, Any]] = None,
    config: Optional[AnnotationConfig] = None
) -> bool:
    """
    Convenience function to create annotations for classifications.

    Args:
        classifications: List of ErrorClassification objects
        error_contexts: Optional dict mapping error_id to ErrorContext
        config: Optional AnnotationConfig (defaults to env-based)

    Returns:
        True if successful
    """
    annotator = GitHubAnnotator(config)
    return annotator.create_check_run_with_annotations(classifications, error_contexts)
