"""
Core error classification orchestration.
"""
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .config import Config
from .claude_client import ClaudeClient
from .error_extractor import ErrorContext


@dataclass
class ErrorClassification:
    """Complete error classification with all metadata."""

    # Identity
    error_id: str

    # Source references
    workflow_id: Optional[str] = None
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    test_name: Optional[str] = None

    # Error source
    error_source: Optional[str] = None  # pytest|buildkit|rust_test|github_annotation
    framework: Optional[str] = None

    # Common context
    user_alias: Optional[str] = None
    repo: Optional[str] = None
    workflow_name: Optional[str] = None
    branch: Optional[str] = None
    pr_id: Optional[str] = None
    commit_sha: Optional[str] = None

    # Error content
    error_snippet: Optional[str] = None  # First 500 chars
    error_full_text: Optional[str] = None

    # AI classification results
    primary_category: Optional[str] = None
    subcategory: Optional[str] = None
    confidence_score: Optional[float] = None
    root_cause_summary: Optional[str] = None

    # Metadata
    classification_method: str = "batch"  # realtime|batch
    model_version: Optional[str] = None
    classified_at: Optional[str] = None
    is_duplicate: bool = False

    # Tracking
    occurrence_count: int = 1
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None

    # API usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    # Timestamp
    timestamp: Optional[str] = None


class ErrorClassifier:
    """Main error classification orchestrator."""

    def __init__(self, config: Config):
        """
        Initialize classifier.

        Args:
            config: Configuration object
        """
        self.config = config
        self.claude = ClaudeClient(config)

    def classify_job_from_full_log(
        self,
        job_log: str,
        job_name: str,
        job_id: str,
        workflow_context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[ErrorClassification]:
        """
        Analyze full job log and classify all errors found.

        Args:
            job_log: Complete raw log from GitHub job
            job_name: Name of the job
            job_id: GitHub job ID
            workflow_context: Optional workflow context (repo, branch, etc.)
            use_cache: Whether to use prompt caching

        Returns:
            List of ErrorClassification objects for all errors found
        """
        # Call Claude to analyze full log
        result = self.claude.analyze_full_job_log(
            job_log=job_log,
            job_name=job_name,
            job_id=job_id,
            use_cache=use_cache
        )

        # Extract workflow context
        workflow_context = workflow_context or {}

        # Convert results to ErrorClassification objects
        classifications = []
        for error_data in result.get("errors_found", []):
            error_text = error_data.get("log_excerpt", "")

            classification = ErrorClassification(
                error_id=str(uuid.uuid4()),
                workflow_id=workflow_context.get("workflow_id"),
                job_id=job_id,
                job_name=job_name,
                step_name=error_data.get("step", "unknown"),
                step_id=None,
                test_name=None,
                error_source="full_log_analysis",
                framework=workflow_context.get("framework"),
                user_alias=workflow_context.get("user_alias"),
                repo=workflow_context.get("repo"),
                workflow_name=workflow_context.get("workflow_name"),
                branch=workflow_context.get("branch"),
                pr_id=workflow_context.get("pr_id"),
                commit_sha=workflow_context.get("commit_sha"),
                error_snippet=error_text[:500] if error_text else "",
                error_full_text=error_text[:self.config.max_error_length] if error_text else "",
                primary_category=error_data["primary_category"],
                confidence_score=error_data["confidence_score"],
                root_cause_summary=error_data["root_cause_summary"],
                classification_method="full_log_analysis",
                model_version=result.get("model_version"),
                classified_at=datetime.now(timezone.utc).isoformat(),
                is_duplicate=False,
                occurrence_count=1,
                first_seen=datetime.now(timezone.utc).isoformat(),
                last_seen=datetime.now(timezone.utc).isoformat(),
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
                cached_tokens=result.get("cached_tokens", 0),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            classifications.append(classification)

        return classifications
