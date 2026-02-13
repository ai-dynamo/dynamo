"""
Core error classification orchestration.
"""
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .config import Config
from .claude_client import ClaudeClient
from .deduplicator import ErrorDeduplicator
from .error_extractor import ErrorContext


@dataclass
class ErrorClassification:
    """Complete error classification with all metadata."""

    # Identity
    error_id: str
    error_hash: str

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

    def to_opensearch_doc(self) -> Dict[str, Any]:
        """Convert to OpenSearch document format with proper field prefixes."""
        doc = {
            "_id": self.error_id,
            "s_error_id": self.error_id,
            "s_error_hash": self.error_hash,

            # Source references
            "s_workflow_id": self.workflow_id,
            "s_job_id": self.job_id,
            "s_job_name": self.job_name,
            "s_step_id": self.step_id,
            "s_step_name": self.step_name,
            "s_test_name": self.test_name,

            # Error source
            "s_error_source": self.error_source,
            "s_framework": self.framework,

            # Common context
            "s_user_alias": self.user_alias,
            "s_repo": self.repo,
            "s_workflow_name": self.workflow_name,
            "s_branch": self.branch,
            "s_pr_id": self.pr_id,
            "s_commit_sha": self.commit_sha,

            # Error content
            "s_error_snippet": self.error_snippet,
            "s_error_full_text": self.error_full_text,

            # AI classification
            "s_primary_category": self.primary_category,
            "s_subcategory": self.subcategory,
            "f_confidence_score": self.confidence_score,
            "s_root_cause_summary": self.root_cause_summary,

            # Metadata
            "s_classification_method": self.classification_method,
            "s_model_version": self.model_version,
            "ts_classified_at": self.classified_at,
            "b_is_duplicate": self.is_duplicate,

            # Tracking
            "l_occurrence_count": self.occurrence_count,
            "ts_first_seen": self.first_seen,
            "ts_last_seen": self.last_seen,

            # API usage
            "l_prompt_tokens": self.prompt_tokens,
            "l_completion_tokens": self.completion_tokens,
            "l_cached_tokens": self.cached_tokens,

            "@timestamp": self.timestamp,
        }

        # Remove None values
        return {k: v for k, v in doc.items() if v is not None}


class ErrorClassifier:
    """Main error classification orchestrator."""

    def __init__(
        self,
        config: Config,
        opensearch_client: Any = None
    ):
        """
        Initialize classifier.

        Args:
            config: Configuration object
            opensearch_client: OpenSearch client for deduplication
        """
        self.config = config
        self.claude = ClaudeClient(config)
        self.deduplicator = ErrorDeduplicator(opensearch_client, config)
        self.opensearch_client = opensearch_client

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
            # Compute error hash
            error_text = error_data.get("log_excerpt", "")
            error_hash = self.deduplicator.compute_error_hash(error_text)

            classification = ErrorClassification(
                error_id=str(uuid.uuid4()),
                error_hash=error_hash,
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
