"""
Core error classification orchestration.
"""
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .config import Config
from .claude_client import ClaudeClient, ClassificationResult
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
    step_id: Optional[str] = None
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
            "s_step_id": self.step_id,
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

    def classify_error(
        self,
        error_context: ErrorContext,
        use_cache: bool = True,
        classification_method: str = "batch"
    ) -> ErrorClassification:
        """
        Classify a single error with deduplication.

        Args:
            error_context: Error context with text and metadata
            use_cache: Whether to use prompt caching
            classification_method: "realtime" or "batch"

        Returns:
            ErrorClassification object
        """
        # Compute error hash for deduplication
        error_hash = self.deduplicator.compute_error_hash(error_context.error_text)

        # Check for existing classification
        existing = None
        if self.opensearch_client and self.config.error_classification_index:
            existing = self.deduplicator.find_similar_classification(
                error_hash,
                self.config.error_classification_index
            )

        # If found and confidence is high, reuse it
        if existing and existing.get('f_confidence_score', 0) >= self.config.min_confidence_for_reuse:
            print(f"  ✓ Reusing existing classification (confidence: {existing['f_confidence_score']:.2f})")

            # Increment occurrence count
            self.deduplicator.increment_occurrence_count(
                error_hash,
                self.config.error_classification_index
            )

            return ErrorClassification(
                error_id=str(uuid.uuid4()),
                error_hash=error_hash,
                workflow_id=error_context.workflow_id,
                job_id=error_context.job_id,
                step_id=error_context.step_id,
                test_name=error_context.test_name,
                error_source=error_context.source_type,
                framework=error_context.framework,
                user_alias=error_context.user_alias,
                repo=error_context.repo,
                workflow_name=error_context.workflow_name,
                branch=error_context.branch,
                pr_id=error_context.pr_id,
                commit_sha=error_context.commit_sha,
                error_snippet=error_context.error_text[:500],
                error_full_text=error_context.error_text[:self.config.max_error_length],
                primary_category=existing.get('s_primary_category'),
                subcategory=existing.get('s_subcategory'),
                confidence_score=existing.get('f_confidence_score'),
                root_cause_summary=existing.get('s_root_cause_summary'),
                classification_method=classification_method,
                model_version=existing.get('s_model_version'),
                classified_at=datetime.now(timezone.utc).isoformat(),
                is_duplicate=True,
                occurrence_count=existing.get('l_occurrence_count', 1) + 1,
                first_seen=existing.get('ts_first_seen'),
                last_seen=datetime.now(timezone.utc).isoformat(),
                cached_tokens=existing.get('l_cached_tokens', 0),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # No existing classification, call Claude
        print(f"  🤖 Classifying with Claude...")
        result = self.claude.classify_error(
            error_context.error_text,
            error_context.to_dict(),
            use_cache=use_cache
        )

        # Create classification object
        classification = ErrorClassification(
            error_id=str(uuid.uuid4()),
            error_hash=error_hash,
            workflow_id=error_context.workflow_id,
            job_id=error_context.job_id,
            step_id=error_context.step_id,
            test_name=error_context.test_name,
            error_source=error_context.source_type,
            framework=error_context.framework,
            user_alias=error_context.user_alias,
            repo=error_context.repo,
            workflow_name=error_context.workflow_name,
            branch=error_context.branch,
            pr_id=error_context.pr_id,
            commit_sha=error_context.commit_sha,
            error_snippet=error_context.error_text[:500],
            error_full_text=error_context.error_text[:self.config.max_error_length],
            primary_category=result.primary_category,
            confidence_score=result.confidence_score,
            root_cause_summary=result.root_cause_summary,
            classification_method=classification_method,
            model_version=result.model_version,
            classified_at=result.classified_at,
            is_duplicate=False,
            occurrence_count=1,
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat(),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cached_tokens=result.cached_tokens,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return classification

    def batch_classify_errors(
        self,
        error_contexts: List[ErrorContext],
        use_cache: bool = True
    ) -> List[ErrorClassification]:
        """
        Classify multiple errors with deduplication.

        Args:
            error_contexts: List of error contexts
            use_cache: Whether to use prompt caching

        Returns:
            List of ErrorClassification objects
        """
        if not error_contexts:
            return []

        print(f"📊 Batch classifying {len(error_contexts)} errors...")

        # Group by hash for deduplication
        error_groups = {}
        for error_context in error_contexts:
            error_hash = self.deduplicator.compute_error_hash(error_context.error_text)
            if error_hash not in error_groups:
                error_groups[error_hash] = []
            error_groups[error_hash].append(error_context)

        print(f"📊 {len(error_contexts)} total errors → {len(error_groups)} unique")

        # Classify one representative per group
        classifications = []

        for i, (error_hash, group) in enumerate(error_groups.items(), 1):
            print(f"  [{i}/{len(error_groups)}] Classifying error hash {error_hash[:8]}...")

            # Use first error as representative
            representative = group[0]

            # Classify
            classification = self.classify_error(
                representative,
                use_cache=use_cache,
                classification_method="batch"
            )

            # Set occurrence count based on group size
            classification.occurrence_count = len(group)

            classifications.append(classification)

        return classifications

    def should_classify_realtime(self, error_context: ErrorContext) -> bool:
        """
        Determine if error should be classified during CI.

        Criteria:
        - Infrastructure/build errors (high impact)
        - Job-blocking failures (imports, collection errors, setup failures)
        - NOT individual test assertion failures (defer to batch)

        Args:
            error_context: Error context

        Returns:
            True if should classify in real-time
        """
        threshold = self.config.classify_realtime_threshold

        if threshold == "all":
            return True

        if threshold == "none":
            return False

        # "infrastructure" threshold (default)
        # Classify infrastructure and build errors in real-time
        source_type = error_context.source_type

        if source_type in ["buildkit", "github_annotation", "infrastructure_error", "github_job_log", "step_failure"]:
            return True

        # For pytest errors, check if they're infrastructure issues
        if source_type == "pytest":
            error_text = error_context.error_text.lower()

            # Critical infrastructure errors that prevent tests from running
            infrastructure_indicators = [
                "importerror",
                "modulenotfounderror",
                "error collecting",
                "collection error",
                "error importing",
                "cannot import",
                "no module named",
                "setup failed",
                "fixture error",
                "fixture not found",
                "session fixture failed",
                "conftest",
            ]

            if any(indicator in error_text for indicator in infrastructure_indicators):
                return True

            # Regular test assertion failures - defer to batch
            if error_context.test_name:
                return False

        return False

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
