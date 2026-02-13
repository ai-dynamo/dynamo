"""
Error context data structure for classification.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ErrorContext:
    """Normalized error context for classification."""

    # Error content
    error_text: str
    source_type: str  # pytest|buildkit|rust_test|github_annotation|github_job_log|full_log_analysis

    # Source references
    workflow_id: Optional[str] = None
    job_id: Optional[str] = None
    step_id: Optional[str] = None
    test_name: Optional[str] = None

    # Framework context
    framework: Optional[str] = None  # vllm|sglang|trtllm|rust

    # Job context
    job_name: Optional[str] = None
    step_name: Optional[str] = None

    # Common metadata
    repo: Optional[str] = None
    workflow_name: Optional[str] = None
    branch: Optional[str] = None
    pr_id: Optional[str] = None
    commit_sha: Optional[str] = None
    user_alias: Optional[str] = None

    # Timestamps
    timestamp: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "error_text": self.error_text,
            "source_type": self.source_type,
            "workflow_id": self.workflow_id,
            "job_id": self.job_id,
            "step_id": self.step_id,
            "test_name": self.test_name,
            "framework": self.framework,
            "job_name": self.job_name,
            "step_name": self.step_name,
            "repo": self.repo,
            "workflow_name": self.workflow_name,
            "branch": self.branch,
            "pr_id": self.pr_id,
            "commit_sha": self.commit_sha,
            "user_alias": self.user_alias,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
