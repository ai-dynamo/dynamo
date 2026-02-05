"""
AI-powered error classification system for CI/CD workflows.
"""

from .config import Config, get_config, ERROR_CATEGORIES
from .classifier import ErrorClassifier, ErrorClassification
from .error_extractor import ErrorExtractor, ErrorContext
from .deduplicator import ErrorDeduplicator
from .claude_client import ClaudeClient, ClassificationResult
from .opensearch_schema import (
    ERROR_CLASSIFICATIONS_INDEX_MAPPING,
    create_index_if_not_exists,
    get_index_mapping,
)
from .prompts import get_system_prompt, get_category_definitions
from .github_annotator import (
    GitHubAnnotator,
    AnnotationConfig,
    create_annotations_for_classifications,
    CATEGORY_SEVERITY,
)

__all__ = [
    # Config
    "Config",
    "get_config",
    "ERROR_CATEGORIES",
    # Classifier
    "ErrorClassifier",
    "ErrorClassification",
    # Extractor
    "ErrorExtractor",
    "ErrorContext",
    # Deduplicator
    "ErrorDeduplicator",
    # Claude client
    "ClaudeClient",
    "ClassificationResult",
    # OpenSearch
    "ERROR_CLASSIFICATIONS_INDEX_MAPPING",
    "create_index_if_not_exists",
    "get_index_mapping",
    # Prompts
    "get_system_prompt",
    "get_category_definitions",
    # GitHub annotations
    "GitHubAnnotator",
    "AnnotationConfig",
    "create_annotations_for_classifications",
    "CATEGORY_SEVERITY",
]

__version__ = "0.1.0"
