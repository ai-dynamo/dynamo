"""
AI-powered error classification system for CI/CD workflows.
"""

from .config import Config, get_config, ERROR_CATEGORIES
from .classifier import ErrorClassifier, ErrorClassification, ErrorContext
from .claude_client import ClaudeClient
from .prompts import get_category_definitions
from .pr_commentator import (
    PRCommentator,
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
    # Error context
    "ErrorContext",
    # Claude client
    "ClaudeClient",
    # Prompts
    "get_category_definitions",
    # PR comments
    "PRCommentator",
    "CATEGORY_SEVERITY",
]

__version__ = "0.1.0"
