"""
AI-powered error classification system for CI/CD workflows.
"""

from .classifier import ErrorClassification, ErrorClassifier, ErrorContext
from .claude_client import ClaudeClient
from .config import ERROR_CATEGORIES, Config, get_config
from .pr_commentator import CATEGORY_SEVERITY, PRCommentator
from .prompts import get_category_definitions

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
