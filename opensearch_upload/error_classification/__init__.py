"""
AI-powered error classification system for CI/CD workflows.
Minimal version for testing Claude API connection.
"""

# Only import what's needed for the simple test
from .config import Config, ERROR_CATEGORIES
from .claude_client import ClaudeClient, ClassificationResult

__all__ = [
    "Config",
    "ERROR_CATEGORIES",
    "ClaudeClient",
    "ClassificationResult",
]

__version__ = "0.1.0"
