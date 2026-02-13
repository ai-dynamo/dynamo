"""
Configuration management for error classification system.
"""
import os
from dataclasses import dataclass
from typing import List

# Core error categories for classification
# Simplified to 2 categories for more consistent classification
ERROR_CATEGORIES = [
    "infrastructure_error",  # Network issues, runner problems, infrastructure failures
    "code_error",  # Errors from code: build, test, compilation, runtime failures
]


@dataclass
class Config:
    """Configuration for error classification system."""

    # Anthropic API settings
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    # API format settings
    api_format: str = "anthropic"  # "anthropic" or "openai"
    api_base_url: str = (
        None  # Custom base URL (e.g., NVIDIA: https://inference-api.nvidia.com/v1)
    )

    # Processing limits
    max_error_length: int = 10000  # Max chars to send to API
    batch_size: int = 10

    # Rate limiting
    max_rpm: int = 50  # Max requests per minute to Claude API

    # Cache settings
    classification_cache_ttl_hours: int = 168  # 1 week
    min_confidence_for_reuse: float = 0.8

    # Real-time classification settings
    enable_realtime_classification: bool = False
    classify_realtime_threshold: str = "infrastructure"  # infrastructure|all

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        return cls(
            anthropic_api_key=anthropic_api_key,
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
            api_format=os.getenv("API_FORMAT", "anthropic"),  # anthropic or openai
            api_base_url=os.getenv(
                "API_BASE_URL"
            ),  # e.g., https://inference-api.nvidia.com/v1
            max_error_length=int(os.getenv("MAX_ERROR_LENGTH", "10000")),
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            max_rpm=int(os.getenv("MAX_RPM", "50")),
            classification_cache_ttl_hours=int(
                os.getenv("CLASSIFICATION_CACHE_TTL_HOURS", "168")
            ),
            min_confidence_for_reuse=float(
                os.getenv("MIN_CONFIDENCE_FOR_REUSE", "0.8")
            ),
            enable_realtime_classification=os.getenv(
                "ENABLE_ERROR_CLASSIFICATION", ""
            ).lower()
            == "true",
            classify_realtime_threshold=os.getenv(
                "CLASSIFY_REALTIME_THRESHOLD", "infrastructure"
            ),
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.anthropic_api_key:
            errors.append("anthropic_api_key is required")

        if self.max_error_length <= 0:
            errors.append("max_error_length must be positive")

        if self.batch_size <= 0:
            errors.append("batch_size must be positive")

        if not (0.0 <= self.min_confidence_for_reuse <= 1.0):
            errors.append("min_confidence_for_reuse must be between 0 and 1")

        return errors


def get_config() -> Config:
    """Get validated configuration from environment."""
    config = Config.from_env()
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    return config
