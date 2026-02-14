# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration management for error classification system.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

# Core error categories for classification
# Simplified to 2 categories for more consistent classification
ERROR_CATEGORIES = [
    "infrastructure_error",  # Network issues, runner problems, infrastructure failures
    "code_error",  # Errors from code: build, test, compilation, runtime failures
]


@dataclass
class Config:
    """Configuration for error classification system."""

    # API key: prefer NVIDIA_INFERENCE_API_KEY (inference.nvidia.com), fallback ANTHROPIC_API_KEY
    api_key: str  # LLM API key (NVIDIA_INFERENCE_API_KEY or ANTHROPIC_API_KEY)
    # Model: prefer NVIDIA_INFERENCE_MODEL, fallback ANTHROPIC_MODEL
    model: str = (
        "aws/anthropic/bedrock-claude-opus-4-6"  # NVIDIA Inference API model id
    )

    # API format settings
    api_format: str = "openai"  # "anthropic" or "openai" (default openai for NVIDIA)
    api_base_url: Optional[
        str
    ] = None  # Custom base URL (e.g., NVIDIA: https://inference-api.nvidia.com/v1)

    # Rate limiting
    max_rpm: int = 50  # Max requests per minute to LLM API
    max_error_length: int = 10000  # Max chars per error to send to API

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables.
        Prefers NVIDIA_INFERENCE_API_KEY and NVIDIA_INFERENCE_MODEL for inference.nvidia.com.
        """
        api_key = os.getenv("NVIDIA_INFERENCE_API_KEY") or os.getenv(
            "ANTHROPIC_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "NVIDIA_INFERENCE_API_KEY or ANTHROPIC_API_KEY environment variable is required"
            )
        # Treat empty string as unset (GitHub Actions vars.X yields "" when not configured)
        model = (
            os.getenv("NVIDIA_INFERENCE_MODEL", "").strip()
            or os.getenv("ANTHROPIC_MODEL", "").strip()
            or "aws/anthropic/bedrock-claude-opus-4-6"  # default NVIDIA Inference API model
        )

        # Parse MAX_RPM with a clear error on invalid values
        max_rpm_raw = os.getenv("MAX_RPM", "").strip()
        try:
            max_rpm = int(max_rpm_raw) if max_rpm_raw else 50
        except ValueError:
            raise ValueError(f"MAX_RPM must be an integer, got: {max_rpm_raw!r}")

        return cls(
            api_key=api_key,
            model=model,
            api_format=os.getenv("API_FORMAT", "openai"),  # default openai for NVIDIA
            api_base_url=os.getenv(
                "API_BASE_URL", "https://inference-api.nvidia.com/v1"
            ),  # default NVIDIA
            max_rpm=max_rpm,
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.api_key:
            errors.append("api_key is required")

        return errors


def get_config() -> Config:
    """Get validated configuration from environment."""
    config = Config.from_env()
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    return config
