# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and CLI argument parsing for the example sample backend."""

import argparse
from typing import Optional

from dynamo.backend import DynamoRuntimeArgGroup, DynamoRuntimeConfig


class ExampleBackendConfig:
    """Backend-specific configuration for the example backend.

    Accessed via ``config.backend.token_delay`` to clearly distinguish
    backend-specific fields from standard Dynamo runtime fields.
    """

    def __init__(self, token_delay: float = 0.0) -> None:
        self.token_delay = token_delay

    def __repr__(self) -> str:
        return f"ExampleBackendConfig(token_delay={self.token_delay!r})"


class Config(DynamoRuntimeConfig):
    """Example backend configuration.

    Extends DynamoRuntimeConfig with backend-specific fields that are
    not part of the shared runtime configuration.
    """

    # Backend-specific fields
    component: str = "backend"
    model: str = "example-model"
    served_model_name: Optional[str] = None
    use_kv_events: bool = False
    backend: Optional[ExampleBackendConfig] = None

    def get_model_name(self) -> str:
        """Get the effective model name for display and metrics."""
        return self.served_model_name or self.model


def parse_args() -> Config:
    """Parse command-line arguments for the example backend."""
    parser = argparse.ArgumentParser(description="Dynamo example sample worker")
    DynamoRuntimeArgGroup().add_arguments(parser)
    parser.add_argument(
        "--model",
        type=str,
        default="example-model",
        help="Model name (default: example-model).",
    )
    parser.add_argument(
        "--token-delay",
        type=float,
        default=0.0,
        help="Delay in seconds between each generated token (default: 0.0).",
    )
    args = parser.parse_args()
    config = Config.from_cli_args(args)
    config.backend = ExampleBackendConfig(
        token_delay=getattr(args, "token_delay", 0.0),
    )
    config.validate()
    return config
