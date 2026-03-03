# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and CLI argument parsing for the example sample backend."""

import argparse

from dynamo.backend import DynamoRuntimeArgGroup, DynamoRuntimeConfig


class Config(DynamoRuntimeConfig):
    """Example backend configuration."""

    token_delay: float = 0.0


def parse_args() -> Config:
    """Parse command-line arguments for the example backend."""
    parser = argparse.ArgumentParser(description="Dynamo example sample worker")
    DynamoRuntimeArgGroup().add_arguments(parser)
    parser.add_argument(
        "--token-delay",
        type=float,
        default=0.0,
        help="Delay in seconds between each generated token (default: 0.0).",
    )
    args = parser.parse_args()
    config = Config.from_cli_args(args)
    config.validate()
    return config
