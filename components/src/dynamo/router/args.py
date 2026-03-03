# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router CLI parsing and config assembly."""

import argparse

from dynamo.llm import KvRouterConfig

from .backend_args import DynamoRouterArgGroup, DynamoRouterConfig


def build_kv_router_config(router_config: DynamoRouterConfig) -> KvRouterConfig:
    """Build KvRouterConfig from DynamoRouterConfig."""
    return KvRouterConfig(**router_config.kv_router_kwargs())


def parse_args(argv=None) -> DynamoRouterConfig:
    """Parse command-line arguments for the standalone router.

    Returns:
        DynamoRouterConfig: Parsed and validated configuration.
    """
    parser = argparse.ArgumentParser(
        description="Dynamo Standalone Router Service: Configurable KV-aware routing for any worker endpoint",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = DynamoRouterArgGroup()
    group.add_arguments(parser)

    args = parser.parse_args(argv)
    config = DynamoRouterConfig.from_cli_args(args)
    config.validate()
    return config
