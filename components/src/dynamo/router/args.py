# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router CLI parsing, config, and assembly for the standalone router."""

import argparse
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.utils import add_argument
from dynamo.llm import KvRouterConfig


class DynamoRouterConfig(KvRouterConfigBase):
    """Typed configuration for the standalone KV router (router-owned options only).

    Examples:
        >>> from dynamo.router.args import parse_args, DynamoRouterConfig
        >>> config = parse_args([
        ...     "--endpoint", "dynamo.prefill.generate",
        ...     "--router-mode", "kv",
        ... ])
        >>> config.endpoint
        'dynamo.prefill.generate'
        >>> config.namespace
        'dynamo'
    """

    namespace: str
    endpoint: str
    router_block_size: int

    def validate(self) -> None:
        """Validate config invariants (aligned with Rust KvRouterConfig where applicable)."""
        if not self.endpoint:
            raise ValueError(
                "endpoint is required (set --endpoint or DYN_ROUTER_ENDPOINT)"
            )

        parts = self.endpoint.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid endpoint format: {self.endpoint!r}. "
                "Expected format: namespace.component.endpoint"
            )
        self.namespace = parts[0]


class DynamoRouterArgGroup(ArgGroup):
    """CLI argument group for standalone router options.

    Examples:
        >>> import argparse
        >>> from dynamo.router.args import DynamoRouterArgGroup, DynamoRouterConfig
        >>>
        >>> parser = argparse.ArgumentParser()
        >>> DynamoRouterArgGroup().add_arguments(parser)
        >>> args = parser.parse_args([
        ...     "--endpoint", "dynamo.prefill.generate",
        ...     "--router-mode", "kv",
        ... ])
        >>> config = DynamoRouterConfig.from_cli_args(args)
        >>> config.endpoint
        'dynamo.prefill.generate'
    """

    name = "dynamo-router"

    def add_arguments(self, parser) -> None:
        """Add router-owned arguments to parser."""
        g = parser.add_argument_group("Dynamo Router Options")

        add_argument(
            g,
            flag_name="--endpoint",
            env_var="DYN_ROUTER_ENDPOINT",
            default=None,
            help="Full endpoint path for workers in the format namespace.component.endpoint (e.g., dynamo.prefill.generate for prefill workers)",
            arg_type=str,
        )

        add_argument(
            g,
            flag_name="--router-block-size",
            env_var="DYN_ROUTER_BLOCK_SIZE",
            default=128,
            help="KV cache block size for routing decisions",
            arg_type=int,
            obsolete_flag="--block-size",
        )

        # KV router options (shared with dynamo.frontend)
        KvRouterArgGroup().add_arguments(parser)


def build_kv_router_config(router_config: DynamoRouterConfig) -> KvRouterConfig:
    """Build KvRouterConfig from DynamoRouterConfig.

    Examples:
        >>> from dynamo.router.args import parse_args, build_kv_router_config
        >>> router_config = parse_args([
        ...     "--endpoint", "dynamo.prefill.generate",
        ... ])
        >>> kv_config = build_kv_router_config(router_config)
    """
    return KvRouterConfig(**router_config.kv_router_kwargs())


def parse_args(argv: Optional[list[str]] = None) -> DynamoRouterConfig:
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
