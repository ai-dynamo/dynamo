# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo standalone router configuration ArgGroup."""

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.utils import add_argument


class DynamoRouterConfig(KvRouterConfigBase):
    """Typed configuration for the standalone KV router (router-owned options only)."""

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
    """CLI argument group for standalone router options."""

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
