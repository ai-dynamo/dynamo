# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router CLI parsing, config, and assembly for the standalone router."""

import argparse
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.aic_perf_args import (
    AicPerfArgGroup,
    AicPerfConfigBase,
)
from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument
from dynamo.llm import AicPerfConfig, KvRouterConfig


class DynamoRouterConfig(KvRouterConfigBase, AicPerfConfigBase):
    """Typed configuration for the standalone KV router (router-owned options only)."""

    namespace: str
    endpoint: str
    router_block_size: int
    serve_indexer: bool = False

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
        if self.serve_indexer and self.use_remote_indexer:
            raise ValueError(
                "--serve-indexer and --use-remote-indexer are mutually exclusive"
            )
        if self.conditional_prefill_max_new_tokens < 0:
            raise ValueError("--router-conditional-prefill-max-new-tokens must be >= 0")
        if self.router_prefill_load_model == "aic":
            missing = [
                flag
                for flag, value in (
                    ("--aic-backend", self.aic_backend),
                    ("--aic-system", self.aic_system),
                    ("--aic-model-path", self.aic_model_path),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "--router-prefill-load-model=aic requires " + ", ".join(missing)
                )
            if not self.router_track_prefill_tokens:
                raise ValueError(
                    "--router-prefill-load-model=aic requires "
                    "--router-track-prefill-tokens"
                )


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

        add_negatable_bool_argument(
            g,
            flag_name="--serve-indexer",
            env_var="DYN_SERVE_INDEXER",
            default=False,
            help="Serve this router's local KV indexer over the request plane.",
            dest="serve_indexer",
        )

        # KV router options (shared with dynamo.frontend)
        KvRouterArgGroup().add_arguments(parser)
        AicPerfArgGroup().add_arguments(parser)


def build_kv_router_config(router_config: DynamoRouterConfig) -> KvRouterConfig:
    """Build KvRouterConfig from DynamoRouterConfig."""
    return KvRouterConfig(**router_config.kv_router_kwargs())


def build_aic_perf_config(
    router_config: DynamoRouterConfig,
) -> AicPerfConfig | None:
    if router_config.router_prefill_load_model != "aic":
        return None
    return AicPerfConfig(**router_config.aic_perf_kwargs())


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
