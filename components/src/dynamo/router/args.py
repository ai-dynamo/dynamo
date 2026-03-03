# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router CLI parsing and config assembly."""

import argparse

from dynamo.llm import KvRouterConfig

from .backend_args import DynamoRouterArgGroup, DynamoRouterConfig


def build_kv_router_config(router_config: DynamoRouterConfig) -> KvRouterConfig:
    """Build KvRouterConfig from DynamoRouterConfig.

    Maps CLI/config attribute names to KvRouterConfig constructor kwargs.
    The only name difference is router_kv_overlap_score_weight -> overlap_score_weight.
    """
    return KvRouterConfig(
        overlap_score_weight=router_config.router_kv_overlap_score_weight,
        router_temperature=router_config.router_temperature,
        use_kv_events=router_config.router_use_kv_events,
        durable_kv_events=router_config.router_durable_kv_events,
        router_replica_sync=router_config.router_replica_sync,
        router_track_active_blocks=router_config.router_track_active_blocks,
        router_track_output_blocks=router_config.router_track_output_blocks,
        router_assume_kv_reuse=router_config.router_assume_kv_reuse,
        router_snapshot_threshold=router_config.router_snapshot_threshold,
        router_reset_states=router_config.router_reset_states,
        router_ttl_secs=router_config.router_ttl_secs,
        router_max_tree_size=router_config.router_max_tree_size,
        router_prune_target_ratio=router_config.router_prune_target_ratio,
        router_event_threads=router_config.router_event_threads,
    )


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
