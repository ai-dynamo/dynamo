# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo standalone router configuration ArgGroup."""


from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument


class DynamoRouterConfig(ConfigBase):
    """Typed configuration for the standalone KV router (router-owned options only)."""

    namespace: str
    endpoint: str
    router_block_size: int
    router_kv_overlap_score_weight: float
    router_temperature: float
    router_use_kv_events: bool
    router_replica_sync: bool
    router_snapshot_threshold: int
    router_reset_states: bool
    router_durable_kv_events: bool
    router_track_active_blocks: bool
    router_assume_kv_reuse: bool
    router_track_output_blocks: bool
    router_ttl_secs: float
    router_max_tree_size: int
    router_prune_target_ratio: float
    router_event_threads: int

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

        add_argument(
            g,
            flag_name="--router-kv-overlap-score-weight",
            env_var="DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT",
            default=1.0,
            help="KV Router: Weight for overlap score in worker selection. Higher values prioritize KV cache reuse",
            arg_type=float,
            obsolete_flag="--kv-overlap-score-weight",
        )

        add_argument(
            g,
            flag_name="--router-temperature",
            env_var="DYN_ROUTER_TEMPERATURE",
            default=0.0,
            help="KV Router: Temperature for worker sampling via softmax. Higher values promote more randomness, and 0 fallbacks to deterministic.",
            arg_type=float,
        )

        add_negatable_bool_argument(
            g,
            flag_name="--router-kv-events",
            env_var="DYN_ROUTER_USE_KV_EVENTS",
            default=True,
            help="KV Router: Enable KV events from workers. When disabled (--no-router-kv-events), the router predicts cache state based on routing decisions with TTL-based expiration and pruning, rather than receiving events from workers.",
            dest="router_use_kv_events",
            obsolete_flag="--kv-events",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--router-replica-sync",
            env_var="DYN_ROUTER_REPLICA_SYNC",
            default=False,
            help="KV Router: Enable replica synchronization across multiple router instances. When true, routers will publish and subscribe to events to maintain consistent state.",
        )

        add_argument(
            g,
            flag_name="--router-snapshot-threshold",
            env_var="DYN_ROUTER_SNAPSHOT_THRESHOLD",
            default=1000000,
            help="KV Router: Number of messages in stream before triggering a snapshot",
            arg_type=int,
        )

        add_negatable_bool_argument(
            g,
            flag_name="--router-reset-states",
            env_var="DYN_ROUTER_RESET_STATES",
            default=False,
            help="KV Router: Reset router state on startup, purging stream and object store. WARNING: Can affect existing router replicas.",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--router-durable-kv-events",
            env_var="DYN_ROUTER_DURABLE_KV_EVENTS",
            default=False,
            help="KV Router: Enable durable KV events using NATS JetStream instead of NATS Core. By default, the router uses the generic event plane (NATS Core or ZMQ) with local_indexer mode. Use this flag when you need durability and multi-replica consistency. Requires NATS with JetStream enabled.",
            obsolete_flag="--durable-kv-events",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--router-track-active-blocks",
            env_var="DYN_ROUTER_TRACK_ACTIVE_BLOCKS",
            default=True,
            help="KV Router: Track active blocks for load balancing. Use --no-router-track-active-blocks to disable",
            obsolete_flag="--track-active-blocks",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--router-assume-kv-reuse",
            env_var="DYN_ROUTER_ASSUME_KV_REUSE",
            default=True,
            help="KV Router: When tracking active blocks, assume KV cache reuse. Use --no-router-assume-kv-reuse to use random hashes, useful when KV cache reuse is not expected.",
            obsolete_flag="--assume-kv-reuse",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--router-track-output-blocks",
            env_var="DYN_ROUTER_TRACK_OUTPUT_BLOCKS",
            default=False,
            help="KV Router: Track output blocks during generation. When enabled, the router adds placeholder blocks as tokens are generated and applies fractional decay based on progress toward expected output sequence length (agent_hints.osl in nvext).",
            obsolete_flag="--track-output-blocks",
        )

        add_argument(
            g,
            flag_name="--router-ttl-secs",
            env_var="DYN_ROUTER_TTL_SECS",
            default=120.0,
            help="KV Router: TTL for blocks in seconds. Only used when --no-router-kv-events is set.  Controls how long cached blocks are considered valid without explicit events.",
            arg_type=float,
        )

        add_argument(
            g,
            flag_name="--router-max-tree-size",
            env_var="DYN_ROUTER_MAX_TREE_SIZE",
            default=2**20,
            help="KV Router: Maximum tree size before pruning. Only used when --no-router-kv-events is set.  When the indexer tree exceeds this size, pruning is triggered.",
            arg_type=int,
        )

        add_argument(
            g,
            flag_name="--router-prune-target-ratio",
            env_var="DYN_ROUTER_PRUNE_TARGET_RATIO",
            default=0.8,
            help="KV Router: Target size ratio after pruning (0.0-1.0). Only used when --no-router-kv-events is set. Determines how aggressively to prune the tree.",
            arg_type=float,
        )

        add_argument(
            g,
            flag_name="--router-event-threads",
            env_var="DYN_ROUTER_EVENT_THREADS",
            default=1,
            help="KV Router: Number of event processing threads. >1 uses concurrent radix tree and thread pool for higher throughput.",
            arg_type=int,
        )
