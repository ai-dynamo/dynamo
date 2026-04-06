# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared KV router configuration ArgGroup.

Defines the shared KvRouterConfig parameters once so that both
``dynamo.frontend`` and ``dynamo.router`` can reuse them without duplication.
Field names on ``KvRouterConfigBase`` match the ``KvRouterConfig`` Python
constructor kwargs 1:1, so ``kv_router_kwargs()`` returns a dict that can be
unpacked directly into ``KvRouterConfig(**config.kv_router_kwargs())``.
"""

from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument

# Authoritative field list — used by kv_router_kwargs() to extract values.
_KV_ROUTER_FIELDS: tuple[str, ...] = (
    "overlap_score_weight",
    "router_temperature",
    "use_kv_events",
    "durable_kv_events",
    "router_replica_sync",
    "router_track_active_blocks",
    "router_track_output_blocks",
    "router_assume_kv_reuse",
    "router_track_prefill_tokens",
    "router_prefill_load_model",
    "router_snapshot_threshold",
    "router_reset_states",
    "router_ttl_secs",
    "router_max_tree_size",
    "router_prune_target_ratio",
    "router_queue_threshold",
    "router_event_threads",
    "router_queue_policy",
    "remote_indexer_component",
)


class KvRouterConfigBase(ConfigBase):
    """Mixin carrying the shared KvRouterConfig fields."""

    overlap_score_weight: float
    router_temperature: float
    use_kv_events: bool
    durable_kv_events: bool
    router_replica_sync: bool
    router_track_active_blocks: bool
    router_track_output_blocks: bool
    router_assume_kv_reuse: bool
    router_track_prefill_tokens: bool
    router_prefill_load_model: str
    router_snapshot_threshold: int
    router_reset_states: bool
    router_ttl_secs: float
    router_max_tree_size: int
    router_prune_target_ratio: float
    router_queue_threshold: Optional[float]
    router_event_threads: int
    router_queue_policy: str
    remote_indexer_component: Optional[str]

    def kv_router_kwargs(self) -> dict:
        """Return a dict suitable for ``KvRouterConfig(**kwargs)``."""
        return {f: getattr(self, f) for f in _KV_ROUTER_FIELDS}


class KvRouterArgGroup(ArgGroup):
    """CLI arguments for the shared KvRouterConfig parameters."""

    def add_arguments(self, parser) -> None:
        g = parser.add_argument_group("KV Router Options")

        add_argument(
            g,
            flag_name="--router-kv-overlap-score-weight",
            env_var="DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT",
            default=1.0,
            help=(
                "KV Router: Weight for overlap score in worker selection. "
                "Higher values prioritize KV cache reuse."
            ),
            arg_type=float,
            dest="overlap_score_weight",
            obsolete_flag="--kv-overlap-score-weight",
        )
        add_argument(
            g,
            flag_name="--router-temperature",
            env_var="DYN_ROUTER_TEMPERATURE",
            default=0.0,
            help=(
                "KV Router: Temperature for worker sampling via softmax. Higher values "
                "promote more randomness, and 0 fallbacks to deterministic."
            ),
            arg_type=float,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-kv-events",
            env_var="DYN_ROUTER_USE_KV_EVENTS",
            default=True,
            help=(
                "KV Router: Enable/disable KV events. Use --router-kv-events to enable "
                "(default, router receives cache state events from workers) or --no-router-kv-events "
                "to disable (router predicts cache state based on routing decisions)."
            ),
            dest="use_kv_events",
            obsolete_flag="--kv-events",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-durable-kv-events",
            env_var="DYN_ROUTER_DURABLE_KV_EVENTS",
            default=False,
            help=(
                "[Deprecated] KV Router: Enable durable KV events using NATS JetStream. "
                "This option will be removed in a future release. The event-plane subscriber "
                "(local_indexer mode) is now the recommended path."
            ),
            dest="durable_kv_events",
            obsolete_flag="--durable-kv-events",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-replica-sync",
            env_var="DYN_ROUTER_REPLICA_SYNC",
            default=False,
            help=(
                "KV Router: Enable replica synchronization across multiple router instances. "
                "When true, routers will publish and subscribe to events to maintain "
                "consistent state."
            ),
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-track-active-blocks",
            env_var="DYN_ROUTER_TRACK_ACTIVE_BLOCKS",
            default=True,
            dest="router_track_active_blocks",
            help=(
                "KV Router: Track active blocks (blocks being used for ongoing generation). "
                "By default, active blocks are tracked for load balancing."
            ),
            obsolete_flag="--track-active-blocks",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-track-output-blocks",
            env_var="DYN_ROUTER_TRACK_OUTPUT_BLOCKS",
            default=False,
            dest="router_track_output_blocks",
            help=(
                "KV Router: Track output blocks during generation. When enabled, the router adds "
                "placeholder blocks as tokens are generated and applies fractional decay based on "
                "progress toward expected output sequence length."
            ),
            obsolete_flag="--track-output-blocks",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-assume-kv-reuse",
            env_var="DYN_ROUTER_ASSUME_KV_REUSE",
            default=True,
            dest="router_assume_kv_reuse",
            help=(
                "KV Router: When tracking active blocks, assume KV cache reuse. "
                "Use --no-router-assume-kv-reuse to generate random hashes instead "
                "(when KV cache reuse is not expected)."
            ),
            obsolete_flag="--assume-kv-reuse",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-track-prefill-tokens",
            env_var="DYN_ROUTER_TRACK_PREFILL_TOKENS",
            default=True,
            dest="router_track_prefill_tokens",
            help=(
                "KV Router: Include prompt-side prefill tokens in active load accounting. "
                "Use --no-router-track-prefill-tokens to ignore prompt tokens in router "
                "prefill-token load, queue pressure, and active_prefill_tokens metrics."
            ),
        )
        add_argument(
            g,
            flag_name="--router-prefill-load-model",
            env_var="DYN_ROUTER_PREFILL_LOAD_MODEL",
            default="none",
            choices=["none", "aic"],
            help=(
                "[EXPERIMENTAL] KV Router: Prompt-side prefill load model. "
                "'none' keeps static prompt load accounting. "
                "'aic' decays the oldest active prefill request using AIC-predicted duration."
            ),
        )
        add_argument(
            g,
            flag_name="--router-snapshot-threshold",
            env_var="DYN_ROUTER_SNAPSHOT_THRESHOLD",
            default=1000000,
            help="KV Router: Number of messages in stream before triggering a snapshot.",
            arg_type=int,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-reset-states",
            env_var="DYN_ROUTER_RESET_STATES",
            default=False,
            help=(
                "KV Router: Reset router state on startup, purging stream and object store. "
                "WARNING: This can affect existing router replicas."
            ),
        )
        add_argument(
            g,
            flag_name="--router-ttl-secs",
            env_var="DYN_ROUTER_TTL_SECS",
            default=120.0,
            help=(
                "KV Router: Time-to-live in seconds for blocks when KV events are disabled. "
                "Only used when --no-router-kv-events is set."
            ),
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--router-max-tree-size",
            env_var="DYN_ROUTER_MAX_TREE_SIZE",
            default=2**20,
            help=(
                "KV Router: Maximum tree size before pruning when KV events are disabled. "
                "Only used when --no-router-kv-events is set."
            ),
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--router-prune-target-ratio",
            env_var="DYN_ROUTER_PRUNE_TARGET_RATIO",
            default=0.8,
            help=(
                "KV Router: Target size ratio after pruning when KV events are disabled. "
                "Only used when --no-router-kv-events is set."
            ),
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--router-queue-threshold",
            env_var="DYN_ROUTER_QUEUE_THRESHOLD",
            default=4.0,
            help=(
                "KV Router: Queue threshold fraction for prefill token capacity. "
                "Requests are queued if all workers exceed this fraction of "
                "max_num_batched_tokens. Must be >= 0. Use 0.0 for maximum "
                "queueing sensitivity (queue as soon as any tokens are active)."
            ),
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--router-event-threads",
            env_var="DYN_ROUTER_EVENT_THREADS",
            default=4,
            help=(
                "KV Router: Number of event processing threads. When > 1, uses a concurrent "
                "radix tree with a thread pool for higher throughput. Ignored when "
                "--no-router-kv-events is set."
            ),
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--router-queue-policy",
            env_var="DYN_ROUTER_QUEUE_POLICY",
            default="fcfs",
            help=(
                "KV Router: Scheduling policy for the router queue. "
                "'fcfs' (default): first-come first-served with priority bumps — optimizes tail TTFT. "
                "'wspt': weighted shortest processing time (Smith's rule) — optimizes average TTFT."
            ),
            arg_type=str,
            choices=["fcfs", "wspt"],
        )
        add_argument(
            g,
            flag_name="--remote-indexer-component",
            env_var="DYN_REMOTE_INDEXER_COMPONENT",
            default=None,
            help=(
                "[EXPERIMENTAL] KV Router: Component name of a standalone KV indexer to use for overlap scoring. "
                "When set, the router queries the standalone indexer via the request plane instead "
                "of maintaining a local radix tree (e.g. 'kv-indexer')."
            ),
            arg_type=str,
        )
