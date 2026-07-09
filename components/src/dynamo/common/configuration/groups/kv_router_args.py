# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared KV router configuration ArgGroup.

Defines the shared KvRouterConfig parameters once so that both
``dynamo.frontend`` and ``dynamo.router`` can reuse them without duplication.
Field names on ``KvRouterConfigBase`` match the ``KvRouterConfig`` Python
constructor kwargs 1:1, so ``kv_router_kwargs()`` returns a dict that can be
unpacked directly into ``KvRouterConfig(**config.kv_router_kwargs())``.
"""

import argparse
import os
import warnings
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.router_valkey_config import (
    legacy_tokenizer_cache_config_json,
    validate_router_valkey_config,
)
from dynamo.common.configuration.utils import (
    add_argument,
    add_negatable_bool_argument,
    nullable_float,
)

# Authoritative field list — used by kv_router_kwargs() to extract values.
_KV_ROUTER_FIELDS: tuple[str, ...] = (
    "overlap_score_weight",
    "overlap_score_credit",
    "overlap_score_credit_decay",
    "prefill_load_scale",
    "host_cache_hit_weight",
    "disk_cache_hit_weight",
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
    "router_queue_threshold",
    "router_policy_config",
    "router_event_threads",
    "router_queue_policy",
    "use_remote_indexer",
    "serve_indexer",
    "valkey_urls",
    "valkey_index_scope",
    "valkey_connection_pool_size",
    "valkey_required_replica_acks",
    "valkey_sentinel_urls",
    "valkey_sentinel_master_name",
    "valkey_sentinel_quorum",
    "valkey_allow_insecure_plaintext",
    "valkey_allow_degraded_writes",
    "valkey_worker_events",
    "valkey_authoritative_admission",
    "valkey_admission_lease_ms",
    "shared_cache_multiplier",
    "shared_cache_type",
    "router_predicted_ttl_secs",
)

_LEGACY_VALKEY_FIELDS: tuple[str, ...] = (
    "valkey_urls",
    "valkey_index_scope",
    "valkey_connection_pool_size",
    "valkey_required_replica_acks",
    "valkey_sentinel_urls",
    "valkey_sentinel_master_name",
    "valkey_sentinel_quorum",
    "valkey_allow_insecure_plaintext",
    "valkey_allow_degraded_writes",
    "valkey_worker_events",
    "valkey_authoritative_admission",
    "valkey_admission_lease_ms",
)

_DEPRECATED_OVERLAP_WEIGHT_MESSAGE = (
    "router KV overlap score weight is deprecated; use "
    "--router-prefill-load-scale or DYN_ROUTER_PREFILL_LOAD_SCALE for equivalent behavior"
)
_LOAD_AWARE_KWARG_OVERRIDES = {
    "overlap_score_credit": 0.0,
    "use_kv_events": False,
    "durable_kv_events": False,
    "router_track_active_blocks": True,
    "router_assume_kv_reuse": False,
    "router_track_prefill_tokens": True,
    "use_remote_indexer": False,
    "serve_indexer": False,
    "valkey_urls": None,
    "valkey_index_scope": None,
    "valkey_connection_pool_size": 4,
    "valkey_required_replica_acks": None,
    "valkey_sentinel_urls": None,
    "valkey_sentinel_master_name": None,
    "valkey_sentinel_quorum": None,
    "valkey_allow_insecure_plaintext": False,
    "valkey_allow_degraded_writes": False,
    "valkey_worker_events": False,
    "valkey_authoritative_admission": False,
    "valkey_admission_lease_ms": 30000,
    "shared_cache_multiplier": 0.0,
    "shared_cache_type": "none",
    "router_predicted_ttl_secs": None,
}


class _DeprecatedOverlapScoreWeightAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        warnings.warn(_DEPRECATED_OVERLAP_WEIGHT_MESSAGE, FutureWarning, stacklevel=2)
        setattr(namespace, self.dest, values)


def _deprecated_overlap_score_weight_from_env() -> Optional[tuple[str, float]]:
    for env_var in ("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "DYN_OVERLAP_SCORE_WEIGHT"):
        if env_var in os.environ:
            return env_var, float(os.environ[env_var])
    return None


def _default_overlap_score_weight() -> Optional[float]:
    legacy = _deprecated_overlap_score_weight_from_env()
    if legacy is None:
        return None

    env_var, value = legacy
    warnings.warn(
        f"{env_var} is deprecated; use DYN_ROUTER_PREFILL_LOAD_SCALE",
        FutureWarning,
        stacklevel=3,
    )

    return value


def _default_prefill_load_scale() -> float:
    return 1.0


class KvRouterConfigBase(ConfigBase):
    """Mixin carrying the shared KvRouterConfig fields."""

    overlap_score_weight: Optional[float] = None
    overlap_score_credit: float
    overlap_score_credit_decay: float
    prefill_load_scale: float
    host_cache_hit_weight: float
    disk_cache_hit_weight: float
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
    router_queue_threshold: Optional[float]
    router_policy_config: Optional[str] = None
    router_event_threads: int
    router_queue_policy: str
    use_remote_indexer: bool = False
    serve_indexer: bool = False
    router_valkey_config: Optional[str] = None
    valkey_urls: Optional[str] = None
    valkey_index_scope: Optional[str] = None
    valkey_connection_pool_size: int = 4
    valkey_required_replica_acks: Optional[int] = None
    valkey_sentinel_urls: Optional[str] = None
    valkey_sentinel_master_name: Optional[str] = None
    valkey_sentinel_quorum: Optional[int] = None
    valkey_allow_insecure_plaintext: bool = False
    valkey_allow_degraded_writes: bool = False
    valkey_worker_events: bool = False
    valkey_authoritative_admission: bool = False
    valkey_admission_lease_ms: int = 30000
    shared_cache_multiplier: float = 0.0
    shared_cache_type: str = "none"
    router_predicted_ttl_secs: Optional[float] = None
    load_aware: bool = False

    def parsed_router_valkey_config(self) -> str | None:
        if self.router_valkey_config is None:
            return None
        return validate_router_valkey_config(self.router_valkey_config)

    def tokenizer_cache_config_json(self) -> str:
        config = self.parsed_router_valkey_config()
        if config is not None:
            return config
        return legacy_tokenizer_cache_config_json()

    def apply_load_aware_preset(self) -> None:
        if not self.load_aware:
            return

        for field, value in _LOAD_AWARE_KWARG_OVERRIDES.items():
            setattr(self, field, value)

    def kv_router_kwargs(self) -> dict:
        """Return a dict suitable for ``KvRouterConfig(**kwargs)``."""
        self.apply_load_aware_preset()
        values = {field: getattr(self, field) for field in _KV_ROUTER_FIELDS}
        if self.router_valkey_config is not None:
            # Validate through the canonical Rust JSON contract here for early
            # CLI errors. RouterConfig parses it once into both Rust configs.
            self.parsed_router_valkey_config()
            for field in _LEGACY_VALKEY_FIELDS:
                values.pop(field)
        return values


class KvRouterArgGroup(ArgGroup):
    """CLI arguments for the shared KvRouterConfig parameters."""

    def add_arguments(self, parser) -> None:
        g = parser.add_argument_group("KV Router Options")

        add_argument(
            g,
            flag_name="--router-valkey-config",
            env_var="DYN_ROUTER_VALKEY_CONFIG",
            default=None,
            help=(
                "[EXPERIMENTAL] KV Router: Inline JSON configuration for the shared "
                "Valkey router index, Sentinel discovery, and optional tokenizer L2 "
                "cache. This object replaces the legacy --router-valkey-* arguments "
                "and tokenizer-cache environment. The tokenizer cache can inherit "
                "the top-level Sentinel "
                "witnesses while selecting a distinct sentinel_master_name."
            ),
            arg_type=str,
            dest="router_valkey_config",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--load-aware",
            env_var="DYN_ROUTER_LOAD_AWARE",
            default=False,
            dest="load_aware",
            help=(
                "KV Router: Enable load-aware routing without cache-reuse signals. "
                "On the frontend, this implies --router-mode kv. "
                "This preset sets overlap_score_credit=0, disables KV events and "
                "durable KV events, disables KV-reuse assumptions, enables active-block "
                "and prefill-token load tracking, and disables remote/shared cache indexers."
            ),
        )
        add_argument(
            g,
            flag_name="--router-kv-overlap-score-credit",
            env_var="DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT",
            default=1.0,
            help=(
                "KV Router: Credit multiplier for device-local prefix overlap. "
                "Range: 0.0 to 1.0; higher values more strongly prefer KV cache reuse. "
                "Use router-prefill-load-scale above 1.0 to weigh TTFT/prompt-side load more heavily."
            ),
            arg_type=float,
            dest="overlap_score_credit",
        )
        add_argument(
            g,
            flag_name="--router-kv-overlap-score-credit-decay",
            env_var="DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY",
            default=0.0,
            help=(
                "KV Router: Decay rate for device-local overlap credit as active "
                "prefill load rises above the least-loaded eligible worker. "
                "0 disables decay; 1 halves credit at one request-equivalent "
                "of excess active prefill load."
            ),
            arg_type=float,
            dest="overlap_score_credit_decay",
        )
        g.add_argument(
            "--router-kv-overlap-score-weight",
            "--kv-overlap-score-weight",
            dest="overlap_score_weight",
            type=float,
            action=_DeprecatedOverlapScoreWeightAction,
            default=_default_overlap_score_weight(),
            help=argparse.SUPPRESS,
        )
        add_argument(
            g,
            flag_name="--router-prefill-load-scale",
            env_var="DYN_ROUTER_PREFILL_LOAD_SCALE",
            default=_default_prefill_load_scale(),
            help=(
                "KV Router: Scale applied to adjusted prompt-side prefill load after "
                "overlap and lower-tier cache-hit credits are subtracted."
            ),
            arg_type=float,
            dest="prefill_load_scale",
        )
        add_argument(
            g,
            flag_name="--router-host-cache-hit-weight",
            env_var="DYN_ROUTER_HOST_CACHE_HIT_WEIGHT",
            default=0.75,
            help=(
                "KV Router: Credit multiplier for host-pinned (CPU offload) prefix overlap. "
                "Range: 0.0 to 1.0; higher values more strongly prefer workers holding the "
                "prefix in CPU-tier KV cache. Symmetric to --router-kv-overlap-score-credit "
                "but applied to host_pinned tier overlap."
            ),
            arg_type=float,
            dest="host_cache_hit_weight",
        )
        add_argument(
            g,
            flag_name="--router-disk-cache-hit-weight",
            env_var="DYN_ROUTER_DISK_CACHE_HIT_WEIGHT",
            default=0.25,
            help=(
                "KV Router: Credit multiplier for disk/lower-tier (e.g. NVMe-backed) prefix overlap. "
                "Range: 0.0 to 1.0. Same semantics as --router-host-cache-hit-weight applied to "
                "the disk tier."
            ),
            arg_type=float,
            dest="disk_cache_hit_weight",
        )
        add_argument(
            g,
            flag_name="--router-temperature",
            env_var="DYN_ROUTER_TEMPERATURE",
            default=0.0,
            help=(
                "KV Router: Temperature for normalized worker sampling via softmax. "
                "Higher values promote more randomness, and 0 falls back to deterministic."
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
            flag_name="--router-queue-threshold",
            env_var="DYN_ROUTER_QUEUE_THRESHOLD",
            default=16.0,
            help=(
                "KV Router: Queue threshold fraction for prefill token capacity. "
                "Requests are queued if all workers exceed this fraction of "
                "max_num_batched_tokens. Must be >= 0. Use 0.0 for maximum "
                "queueing sensitivity (queue as soon as any tokens are active). "
                "Pass 'None' to disable router queueing. "
                "Note (SGLang backend): when --max-prefill-tokens is not set, MDC's "
                "max_num_batched_tokens falls back to max_total_num_tokens (the KV "
                "cache pool size), not the per-step prefill window, which inflates "
                "the threshold's effective denominator. Set --max-prefill-tokens "
                "explicitly for predictable semantics, or use a smaller threshold."
            ),
            arg_type=nullable_float,
        )
        add_argument(
            g,
            flag_name="--router-policy-config",
            env_var="DYN_ROUTER_POLICY_CONFIG",
            default=None,
            help=(
                "KV Router: Startup-only YAML policy-family and cache-bucket "
                "queue configuration. "
                "When omitted, router_queue_threshold and router_queue_policy define "
                "the existing single default queue."
            ),
            arg_type=str,
        )
        add_argument(
            g,
            flag_name="--router-event-threads",
            env_var="DYN_ROUTER_EVENT_THREADS",
            default=4,
            help=(
                "KV Router: Number of KV indexer worker threads. When > 1, uses a concurrent "
                "radix tree with a thread pool for higher throughput, including "
                "approximate routing when --no-router-kv-events is set."
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
        add_negatable_bool_argument(
            g,
            flag_name="--use-remote-indexer",
            env_var="DYN_USE_REMOTE_INDEXER",
            default=False,
            help=(
                "[EXPERIMENTAL] KV Router: Query a remote KV indexer served from the worker "
                "component via the request plane instead of maintaining a local radix tree."
            ),
            dest="use_remote_indexer",
        )
        add_argument(
            g,
            flag_name="--router-valkey-urls",
            env_var="DYN_ROUTER_VALKEY_URLS",
            default=None,
            help=(
                "[EXPERIMENTAL] KV Router: Comma-separated Valkey endpoints for a persistent "
                "shared device-tier prefix index (for example, valkey://valkey-0:6379,valkey://valkey-1:6379). "
                "Only the first endpoint receives traffic; a Sentinel/operator-managed stable-primary "
                "endpoint can be used with --router-valkey-required-replica-acks. "
                "Use --router-valkey-authoritative-admission to make the replicated module "
                "also own immediate cross-frontend admission."
            ),
            arg_type=str,
            dest="valkey_urls",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-valkey-worker-events",
            env_var="DYN_ROUTER_VALKEY_WORKER_EVENTS",
            default=False,
            help=(
                "[EXPERIMENTAL] KV Router: Workers publish GPU KV metadata directly "
                "to the primary Valkey index. Set the same DYN_ROUTER_VALKEY_URLS "
                "and DYN_ROUTER_VALKEY_WORKER_EVENTS environment variables in every worker process. "
                "Frontends route from the shared Valkey index; legacy event-plane delivery is a post-commit compatibility relay."
            ),
            dest="valkey_worker_events",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-valkey-authoritative-admission",
            env_var="DYN_ROUTER_VALKEY_AUTHORITATIVE_ADMISSION",
            default=False,
            help=(
                "[EXPERIMENTAL] KV Router: Atomically select and reserve workers in the "
                "replicated Valkey module, rather than making frontend-local admission decisions. "
                "Requires direct Valkey worker events and immediate-routing mode: disable "
                "router queues, replica sync, frontend-local prefill/output accounting, "
                "shared-cache credits, overlap decay, and temperature."
            ),
            dest="valkey_authoritative_admission",
        )
        add_argument(
            g,
            flag_name="--router-valkey-admission-lease-ms",
            env_var="DYN_ROUTER_VALKEY_ADMISSION_LEASE_MS",
            default=30000,
            help=(
                "KV Router: Valkey authoritative-admission lease duration in milliseconds "
                "(10000-600000). Normal request cleanup releases the lease sooner; expiry "
                "recovers a reservation after a frontend failure."
            ),
            arg_type=int,
            dest="valkey_admission_lease_ms",
        )
        add_argument(
            g,
            flag_name="--router-valkey-index-scope",
            env_var="DYN_ROUTER_VALKEY_INDEX_SCOPE",
            default=None,
            help=(
                "KV Router: Shared Valkey index scope used by every frontend and worker. "
                "Required with --router-valkey-worker-events when worker and frontend "
                "component names differ (for example DGD prefill/decode workers)."
            ),
            arg_type=str,
            dest="valkey_index_scope",
        )
        add_argument(
            g,
            flag_name="--router-valkey-connection-pool-size",
            env_var="DYN_ROUTER_VALKEY_CONNECTION_POOL_SIZE",
            default=4,
            help=(
                "KV Router: Number of persistent match connections maintained to the Valkey primary."
            ),
            arg_type=int,
            dest="valkey_connection_pool_size",
        )
        add_argument(
            g,
            flag_name="--router-valkey-required-replica-acks",
            env_var="DYN_ROUTER_VALKEY_REQUIRED_REPLICA_ACKS",
            default=None,
            help=(
                "KV Router: Number of replicas that must acknowledge every Valkey "
                "mutation. Set this to 1 with a single stable-primary URL for strict "
                "two-node durability. When omitted, one acknowledgement is inferred "
                "from a URL list containing two or more endpoints; otherwise zero."
            ),
            arg_type=int,
            dest="valkey_required_replica_acks",
        )
        add_argument(
            g,
            flag_name="--router-valkey-sentinel-urls",
            env_var="DYN_ROUTER_VALKEY_SENTINEL_URLS",
            default=None,
            help=(
                "KV Router: Comma-separated Valkey Sentinel witness endpoints. "
                "Clients require a strict witness majority before switching primaries."
            ),
            arg_type=str,
            dest="valkey_sentinel_urls",
        )
        add_argument(
            g,
            flag_name="--router-valkey-sentinel-master-name",
            env_var="DYN_ROUTER_VALKEY_SENTINEL_MASTER_NAME",
            default=None,
            help="KV Router: Sentinel master-set name used for primary discovery.",
            arg_type=str,
            dest="valkey_sentinel_master_name",
        )
        add_argument(
            g,
            flag_name="--router-valkey-sentinel-quorum",
            env_var="DYN_ROUTER_VALKEY_SENTINEL_QUORUM",
            default=None,
            help=(
                "KV Router: Sentinel witnesses that must agree on one primary. "
                "Defaults to a strict majority."
            ),
            arg_type=int,
            dest="valkey_sentinel_quorum",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-valkey-allow-insecure-plaintext",
            env_var="DYN_ROUTER_VALKEY_ALLOW_INSECURE_PLAINTEXT",
            default=False,
            help=(
                "KV Router: Explicitly allow unauthenticated plaintext Valkey only "
                "on a separate tenant-isolated trusted network. TLS and ACL "
                "credentials are not yet supported."
            ),
            dest="valkey_allow_insecure_plaintext",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-valkey-allow-degraded-writes",
            env_var="DYN_ROUTER_VALKEY_ALLOW_DEGRADED_WRITES",
            default=False,
            help=(
                "KV Router: Preserve two-node availability after failover by accepting "
                "writes on a Sentinel-quorum-confirmed singleton when WAIT returns zero. "
                "Those writes have no redundant copy until the replica rejoins."
            ),
            dest="valkey_allow_degraded_writes",
        )
        add_argument(
            g,
            flag_name="--shared-cache-multiplier",
            env_var="DYN_SHARED_CACHE_MULTIPLIER",
            default=0.5,
            help=(
                "[EXPERIMENTAL] KV Router: Multiplier for shared cache hits (0.0-1.0). "
                "Blocks in the shared cache are less valuable than device-local blocks. "
                "E.g. 0.5 means each shared hit counts as half a device-local hit. "
                "Default 0.5."
            ),
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--shared-cache-type",
            env_var="DYN_SHARED_CACHE_TYPE",
            default="none",
            help=(
                "[EXPERIMENTAL] KV Router: Type of external shared KV cache to query. "
                "'none' (default): disabled. "
                "'hicache': query Mooncake master directly for SGLang L3 (HiCache) state "
                "using SGLang-compatible Mooncake key derivation."
            ),
            arg_type=str,
            choices=["none", "hicache"],
        )
        add_argument(
            g,
            flag_name="--router-predicted-ttl-secs",
            env_var="DYN_ROUTER_PREDICTED_TTL_SECS",
            default=None,
            help=(
                "KV Router: Enable predict-on-route with this TTL in seconds for entries "
                "in the local side indexer. Requires KV events; omit to disable. "
                "Independent of --router-ttl-secs, which covers pure approximate mode."
            ),
            arg_type=float,
        )
