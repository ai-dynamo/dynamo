# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pathlib
from typing import Any, Dict, Optional

from dynamo.common.config_dump import register_encoder
from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import (
    add_argument,
    add_negatable_bool_argument,
    env_or_default,
)

from . import __version__


def validate_model_name(value: str) -> str:
    """Validate that model-name is a non-empty string."""
    if not value or not isinstance(value, str) or len(value.strip()) == 0:
        raise argparse.ArgumentTypeError(
            f"model-name must be a non-empty string, got: {value}"
        )
    return value.strip()


def validate_model_path(value: str) -> str:
    """Validate that model-path is a valid directory on disk."""
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError(
            f"model-path must be a valid directory on disk, got: {value}"
        )
    return value


class FrontendConfig(ConfigBase):
    """Configuration for the Dynamo frontend."""

    interactive: bool
    kv_cache_block_size: Optional[int]
    http_host: str
    http_port: int
    tls_cert_path: Optional[pathlib.Path]
    tls_key_path: Optional[pathlib.Path]

    router_mode: str
    kv_overlap_score_weight: float
    router_temperature: float
    use_kv_events: bool
    router_ttl: float
    router_max_tree_size: int
    router_prune_target_ratio: float
    namespace: Optional[str] = None
    namespace_prefix: Optional[str] = None
    router_replica_sync: bool
    router_snapshot_threshold: int
    router_reset_states: bool
    durable_kv_events: bool
    router_track_active_blocks: bool
    router_assume_kv_reuse: bool
    router_track_output_blocks: bool
    router_event_threads: int
    router_queue_threshold: Optional[float]
    enforce_disagg: bool

    migration_limit: int
    active_decode_blocks_threshold: Optional[float]
    active_prefill_tokens_threshold: Optional[int]
    active_prefill_tokens_threshold_frac: Optional[float]
    model_name: Optional[str]
    model_path: Optional[str]
    metrics_prefix: Optional[str] = None

    kserve_grpc_server: bool
    grpc_metrics_port: int
    dump_config_to: Optional[str]

    discovery_backend: str
    request_plane: str
    event_plane: str
    chat_processor: str
    enable_anthropic_api: bool
    debug_perf: bool
    preprocess_workers: int

    def validate(self) -> None:
        if bool(self.tls_cert_path) ^ bool(self.tls_key_path):  # ^ is XOR
            raise ValueError(
                "--tls-cert-path and --tls-key-path must be provided together"
            )
        if self.migration_limit < 0 or self.migration_limit > 4294967295:
            raise ValueError(
                "--migration-limit must be between 0 and 4294967295 (0=disabled)"
            )


@register_encoder(FrontendConfig)
def _preprocess_for_encode_config(config: FrontendConfig) -> Dict[str, Any]:
    """Convert FrontendConfig object to dictionary for encoding."""
    return config.__dict__


class FrontendArgGroup(ArgGroup):
    """Frontend configuration parameters."""

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--version", action="version", version=f"Dynamo Frontend {__version__}"
        )

        g = parser.add_argument_group("Dynamo Frontend Options")

        # Interactive needs -i short option; use raw add_argument with BooleanOptionalAction
        g.add_argument(
            "-i",
            "--interactive",
            dest="interactive",
            action=argparse.BooleanOptionalAction,
            default=env_or_default("DYN_INTERACTIVE", False),
            help="Interactive text chat.\nenv var: DYN_INTERACTIVE",
        )

        add_argument(
            g,
            flag_name="--namespace",
            env_var="DYN_NAMESPACE",
            default=None,
            help=(
                "Dynamo namespace for model discovery scoping. Use for exact namespace matching. "
                "If --namespace-prefix is also specified, prefix takes precedence."
            ),
        )

        add_argument(
            g,
            flag_name="--kv-cache-block-size",
            env_var="DYN_KV_CACHE_BLOCK_SIZE",
            default=None,
            help="KV cache block size (u32).",
            arg_type=int,
        )

        add_argument(
            g,
            flag_name="--http-host",
            env_var="DYN_HTTP_HOST",
            default="0.0.0.0",
            help="HTTP host for the engine (str).",
        )
        add_argument(
            g,
            flag_name="--http-port",
            env_var="DYN_HTTP_PORT",
            default=8000,
            help="HTTP port for the engine (u16).",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--tls-cert-path",
            env_var="DYN_TLS_CERT_PATH",
            default=None,
            help="TLS certificate path, PEM format.",
            arg_type=pathlib.Path,
        )
        add_argument(
            g,
            flag_name="--tls-key-path",
            env_var="DYN_TLS_KEY_PATH",
            default=None,
            help="TLS certificate key path, PEM format.",
            arg_type=pathlib.Path,
        )

        add_argument(
            g,
            flag_name="--router-mode",
            env_var="DYN_ROUTER_MODE",
            default="round-robin",
            help="How to route the request.",
            choices=["round-robin", "random", "kv", "direct"],
        )
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
            dest="kv_overlap_score_weight",
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
            dest="router_ttl",
            obsolete_flag="--router-ttl",
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
            flag_name="--namespace-prefix",
            env_var="DYN_NAMESPACE_PREFIX",
            default=None,
            help=(
                "Dynamo namespace prefix for model discovery scoping. Discovers models from "
                "namespaces starting with this prefix (e.g., 'ns' matches 'ns', 'ns-abc123', "
                "'ns-def456'). Takes precedence over --namespace if both are specified."
            ),
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
        add_argument(
            g,
            flag_name="--router-snapshot-threshold",
            env_var="DYN_ROUTER_SNAPSHOT_THRESHOLD",
            default=1000000,
            help=(
                "KV Router: Number of messages in stream before triggering a snapshot. "
            ),
            arg_type=int,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-reset-states",
            env_var="DYN_ROUTER_RESET_STATES",
            default=False,
            help=(
                "KV Router: Reset router state on startup, purging stream and object store. "
                "By default, states are persisted. WARNING: This can affect existing router "
                "replicas."
            ),
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-durable-kv-events",
            env_var="DYN_ROUTER_DURABLE_KV_EVENTS",
            default=False,
            help=(
                "KV Router: Enable durable KV events using NATS JetStream instead of NATS Core. "
                "By default, the router uses the generic event plane (NATS Core or ZMQ) with "
                "local_indexer mode. Use this flag when you need durability and multi-replica "
                "consistency. Requires NATS with JetStream enabled."
            ),
            dest="durable_kv_events",
            obsolete_flag="--durable-kv-events",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-track-active-blocks",
            env_var="DYN_ROUTER_TRACK_ACTIVE_BLOCKS",
            default=True,
            dest="router_track_active_blocks",
            help=(
                "KV Router: Track active blocks (blocks being used for ongoing generation). "
                "By default, active blocks are tracked for load balancing. "
            ),
            obsolete_flag="--track-active-blocks",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-assume-kv-reuse",
            env_var="DYN_ROUTER_ASSUME_KV_REUSE",
            default=True,
            dest="router_assume_kv_reuse",
            help=(
                "KV Router: When tracking active blocks, assume KV cache reuse. "
                "Use --no-router-assume-kv-reuse to generate random hashes instead (when KV cache reuse is not expected)."
            ),
            obsolete_flag="--assume-kv-reuse",
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
                "progress toward expected_output_tokens."
            ),
            obsolete_flag="--track-output-blocks",
        )
        add_argument(
            g,
            flag_name="--router-event-threads",
            env_var="DYN_ROUTER_EVENT_THREADS",
            default=1,
            help=(
                "KV Router: Number of event processing threads. When > 1, uses a concurrent radix tree with a thread pool for higher throughput."
            ),
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--router-queue-threshold",
            env_var="DYN_ROUTER_QUEUE_THRESHOLD",
            default=None,
            help=(
                "KV Router: Queue threshold fraction for prefill token capacity. "
                "When set, requests are queued if all workers exceed this fraction of "
                "max_num_batched_tokens. Enables priority scheduling via latency_sensitivity "
                "hints. Must be > 0. If not set, queueing is disabled."
            ),
            arg_type=float,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enforce-disagg",
            env_var="DYN_ENFORCE_DISAGG",
            default=False,
            help=(
                "Enforce disaggregated prefill-decode. When set, unactivated prefill router will "
                "return an error instead of falling back to decode-only mode."
            ),
        )

        add_argument(
            g,
            flag_name="--migration-limit",
            env_var="DYN_MIGRATION_LIMIT",
            default=0,
            help=(
                "Maximum number of times a request may be migrated to a different engine worker. "
                "When > 0, enables request migration on worker disconnect."
            ),
            arg_type=int,
        )

        add_argument(
            g,
            flag_name="--active-decode-blocks-threshold",
            env_var="DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
            default=None,
            help=(
                "Threshold percentage (0.0-1.0) for determining when a worker is considered busy "
                "based on KV cache block utilization. If not set, blocks-based busy detection is disabled."
            ),
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
            default=None,
            help=(
                "Literal token count threshold for determining when a worker is considered busy "
                "based on prefill token utilization. When active prefill tokens exceed this "
                "threshold, the worker is marked as busy. If not set, tokens-based busy detection is disabled."
            ),
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold-frac",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
            default=None,
            help=(
                "Fraction of max_num_batched_tokens for busy detection. Worker is busy when "
                "active_prefill_tokens > frac * max_num_batched_tokens. Default 1.5 (disabled). "
                "Uses OR logic with --active-prefill-tokens-threshold."
            ),
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--model-name",
            env_var="DYN_MODEL_NAME",
            default=None,
            help="Model name as a string (e.g., 'Llama-3.2-1B-Instruct')",
            arg_type=validate_model_name,
        )
        add_argument(
            g,
            flag_name="--model-path",
            env_var="DYN_MODEL_PATH",
            default=None,
            help="Path to model directory on disk (e.g., /tmp/model_cache/llama3.2_1B/)",
            arg_type=validate_model_path,
        )
        add_argument(
            g,
            flag_name="--metrics-prefix",
            env_var="DYN_METRICS_PREFIX",
            default=None,
            help=(
                "Prefix for Dynamo frontend metrics. If unset, uses DYN_METRICS_PREFIX env var "
                "or 'dynamo_frontend'."
            ),
        )
        add_negatable_bool_argument(
            g,
            flag_name="--kserve-grpc-server",
            env_var="DYN_KSERVE_GRPC_SERVER",
            default=False,
            help="Start KServe gRPC server.",
        )
        add_argument(
            g,
            flag_name="--grpc-metrics-port",
            env_var="DYN_GRPC_METRICS_PORT",
            default=8788,
            help=(
                "HTTP metrics port for gRPC service (u16). Only used with --kserve-grpc-server. "
                "Defaults to 8788."
            ),
            arg_type=int,
        )

        add_argument(
            g,
            flag_name="--dump-config-to",
            env_var="DYN_DUMP_CONFIG_TO",
            default=None,
            help="Dump config to the specified file path.",
        )

        add_argument(
            g,
            flag_name="--discovery-backend",
            env_var="DYN_DISCOVERY_BACKEND",
            default="etcd",
            help=(
                "Discovery backend: kubernetes (K8s API), etcd (distributed KV), file (local filesystem), "
                "mem (in-memory). Etcd uses the ETCD_* env vars (e.g. ETCD_ENDPOINTS) for connection details. "
                "File uses root dir from env var DYN_FILE_KV or defaults to $TMPDIR/dynamo_store_kv."
            ),
            choices=["kubernetes", "etcd", "file", "mem"],
        )
        add_argument(
            g,
            flag_name="--request-plane",
            env_var="DYN_REQUEST_PLANE",
            default="tcp",
            help=(
                "Determines how requests are distributed from routers to workers. "
                "'tcp' is fastest [nats|http|tcp]"
            ),
            choices=["nats", "http", "tcp"],
        )
        add_argument(
            g,
            flag_name="--event-plane",
            env_var="DYN_EVENT_PLANE",
            default="nats",
            help="Determines how events are published [nats|zmq]",
            choices=["nats", "zmq"],
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-anthropic-api",
            env_var="DYN_ENABLE_ANTHROPIC_API",
            default=False,
            help=(
                "[EXPERIMENTAL] Enable Anthropic Messages API endpoint (/v1/messages). "
                "This feature is experimental and may change."
            ),
        )
        add_argument(
            g,
            flag_name="--dyn-chat-processor",
            env_var="DYN_CHAT_PROCESSOR",
            default="dynamo",
            dest="chat_processor",
            help=(
                "[EXPERIMENTAL] When set to 'vllm', use local vllm for the pre and post "
                "processor."
            ),
            choices=["dynamo", "vllm"],
        )

        add_negatable_bool_argument(
            g,
            flag_name="--dyn-debug-perf",
            env_var="DYN_DEBUG_PERF",
            default=False,
            dest="debug_perf",
            help=(
                "[EXPERIMENTAL] Enable performance instrumentation for diagnosing preprocessing bottlenecks. "
                "Logs per-function timing, request concurrency, and hot-path section durations. "
                "'--dyn-chat-processor vllm' only."
            ),
        )

        add_argument(
            g,
            flag_name="--dyn-preprocess-workers",
            env_var="DYN_PREPROCESS_WORKERS",
            default=0,
            dest="preprocess_workers",
            help=(
                "[EXPERIMENTAL] Number of worker processes for preprocessing and output processing. "
                "When > 0, offloads CPU-bound work (tokenization, template rendering, "
                "detokenization) to a ProcessPoolExecutor with N workers, each with its "
                "own GIL. 0 (default) keeps all processing on the main event loop. '--dyn-chat-processor vllm' only."
            ),
            arg_type=int,
        )
