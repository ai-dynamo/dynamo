#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.frontend [args]`
#
# Start a frontend node. This runs:
# - OpenAI HTTP server.
# - Auto-discovery: Watches etcd for engine/worker registration (via `register_llm`).
# - Pre-processor: Prompt templating and tokenization.
# - Router, defaulting to round-robin (TODO: Add flags to enable KV routing).
#
# Pass `--interactive` or `-i` for text chat instead of HTTP server.
#
# For TLS:
# - python -m dynamo.frontend --http-port 8443 --tls-cert-path cert.pem --tls-key-path key.pem
#

import argparse
import asyncio
import logging
import os
import pathlib
import signal

import uvloop

from dynamo.common.config_dump import dump_config
from dynamo.common.config_dump.config_dumper import add_config_dump_args
from dynamo.llm import (
    EngineType,
    EntrypointArgs,
    KvRouterConfig,
    ModelDeploymentCard,
    PythonAsyncEngine,
    RouterConfig,
    RouterMode,
    make_engine,
    run_input,
)
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

from . import __version__

DYN_NAMESPACE_ENV_VAR = "DYN_NAMESPACE"
CUSTOM_BACKEND_METRICS_POLLING_INTERVAL_ENV_VAR = (
    "CUSTOM_BACKEND_METRICS_POLLING_INTERVAL"
)
CUSTOM_BACKEND_ENDPOINT_ENV_VAR = "CUSTOM_BACKEND_ENDPOINT"

configure_dynamo_logging()
logger = logging.getLogger(__name__)


async def _dummy_generator(request):
    """Minimal generator that yields nothing. Work in progress."""
    return
    yield  # Makes this an async generator


async def engine_factory(mdc: ModelDeploymentCard) -> PythonAsyncEngine:
    """
    Called by Rust when a model is discovered.
    """
    loop = asyncio.get_running_loop()
    logger.info(f"Engine_factory called with MDC: {mdc.to_json_str()[:100]}...")
    return PythonAsyncEngine(_dummy_generator, loop)


def validate_model_name(value):
    """Validate that model-name is a non-empty string."""
    if not value or not isinstance(value, str) or len(value.strip()) == 0:
        raise argparse.ArgumentTypeError(
            f"model-name must be a non-empty string, got: {value}"
        )
    return value.strip()


def validate_model_path(value):
    """Validate that model-path is a valid directory on disk."""
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError(
            f"model-path must be a valid directory on disk, got: {value}"
        )
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamo Frontend: HTTP+Pre-processor+Router",
        formatter_class=argparse.RawTextHelpFormatter,  # To preserve multi-line help formatting
    )
    parser.add_argument(
        "--version", action="version", version=f"Dynamo Frontend {__version__}"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive text chat"
    )
    parser.add_argument(
        "--kv-cache-block-size",
        type=int,
        default=os.environ.get("DYN_KV_CACHE_BLOCK_SIZE"),
        help="KV cache block size (u32). Can be set via DYN_KV_CACHE_BLOCK_SIZE env var.",
    )
    parser.add_argument(
        "--http-host",
        type=str,
        default=os.environ.get("DYN_HTTP_HOST", "0.0.0.0"),
        help="HTTP host for the engine (str). Can be set via DYN_HTTP_HOST env var.",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=int(os.environ.get("DYN_HTTP_PORT", "8000")),
        help="HTTP port for the engine (u16). Can be set via DYN_HTTP_PORT env var.",
    )
    parser.add_argument(
        "--tls-cert-path",
        type=pathlib.Path,
        default=None,
        help="TLS certificate path, PEM format.",
    )
    parser.add_argument(
        "--tls-key-path",
        type=pathlib.Path,
        default=None,
        help="TLS certificate key path, PEM format.",
    )
    parser.add_argument(
        "--router-mode",
        type=str,
        choices=["round-robin", "random", "kv"],
        default=os.environ.get("DYN_ROUTER_MODE", "round-robin"),
        help="How to route the request. Can be set via DYN_ROUTER_MODE env var.",
    )
    parser.add_argument(
        "--kv-overlap-score-weight",
        type=float,
        default=float(os.environ.get("DYN_KV_OVERLAP_SCORE_WEIGHT", "1.0")),
        help="KV Router: Weight for overlap score in worker selection. Higher values prioritize KV cache reuse.",
    )
    parser.add_argument(
        "--router-temperature",
        type=float,
        default=float(os.environ.get("DYN_ROUTER_TEMPERATURE", "0.0")),
        help="KV Router: Temperature for worker sampling via softmax. Higher values promote more randomness, and 0 fallbacks to deterministic.",
    )
    parser.add_argument(
        "--no-kv-events",
        action="store_false",
        dest="use_kv_events",
        default=os.environ.get("DYN_KV_EVENTS", "true").lower() != "false",
        help="KV Router: Disable KV events. When set, the router predicts cache state based on routing decisions with TTL-based expiration and pruning, rather than receiving events from workers. By default, KV events are enabled.",
    )
    parser.add_argument(
        "--router-ttl",
        type=float,
        default=float(os.environ.get("DYN_ROUTER_TTL", "120.0")),
        help="KV Router: Time-to-live in seconds for blocks when KV events are disabled. Only used when --no-kv-events is set. Can be set via DYN_ROUTER_TTL env var (default: 120.0).",
    )
    parser.add_argument(
        "--router-max-tree-size",
        type=int,
        default=int(os.environ.get("DYN_ROUTER_MAX_TREE_SIZE", str(2**20))),
        help="KV Router: Maximum tree size before pruning when KV events are disabled. Only used when --no-kv-events is set. Can be set via DYN_ROUTER_MAX_TREE_SIZE env var (default: 1048576, which is 2^20).",
    )
    parser.add_argument(
        "--router-prune-target-ratio",
        type=float,
        default=float(os.environ.get("DYN_ROUTER_PRUNE_TARGET_RATIO", "0.8")),
        help="KV Router: Target size ratio after pruning when KV events are disabled. Only used when --no-kv-events is set. Can be set via DYN_ROUTER_PRUNE_TARGET_RATIO env var (default: 0.8).",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=os.environ.get(DYN_NAMESPACE_ENV_VAR),
        help="Dynamo namespace for model discovery scoping. If specified, models will only be discovered from this namespace. If not specified, discovers models from all namespaces (global discovery).",
    )
    parser.add_argument(
        "--router-replica-sync",
        action="store_true",
        default=False,
        help="KV Router: Enable replica synchronization across multiple router instances. When true, routers will publish and subscribe to events to maintain consistent state.",
    )
    parser.add_argument(
        "--router-snapshot-threshold",
        type=int,
        default=1000000,
        help="KV Router: Number of messages in stream before triggering a snapshot. Defaults to 1000000.",
    )
    parser.add_argument(
        "--router-reset-states",
        action="store_true",
        dest="router_reset_states",
        default=False,
        help="KV Router: Reset router state on startup, purging stream and object store. By default, states are persisted. WARNING: This can affect existing router replicas.",
    )
    parser.add_argument(
        "--no-track-active-blocks",
        action="store_false",
        dest="router_track_active_blocks",
        default=True,
        help="KV Router: Disable tracking of active blocks (blocks being used for ongoing generation). By default, active blocks are tracked for load balancing.",
    )
    parser.add_argument(
        "--enforce-disagg",
        action="store_true",
        default=False,
        help="Enforce disaggregated prefill-decode. When set, unactivated prefill router will return an error instead of falling back to decode-only mode.",
    )
    parser.add_argument(
        "--active-decode-blocks-threshold",
        type=float,
        default=None,
        help="Threshold percentage (0.0-1.0) for determining when a worker is considered busy based on KV cache block utilization. If not set, blocks-based busy detection is disabled.",
    )
    parser.add_argument(
        "--active-prefill-tokens-threshold",
        type=int,
        default=None,
        help="Literal token count threshold for determining when a worker is considered busy based on prefill token utilization. When active prefill tokens exceed this threshold, the worker is marked as busy. If not set, tokens-based busy detection is disabled.",
    )
    parser.add_argument(
        "--model-name",
        type=validate_model_name,
        help="Model name as a string (e.g., 'Llama-3.2-1B-Instruct')",
    )
    parser.add_argument(
        "--model-path",
        type=validate_model_path,
        help="Path to model directory on disk (e.g., /tmp/model_cache/lama3.2_1B/)",
    )
    parser.add_argument(
        "--metrics-prefix",
        type=str,
        default=None,
        help="Prefix for Dynamo frontend metrics. If unset, uses DYN_METRICS_PREFIX env var or 'dynamo_frontend'.",
    )
    parser.add_argument(
        "--kserve-grpc-server",
        action="store_true",
        default=False,
        help="Start KServe gRPC server.",
    )
    parser.add_argument(
        "--grpc-metrics-port",
        type=int,
        default=8788,
        help="HTTP metrics port for gRPC service (u16). Only used with --kserve-grpc-server. Defaults to 8788.",
    )
    add_config_dump_args(parser)
    parser.add_argument(
        "--custom-backend-metrics-endpoint",
        type=str,
        default=os.environ.get(
            CUSTOM_BACKEND_ENDPOINT_ENV_VAR, "nim.backend.runtime_stats"
        ),
        help=f"Custom backend endpoint to poll for metrics in format 'namespace.component.endpoint' (default: 'nim.backend.runtime_stats'). Required if --custom-backend-metrics-polling-interval is specified. All metrics will be prefixed with 'dynamo_component_' in Prometheus. Can be set via {CUSTOM_BACKEND_ENDPOINT_ENV_VAR} env var.",
    )
    parser.add_argument(
        "--custom-backend-metrics-polling-interval",
        type=float,
        default=float(
            os.environ.get(CUSTOM_BACKEND_METRICS_POLLING_INTERVAL_ENV_VAR, "0")
        ),
        help=f"Interval in seconds for polling custom backend metrics. Set to > 0 to enable polling (default: 0=disabled, suggested: 9.2s which is less than typical Prometheus scrape interval). Can be set via {CUSTOM_BACKEND_METRICS_POLLING_INTERVAL_ENV_VAR} env var.",
    )
    parser.add_argument(
        "--store-kv",
        type=str,
        choices=["etcd", "file", "mem"],
        default=os.environ.get("DYN_STORE_KV", "etcd"),
        help="Which key-value backend to use: etcd, mem, file. Etcd uses the ETCD_* env vars (e.g. ETCD_ENPOINTS) for connection details. File uses root dir from env var DYN_FILE_KV or defaults to $TMPDIR/dynamo_store_kv.",
    )
    parser.add_argument(
        "--request-plane",
        type=str,
        choices=["nats", "http", "tcp"],
        default=os.environ.get("DYN_REQUEST_PLANE", "tcp"),
        help="Determines how requests are distributed from routers to workers. 'tcp' is fastest [nats|http|tcp]",
    )
    parser.add_argument(
        "--exp-python-factory",
        action="store_true",
        default=False,
        help="[EXPERIMENTAL] Enable Python-based engine factory. When set, engines will be created via a Python callback instead of the default Rust pipeline.",
    )

    flags = parser.parse_args()

    if bool(flags.tls_cert_path) ^ bool(flags.tls_key_path):  # ^ is XOR
        parser.error("--tls-cert-path and --tls-key-path must be provided together")
    if flags.custom_backend_metrics_polling_interval < 0:
        parser.error(
            "--custom-backend-metrics-polling-interval must be >= 0 (0=disabled)"
        )

    return flags


async def async_main():
    # The system status server port is a worker concern.
    #
    # Serve tests set DYN_SYSTEM_PORT for the worker, but aggregated launch scripts
    # start `dynamo.frontend` first. If the frontend inherits DYN_SYSTEM_PORT, it can
    # bind that port before the worker, causing port conflicts and/or scraping the
    # wrong metrics endpoint.
    os.environ.pop("DYN_SYSTEM_PORT", None)
    flags = parse_args()
    dump_config(flags.dump_config_to, flags)

    # Warn if DYN_SYSTEM_PORT is set (frontend doesn't use system metrics server)
    if os.environ.get("DYN_SYSTEM_PORT"):
        logger.warning(
            "=" * 80 + "\n"
            "WARNING: DYN_SYSTEM_PORT is set but NOT used by the frontend!\n"
            "The frontend does not expose a system metrics server.\n"
            "Only backend workers should set DYN_SYSTEM_PORT.\n"
            "Use --http-port to configure the frontend HTTP API port.\n" + "=" * 80
        )

    # Configure Dynamo frontend HTTP service metrics prefix
    if flags.metrics_prefix is not None:
        prefix = flags.metrics_prefix.strip()
        if prefix:
            os.environ["DYN_METRICS_PREFIX"] = flags.metrics_prefix

    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, flags.store_kv, flags.request_plane)

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    if flags.router_mode == "kv":
        router_mode = RouterMode.KV
        kv_router_config = KvRouterConfig(
            overlap_score_weight=flags.kv_overlap_score_weight,
            router_temperature=flags.router_temperature,
            use_kv_events=flags.use_kv_events,
            router_replica_sync=flags.router_replica_sync,
            router_snapshot_threshold=flags.router_snapshot_threshold,
            router_reset_states=flags.router_reset_states,
            router_track_active_blocks=flags.router_track_active_blocks,
            router_ttl_secs=flags.router_ttl,
            router_max_tree_size=flags.router_max_tree_size,
            router_prune_target_ratio=flags.router_prune_target_ratio,
        )
    elif flags.router_mode == "random":
        router_mode = RouterMode.Random
        kv_router_config = None
    else:
        router_mode = RouterMode.RoundRobin
        kv_router_config = None

    kwargs = {
        "http_host": flags.http_host,
        "http_port": flags.http_port,
        "kv_cache_block_size": flags.kv_cache_block_size,
        "router_config": RouterConfig(
            router_mode,
            kv_router_config,
            active_decode_blocks_threshold=flags.active_decode_blocks_threshold,
            active_prefill_tokens_threshold=flags.active_prefill_tokens_threshold,
            enforce_disagg=flags.enforce_disagg,
        ),
    }

    if flags.model_name:
        kwargs["model_name"] = flags.model_name
    if flags.model_path:
        kwargs["model_path"] = flags.model_path
    if flags.tls_cert_path:
        kwargs["tls_cert_path"] = flags.tls_cert_path
    if flags.tls_key_path:
        kwargs["tls_key_path"] = flags.tls_key_path
    if flags.namespace:
        kwargs["namespace"] = flags.namespace
    if flags.kserve_grpc_server and flags.grpc_metrics_port:
        kwargs["http_metrics_port"] = flags.grpc_metrics_port
    if flags.custom_backend_metrics_endpoint:
        kwargs[
            "custom_backend_metrics_endpoint"
        ] = flags.custom_backend_metrics_endpoint
    if flags.custom_backend_metrics_polling_interval:
        kwargs[
            "custom_backend_metrics_polling_interval"
        ] = flags.custom_backend_metrics_polling_interval

    if flags.exp_python_factory:
        kwargs["engine_factory"] = engine_factory

    e = EntrypointArgs(EngineType.Dynamic, **kwargs)
    engine = await make_engine(runtime, e)

    try:
        if flags.interactive:
            await run_input(runtime, "text", engine)
        elif flags.kserve_grpc_server:
            await run_input(runtime, "grpc", engine)
        else:
            await run_input(runtime, "http", engine)
    except asyncio.exceptions.CancelledError:
        pass


async def graceful_shutdown(runtime):
    runtime.shutdown()


def main():
    uvloop.run(async_main())


if __name__ == "__main__":
    main()
