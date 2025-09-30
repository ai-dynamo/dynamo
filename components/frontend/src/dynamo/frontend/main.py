#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# For static mode (no etcd auto-discovery):
# - python -m dynamo.frontend --model-name Qwen3-0.6B-Q8_0.gguf --model-path ~/llms/Qwen3-0.6B --static-endpoint dynamo.backend.generate
# Worker example:
# - cd lib/bindings/python/examples/hello_world
# - python server_sglang_static.py
#
# For TLS:
# - python -m dynamo.frontend --http-port 8443 --tls-cert-path cert.pem --tls-key-path key.pem
#

import argparse
import asyncio
import logging
import os
import pathlib
import re

import uvloop

from dynamo.llm import (
    EngineType,
    EntrypointArgs,
    KvRouterConfig,
    RouterConfig,
    RouterMode,
    make_engine,
    run_input,
)
from dynamo.runtime import DistributedRuntime

from . import __version__

DYNAMO_NAMESPACE_ENV_VAR = "DYN_NAMESPACE"

logger = logging.getLogger(__name__)


def validate_static_endpoint(value):
    """Validate that static-endpoint is three words separated by dots."""
    if not re.match(
        r"^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$",
        value,
    ):
        raise argparse.ArgumentTypeError(
            f"static-endpoint must be three words separated by dots, got: {value}"
        )
    return value


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
        "--kv-cache-block-size", type=int, help="KV cache block size (u32)."
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
        default=1.0,
        help="KV Router: Weight for overlap score in worker selection. Higher values prioritize KV cache reuse.",
    )
    parser.add_argument(
        "--router-temperature",
        type=float,
        default=0.0,
        help="KV Router: Temperature for worker sampling via softmax. Higher values promote more randomness, and 0 fallbacks to deterministic.",
    )
    parser.add_argument(
        "--no-kv-events",
        action="store_false",
        dest="use_kv_events",
        default=True,
        help="KV Router: Disable KV events. When set, uses ApproxKvRouter for predicting block creation/deletion based only on incoming requests at a timer. By default, KV events are enabled.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=os.environ.get(DYNAMO_NAMESPACE_ENV_VAR),
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
        "--busy-threshold",
        type=float,
        default=None,
        help="Threshold (0.0-1.0) for determining when a worker is considered busy based on KV cache usage. If not set, busy detection is disabled.",
    )
    parser.add_argument(
        "--static-endpoint",
        type=validate_static_endpoint,
        help="Static endpoint in format: word.word.word (e.g., dynamo.backend.generate)",
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

    flags = parser.parse_args()

    if flags.static_endpoint and (not flags.model_name or not flags.model_path):
        parser.error("--static-endpoint requires both --model-name and --model-path")
    if bool(flags.tls_cert_path) ^ bool(flags.tls_key_path):  # ^ is XOR
        parser.error("--tls-cert-path and --tls-key-path must be provided together")

    return flags


async def async_main():
    flags = parse_args()
    is_static = bool(flags.static_endpoint)  # true if the string has a value

    # Configure Dynamo frontend HTTP service metrics prefix
    if flags.metrics_prefix is not None:
        prefix = flags.metrics_prefix.strip()
        if prefix:
            os.environ["DYN_METRICS_PREFIX"] = flags.metrics_prefix

    runtime = DistributedRuntime(asyncio.get_running_loop(), is_static)

    if flags.router_mode == "kv":
        router_mode = RouterMode.KV
        kv_router_config = KvRouterConfig(
            overlap_score_weight=flags.kv_overlap_score_weight,
            router_temperature=flags.router_temperature,
            use_kv_events=flags.use_kv_events,
            router_replica_sync=flags.router_replica_sync,
            router_snapshot_threshold=flags.router_snapshot_threshold,
            router_reset_states=flags.router_reset_states,
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
            router_mode, kv_router_config, flags.busy_threshold
        ),
    }

    if flags.static_endpoint:
        kwargs["endpoint_id"] = flags.static_endpoint

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

    if is_static:
        # out=dyn://<static_endpoint>
        engine_type = EngineType.Static
    else:
        # out=auto, most common
        engine_type = EngineType.Dynamic
    e = EntrypointArgs(engine_type, **kwargs)
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


def main():
    uvloop.run(async_main())


if __name__ == "__main__":
    main()
