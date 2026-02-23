#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.frontend [args]`
#
# Start a frontend node. This runs:
# - OpenAI HTTP server.
# - Auto-discovery: Watches etcd for engine/worker registration (via `register_model`).
# - Pre-processor: Prompt templating and tokenization.
# - Router, defaulting to round-robin. Use --router-mode to switch (round-robin, random, kv, direct).
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
import signal
import sys
from argparse import Namespace
from typing import Optional

import uvloop

from dynamo.common.config_dump import dump_config
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
from dynamo.runtime.logging import configure_dynamo_logging

from .frontend_args import FrontendArgGroup, FrontendConfig

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def setup_engine_factory(
    runtime: DistributedRuntime,
    router_config: RouterConfig,
    config: FrontendConfig,
    vllm_flags: Namespace,
):
    """
    When using vllm pre and post processor, create the EngineFactory that
    creates the engines that run requests.
    """
    from .vllm_processor import EngineFactory

    return EngineFactory(runtime, router_config, config, vllm_flags, config.debug_perf)


def parse_args() -> tuple[FrontendConfig, Optional[Namespace]]:
    """Parse command-line arguments for the Dynamo frontend.

    Returns:
        FrontendConfig: Parsed configuration object.
    """

    parser = argparse.ArgumentParser(
        description="Dynamo Frontend: HTTP+Pre-processor+Router",
        formatter_class=argparse.RawTextHelpFormatter,  # To preserve multi-line help formatting
    )

    FrontendArgGroup().add_arguments(parser)

    args, unknown = parser.parse_known_args()

    config = FrontendConfig.from_cli_args(args)
    config.validate()

    vllm_flags = None

    # parse extra vllm flags using vllm native parser.
    if config.chat_processor == "vllm":
        try:
            from vllm.utils import FlexibleArgumentParser
        except ImportError:
            try:
                from vllm.utils.argparse_utils import FlexibleArgumentParser
            except ModuleNotFoundError:
                logger.exception(
                    "Flag '--chat-processor vllm' requires vllm be installed."
                )
                sys.exit(1)
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.entrypoints.openai.cli_args import FrontendArgs
        except ModuleNotFoundError:
            logger.exception("Flag '--chat-processor vllm' requires vllm be installed.")
            sys.exit(1)

        vllm_parser = FlexibleArgumentParser(add_help=False)
        vllm_parser = FrontendArgs.add_cli_args(vllm_parser)
        vllm_parser = AsyncEngineArgs.add_cli_args(vllm_parser)
        # the result is returned as Namespace object rather than AsyncEngineArgs object to avoid import error for non-vllm users.
        vllm_flags = vllm_parser.parse_args(unknown)
    else:
        if unknown:
            logger.error(f"Unknown arguments specified: {unknown}")
            sys.exit(1)
    return config, vllm_flags


async def async_main():
    """Main async entry point for the Dynamo frontend.

    Initializes the distributed runtime, configures routing, and starts
    the HTTP server or interactive mode based on command-line arguments.
    """
    # The system status server port is a worker concern.
    #
    # Serve tests set DYN_SYSTEM_PORT for the worker, but aggregated launch scripts
    # start `dynamo.frontend` first. If the frontend inherits DYN_SYSTEM_PORT, it can
    # bind that port before the worker, causing port conflicts and/or scraping the
    # wrong metrics endpoint.
    os.environ.pop("DYN_SYSTEM_PORT", None)
    config, vllm_flags = parse_args()
    dump_config(config.dump_config_to, config)
    os.environ["DYN_EVENT_PLANE"] = config.event_plane
    logger.info(
        f"Request migration {'enabled' if config.migration_limit > 0 else 'disabled'} "
        f"(limit: {config.migration_limit})"
    )
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
    if config.metrics_prefix is not None:
        prefix = config.metrics_prefix.strip()
        if prefix:
            os.environ["DYN_METRICS_PREFIX"] = config.metrics_prefix

    # NATS is needed when:
    # 1. Request plane is NATS, OR
    # 2. Durable KV events (JetStream) is explicitly requested, OR
    # 3. Event plane is NATS AND KV router mode AND (KV events OR replica sync enabled)
    # Note: NATS Core (without JetStream) is the default for KV events when durable_kv_events=False
    enable_nats = config.request_plane == "nats" or (
        config.router_mode == "kv"
        and (
            config.durable_kv_events
            or (
                config.event_plane == "nats"
                and (config.use_kv_events or config.router_replica_sync)
            )
        )
    )

    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(
        loop, config.discovery_backend, config.request_plane, enable_nats
    )

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    if config.router_mode == "kv":
        router_mode = RouterMode.KV
        kv_router_config = KvRouterConfig(
            overlap_score_weight=config.kv_overlap_score_weight,
            router_temperature=config.router_temperature,
            use_kv_events=config.use_kv_events,
            durable_kv_events=config.durable_kv_events,
            router_replica_sync=config.router_replica_sync,
            router_track_active_blocks=config.router_track_active_blocks,
            router_track_output_blocks=config.router_track_output_blocks,
            router_assume_kv_reuse=config.router_assume_kv_reuse,
            router_snapshot_threshold=config.router_snapshot_threshold,
            router_reset_states=config.router_reset_states,
            router_ttl_secs=config.router_ttl,
            router_max_tree_size=config.router_max_tree_size,
            router_prune_target_ratio=config.router_prune_target_ratio,
            router_queue_threshold=config.router_queue_threshold,
            router_event_threads=config.router_event_threads,
        )
    elif config.router_mode == "random":
        router_mode = RouterMode.Random
        kv_router_config = None
    elif config.router_mode == "direct":
        router_mode = RouterMode.Direct
        kv_router_config = None
    else:
        router_mode = RouterMode.RoundRobin
        kv_router_config = None

    router_config = RouterConfig(
        router_mode,
        kv_router_config,
        active_decode_blocks_threshold=config.active_decode_blocks_threshold,
        active_prefill_tokens_threshold=config.active_prefill_tokens_threshold,
        active_prefill_tokens_threshold_frac=config.active_prefill_tokens_threshold_frac,
        enforce_disagg=config.enforce_disagg,
    )
    kwargs = {
        "http_host": config.http_host,
        "http_port": config.http_port,
        "kv_cache_block_size": config.kv_cache_block_size,
        "router_config": router_config,
        "migration_limit": config.migration_limit,
    }

    if config.model_name:
        kwargs["model_name"] = config.model_name
    if config.model_path:
        kwargs["model_path"] = config.model_path
    if config.tls_cert_path:
        kwargs["tls_cert_path"] = config.tls_cert_path
    if config.tls_key_path:
        kwargs["tls_key_path"] = config.tls_key_path
    if config.namespace:
        kwargs["namespace"] = config.namespace
    if config.namespace_prefix:
        kwargs["namespace_prefix"] = config.namespace_prefix
    if config.kserve_grpc_server and config.grpc_metrics_port:
        kwargs["http_metrics_port"] = config.grpc_metrics_port

    if config.enable_anthropic_api:
        os.environ["DYN_ENABLE_ANTHROPIC_API"] = "1"

    if config.chat_processor == "vllm":
        assert (
            vllm_flags is not None
        ), "vllm_flags is required when chat_processor is vllm"
        chat_engine_factory = setup_engine_factory(
            runtime, router_config, config, vllm_flags
        ).chat_engine_factory
        kwargs["chat_engine_factory"] = chat_engine_factory

    e = EntrypointArgs(EngineType.Dynamic, **kwargs)
    engine = await make_engine(runtime, e)

    try:
        if config.interactive:
            await run_input(runtime, "text", engine)
        elif config.kserve_grpc_server:
            await run_input(runtime, "grpc", engine)
        else:
            await run_input(runtime, "http", engine)
    except asyncio.exceptions.CancelledError:
        pass


async def graceful_shutdown(runtime):
    """Handle graceful shutdown of the distributed runtime.

    Args:
        runtime: The DistributedRuntime instance to shut down.
    """
    runtime.shutdown()


def main():
    """Entry point for the Dynamo frontend CLI."""
    uvloop.run(async_main())


if __name__ == "__main__":
    main()
