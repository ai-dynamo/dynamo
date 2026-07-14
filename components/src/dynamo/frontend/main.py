#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.frontend [args]`
#
# Start a frontend node. This runs:
# - OpenAI HTTP server.
# - Auto-discovery: Watches etcd for engine/worker registration (via `register_model`).
# - Pre-processor: Prompt templating and tokenization.
# - Router, defaulting to round-robin. Use --router-mode to switch
#   (round-robin, random, kv, direct, least-loaded, device-aware-weighted).
#
# Pass `--interactive` or `-i` for text chat instead of HTTP server.
#
# For TLS:
# - python -m dynamo.frontend --http-port 8443 --tls-cert-path cert.pem --tls-key-path key.pem
#

import argparse
import asyncio
import importlib.metadata
import logging
import os
import signal
import sys
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Optional

import uvloop

from dynamo.common.config_dump import dump_config
from dynamo.llm import (
    AicPerfConfig,
    EngineType,
    EntrypointArgs,
    FrontendRoute,
    KvRouterConfig,
    RouterConfig,
    RouterMode,
    make_engine,
    run_input,
)
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

from .frontend_args import FrontendArgGroup, FrontendConfig

if TYPE_CHECKING:
    from .vllm_processor import EngineFactory

configure_dynamo_logging()
logger = logging.getLogger(__name__)

MIN_INITIAL_WORKERS_ENV = "DYN_ROUTER_MIN_INITIAL_WORKERS"
FRONTEND_ROUTE_ENTRYPOINT_GROUP = "dynamo.frontend_routes"


def setup_engine_factory(
    config: FrontendConfig,
    vllm_flags: Namespace,
) -> "EngineFactory":
    """
    When using vllm pre and post processor, create the EngineFactory that
    creates the engines that run requests.
    """
    from .vllm_processor import EngineFactory

    return EngineFactory(config, vllm_flags)


def setup_sglang_engine_factory(
    config: FrontendConfig,
    sglang_flags: Optional[Namespace] = None,
):
    """
    When using sglang pre and post processor, create the SglangEngineFactory
    that creates the engines that run requests.
    """
    from .sglang_processor import SglangEngineFactory

    tool_call_parser = getattr(sglang_flags, "tool_call_parser", None)
    reasoning_parser = getattr(sglang_flags, "reasoning_parser", None)
    chat_template = getattr(sglang_flags, "chat_template", None)

    return SglangEngineFactory(
        config,
        debug_perf=config.debug_perf,
        tool_call_parser_name=tool_call_parser,
        reasoning_parser_name=reasoning_parser,
        chat_template=chat_template,
    )


def _frontend_route_extension_entry_points():
    """Return the entry points registered under the frontend-route group.

    Handles both the ``EntryPoints.select`` API (Python >= 3.10) and the older
    mapping-style ``entry_points().get`` fallback.
    """
    entry_points = importlib.metadata.entry_points()
    if hasattr(entry_points, "select"):
        return list(entry_points.select(group=FRONTEND_ROUTE_ENTRYPOINT_GROUP))
    return list(entry_points.get(FRONTEND_ROUTE_ENTRYPOINT_GROUP, ()))


def _normalize_frontend_routes(
    extension_name: str, provided: Any
) -> list[FrontendRoute]:
    """Coerce a provider's return value into a list of ``FrontendRoute``.

    Accepts a single ``FrontendRoute`` or any iterable of them, and raises
    ``TypeError`` if the provider returns anything else.
    """
    if provided is None:
        return []
    if isinstance(provided, FrontendRoute):
        return [provided]
    try:
        routes = list(provided)
    except TypeError as exc:
        raise TypeError(
            f"Frontend route extension '{extension_name}' must return a FrontendRoute "
            "or an iterable of FrontendRoute objects"
        ) from exc
    for route in routes:
        if not isinstance(route, FrontendRoute):
            raise TypeError(
                f"Frontend route extension '{extension_name}' returned unsupported route "
                f"object {route!r}; expected dynamo.llm.FrontendRoute"
            )
    return routes


def load_frontend_route_extensions(extension_names: list[str]) -> list[FrontendRoute]:
    """Load trusted frontend route extensions by entry point name."""

    if not extension_names:
        return []

    # Deduplicate names (repeated --frontend-route-extension, or a flag that
    # overlaps DYN_FRONTEND_ROUTE_EXTENSIONS) so a provider is loaded and its
    # routes appended only once, preserving first-seen order.
    unique_names = list(dict.fromkeys(extension_names))

    entry_points = _frontend_route_extension_entry_points()
    routes: list[FrontendRoute] = []
    for extension_name in unique_names:
        matches = [ep for ep in entry_points if ep.name == extension_name]
        if not matches:
            available = ", ".join(sorted(ep.name for ep in entry_points)) or "<none>"
            raise ValueError(
                f"Unknown frontend route extension '{extension_name}' in entry point "
                f"group '{FRONTEND_ROUTE_ENTRYPOINT_GROUP}'. Available: {available}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous frontend route extension '{extension_name}' in entry point "
                f"group '{FRONTEND_ROUTE_ENTRYPOINT_GROUP}'"
            )
        provider = matches[0].load()
        if not callable(provider):
            raise TypeError(
                f"Frontend route extension '{extension_name}' entry point must load a callable"
            )
        routes.extend(_normalize_frontend_routes(extension_name, provider()))

    return routes


def parse_args() -> tuple[FrontendConfig, Optional[Namespace], Optional[Namespace]]:
    """Parse command-line arguments for the Dynamo frontend.

    Returns:
        Tuple of (FrontendConfig, vllm_flags, sglang_flags).
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
    sglang_flags = None

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

        # On a host with no GPU and a CUDA-built wheel, vllm.platforms
        # auto-detection picks UnspecifiedPlatform and the
        # AsyncEngineArgs.add_cli_args call below crashes inside
        # DeviceConfig.__post_init__. Frontend uses vLLM for parsers
        # only and never constructs an engine, so coerce CpuPlatform.
        # Must run before importing vllm.engine.arg_utils, which binds
        # current_platform at module scope.
        import vllm.platforms

        if vllm.platforms.current_platform.device_type == "":
            from vllm.platforms.cpu import CpuPlatform

            vllm.platforms.current_platform = CpuPlatform()

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
    elif config.chat_processor == "sglang":
        sglang_parser = argparse.ArgumentParser(add_help=False)
        sglang_parser.add_argument("--tool-call-parser", default=None)
        sglang_parser.add_argument("--reasoning-parser", default=None)
        sglang_parser.add_argument("--chat-template", default=None)
        sglang_flags, remaining = sglang_parser.parse_known_args(unknown)
        if remaining:
            logger.error(f"Unknown arguments specified: {remaining}")
            sys.exit(1)
    else:
        if unknown:
            logger.error(f"Unknown arguments specified: {unknown}")
            sys.exit(1)
    return config, vllm_flags, sglang_flags


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
    config, vllm_flags, sglang_flags = parse_args()
    dump_config(config.dump_config_to, config)
    max_seq_info = (
        f", max_seq_len: {config.migration_max_seq_len}"
        if config.migration_max_seq_len is not None
        else ""
    )
    logger.info(
        f"Request migration {'enabled' if config.migration_limit > 0 else 'disabled'} "
        f"(limit: {config.migration_limit}{max_seq_info})"
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

    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(
        loop,
        config.discovery_backend,
        config.request_plane,
        event_plane=config.event_plane,
    )

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    if config.router_mode == "kv":
        router_mode = RouterMode.KV
        kv_router_config = KvRouterConfig(**config.kv_router_kwargs())
    elif config.router_mode == "random":
        router_mode = RouterMode.Random
        kv_router_config = None
    elif config.router_mode == "direct":
        router_mode = RouterMode.Direct
        kv_router_config = None
    elif config.router_mode == "power-of-two":
        router_mode = RouterMode.PowerOfTwoChoices
        kv_router_config = None
    elif config.router_mode == "least-loaded":
        router_mode = RouterMode.LeastLoaded
        kv_router_config = None
    elif config.router_mode == "device-aware-weighted":
        router_mode = RouterMode.DeviceAwareWeighted
        kv_router_config = None
    else:
        router_mode = RouterMode.RoundRobin
        kv_router_config = None

    os.environ[MIN_INITIAL_WORKERS_ENV] = str(config.min_initial_workers)
    router_config = RouterConfig(
        router_mode, kv_router_config, **config.router_kwargs()
    )

    metrics_prefix = (
        config.metrics_prefix
        if config.metrics_prefix is not None and config.metrics_prefix.strip()
        else None
    )
    kwargs: dict[str, Any] = {
        "http_host": config.http_host,
        "http_port": config.http_port,
        "kv_cache_block_size": config.kv_cache_block_size,
        "router_config": router_config,
        "migration_limit": config.migration_limit,
        "metrics_prefix": metrics_prefix,
        "enable_anthropic_api": config.enable_anthropic_api,
        "strip_anthropic_preamble": config.strip_anthropic_preamble,
        "enable_streaming_tool_dispatch": config.enable_streaming_tool_dispatch,
        "enable_streaming_reasoning_dispatch": config.enable_streaming_reasoning_dispatch,
        "tokenizer_backend": config.tokenizer_backend,
    }
    if config.migration_max_seq_len is not None:
        kwargs["migration_max_seq_len"] = config.migration_max_seq_len

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

    if config.chat_processor == "vllm":
        assert (
            vllm_flags is not None
        ), "vllm_flags is required when chat processor is vllm"
        chat_engine_factory = setup_engine_factory(
            config, vllm_flags
        ).chat_engine_factory
        kwargs["chat_engine_factory"] = chat_engine_factory
    elif config.chat_processor == "sglang":
        chat_engine_factory = setup_sglang_engine_factory(
            config, sglang_flags
        ).chat_engine_factory
        kwargs["chat_engine_factory"] = chat_engine_factory

    if config.router_prefill_load_model == "aic":
        kwargs["aic_perf_config"] = AicPerfConfig(**config.aic_perf_kwargs())

    e = EntrypointArgs(EngineType.Dynamic, **kwargs)
    engine = await make_engine(runtime, e)
    # Validate mode compatibility before loading extensions, so an incompatible
    # mode fails fast without importing/executing third-party provider code.
    if config.frontend_route_extensions and (
        config.interactive or config.kserve_grpc_server
    ):
        raise ValueError(
            "frontend route extensions are only supported by HTTP frontend mode"
        )
    frontend_route_extensions = load_frontend_route_extensions(
        config.frontend_route_extensions
    )

    try:
        if config.interactive:
            await run_input(runtime, "text", engine)
        elif config.kserve_grpc_server:
            await run_input(runtime, "grpc", engine)
        else:
            await run_input(runtime, "http", engine, frontend_route_extensions)
    except asyncio.exceptions.CancelledError:
        pass


async def graceful_shutdown(runtime: DistributedRuntime) -> None:
    """Handle graceful shutdown of the distributed runtime.

    Args:
        runtime: The DistributedRuntime instance to shut down.
    """
    runtime.shutdown()


def main() -> None:
    """Entry point for the Dynamo frontend CLI."""
    uvloop.run(async_main())


if __name__ == "__main__":
    main()
