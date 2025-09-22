# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import signal

import uvloop
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    register_llm,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .args import ENABLE_LMCACHE, Config, configure_ports, overwrite_args, parse_args
from .handlers import DecodeWorkerHandler, PrefillWorkerHandler
from .health_check import VllmHealthCheckPayload
from .publisher import StatLoggerFactory

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def setup_lmcache_environment():
    """Setup LMCache environment variables for KV cache offloading"""
    # LMCache configuration for matching logic
    lmcache_config = {
        "LMCACHE_CHUNK_SIZE": "256",  # Token chunk size
        "LMCACHE_LOCAL_CPU": "True",  # Enable CPU memory backend
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "20",  # CPU memory limit in GB
    }

    # Set environment variables
    for key, value in lmcache_config.items():
        if key not in os.environ:  # Only set if not already configured
            os.environ[key] = value
            logger.info(f"Set LMCache environment variable: {key}={value}")


async def graceful_shutdown(runtime):
    """
    Shutdown dynamo distributed runtime.
    The endpoints will be immediately invalidated so no new requests will be accepted.
    For endpoints served with graceful_shutdown=True, the serving function will wait until all in-flight requests are finished.
    For endpoints served with graceful_shutdown=False, the serving function will return immediately.
    """
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    config = parse_args()

    await configure_ports(runtime, config)
    overwrite_args(config)

    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.debug("Signal handlers set up for graceful shutdown")

    if config.is_prefill_worker:
        await init_prefill(runtime, config)
        logger.debug("init_prefill completed")
    else:
        await init(runtime, config)
        logger.debug("init completed")

    logger.debug("Worker function completed, exiting...")


def setup_vllm_engine(config, stat_logger=None):
    os.environ["VLLM_NO_USAGE_STATS"] = "1"  # Avoid internal HTTP requests
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    engine_args = config.engine_args

    # KV transfer config is now handled by args.py based on ENABLE_LMCACHE env var
    if ENABLE_LMCACHE:
        setup_lmcache_environment()
        logger.info("LMCache enabled for VllmWorker")
    else:
        logger.debug("LMCache is disabled")

    # Load default sampling params from `generation_config.json`
    default_sampling_params = (
        engine_args.create_model_config().get_diff_sampling_param()
    )

    # Taken from build_async_engine_client_from_engine_args()
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    factory = []
    if stat_logger:
        factory.append(stat_logger)

    engine_client = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=usage_context,
        stat_loggers=factory,
        disable_log_requests=engine_args.disable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )
    if ENABLE_LMCACHE:
        logger.info(f"VllmWorker for {config.model} has been initialized with LMCache")
    else:
        logger.info(f"VllmWorker for {config.model} has been initialized")
    return engine_client, vllm_config, default_sampling_params


def _maybe_setup_kv_publisher(component, endpoint, vllm_config, engine_args):
    """Create and return a ZMQ KV events publisher if prefix caching is enabled.

    DESIGN DECISION: Unified KV publisher setup for both decode and prefill workers.
    This shared helper ensures consistent KV cache event publishing across all worker types,
    enabling the router to track cache states from both prefill-only and decode workers.

    The KV publisher sends cache block allocation/deallocation events to the router's
    indexer, which maintains a global view of cache states across all workers for
    intelligent routing decisions.

    Args:
        component: Dynamo component for NATS communication
        endpoint: Worker endpoint with lease_id for unique worker identification
        vllm_config: vLLM configuration containing cache settings
        engine_args: Engine arguments including prefix caching settings

    Returns:
        ZmqKvEventPublisher instance if prefix caching enabled, None otherwise
    """
    if not engine_args.enable_prefix_caching:
        logger.debug("Prefix caching disabled; KV publisher not started")
        return None

    zmq_endpoint = ZmqEventPublisher.offset_endpoint_port(
        engine_args.kv_events_config.endpoint,
        data_parallel_rank=engine_args.data_parallel_rank or 0,
    ).replace("*", "127.0.0.1")

    zmq_config = ZmqKvEventPublisherConfig(
        worker_id=endpoint.lease_id(),
        kv_block_size=vllm_config.cache_config.block_size,
        zmq_endpoint=zmq_endpoint,
    )
    kv_publisher = ZmqKvEventPublisher(component=component, config=zmq_config)
    logger.info(f"Reading Events from {zmq_endpoint}")
    return kv_publisher


async def init_prefill(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    engine_client, vllm_config, default_sampling_params = setup_vllm_engine(config)

    # TODO register_prefill in similar vein to register_llm

    handler = PrefillWorkerHandler(
        runtime, component, engine_client, default_sampling_params
    )

    # Setup KV publisher for prefill worker (consistent with decode workers)
    # This enables the router to track KV cache states from prefill workers
    kv_publisher = _maybe_setup_kv_publisher(
        component, generate_endpoint, vllm_config, config.engine_args
    )
    if kv_publisher:
        handler.kv_publisher = kv_publisher

    # CRITICAL CHANGE: Register prefill workers with the router discovery system
    # Previously, only decode workers were registered, making them invisible to the router.
    # This registration enables the router to:
    # 1. Discover prefill workers as available instances
    # 2. Track their runtime capabilities (memory, slots, batching limits)
    # 3. Include them in intelligent routing decisions
    # 4. Monitor their load and availability
    try:
        from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_llm

        # Create runtime configuration matching the actual vLLM worker capabilities
        # This provides the router with essential information for load balancing:
        runtime_config = ModelRuntimeConfig()
        runtime_config.total_kv_blocks = (
            vllm_config.cache_config.num_gpu_blocks
        )  # GPU memory capacity
        runtime_config.max_num_seqs = (
            vllm_config.scheduler_config.max_num_seqs
        )  # Concurrent request limit
        runtime_config.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )  # Batch size limit

        # Register with the same parameters as decode workers to ensure compatibility
        # The router will distinguish prefill vs decode workers by component name ("prefill" vs "backend")
        await register_llm(
            ModelInput.Tokens,
            ModelType.Chat | ModelType.Completions,
            generate_endpoint,
            config.model,
            config.served_model_name,
            kv_cache_block_size=config.engine_args.block_size,
            migration_limit=0,  # Prefill workers don't support migration
            runtime_config=runtime_config,
            custom_template_path=config.custom_jinja_template,
        )
    except Exception:
        # Non-fatal: prefill workers can still operate without router awareness
        logger.exception("Prefill registration failed (continuing)")

    # Get health check payload (checks env var and falls back to vLLM default)
    health_check_payload = VllmHealthCheckPayload().to_dict()

    try:
        logger.debug("Starting serve_endpoint for prefill worker")
        await asyncio.gather(
            # for prefill, we want to shutdown the engine after all prefill requests are finished because
            #     (temp reason): we don't support re-routing prefill requests
            #     (long-term reason): prefill engine should pull from a global queue so there is
            #                         only a few in-flight requests that can be quickly finished
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("model", config.model)],
                health_check_payload=health_check_payload,
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks, metrics_labels=[("model", config.model)]
            ),
        )
        logger.debug("serve_endpoint completed for prefill worker")
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        logger.debug("Cleaning up prefill worker")
        handler.cleanup()


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    prefill_worker_client = (
        await runtime.namespace(config.namespace)
        .component("prefill")  # TODO don't hardcode
        .endpoint("generate")
        .client()
    )

    # NEW: Create NATS client for intelligent prefill worker selection in background
    # This prevents startup blocking if NATS is slow/unavailable while enabling
    # graceful upgrade from random to intelligent selection when router becomes available
    import asyncio

    from dynamo.llm import FrontendRouterClient

    # Shared container that handler can check dynamically
    class RouterClientContainer:
        def __init__(self):
            self.client = None
            self.ready = False

    router_container = RouterClientContainer()

    async def setup_router_client():
        """Background task to create router client without blocking startup."""
        try:
            # FrontendRouterClient provides NATS-based communication with the KV router
            # It sends prefill selection requests and receives intelligent worker recommendations
            # based on KV cache overlap analysis and load balancing
            client = FrontendRouterClient(component)
            router_container.client = client
            router_container.ready = True
            logger.info(
                "Frontend router NATS client created for intelligent prefill selection"
            )
        except Exception as e:
            # Non-fatal: system continues with random prefill selection
            logger.warning(
                f"Failed to create frontend router client: {e}. Continuing with random prefill selection."
            )

    # Start router client creation in background - doesn't block startup
    asyncio.create_task(setup_router_client())

    factory = StatLoggerFactory(
        component,
        config.engine_args.data_parallel_rank or 0,
        metrics_labels=[("model", config.model)],
    )
    engine_client, vllm_config, default_sampling_params = setup_vllm_engine(
        config, factory
    )

    # TODO Hack to get data, move this to registering in TBD
    factory.set_num_gpu_blocks_all(vllm_config.cache_config.num_gpu_blocks)
    factory.set_request_total_slots_all(vllm_config.scheduler_config.max_num_seqs)
    factory.init_publish()

    logger.info(f"VllmWorker for {config.model} has been initialized")

    # Create decode worker handler with intelligent prefill routing capability
    # The router_container enables KV cache-aware prefill worker selection when ready
    handler = DecodeWorkerHandler(
        runtime,
        component,
        engine_client,
        default_sampling_params,
        prefill_worker_client,
        router_container,  # NEW: Container that will be populated by background task
    )

    # Setup KV publisher for decode worker (consistent with prefill workers)
    # This enables the router to track KV cache states from decode workers
    kv_publisher = _maybe_setup_kv_publisher(
        component, generate_endpoint, vllm_config, config.engine_args
    )
    if kv_publisher:
        handler.kv_publisher = kv_publisher

    if not config.engine_args.data_parallel_rank:  # if rank is 0 or None then register
        runtime_config = ModelRuntimeConfig()

        # make a `collective_rpc` call to get runtime configuration values
        logging.info(
            "Getting engine runtime configuration metadata from vLLM engine..."
        )
        runtime_values = get_engine_cache_info(engine_client)
        runtime_config.total_kv_blocks = runtime_values["num_gpu_blocks"]
        runtime_config.max_num_seqs = runtime_values["max_num_seqs"]
        runtime_config.max_num_batched_tokens = runtime_values["max_num_batched_tokens"]
        runtime_config.tool_call_parser = config.tool_call_parser
        runtime_config.reasoning_parser = config.reasoning_parser

        await register_llm(
            ModelInput.Tokens,
            ModelType.Chat | ModelType.Completions,
            generate_endpoint,
            config.model,
            config.served_model_name,
            kv_cache_block_size=config.engine_args.block_size,
            migration_limit=config.migration_limit,
            runtime_config=runtime_config,
            custom_template_path=config.custom_jinja_template,
        )

    # Get health check payload (checks env var and falls back to vLLM default)
    health_check_payload = VllmHealthCheckPayload().to_dict()

    try:
        logger.debug("Starting serve_endpoint for decode worker")
        await asyncio.gather(
            # for decode, we want to transfer the in-flight requests to other decode engines,
            # because waiting them to finish can take a long time for long OSLs
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=config.migration_limit <= 0,
                metrics_labels=[("model", config.model)],
                health_check_payload=health_check_payload,
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks, metrics_labels=[("model", config.model)]
            ),
        )
        logger.debug("serve_endpoint completed for decode worker")
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        logger.debug("Cleaning up decode worker")
        # Cleanup background tasks
        handler.cleanup()


def get_engine_cache_info(engine: AsyncLLM):
    """Retrieve cache configuration information from [`AsyncLLM`] engine."""

    try:
        # Get values directly from vllm_config instead of collective_rpc
        cache_values = {
            "num_gpu_blocks": engine.vllm_config.cache_config.num_gpu_blocks,
        }

        scheduler_values = {
            "max_num_seqs": engine.vllm_config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": engine.vllm_config.scheduler_config.max_num_batched_tokens,
        }

        logging.info(f"Cache config values: {cache_values}")
        logging.info(f"Scheduler config values: {scheduler_values}")
        return {
            "num_gpu_blocks": cache_values["num_gpu_blocks"],
            "max_num_seqs": scheduler_values["max_num_seqs"],
            "max_num_batched_tokens": scheduler_values["max_num_batched_tokens"],
        }
    except Exception as e:
        logging.error(f"Failed to get configuration values from vLLM config: {e}")
        raise


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
