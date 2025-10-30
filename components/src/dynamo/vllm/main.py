# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import signal

import uvloop
from prometheus_client import REGISTRY
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus

from dynamo.common.config_dump import dump_config
from dynamo.common.utils.prometheus import register_engine_metrics_callback
from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    fetch_llm,
    register_llm,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.multimodal_handlers import (
    EncodeWorkerHandler,
    MultimodalPDWorkerHandler,
    ProcessorHandler,
)

from .args import ENABLE_LMCACHE, Config, configure_ports, overwrite_args, parse_args
from .handlers import DecodeWorkerHandler, PrefillWorkerHandler
from .health_check import VllmHealthCheckPayload, VllmPrefillHealthCheckPayload
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

    dump_config(config.dump_config_to, config)

    # Download the model if necessary.
    # register_llm would do this for us, but we want it on disk before we start vllm.
    # Ensure the original HF name (e.g. "Qwen/Qwen3-0.6B") is used as the served_model_name.
    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model
    if not os.path.exists(config.model):
        config.model = config.engine_args.model = await fetch_llm(config.model)

    # Route to appropriate initialization based on config flags
    if config.multimodal_processor:
        await init_multimodal_processor(runtime, config)
        logger.debug("init_multimodal_processor completed")
    elif config.multimodal_encode_worker:
        await init_multimodal_encode_worker(runtime, config)
        logger.debug("init_multimodal_encode_worker completed")
    elif config.multimodal_worker:
        await init_multimodal_worker(runtime, config)
        logger.debug("init_multimodal_worker completed")
    elif config.is_prefill_worker:
        await init_prefill(runtime, config)
        logger.debug("init_prefill completed")
    else:
        await init(runtime, config)
        logger.debug("init completed")

    logger.debug("Worker function completed, exiting...")


def setup_kv_event_publisher(
    config: Config,
    component,
    generate_endpoint,
    vllm_config,
):
    """
    Set up KV event publishers for prefix caching if enabled.
    Creates one publisher per dp_rank since each dp_rank publishes to a different port.

    Returns:
        List of ZmqKvEventPublisher instances (one per dp_rank) if prefix caching is enabled, None otherwise.
    """
    if not config.engine_args.enable_prefix_caching:
        return None

    # Skip KV event publishing for decode workers
    if config.is_decode_worker:
        logger.info("Skipping KV event publisher setup for decode worker")
        return None

    # Get data_parallel_size to create publishers for all dp_ranks
    data_parallel_size = getattr(vllm_config.parallel_config, "data_parallel_size", 1)
    kv_publishers = []

    for dp_rank in range(data_parallel_size):
        # Each dp_rank publishes to a different port
        zmq_endpoint = ZmqEventPublisher.offset_endpoint_port(
            config.engine_args.kv_events_config.endpoint,
            data_parallel_rank=dp_rank,
        ).replace("*", "127.0.0.1")

        zmq_config = ZmqKvEventPublisherConfig(
            worker_id=generate_endpoint.connection_id(),
            kv_block_size=vllm_config.cache_config.block_size,
            zmq_endpoint=zmq_endpoint,
        )
        kv_publisher = ZmqKvEventPublisher(component=component, config=zmq_config)
        kv_publishers.append(kv_publisher)

        logger.info(
            f"Worker reading KV events for dp_rank={dp_rank} from {zmq_endpoint}"
        )

    return kv_publishers if kv_publishers else None


def setup_vllm_engine(config, stat_logger=None):
    setup_multiprocess_prometheus()
    logger.debug(
        f"Prometheus multiproc dir set to: {os.environ.get('PROMETHEUS_MULTIPROC_DIR')}"
    )

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
        logger.info(
            f"VllmWorker for {config.served_model_name} has been initialized with LMCache"
        )
    else:
        logger.info(f"VllmWorker for {config.served_model_name} has been initialized")

    return engine_client, vllm_config, default_sampling_params


async def register_vllm_model(
    model_input: ModelInput,
    model_type: ModelType,
    generate_endpoint,
    config: Config,
    engine_client: AsyncLLM,
    vllm_config,
    migration_limit: int,
):
    """
    Helper function to register a vLLM model with runtime configuration.

    Args:
        model_input: Input type for the model (e.g., ModelInput.Tokens)
        model_type: Type of model (e.g., ModelType.Chat, ModelType.Prefill)
        generate_endpoint: Endpoint to register
        config: Configuration object
        engine_client: vLLM engine client
        vllm_config: vLLM configuration
        migration_limit: Migration limit for the model
    """
    runtime_config = ModelRuntimeConfig()

    # Get runtime configuration from vLLM engine
    logging.info(
        f"Getting engine runtime configuration metadata from vLLM engine for {model_type}..."
    )
    runtime_values = get_engine_cache_info(engine_client)
    runtime_config.total_kv_blocks = runtime_values["num_gpu_blocks"]
    runtime_config.max_num_seqs = runtime_values["max_num_seqs"]
    runtime_config.max_num_batched_tokens = runtime_values["max_num_batched_tokens"]

    # Add tool/reasoning parsers for decode models
    if model_type != ModelType.Prefill:
        runtime_config.tool_call_parser = config.tool_call_parser
        runtime_config.reasoning_parser = config.reasoning_parser

    # Get data_parallel_size from vllm_config (defaults to 1)
    data_parallel_size = getattr(vllm_config.parallel_config, "data_parallel_size", 1)
    runtime_config.data_parallel_size = data_parallel_size

    await register_llm(
        model_input,
        model_type,
        generate_endpoint,
        config.model,
        config.served_model_name,
        kv_cache_block_size=config.engine_args.block_size,
        migration_limit=migration_limit,
        runtime_config=runtime_config,
        custom_template_path=config.custom_jinja_template,
    )


async def init_prefill(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    engine_client, vllm_config, default_sampling_params = setup_vllm_engine(config)

    handler = PrefillWorkerHandler(
        runtime, component, engine_client, default_sampling_params
    )

    # Set up KV event publishers for prefix caching if enabled (one per dp_rank)
    kv_publishers = setup_kv_event_publisher(
        config, component, generate_endpoint, vllm_config
    )
    if kv_publishers:
        handler.kv_publishers = kv_publishers

    if config.engine_args.disable_log_stats is False:
        register_engine_metrics_callback(
            endpoint=generate_endpoint, registry=REGISTRY, metric_prefix_filter="vllm:"
        )

    # Register prefill model with ModelType.Prefill
    if not config.engine_args.data_parallel_rank:  # if rank is 0 or None then register
        await register_vllm_model(
            ModelInput.Tokens,
            ModelType.Prefill,
            generate_endpoint,
            config,
            engine_client,
            vllm_config,
            migration_limit=0,  # Prefill doesn't support migration
        )

    health_check_payload = VllmPrefillHealthCheckPayload(engine_client).to_dict()

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
                # In practice config.served_model_name is always set, but mypy needs the "or" here.
                metrics_labels=[("model", config.served_model_name or config.model)],
                health_check_payload=health_check_payload,
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks,
                metrics_labels=[("model", config.served_model_name)],
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

    factory = StatLoggerFactory(
        component,
        config.engine_args.data_parallel_rank or 0,
        metrics_labels=[("model", config.served_model_name or config.model)],
    )
    engine_client, vllm_config, default_sampling_params = setup_vllm_engine(
        config, factory
    )

    # TODO Hack to get data, move this to registering in TBD
    factory.set_num_gpu_blocks_all(vllm_config.cache_config.num_gpu_blocks)
    factory.set_request_total_slots_all(vllm_config.scheduler_config.max_num_seqs)
    factory.init_publish()

    handler = DecodeWorkerHandler(
        runtime,
        component,
        engine_client,
        default_sampling_params,
    )

    # Set up KV event publishers for prefix caching if enabled (one per dp_rank)
    kv_publishers = setup_kv_event_publisher(
        config, component, generate_endpoint, vllm_config
    )
    if kv_publishers:
        handler.kv_publishers = kv_publishers

    if config.engine_args.disable_log_stats is False:
        register_engine_metrics_callback(
            endpoint=generate_endpoint, registry=REGISTRY, metric_prefix_filter="vllm:"
        )

    if not config.engine_args.data_parallel_rank:  # if rank is 0 or None then register
        await register_vllm_model(
            ModelInput.Tokens,
            ModelType.Chat | ModelType.Completions,
            generate_endpoint,
            config,
            engine_client,
            vllm_config,
            migration_limit=config.migration_limit,
        )

    health_check_payload = VllmHealthCheckPayload(engine_client).to_dict()

    try:
        logger.debug("Starting serve_endpoint for decode worker")
        await asyncio.gather(
            # for decode, we want to transfer the in-flight requests to other decode engines,
            # because waiting them to finish can take a long time for long OSLs
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=config.migration_limit <= 0,
                metrics_labels=[("model", config.served_model_name or config.model)],
                health_check_payload=health_check_payload,
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks,
                metrics_labels=[("model", config.served_model_name or config.model)],
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


async def init_multimodal_processor(runtime: DistributedRuntime, config: Config):
    """Initialize multimodal processor component"""
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)

    # Get encode worker client
    encode_worker_client = (
        await runtime.namespace(config.namespace)
        .component("encoder")
        .endpoint("generate")
        .client()
    )

    # Get prompt template from args (must be passed via environment or command line)
    mm_prompt_template = config.mm_prompt_template

    handler = ProcessorHandler(
        config.engine_args,
        encode_worker_client,
        mm_prompt_template,
    )

    logger.info("Waiting for Encoder Worker Instances ...")
    await encode_worker_client.wait_for_instances()

    # Register the endpoint as entrypoint to a model
    await register_llm(
        ModelInput.Text,  # Custom processor is used and this type bypasses SDK processor
        ModelType.Chat,
        generate_endpoint,
        config.model,
        config.served_model_name,
        kv_cache_block_size=config.engine_args.block_size,
    )

    logger.info("Starting to serve the processor endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=[("model", config.model)]
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init_multimodal_encode_worker(runtime: DistributedRuntime, config: Config):
    """Initialize multimodal encode worker component"""
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)

    # Get PD worker client
    # In multimodal mode, the PD worker always registers as "backend"
    # (even in disaggregated mode with prefill/decode split, we still connect to "backend")
    pd_worker_client = (
        await runtime.namespace(config.namespace)
        .component("backend")
        .endpoint("generate")
        .client()
    )

    handler = EncodeWorkerHandler(
        config.engine_args,
        pd_worker_client,
    )
    await handler.async_init(runtime)
    logger.info("Waiting for PD Worker Instances ...")
    await pd_worker_client.wait_for_instances()
    logger.info("Starting to serve the encode worker endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=[("model", config.model)]
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init_multimodal_worker(runtime: DistributedRuntime, config: Config):
    """Initialize multimodal worker component for aggregated or disaggregated mode"""

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    engine_client, vllm_config, default_sampling_params = setup_vllm_engine(config)

    # TODO: Support Disaggregated mode separately
    client = (
        await runtime.namespace(config.namespace)
        .component("backend")
        .endpoint("generate")
        .client()
    )

    handler = MultimodalPDWorkerHandler(
        runtime, component, engine_client, config, client
    )

    await handler.async_init(runtime)

    # Set up KV event publisher for prefix caching if enabled
    kv_publisher = setup_kv_event_publisher(
        config, component, generate_endpoint, vllm_config
    )
    if kv_publisher:
        handler.kv_publisher = kv_publisher

    metrics_labels = [("model", config.model)]

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=metrics_labels
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks, metrics_labels=metrics_labels
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
