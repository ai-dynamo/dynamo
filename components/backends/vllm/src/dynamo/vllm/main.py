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


async def init_prefill(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    engine_client, _, default_sampling_params = setup_vllm_engine(config)

    # TODO register_prefill in similar vein to register_llm

    handler = PrefillWorkerHandler(
        runtime, component, engine_client, default_sampling_params
    )

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

    handler = DecodeWorkerHandler(
        runtime,
        component,
        engine_client,
        default_sampling_params,
        prefill_worker_client,
    )

    if config.engine_args.enable_prefix_caching:
        # TODO: We start off with a valid endpoint, then we increment it by dp_rank
        # May no longer be valid. Lets remove the increment behavior from vLLM and here
        zmq_endpoint = ZmqEventPublisher.offset_endpoint_port(
            config.engine_args.kv_events_config.endpoint,
            data_parallel_rank=config.engine_args.data_parallel_rank or 0,
        ).replace("*", "127.0.0.1")

        zmq_config = ZmqKvEventPublisherConfig(
            worker_id=generate_endpoint.lease_id(),
            kv_block_size=vllm_config.cache_config.block_size,
            zmq_endpoint=zmq_endpoint,
        )
        kv_publisher = ZmqKvEventPublisher(component=component, config=zmq_config)

        logger.info(f"Reading Events from {zmq_endpoint}")

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
