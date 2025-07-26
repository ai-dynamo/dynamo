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
    ModelType,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    register_llm,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .args import Config, configure_ports_with_etcd, overwrite_args, parse_args
from .handlers import DecodeWorkerHandler, PrefillWorkerHandler
from .publisher import StatLoggerFactory

configure_dynamo_logging()
logger = logging.getLogger(__name__)

def setup_lmcache_environment():
    """Setup LMCache environment variables for KV cache offloading
    
    When ENABLE_LMCACHE is enabled, LMCache will be used with MultiConnector:
    - LMCacheConnector: Handles get_num_new_matched_tokens (cache matching logic)
    - NixlConnector: Handles actual KV cache transfer between workers
    """
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
    By calling `runtime.shutdown()`, the endpoints will immediately be unavailable.
    However, in-flight requests will still be processed until they are finished.
    After all in-flight requests are finished, the `serve_endpoint` functions will return
    and the engine will be shutdown by Python's garbage collector.
    """
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    config = parse_args()

    etcd_client = runtime.etcd_client()
    await configure_ports_with_etcd(config, etcd_client)
    overwrite_args(config)

    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    if config.is_prefill_worker:
        await init_prefill(runtime, config)
    else:
        await init(runtime, config)


def setup_vllm_engine(config, stat_logger=None):
    os.environ["VLLM_NO_USAGE_STATS"] = "1"  # Avoid internal HTTP requests
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    engine_args = config.engine_args

    # KV transfer config is now handled by args.py based on ENABLE_LMCACHE env var
    # Check if LMCache is enabled
    enable_lmcache = os.getenv("ENABLE_LMCACHE", "0").lower() in ("1", "true", "yes")

    if enable_lmcache:
        setup_lmcache_environment()
        logger.info("LMCache enabled for VllmWorker")
    else:
        logger.info("LMCache is disabled")

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
    if enable_lmcache:
        logger.info(f"VllmWorker for {config.model} has been initialized with LMCache")
    else:
        logger.info(f"VllmWorker for {config.model} has been initialized")
    return engine_client, vllm_config, default_sampling_params


async def init_prefill(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
# Check if LMCache is enabled
    enable_lmcache = os.getenv("ENABLE_LMCACHE", "0").lower() in ("1", "true", "yes")
    enable_lmcache_disag = os.getenv("ENABLE_LMCACHE_DISAG", "0").lower() in ("1", "true", "yes")
    enable_lmcache_connector = os.getenv("ENABLE_LMCACHE_CONNECTOR", "0").lower() in ("1", "true", "yes")
    if enable_lmcache and enable_lmcache_disag and enable_lmcache_connector:
        # NIXL configuration for KV cache transfer
        # enable_nixl: True
        # enable_xpyd: True
        # nixl_role: "sender"
        # nixl_proxy_host: "localhost"
        # nixl_proxy_port: 7500
        # nixl_buffer_size: 1073741824 # 1GB
        # nixl_buffer_device: "cuda"
        lmcache_prefill_config = {
            "enable_nixl": "True",
            "enable_xpyd": "True",
            "nixl_role": "sender",
            "nixl_proxy_host": "localhost",
            "nixl_proxy_port": "7500",
            "nixl_buffer_size": "1073741824",
            "nixl_buffer_device": "cuda",
        }
        # set the environment variables
        for key, value in lmcache_prefill_config.items():
            if key not in os.environ:  # Only set if not already configured
                os.environ[key] = value
                logging.info(f"Set LMCache environment variable: {key}={value}")

        # Force local cpu is true
        os.environ["LMCACHE_CHUNK_SIZE"] = "256"
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "30"
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    engine_client, _, default_sampling_params = setup_vllm_engine(config)

    # TODO register_prefill in similar vein to register_llm

    handler = PrefillWorkerHandler(component, engine_client, default_sampling_params)

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(handler.generate),
            clear_endpoint.serve_endpoint(handler.clear_kv_blocks),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    # Check if LMCache is enabled
    enable_lmcache = os.getenv("ENABLE_LMCACHE", "0").lower() in ("1", "true", "yes")
    enable_lmcache_disag = os.getenv("ENABLE_LMCACHE_DISAG", "0").lower() in ("1", "true", "yes")
    enable_lmcache_connector = os.getenv("ENABLE_LMCACHE_CONNECTOR", "0").lower() in ("1", "true", "yes")
    if enable_lmcache and enable_lmcache_disag and enable_lmcache_connector:
        # NIXL configuration for KV cache transfer
        # enable_nixl: True
        # enable_xpyd: True
        # nixl_role: "receiver"
        # nixl_peer_host: "localhost"
        # nixl_peer_init_port: 7300
        # nixl_peer_alloc_port: 7400
        # nixl_buffer_size: 2147483648 # 2GB
        # nixl_buffer_device: "cuda"
        lmcache_decode_config = {
            "enable_xpyd": "True",
            "nixl_role": "receiver",
            "nixl_peer_host": "localhost",
            "nixl_peer_init_port": "7300",
            "nixl_peer_alloc_port": "7400",
            "nixl_buffer_size": "2147483648",
            "nixl_buffer_device": "cuda",
        }
        # set the environment variables
        for key, value in lmcache_decode_config.items():
            if key not in os.environ:  # Only set if not already configured
                os.environ[key] = value
                logging.info(f"Set LMCache environment variable: {key}={value}")
        # Force local cpu is false
        os.environ["LMCACHE_LOCAL_CPU"] = "False"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "0"

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

    if not config.engine_args.data_parallel_rank:  # if rank is 0 or None then register
        await register_llm(
            ModelType.Backend,
            generate_endpoint,
            config.model,
            config.served_model_name,
            kv_cache_block_size=config.engine_args.block_size,
        )

    factory = StatLoggerFactory(component, config.engine_args.data_parallel_rank or 0)
    engine_client, vllm_config, default_sampling_params = setup_vllm_engine(
        config, factory
    )

    # TODO Hack to get data, move this to registering in ETCD
    factory.set_num_gpu_blocks_all(vllm_config.cache_config.num_gpu_blocks)
    factory.set_request_total_slots_all(vllm_config.scheduler_config.max_num_seqs)
    factory.init_publish()

    logger.info(f"VllmWorker for {config.model} has been initialized")

    handler = DecodeWorkerHandler(
        component, engine_client, default_sampling_params, prefill_worker_client
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

    print(f"FINAL: {engine_client.vllm_config.cache_config.enable_prefix_caching}")
    print(f"FINAL: {engine_client.vllm_config.kv_events_config}")
    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(handler.generate),
            clear_endpoint.serve_endpoint(handler.clear_kv_blocks),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        # Cleanup background tasks
        handler.cleanup()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
