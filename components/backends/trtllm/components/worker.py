# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import signal
import sys
from typing import TYPE_CHECKING

import uvloop
from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory

from dynamo.llm import (
    ModelType,
    get_tensorrtllm_engine,
    get_tensorrtllm_publisher,
    register_llm,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

if TYPE_CHECKING:
    from utils.trtllm_utils import Config


def _setup_path_and_imports():
    """Setup path and import utils modules"""
    # Add the parent directory to the Python path so we can import utils
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from utils.request_handlers.handlers import (
        RequestHandlerConfig,
        RequestHandlerFactory,
    )
    from utils.trtllm_utils import (
        Config,
        cmd_line_args,
        is_first_worker,
        parse_endpoint,
    )

    return (
        RequestHandlerConfig,
        RequestHandlerFactory,
        Config,
        cmd_line_args,
        is_first_worker,
        parse_endpoint,
    )


# Import utils modules
(
    RequestHandlerConfig,
    RequestHandlerFactory,
    Config,
    cmd_line_args,
    is_first_worker,
    parse_endpoint,
) = _setup_path_and_imports()

# Default buffer size for kv cache events.
DEFAULT_KV_EVENT_BUFFER_MAX_SIZE = 1024

configure_dynamo_logging()


async def graceful_shutdown(runtime):
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        # Schedule the shutdown coroutine instead of calling it directly
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    config = cmd_line_args()
    await init(runtime, config)


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    logging.info(f"Initializing the worker with config: {config}")

    next_client = None
    if config.next_endpoint:
        logging.info(
            f"Initializing next worker client for endpoint: {config.next_endpoint}"
        )
        parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
            config.next_endpoint
        )
        next_client = (
            await runtime.namespace(parsed_namespace)
            .component(parsed_component_name)
            .endpoint(parsed_endpoint_name)
            .client()
        )

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    # Convert model path to Path object if it's a local path, otherwise keep as string
    model_path = str(config.model_path)

    arg_map = {
        "model": model_path,
        "tensor_parallel_size": config.tensor_parallel_size,
        "backend": "pytorch",
        "skip_tokenizer_init": True,
    }
    if config.extra_engine_args != "":
        # TODO: Support extra engine args from json file as well.
        arg_map = update_llm_args_with_extra_options(arg_map, config.extra_engine_args)
    if config.publish_events_and_metrics:
        # 'event_buffer_max_size' is required to enable TRTLLM to publish kv cache events.
        kv_cache_config = None
        if "kv_cache_config" not in arg_map:
            kv_cache_config = {}
            kv_cache_config["event_buffer_max_size"] = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
        else:
            kv_cache_config = arg_map["kv_cache_config"]
            if not kv_cache_config.event_buffer_max_size:
                kv_cache_config.event_buffer_max_size = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
        arg_map["kv_cache_config"] = kv_cache_config

        # Only pytorch backend is supported for now to publish events and metrics.
        if "backend" not in arg_map:
            arg_map["backend"] = "pytorch"
        elif arg_map["backend"] != "pytorch":
            logging.error(
                "Only pytorch backend is supported for now to publish events and metrics."
            )
            sys.exit(1)

    logging.info(f"TensorRT-LLM engine args: {arg_map}")
    engine_args = arg_map

    # Populate default sampling params from the model
    tokenizer = tokenizer_factory(arg_map["model"])
    default_sampling_params = SamplingParams()
    default_sampling_params._setup(tokenizer)
    default_sampling_params.stop = None

    async with get_tensorrtllm_engine(engine_args) as engine:
        endpoint = component.endpoint(config.endpoint)

        if is_first_worker(config):
            # Register the model with the endpoint if only the worker is first in the disaggregation chain.
            await register_llm(
                ModelType.Backend,
                endpoint,
                config.model_path,
                config.served_model_name,
                kv_cache_block_size=config.kv_block_size,
            )

        # publisher will be set later if publishing is enabled.
        handler_config = RequestHandlerConfig(
            component=component,
            engine=engine,
            default_sampling_params=default_sampling_params,
            publisher=None,
            disaggregation_mode=config.disaggregation_mode,
            disaggregation_strategy=config.disaggregation_strategy,
            next_client=next_client,
        )

        if config.publish_events_and_metrics and is_first_worker(config):
            # Initialize and pass in the publisher to the request handler to
            # publish events and metrics.
            kv_listener = runtime.namespace(config.namespace).component(
                config.component
            )
            async with get_tensorrtllm_publisher(
                component,
                engine,
                kv_listener,
                int(endpoint.lease_id()),
                config.kv_block_size,
            ) as publisher:
                handler_config.publisher = publisher
                handler = RequestHandlerFactory().get_request_handler(handler_config)
                await endpoint.serve_endpoint(handler.generate)
        else:
            handler = RequestHandlerFactory().get_request_handler(handler_config)
            await endpoint.serve_endpoint(handler.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
