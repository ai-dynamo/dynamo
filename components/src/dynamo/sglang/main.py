# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import signal
import sys

import sglang as sgl
import uvloop
from sglang.srt.utils import get_ip

from dynamo.llm import ModelInput, ZmqKvEventPublisher, ZmqKvEventPublisherConfig
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.args import Config, DisaggregationMode, parse_args, parse_endpoint
from dynamo.sglang.health_check import (
    SglangHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.publisher import setup_sgl_metrics
from dynamo.sglang.register import register_llm_with_runtime_config
from dynamo.sglang.request_handlers import (
    DecodeWorkerHandler,
    MultimodalEncodeWorkerHandler,
    MultimodalPrefillWorkerHandler,
    MultimodalProcessorHandler,
    MultimodalWorkerHandler,
    PrefillWorkerHandler,
)

configure_dynamo_logging()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers will trigger a graceful shutdown of the runtime")

    config = parse_args(sys.argv[1:])
    if config.dynamo_args.multimodal_processor:
        await init_multimodal_processor(runtime, config)
    elif config.dynamo_args.multimodal_encode_worker:
        await init_multimodal_encode_worker(runtime, config)
    elif config.dynamo_args.multimodal_worker:
        if config.serving_mode != DisaggregationMode.PREFILL:
            await init_multimodal_worker(runtime, config)
        else:
            await init_multimodal_prefill_worker(runtime, config)
    elif config.serving_mode != DisaggregationMode.PREFILL:
        await init(runtime, config)
    else:
        await init_prefill(runtime, config)


async def init(runtime: DistributedRuntime, config: Config):
    server_args, dynamo_args = config.server_args, config.dynamo_args

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )
    await component.create_service()

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # TODO: think about implementing DisaggregationStrategy for P->D
    # TODO: implement a `next` field in the config to dynamically set the next client
    prefill_client = None
    if config.serving_mode == DisaggregationMode.DECODE:
        logging.info("Initializing prefill client")
        prefill_client = (
            await runtime.namespace(dynamo_args.namespace)
            .component("prefill")
            .endpoint("generate")
            .client()
        )

    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(engine, component)

    kv_publisher = None
    if server_args.kv_events_config:
        kv_events = json.loads(server_args.kv_events_config)
        ep = kv_events.get("endpoint")
        zmq_ep = ep.replace("*", get_ip()) if ep else None

        zmq_config = ZmqKvEventPublisherConfig(
            worker_id=generate_endpoint.lease_id(),
            kv_block_size=server_args.page_size,
            zmq_endpoint=zmq_ep,
        )
        logging.info(f"Setting up ZMQ kv event publisher at {zmq_ep}")
        kv_publisher = ZmqKvEventPublisher(component=component, config=zmq_config)

    # Readiness gate: requests wait until model is registered
    ready_event = asyncio.Event()

    handler = DecodeWorkerHandler(
        component, engine, config, publisher, kv_publisher, prefill_client
    )

    async def register_model():
        """Register the model and signal readiness"""
        registration_success = await register_llm_with_runtime_config(
            engine,
            generate_endpoint,
            server_args,
            dynamo_args,
        )

        if not registration_success:
            logging.error("Model registration failed; shutting down")
            runtime.shutdown()
            raise RuntimeError("Model registration failed")

        # Model is ready - allow queued requests to proceed
        ready_event.set()
        logging.info("Model registration succeeded; processing queued requests")

    health_check_payload = SglangHealthCheckPayload(engine).to_dict()

    try:
        # Start endpoint immediately and register model concurrently
        # Requests queue until ready_event is set
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            ),
            register_model(),
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            logging.info("Metrics task succesfully cancelled")
            pass
        handler.cleanup()


async def init_prefill(runtime: DistributedRuntime, config: Config):
    server_args, dynamo_args = config.server_args, config.dynamo_args

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )
    await component.create_service()

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    handler = PrefillWorkerHandler(component, engine, config)

    health_check_payload = SglangPrefillHealthCheckPayload(engine).to_dict()

    tasks = [
        generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[("model", server_args.served_model_name)],
            health_check_payload=health_check_payload,
        )
    ]

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init_multimodal_processor(runtime: DistributedRuntime, config: Config):
    """Initialize multimodal processor component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args
    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )
    await component.create_service()

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # For processor, we need to connect to the encode worker
    # Default endpoint for encode worker
    encode_endpoint = f"dyn://{dynamo_args.namespace}.encoder.generate"
    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        encode_endpoint
    )

    encode_worker_client = (
        await runtime.namespace(parsed_namespace)
        .component(parsed_component_name)
        .endpoint(parsed_endpoint_name)
        .client()
    )

    handler = MultimodalProcessorHandler(component, config, encode_worker_client)

    logging.info("Waiting for Encoder Worker Instances ...")
    await encode_worker_client.wait_for_instances()

    async def register_model():
        """Register the model and signal readiness"""
        registration_success = await register_llm_with_runtime_config(
            None,  # engine,
            generate_endpoint,
            server_args,
            dynamo_args,
            input_type=ModelInput.Text,
        )

        if not registration_success:
            logging.error("Model registration failed; shutting down")
            runtime.shutdown()
            raise RuntimeError("Model registration failed")

        logging.info("Model registration succeeded; processing queued requests")

    try:
        # Start endpoint immediately and register model concurrently
        # Requests queue until ready_event is set
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("model", server_args.served_model_name)],
            ),
            register_model(),
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init_multimodal_encode_worker(runtime: DistributedRuntime, config: Config):
    """Initialize multimodal encode worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )
    await component.create_service()

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # For encode worker, we need to connect to the downstream worker (LLM worker)
    # Default endpoint for LLM worker
    llm_endpoint = f"dyn://{dynamo_args.namespace}.backend.generate"
    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        llm_endpoint
    )

    pd_worker_client = (
        await runtime.namespace(parsed_namespace)
        .component(parsed_component_name)
        .endpoint(parsed_endpoint_name)
        .client()
    )

    handler = MultimodalEncodeWorkerHandler(component, config, pd_worker_client)
    await handler.async_init(runtime)

    await pd_worker_client.wait_for_instances()

    tasks = [
        generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[("model", server_args.served_model_name)],
        )
    ]

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init_multimodal_worker(runtime: DistributedRuntime, config: Config):
    """Initialize multimodal worker component for aggregated or decode mode"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )
    await component.create_service()

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    engine = sgl.Engine(server_args=server_args)

    # Setup handler based on serving mode
    if config.serving_mode == DisaggregationMode.DECODE:
        # Decode mode: create prefill client
        prefill_endpoint = f"dyn://{dynamo_args.namespace}.prefill.generate"
        parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
            prefill_endpoint
        )

        logging.info("Initializing prefill client for multimodal decode worker")
        prefill_client = (
            await runtime.namespace(parsed_namespace)
            .component(parsed_component_name)
            .endpoint(parsed_endpoint_name)
            .client()
        )
        handler = MultimodalWorkerHandler(
            component, engine, config, None, None, prefill_client
        )
    else:
        # Aggregated mode: no prefill client needed
        handler = MultimodalWorkerHandler(component, engine, config)

    # Initialize async components
    await handler.async_init()

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            metrics_labels=[("model", server_args.served_model_name)],
            graceful_shutdown=True,
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init_multimodal_prefill_worker(runtime: DistributedRuntime, config: Config):
    """Initialize multimodal prefill worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )
    await component.create_service()

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    handler = MultimodalPrefillWorkerHandler(component, engine, config)
    await handler.async_init()

    health_check_payload = SglangPrefillHealthCheckPayload(engine).to_dict()

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("model", server_args.served_model_name)],
                health_check_payload=health_check_payload,
            )
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def graceful_shutdown(runtime):
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
