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

from dynamo.llm import ZmqKvEventPublisher, ZmqKvEventPublisherConfig
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.args import Config, DisaggregationMode, parse_args
from dynamo.sglang.publisher import setup_sgl_metrics
from dynamo.sglang.register import register_llm_with_runtime_config
from dynamo.sglang.request_handlers import DecodeWorkerHandler, PrefillWorkerHandler

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
    if config.serving_mode != DisaggregationMode.PREFILL:
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

    publisher, metrics_task = await setup_sgl_metrics(engine, component)

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

    handler = DecodeWorkerHandler(
        component, engine, config, publisher, kv_publisher, prefill_client
    )

    # Start serving endpoint first (this does instance registration)
    # TODO: add in native endpoints
    ready_evt = asyncio.Event()

    class GatedHandler:
        def __init__(self, original_handler, ready_event):
            self.original_handler = original_handler
            self.ready_event = ready_event

        async def generate(self, request):
            # Do not process any requests until registration completes.
            await self.ready_event.wait()
            async for out in self.original_handler.generate(request):
                yield out

    gated_handler = GatedHandler(handler, ready_evt)

    async def registration_task():
        # Wait for endpoint to be fully established then do model registration
        try:
            # Create a client to check if the endpoint is ready
            client = await generate_endpoint.client()
            logging.info("Waiting for endpoint instances to be ready...")
            await client.wait_for_instances()
            logging.info("Endpoint is ready, proceeding with model registration")
        except Exception as e:
            logging.error(f"Failed to wait for endpoint readiness: {e}")
            raise RuntimeError(f"Endpoint readiness check failed: {e}")

        registration_success = await register_llm_with_runtime_config(
            engine, generate_endpoint, server_args, dynamo_args.migration_limit
        )

        # If registration failed, shut down serving and fail fast
        if not registration_success:
            logging.error("Model registration failed; shutting down server")
            # Trigger graceful shutdown of the runtime
            runtime.shutdown()
            raise RuntimeError("Model registration failed")

        # Registration succeeded; allow traffic to flow
        ready_evt.set()
        logging.info("Model registration succeeded; service is ready")

    try:
        await asyncio.gather(
            registration_task(),
            generate_endpoint.serve_endpoint(
                gated_handler.generate, graceful_shutdown=False
            ),
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

    tasks = [generate_endpoint.serve_endpoint(handler.generate, graceful_shutdown=True)]

    try:
        await asyncio.gather(*tasks)
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
