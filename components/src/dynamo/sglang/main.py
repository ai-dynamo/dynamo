# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import signal
import sys

import sglang as sgl
import uvloop

from dynamo.common.config_dump import dump_config
from dynamo.llm import ModelInput, ModelType
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.args import Config, DisaggregationMode, parse_args
from dynamo.sglang.health_check import (
    SglangHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.publisher import setup_sgl_metrics
from dynamo.sglang.register import register_llm_with_readiness_gate
from dynamo.sglang.request_handlers import (
    DecodeWorkerHandler,
    EmbeddingWorkerHandler,
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
    dump_config(config.dynamo_args.dump_config_to, config)
    if config.dynamo_args.embedding_worker:
        await init_embedding(runtime, config)
    elif config.dynamo_args.multimodal_processor:
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

    prefill_client = None
    prefill_router_client = None
    if config.serving_mode == DisaggregationMode.DECODE:
        prefill_router_client = (
            await runtime.namespace(dynamo_args.namespace)
            .component("router")
            .endpoint("best_worker_id")
            .client()
        )
        prefill_client = (
            await runtime.namespace(dynamo_args.namespace)
            .component("prefill")
            .endpoint("generate")
            .client()
        )

    # publisher instantiates the metrics and kv event publishers
    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, component, generate_endpoint
    )

    # Readiness gate: requests wait until model is registered
    ready_event = asyncio.Event()

    handler = DecodeWorkerHandler(
        component, engine, config, publisher, prefill_client, prefill_router_client
    )

    health_check_payload = SglangHealthCheckPayload(engine).to_dict()

    try:
        # Start endpoint immediately and register model concurrently
        # Requests queue until ready_event is set (TODO: Part of new PR)
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            ),
            register_llm_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                readiness_gate=ready_event,
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

    # publisher instantiates the metrics and kv event publishers
    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, component, generate_endpoint
    )

    handler = PrefillWorkerHandler(component, engine, config, publisher)

    health_check_payload = SglangPrefillHealthCheckPayload(engine).to_dict()

    tasks = [
        generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=metrics_labels,
            health_check_payload=health_check_payload,
        )
    ]

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            logging.info("Metrics task successfully cancelled")
            pass
        handler.cleanup()


async def init_embedding(runtime: DistributedRuntime, config: Config):
    """Initialize embedding worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )
    await component.create_service()

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # publisher instantiates the metrics and kv event publishers
    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, component, generate_endpoint
    )

    # Readiness gate: requests wait until model is registered
    ready_event = asyncio.Event()

    handler = EmbeddingWorkerHandler(component, engine, config, publisher)
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
            register_llm_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                input_type=ModelInput.Text,
                output_type=ModelType.Embedding,
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve embedding endpoints: {e}")
        raise
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            logging.info("Metrics task successfully cancelled")
            pass
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
    encode_worker_client = (
        await runtime.namespace(dynamo_args.namespace)
        .component("encoder")
        .endpoint("generate")
        .client()
    )

    ready_event = asyncio.Event()

    handler = MultimodalProcessorHandler(component, config, encode_worker_client)

    logging.info("Waiting for Encoder Worker Instances ...")
    await encode_worker_client.wait_for_instances()

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("model", server_args.served_model_name)],
            ),
            register_llm_with_readiness_gate(
                None,  # engine
                generate_endpoint,
                server_args,
                dynamo_args,
                input_type=ModelInput.Text,
                readiness_gate=ready_event,
            ),
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

    # For encode worker, we need to connect to the downstream LLM worker
    pd_worker_client = (
        await runtime.namespace(dynamo_args.namespace)
        .component("backend")
        .endpoint("generate")
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

    if config.serving_mode == DisaggregationMode.DECODE:
        logging.info("Initializing prefill client for multimodal decode worker")
        prefill_client = (
            await runtime.namespace(dynamo_args.namespace)
            .component("prefill")
            .endpoint("generate")
            .client()
        )
        handler = MultimodalWorkerHandler(component, engine, config, prefill_client)
    else:
        handler = MultimodalWorkerHandler(component, engine, config)

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
