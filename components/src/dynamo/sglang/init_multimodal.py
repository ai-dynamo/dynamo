# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Awaitable, Callable

import sglang as sgl

from dynamo import prometheus_names
from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.health_check import (
    SglangHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.register import register_model_with_readiness_gate
from dynamo.sglang.request_handlers import (
    MultimodalEncodeWorkerHandler,
    MultimodalPrefillWorkerHandler,
    MultimodalProcessorHandler,
    MultimodalWorkerHandler,
)


async def init_multimodal_processor(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
):
    """Initialize multimodal processor component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args
    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    encode_worker_client = await runtime.endpoint(
        f"{dynamo_args.namespace}.encoder.generate"
    ).client()

    ready_event = asyncio.Event()

    handler = MultimodalProcessorHandler(config, encode_worker_client, shutdown_event)

    logging.info("Waiting for Encoder Worker Instances ...")
    await encode_worker_client.wait_for_instances()

    try:
        _ = await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[
                    (prometheus_names.labels.MODEL, server_args.served_model_name),
                    (prometheus_names.labels.MODEL_NAME, server_args.served_model_name),
                ],
            ),
            register_model_with_readiness_gate(
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
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()


async def init_multimodal_encode_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
):
    """Initialize multimodal encode worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    pd_worker_client = await runtime.endpoint(
        f"{dynamo_args.namespace}.backend.generate"
    ).client()

    handler = MultimodalEncodeWorkerHandler(config, pd_worker_client, shutdown_event)
    await handler.async_init(runtime)

    await pd_worker_client.wait_for_instances()

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[
                (prometheus_names.labels.MODEL, server_args.served_model_name),
                (prometheus_names.labels.MODEL_NAME, server_args.served_model_name),
            ],
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()


async def init_multimodal_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
):
    """Initialize multimodal worker component.

    This worker is always an internal component that should not register with
    the Frontend. Public registration is handled by the Processor component
    (--multimodal-processor). For standalone serving, use init() (default).
    """
    server_args, dynamo_args = config.server_args, config.dynamo_args

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    engine = sgl.Engine(server_args=server_args)

    if config.serving_mode == DisaggregationMode.DECODE:
        logging.info("Initializing prefill client for multimodal decode worker")
        prefill_client = await runtime.endpoint(
            f"{dynamo_args.namespace}.prefill.generate"
        ).client()
        handler = MultimodalWorkerHandler(
            engine, config, prefill_client, shutdown_event
        )
    else:
        handler = MultimodalWorkerHandler(engine, config, None, shutdown_event)

    await handler.async_init()

    health_check_payload = SglangHealthCheckPayload(engine).to_dict()

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            metrics_labels=[("model", server_args.served_model_name)],
            graceful_shutdown=True,
            health_check_payload=health_check_payload,
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()


async def init_multimodal_prefill_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
):
    """Initialize multimodal prefill worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    engine = sgl.Engine(server_args=server_args)

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    handler = MultimodalPrefillWorkerHandler(engine, config, shutdown_event)

    shutdown_endpoints[:] = [generate_endpoint]

    await handler.async_init()

    health_check_payload = SglangPrefillHealthCheckPayload(engine).to_dict()

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[("model", server_args.served_model_name)],
            health_check_payload=health_check_payload,
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()
