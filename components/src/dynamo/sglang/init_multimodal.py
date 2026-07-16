# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Awaitable, Callable

import sglang as sgl

from dynamo import prometheus_names
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.prometheus import register_embedding_cache_metrics
from dynamo.llm import (
    ModelInput,
    ModelType,
    MultimodalEmbeddingCachePublisher,
    WorkerType,
)
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.register import register_model_with_readiness_gate
from dynamo.sglang.request_handlers import (
    MultimodalEncodeWorkerHandler,
    MultimodalPrefillWorkerHandler,
    MultimodalWorkerHandler,
)


async def init_multimodal_encode_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Initialize multimodal encode worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    pd_worker_client = await runtime.endpoint(
        f"{dynamo_args.namespace}.backend.generate"
    ).client()

    cache_publisher = None
    if (
        config.dynamo_args.multimodal_embedding_cache_capacity_gb > 0
        and config.dynamo_args.multimodal_embedding_cache_publisher
    ):
        cache_publisher = MultimodalEmbeddingCachePublisher()
        await cache_publisher.create_endpoint(generate_endpoint)

    handler = MultimodalEncodeWorkerHandler(
        config,
        pd_worker_client,
        cache_publisher,
        shutdown_event,
    )

    if handler._embedding_cache is not None:
        register_embedding_cache_metrics(
            endpoint=generate_endpoint,
            cache=handler._embedding_cache,
            model_name=server_args.served_model_name,
            component_name=dynamo_args.component,
        )

    await pd_worker_client.wait_for_instances()

    ready_event = asyncio.Event()

    try:
        await register_model_with_readiness_gate(
            None,  # engine
            generate_endpoint,
            server_args,
            dynamo_args,
            input_type=ModelInput.Tokens,
            # The encode worker is the OpenAI front door for sglang
            # multimodal: it carries the Chat/Completions surface and
            # delegates token generation to the internal PD worker via
            # pd_worker_client. Its `worker_type=Encode` topology role gates
            # serving on the downstream worker(s) — `needs` is a DNF: a P+D
            # pair OR a single Aggregated peer — so the model is not
            # advertised until the whole pipeline is live.
            output_type=ModelType.Chat | ModelType.Completions,
            readiness_gate=ready_event,
            worker_type=WorkerType.Encode,
            needs=[
                [WorkerType.Prefill, WorkerType.Decode],
                [WorkerType.Aggregated],
            ],
        )
        _ = await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[
                    (prometheus_names.labels.MODEL, server_args.served_model_name),
                    (prometheus_names.labels.MODEL_NAME, server_args.served_model_name),
                ],
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


async def init_multimodal_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Initialize multimodal worker component.

    This worker is always an internal component that should not register with
    the Frontend. Public registration is handled by the Encode Worker component
    (--enable-multimodal --disaggregation-mode encode). For standalone serving,
    use init() (default).
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

    if config.serving_mode == DisaggregationMode.DECODE:
        health_check_payload = SglangDisaggHealthCheckPayload(engine).to_dict()
    else:
        health_check_payload = SglangHealthCheckPayload(engine).to_dict()

    # This worker has no OpenAI surface (ModelType.Empty); it is reached only
    # through the encode worker's client, never by the frontend. It registers a
    # topology card so the serving-readiness gate counts it — the model is not
    # advertised until this worker AND the encode worker (and the prefill
    # worker, in disaggregated mode) are live.
    if config.serving_mode == DisaggregationMode.DECODE:
        readiness_worker_type = WorkerType.Decode
        readiness_needs: list[list[WorkerType]] = [[WorkerType.Prefill]]
    else:
        readiness_worker_type = WorkerType.Aggregated
        readiness_needs = [[WorkerType.Encode]]

    try:
        await register_model_with_readiness_gate(
            engine,
            generate_endpoint,
            server_args,
            dynamo_args,
            input_type=ModelInput.Tokens,
            output_type=ModelType.Empty,
            worker_type=readiness_worker_type,
            needs=readiness_needs,
        )
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                metrics_labels=[("model", server_args.served_model_name)],
                graceful_shutdown=True,
                health_check_payload=health_check_payload,
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


async def init_multimodal_prefill_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Initialize multimodal prefill worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    engine = sgl.Engine(server_args=server_args)

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    handler = MultimodalPrefillWorkerHandler(engine, config, shutdown_event)

    shutdown_endpoints[:] = [generate_endpoint]

    health_check_payload = SglangPrefillHealthCheckPayload(engine).to_dict()

    # No OpenAI surface (ModelType.Empty): internal prefill worker, reached via
    # the decode worker / prefill router, never by the frontend. Registers a
    # topology card so the serving-readiness gate counts it.
    try:
        await register_model_with_readiness_gate(
            engine,
            generate_endpoint,
            server_args,
            dynamo_args,
            input_type=ModelInput.Tokens,
            output_type=ModelType.Empty,
            worker_type=WorkerType.Prefill,
            needs=[[WorkerType.Decode]],
        )
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("model", server_args.served_model_name)],
                health_check_payload=health_check_payload,
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
