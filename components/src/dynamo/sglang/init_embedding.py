# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Awaitable, Callable

import sglang as sgl

from dynamo.common.utils.prometheus import register_engine_metrics_callback
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.health_check import SglangHealthCheckPayload
from dynamo.sglang.publisher import (
    set_forward_pass_metrics_worker_id,
    setup_sgl_metrics,
)
from dynamo.sglang.register import register_model_with_readiness_gate
from dynamo.sglang.request_handlers import EmbeddingWorkerHandler
from dynamo.sglang.request_handlers.embedding.metrics import init_embedding_metrics


async def init_embedding(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Initialize embedding worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )
    set_forward_pass_metrics_worker_id(server_args, generate_endpoint)

    engine = sgl.Engine(server_args=server_args)

    shutdown_endpoints[:] = [generate_endpoint]

    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, generate_endpoint
    )

    # Wire ``dynamo_embedding_*`` histograms to the worker's /metrics
    # endpoint via a dedicated CollectorRegistry — the default Prometheus
    # global registry is not exposed by the Dynamo SGLang publisher.
    # Done AFTER engine init (and after setup_sgl_metrics) so the lazy
    # ``prometheus_client`` import in metrics.py doesn't interfere with
    # SGLang's multiprocess Prometheus setup.
    from prometheus_client import CollectorRegistry

    embedding_metrics_registry = CollectorRegistry()
    register_engine_metrics_callback(
        endpoint=generate_endpoint,
        registry=embedding_metrics_registry,
        metric_prefix_filters=["dynamo_embedding_"],
        namespace_name=dynamo_args.namespace,
        component_name=dynamo_args.component,
        endpoint_name=dynamo_args.endpoint,
        model_name=server_args.served_model_name,
    )
    init_embedding_metrics(embedding_metrics_registry)

    ready_event = asyncio.Event()

    handler = EmbeddingWorkerHandler(engine, config, publisher, shutdown_event)
    health_check_payload = SglangHealthCheckPayload(
        engine, use_text_input=dynamo_args.use_sglang_tokenizer
    ).to_dict()

    try:
        await register_model_with_readiness_gate(
            engine,
            generate_endpoint,
            server_args,
            dynamo_args,
            input_type=ModelInput.Text,
            output_type=ModelType.Embedding,
            readiness_gate=ready_event,
            worker_type=WorkerType.Aggregated,
            needs=[],
        )
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
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
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()
