# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AFD (Attention-FFN Disaggregation) initialization module.

This module provides initialization functions for Attention and FFN workers
in AFD disaggregated serving mode.

Architecture: r Attention instances -> 1 shared FFN instance
Reference: https://arxiv.org/abs/2601.21351
"""

import asyncio
import logging
from typing import List, Optional, Tuple

import sglang as sgl

from dynamo._core import Component, DistributedRuntime
from dynamo.common.constants import DisaggregationMode
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import setup_sgl_metrics
from dynamo.sglang.request_handlers import AFDAttentionHandler, AFDFFNHandler
from dynamo.sglang.shutdown import run_deferred_handlers


async def init_attention_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: List,
    run_deferred_handlers: List,
) -> None:
    """Initialize an Attention worker in AFD disaggregated mode.

    The Attention worker maintains KV cache and performs attention computation.
    Multiple Attention workers feed into a single shared FFN worker.

    Args:
        runtime: The distributed runtime instance.
        config: SGLang and Dynamo configuration.
        shutdown_event: Event to signal shutdown.
        shutdown_endpoints: List of endpoints to unregister on shutdown.
        run_deferred_handlers: List of handlers to run on shutdown.
    """
    dynamo_args = config.dynamo_args
    server_args = config.server_args

    logging.info(
        f"Initializing AFD Attention worker - "
        f"namespace={dynamo_args.namespace}, component={dynamo_args.component}"
    )

    # Create component and endpoint
    component = runtime.namespace(dynamo_args.namespace).component(dynamo_args.component)
    endpoint = component.endpoint(dynamo_args.endpoint)

    # Initialize SGLang engine with attention-specific configuration
    # Note: AFD mode requires splitting the model into attention and FFN parts
    # This is a placeholder - actual implementation needs model partitioning
    engine = sgl.Engine(
        server_args=server_args,
    )

    # Set up metrics publisher
    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, endpoint
    )

    # Create Attention handler
    handler = AFDAttentionHandler(
        component=component,
        engine=engine,
        config=config,
        publisher=publisher,
        generate_endpoint=endpoint,
        shutdown_event=shutdown_event,
        ffn_endpoint=dynamo_args.afd_ffn_endpoint,
        attention_ratio=dynamo_args.afd_attention_ratio or 1,
    )

    logging.info(
        f"AFD Attention worker ready - "
        f"attention_ratio={dynamo_args.afd_attention_ratio or 1}, "
        f"ffn_endpoint={dynamo_args.afd_ffn_endpoint}"
    )

    # Wait for shutdown signal
    try:
        await shutdown_event.wait()
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass
        handler.cleanup()
        publisher.cleanup()
        await run_deferred_handlers(shutdown_endpoints)


async def init_ffn_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: List,
    run_deferred_handlers: List,
) -> None:
    """Initialize an FFN worker in AFD disaggregated mode.

    The FFN worker is stateless and compute-intensive, receiving activations
    from multiple Attention workers and performing FFN computation.

    Args:
        runtime: The distributed runtime instance.
        config: SGLang and Dynamo configuration.
        shutdown_event: Event to signal shutdown.
        shutdown_endpoints: List of endpoints to unregister on shutdown.
        run_deferred_handlers: List of handlers to run on shutdown.
    """
    dynamo_args = config.dynamo_args
    server_args = config.server_args

    logging.info(
        f"Initializing AFD FFN worker - "
        f"namespace={dynamo_args.namespace}, component={dynamo_args.component}"
    )

    # Create component and endpoint
    component = runtime.namespace(dynamo_args.namespace).component(dynamo_args.component)
    endpoint = component.endpoint(dynamo_args.endpoint)

    # Initialize SGLang engine with FFN-specific configuration
    # Note: AFD mode requires splitting the model into attention and FFN parts
    # This is a placeholder - actual implementation needs model partitioning
    engine = sgl.Engine(
        server_args=server_args,
    )

    # Set up metrics publisher
    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, endpoint
    )

    # Create FFN handler
    handler = AFDFFNHandler(
        component=component,
        engine=engine,
        config=config,
        publisher=publisher,
        generate_endpoint=endpoint,
        shutdown_event=shutdown_event,
        attention_ratio=dynamo_args.afd_attention_ratio or 1,
    )

    logging.info(
        f"AFD FFN worker ready - "
        f"attention_ratio={dynamo_args.afd_attention_ratio or 1} "
        f"(shared by {dynamo_args.afd_attention_ratio or 1} Attention workers)"
    )

    # Wait for shutdown signal
    try:
        await shutdown_event.wait()
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass
        handler.cleanup()
        publisher.cleanup()
        await run_deferred_handlers(shutdown_endpoints)
