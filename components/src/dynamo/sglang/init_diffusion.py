# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from typing import Awaitable, Callable

import sglang as sgl

from dynamo.common.storage import get_fs
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.health_check import (
    ImageDiffusionHealthCheckPayload,
    SglangHealthCheckPayload,
    VideoGenerationHealthCheckPayload,
)
from dynamo.sglang.publisher import handle_non_leader_node, setup_sgl_metrics
from dynamo.sglang.register import (
    register_image_diffusion_model,
    register_model_with_readiness_gate,
    register_video_generation_model,
)
from dynamo.sglang.request_handlers import (
    DiffusionWorkerHandler,
    ImageDiffusionWorkerHandler,
    VideoGenerationWorkerHandler,
)


async def init_llm_diffusion(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
):
    """Initialize diffusion language model worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    logging.info(
        f"Initializing diffusion worker with algorithm: {server_args.dllm_algorithm}"
    )
    if server_args.dllm_algorithm_config:
        logging.info(
            f"Using diffusion algorithm config: {server_args.dllm_algorithm_config}"
        )

    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    engine = sgl.Engine(server_args=server_args)

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, generate_endpoint
    )

    if server_args.node_rank >= 1:
        await handle_non_leader_node(engine, publisher, metrics_task)
        return

    ready_event = asyncio.Event()

    handler = DiffusionWorkerHandler(
        engine, config, publisher, generate_endpoint, shutdown_event
    )
    handler.register_engine_routes(runtime)

    health_check_payload = SglangHealthCheckPayload(
        engine, use_text_input=dynamo_args.use_sglang_tokenizer
    ).to_dict()

    logging.info(
        f"Registering diffusion model with endpoint types: {dynamo_args.endpoint_types}"
    )

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            ),
            register_model_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                output_type=parse_endpoint_types(dynamo_args.endpoint_types),
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve diffusion endpoints: {e}")
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


async def init_image_diffusion(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
):
    """Initialize image diffusion worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    from sglang.multimodal_gen import DiffGenerator

    if not server_args.model_path:
        raise ValueError("--model is required for diffusion workers")

    tp_size = getattr(server_args, "tp_size", 1)
    dp_size = getattr(server_args, "dp_size", 1)
    num_gpus = tp_size * dp_size

    dist_timeout = getattr(server_args, "dist_timeout", None)

    generator = DiffGenerator.from_pretrained(
        model_path=server_args.model_path,
        num_gpus=num_gpus,
        tp_size=tp_size,
        dp_size=dp_size,
        dist_timeout=dist_timeout,
    )

    fs_url = dynamo_args.media_output_fs_url

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    handler = ImageDiffusionWorkerHandler(
        generator,
        config,
        publisher=None,
        fs=get_fs(fs_url),
    )

    health_check_payload = ImageDiffusionHealthCheckPayload(
        model_path=server_args.model_path
    ).to_dict()

    ready_event = asyncio.Event()

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[],
                health_check_payload=health_check_payload,
            ),
            register_image_diffusion_model(
                generator,
                generate_endpoint,
                server_args,
                output_modalities=dynamo_args.output_modalities,
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve image diffusion endpoints: {e}")
        raise
    finally:
        handler.cleanup()
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()


async def init_video_diffusion(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
):
    """Initialize video generation worker component"""
    server_args, dynamo_args = config.server_args, config.dynamo_args

    from sglang.multimodal_gen import DiffGenerator

    if not server_args.model_path:
        raise ValueError("--model is required for video generation workers")

    tp_size = getattr(server_args, "tp_size", 1)
    dp_size = getattr(server_args, "dp_size", 1)
    num_gpus = tp_size * dp_size

    dist_timeout = getattr(server_args, "dist_timeout", None)

    generator = DiffGenerator.from_pretrained(
        model_path=server_args.model_path,
        num_gpus=num_gpus,
        tp_size=tp_size,
        dp_size=dp_size,
        dist_timeout=dist_timeout,
    )

    fs_url = dynamo_args.media_output_fs_url

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    handler = VideoGenerationWorkerHandler(
        generator,
        config,
        publisher=None,
        fs=get_fs(fs_url),
    )

    health_check_payload = VideoGenerationHealthCheckPayload(
        model_path=server_args.model_path
    ).to_dict()

    ready_event = asyncio.Event()

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[],
                health_check_payload=health_check_payload,
            ),
            register_video_generation_model(
                generator,
                generate_endpoint,
                server_args,
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve video generation endpoints: {e}")
        raise
    finally:
        handler.cleanup()
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()
