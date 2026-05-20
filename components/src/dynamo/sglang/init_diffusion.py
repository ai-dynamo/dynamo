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
    RealtimeVideoHealthCheckPayload,
    SglangHealthCheckPayload,
    VideoGenerationHealthCheckPayload,
)
from dynamo.sglang.publisher import (
    handle_non_leader_node,
    set_forward_pass_metrics_worker_id,
    setup_sgl_metrics,
)
from dynamo.sglang.register import (
    register_image_diffusion_model,
    register_model_with_readiness_gate,
    register_video_generation_model,
)
from dynamo.sglang.request_handlers import (
    DiffusionWorkerHandler,
    ImageDiffusionWorkerHandler,
    RealtimeVideoWorkerHandler,
    VideoGenerationWorkerHandler,
)


async def init_llm_diffusion(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
) -> None:
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

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )
    set_forward_pass_metrics_worker_id(server_args, generate_endpoint)

    engine = sgl.Engine(server_args=server_args)

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
) -> None:
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

    # The global --output-modalities default is ["text"] which is wrong for
    # image diffusion workers -- it causes the Rust registration path to look
    # for config.json (LLM artefacts).  Only override when the user hasn't
    # explicitly chosen a non-default value.
    output_modalities = dynamo_args.output_modalities
    if output_modalities is None or output_modalities == ["text"]:
        output_modalities = ["image"]
        logging.info(
            "Overriding output_modalities to ['image'] for image diffusion worker"
        )

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
                output_modalities=output_modalities,
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
) -> None:
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


async def init_realtime_video_diffusion(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Initialize the Krea realtime video worker component.

    Reuses `DiffGenerator.from_pretrained()` to bootstrap the SGLang scheduler
    subprocess that the realtime pipeline talks to via ZMQ. The handler then
    drives `process_generation_batch` per chunk and yields one MP4 chunk per
    response so the Rust frontend's `/v1/videos` SSE route can forward each
    chunk to the client.

    The realtime path goes through `async_scheduler_client`, a module-level
    singleton that is normally initialized by sglang's diffusion FastAPI
    lifespan (sgl-project/sglang#19817 `http_server.py:lifespan`). Dynamo's
    worker bypasses FastAPI, so we replicate that bootstrap here: after
    `DiffGenerator.from_pretrained` has spawned the scheduler, we pull the
    constructed `ServerArgs` off the generator (`generator.server_args`,
    same instance the sync client already uses), publish it as the
    process-global via `set_global_server_args`, and feed it to
    `async_scheduler_client.initialize(...)`, paired with `close()` on
    teardown. The `run_zeromq_broker` background task from the upstream
    lifespan is intentionally omitted — it only services external offline
    clients connecting to the FastAPI process, which doesn't exist here.

    The global publish is load-bearing: antgroup's per-chunk code path
    (`session.build_sampling_params` → `build_sampling_params` in
    sglang/multimodal_gen/runtime/entrypoints/openai/utils.py) calls
    `get_global_server_args()` to read fields like attention backend config.
    `launch_server(..., launch_http_server=False)` — the path
    `DiffGenerator.from_pretrained` takes — does NOT set the global in this
    process; the only call to `set_global_server_args` upstream is inside
    `launch_http_server_only`. So without this explicit set, requests fail
    at the first chunk with "Global sgl_diffusion args is not set."
    """
    server_args, dynamo_args = config.server_args, config.dynamo_args

    from sglang.multimodal_gen import DiffGenerator
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args

    if not server_args.model_path:
        raise ValueError("--model is required for realtime video workers")

    tp_size = getattr(server_args, "tp_size", 1)
    dp_size = getattr(server_args, "dp_size", 1)
    num_gpus = tp_size * dp_size

    dist_timeout = getattr(server_args, "dist_timeout", None)

    # Disable all CPU offload paths. Antgroup's __post_init__ auto-enables
    # them on memory-constrained GPUs; we target a GPU that can hold the
    # full Wan 14B + UMT5 + VAE resident, and the offloaded code paths have
    # latent-vs-generator device-mismatch issues that surface randomly. The
    # realtime worker is designed for a large GPU — we own that assumption.
    # pin_cpu_memory is inert when all offload paths are False (its only
    # consumers are gated by cpu_offload/use_fsdp_inference/layerwise/comfyui),
    # but we set it False for symmetry so the server_args dump matches intent.
    generator = DiffGenerator.from_pretrained(
        model_path=server_args.model_path,
        num_gpus=num_gpus,
        tp_size=tp_size,
        dp_size=dp_size,
        dist_timeout=dist_timeout,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        pin_cpu_memory=False,
    )

    # Mirror the FastAPI lifespan init, but read the ServerArgs off the
    # generator (the global is unset in this process — see docstring).
    set_global_server_args(generator.server_args)
    async_scheduler_client.initialize(generator.server_args)

    fs_url = dynamo_args.media_output_fs_url

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    handler = RealtimeVideoWorkerHandler(
        generator,
        config,
        publisher=None,
        fs=get_fs(fs_url),
    )

    health_check_payload = RealtimeVideoHealthCheckPayload(
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
        logging.error(f"Failed to serve realtime video endpoints: {e}")
        raise
    finally:
        try:
            async_scheduler_client.close()
        except Exception as e:
            logging.warning(f"async_scheduler_client.close() failed: {e}")
        handler.cleanup()
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()
