# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video diffusion worker initialization for TensorRT-LLM backend.

This module handles the initialization and lifecycle of video generation
workers using diffusion models (Wan, Flux, Cosmos, etc.).
"""

import asyncio
import logging

from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.trtllm.utils.trtllm_utils import Config


async def init_video_diffusion_worker(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
) -> None:
    """Initialize and run the video diffusion worker.

    This function handles video_diffusion modality, loading the appropriate
    diffusion model and serving video generation requests.

    Args:
        runtime: The Dynamo distributed runtime.
        config: Configuration parsed from command line.
        shutdown_event: Event to signal shutdown.
    """
    # Import diffusion-specific modules (lazy import to avoid loading heavy deps early)
    from dynamo.trtllm.configs.diffusion_config import DiffusionConfig
    from dynamo.trtllm.engines.diffusion_engine import DiffusionEngine
    from dynamo.trtllm.request_handlers.video_diffusion import VideoGenerationHandler

    logging.info(f"Initializing video diffusion worker with config: {config}")

    # Build DiffusionConfig from the main Config
    diffusion_config = DiffusionConfig(
        namespace=config.namespace,
        component=config.component,
        endpoint=config.endpoint,
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
        model_path=config.model_path,
        served_model_name=config.served_model_name,
        output_dir=config.output_dir,
        default_height=config.default_height,
        default_width=config.default_width,
        default_num_frames=config.default_num_frames,
        default_num_inference_steps=config.default_num_inference_steps,
        default_guidance_scale=config.default_guidance_scale,
        enable_teacache=config.enable_teacache,
        teacache_thresh=config.teacache_thresh,
        attn_type=config.attn_type,
        linear_type=config.linear_type,
        disable_torch_compile=config.disable_torch_compile,
        torch_compile_mode=config.torch_compile_mode,
        dit_dp_size=config.dit_dp_size,
        dit_tp_size=config.dit_tp_size,
        dit_ulysses_size=config.dit_ulysses_size,
        dit_ring_size=config.dit_ring_size,
        dit_cfg_size=config.dit_cfg_size,
        dit_fsdp_size=config.dit_fsdp_size,
        enable_async_cpu_offload=config.enable_async_cpu_offload,
    )

    # Get the component and endpoint from the runtime
    component = runtime.namespace(config.namespace).component(config.component)
    endpoint = component.endpoint(config.endpoint)

    # Initialize the diffusion engine (auto-detects pipeline from model_index.json)
    engine = DiffusionEngine(diffusion_config)
    await engine.initialize()

    # Create the request handler
    handler = VideoGenerationHandler(component, engine, diffusion_config)

    # Register the model with Dynamo's discovery system
    model_name = config.served_model_name or config.model_path

    # Use ModelType.Videos for video generation
    if not hasattr(ModelType, "Videos"):
        raise RuntimeError(
            "ModelType.Videos not available in dynamo-runtime. "
            "Video diffusion requires a compatible dynamo-runtime version. "
            "See docs/pages/backends/trtllm/README.md for setup instructions."
        )
    model_type = ModelType.Videos

    logging.info(f"Registering model '{model_name}' with ModelType={model_type}")

    # register_model is Dynamo's generic model registration function
    await register_model(
        ModelInput.Text,
        model_type,
        endpoint,
        config.model_path,
        model_name,
    )

    logging.info(f"Model registered, serving endpoint: {config.endpoint}")

    # Serve the endpoint
    try:
        await endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
        )
    except asyncio.CancelledError:
        logging.info("Endpoint serving cancelled")
    except Exception as e:
        logging.error(f"Error serving endpoint: {e}", exc_info=True)
        raise
    finally:
        handler.cleanup()
        engine.cleanup()
