# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video diffusion worker initialization for TensorRT-LLM backend.

This module handles the initialization and lifecycle of video generation
workers using diffusion models (Wan, Flux, Cosmos, etc.).
"""

import asyncio
import logging
from typing import Optional

from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.trtllm.args import Config


async def init_video_diffusion_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: Optional[list] = None,
) -> None:
    """Initialize and run the video diffusion worker.

    This function handles video_diffusion modality, loading the appropriate
    diffusion model and serving video generation requests.

    Args:
        runtime: The Dynamo distributed runtime.
        config: Configuration parsed from command line.
        shutdown_event: Event to signal shutdown.
        shutdown_endpoints: Optional list to populate with endpoints for graceful shutdown.
    """
    # Check visual_gen availability early with a clear error message.
    # visual_gen is part of TensorRT-LLM but only available on the feat/visual_gen
    # branch — not yet in any release. Without this check, users would get a cryptic
    # ImportError deep inside DiffusionEngine.initialize().
    try:
        import visual_gen  # noqa: F401
    except ImportError:
        raise ImportError(
            "Video diffusion requires the 'visual_gen' package from TensorRT-LLM's "
            "feat/visual_gen branch. Install with:\n"
            "  git clone https://github.com/NVIDIA/TensorRT-LLM.git\n"
            "  cd TensorRT-LLM && git checkout feat/visual_gen\n"
            "  cd tensorrt_llm/visual_gen && pip install -e .\n"
            "See: https://github.com/NVIDIA/TensorRT-LLM/tree/feat/visual_gen/tensorrt_llm/visual_gen"
        ) from None

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
        model_path=config.model,
        served_model_name=config.served_model_name,
        media_output_fs_url=config.media_output_fs_url,
        media_output_http_url=config.media_output_http_url,
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

    # Get the endpoint from the runtime
    endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint}"
    )

    if shutdown_endpoints is not None:
        shutdown_endpoints[:] = [endpoint]

    # Initialize the diffusion engine (auto-detects pipeline from model_index.json)
    engine = DiffusionEngine(diffusion_config)
    await engine.initialize()

    # Create the request handler
    handler = VideoGenerationHandler(engine, diffusion_config)

    # Register the model with Dynamo's discovery system
    model_name = config.served_model_name or config.model

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
        config.model,
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
