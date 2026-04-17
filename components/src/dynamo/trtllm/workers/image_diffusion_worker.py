# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image diffusion worker initialization for TensorRT-LLM backend.

This module handles the initialization and lifecycle of image generation
workers using diffusion models (Wan, Flux, Cosmos, etc.).
"""

import asyncio
import logging
from typing import Optional

from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.trtllm.args import Config


async def init_image_diffusion_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: Optional[list] = None,
) -> None:
    # [gluo TODO] this can be the same as video diffusion worker, just need to update the handler and model type
    """Initialize and run the image diffusion worker.

    This function handles image_diffusion modality, loading the appropriate
    diffusion model and serving image generation requests.

    Args:
        runtime: The Dynamo distributed runtime.
        config: Configuration parsed from command line.
        shutdown_event: Event to signal shutdown.
        shutdown_endpoints: Optional list to populate with endpoints for graceful shutdown.
    """
    # Check tensorrt_llm visual_gen availability early with a clear error message.
    # visual_gen is part of TensorRT-LLM (tensorrt_llm._torch.visual_gen).
    # Without this check, users would get a cryptic ImportError deep inside
    # DiffusionEngine.initialize().
    try:
        import tensorrt_llm._torch.visual_gen  # noqa: F401
    except ImportError:
        raise ImportError(
            "Image diffusion requires TensorRT-LLM with visual_gen support.\n"
            "The visual_gen module is at tensorrt_llm._torch.visual_gen.\n"
            "Install TensorRT-LLM with AIGV support:\n"
            "  pip install tensorrt_llm\n"
            "See: https://github.com/NVIDIA/TensorRT-LLM"
        ) from None

    from dynamo.trtllm.configs.diffusion_config import DiffusionConfig
    from dynamo.trtllm.engines.diffusion_engine import DiffusionEngine
    from dynamo.trtllm.request_handlers.diffusion import ImageGenerationHandler

    logging.info(f"Initializing image diffusion worker with config: {config}")

    # Parse skip_components from comma-separated string to list
    skip_components = (
        [c.strip() for c in config.skip_components.split(",") if c.strip()]
        if config.skip_components
        else []
    )

    if not config.endpoint:
        raise ValueError("endpoint must be configured for image diffusion worker")

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
        torch_dtype=config.torch_dtype,
        revision=config.revision,
        media_output_fs_url=config.media_output_fs_url,
        media_output_http_url=config.media_output_http_url,
        default_height=config.default_height,
        default_width=config.default_width,
        default_num_frames=config.default_num_frames,
        default_num_images_per_prompt=config.default_num_images_per_prompt,
        default_num_inference_steps=config.default_num_inference_steps,
        default_guidance_scale=config.default_guidance_scale,
        # Pipeline optimization
        disable_torch_compile=config.disable_torch_compile,
        enable_fullgraph=config.enable_fullgraph,
        fuse_qkv=config.fuse_qkv,
        enable_cuda_graph=config.enable_cuda_graph,
        enable_layerwise_nvtx_marker=config.enable_layerwise_nvtx_marker,
        skip_warmup=config.skip_warmup,
        # Attention
        attn_backend=config.attn_backend,
        # Quantization
        quant_algo=config.quant_algo,
        quant_dynamic=config.quant_dynamic,
        # TeaCache
        enable_teacache=config.enable_teacache,
        teacache_use_ret_steps=config.teacache_use_ret_steps,
        teacache_thresh=config.teacache_thresh,
        # Parallelism
        dit_dp_size=config.dit_dp_size,
        dit_tp_size=config.dit_tp_size,
        dit_ulysses_size=config.dit_ulysses_size,
        dit_ring_size=config.dit_ring_size,
        dit_cfg_size=config.dit_cfg_size,
        dit_fsdp_size=config.dit_fsdp_size,
        # Offloading
        enable_async_cpu_offload=config.enable_async_cpu_offload,
        # Component loading
        skip_components=skip_components,
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
    handler = ImageGenerationHandler(engine, diffusion_config)

    # Register the model with Dynamo's discovery system
    model_name = config.served_model_name or config.model

    # Use ModelType.Images for image generation
    if not hasattr(ModelType, "Images"):
        raise RuntimeError(
            "ModelType.Images not available in dynamo-runtime. "
            "Image diffusion requires a compatible dynamo-runtime version. "
            "See docs/backends/trtllm/README.md for setup instructions."
        )
    model_type = ModelType.Images

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
