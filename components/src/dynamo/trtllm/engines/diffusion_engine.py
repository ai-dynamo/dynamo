# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic Diffusion Engine wrapper for TensorRT-LLM visual_gen pipelines.

This module provides a unified interface for various diffusion models
(Wan, Flux, Cosmos, etc.) through TensorRT-LLM's AutoPipeline system.

The pipeline type is auto-detected from model_index.json (shipped with every
HuggingFace Diffusers model), eliminating the need for a --model-type flag.

Requirements:
    - tensorrt_llm with visual_gen support (tensorrt_llm._torch.visual_gen).
      See: https://github.com/NVIDIA/TensorRT-LLM
    - See docs/pages/backends/trtllm/README.md for setup instructions.

Note on imports:
    tensorrt_llm._torch.visual_gen is imported lazily in initialize() because:
    1. It's a heavy package that may not be installed in all environments
    2. Importing at module load would fail if tensorrt_llm is not available
    3. This allows the module to be imported for type checking and validation
       without requiring tensorrt_llm to be installed
"""

import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen import DiffusionArgs
    from tensorrt_llm._torch.visual_gen.output import MediaOutput
    from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline

    from dynamo.trtllm.configs.diffusion_config import DiffusionConfig

logger = logging.getLogger(__name__)


class DiffusionEngine:
    """Generic wrapper for TensorRT-LLM visual_gen diffusion pipelines.

    This engine provides:
    - Auto-detection of pipeline class from model_index.json via AutoPipeline
    - Loading and initialization through PipelineLoader
    - Common interface for video/image generation via pipeline.infer()

    The old visual_gen standalone package (setup_configs + from_pretrained +
    PIPELINE_REGISTRY) has been replaced by TensorRT-LLM's integrated
    visual_gen module which uses:
    - DiffusionArgs for configuration
    - PipelineLoader for model loading (handles MetaInit, weight loading,
      quantization, torch.compile, and warmup)
    - AutoPipeline for pipeline type auto-detection
    - MediaOutput for typed output (video/image/audio torch tensors)

    Example:
        >>> engine = DiffusionEngine(config)
        >>> await engine.initialize()
        >>> output = engine.generate(prompt="A cat playing piano", ...)
        >>> output.video  # torch.Tensor (num_frames, H, W, 3) uint8
    """

    def __init__(self, config: "DiffusionConfig"):
        """Initialize the engine with configuration.

        Args:
            config: Diffusion generation configuration.
        """
        self.config = config
        self._pipeline: Optional["BasePipeline"] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load and configure the diffusion pipeline via PipelineLoader.

        This is called once at worker startup to load the model.
        PipelineLoader handles:
        1. Loading config via DiffusionModelConfig.from_pretrained()
        2. Creating pipeline via AutoPipeline.from_config() (auto-detects type)
        3. Loading weights with optional on-the-fly quantization
        4. Post-load hooks (TeaCache setup, etc.)
        5. torch.compile (if enabled)
        6. Warmup inference
        """
        if self._initialized:
            logger.warning("Engine already initialized, skipping")
            return

        logger.info(
            f"Initializing DiffusionEngine: model_path={self.config.model_path}"
        )

        # Import TensorRT-LLM visual_gen components
        from tensorrt_llm._torch.visual_gen import DiffusionArgs, PipelineLoader

        # Build DiffusionArgs from DiffusionConfig
        diffusion_args = self._build_diffusion_args()
        logger.info(f"DiffusionArgs: {diffusion_args}")

        # Use PipelineLoader for the full loading flow:
        #   DiffusionArgs → DiffusionModelConfig → AutoPipeline → BasePipeline
        loader = PipelineLoader(diffusion_args)
        self._pipeline = loader.load()

        self._initialized = True
        logger.info(
            f"DiffusionEngine initialization complete: "
            f"{self._pipeline.__class__.__name__}"
        )

    def _build_diffusion_args(self) -> "DiffusionArgs":
        """Build DiffusionArgs from DiffusionConfig.

        Maps dynamo's DiffusionConfig fields to TensorRT-LLM's DiffusionArgs
        structure with its nested sub-configs (PipelineConfig, AttentionConfig,
        ParallelConfig, TeaCacheConfig, quant_config).

        Returns:
            DiffusionArgs instance for PipelineLoader.
        """
        from tensorrt_llm._torch.visual_gen import (
            DiffusionArgs,
            ParallelConfig,
            PipelineConfig,
            TeaCacheConfig,
        )
        from tensorrt_llm._torch.visual_gen.config import AttentionConfig

        # Build quant_config dict if quantization is requested
        # DiffusionArgs accepts a dict in ModelOpt format and parses it via model_validator
        quant_config: dict | None = None
        if self.config.quant_algo:
            quant_config = {
                "quant_algo": self.config.quant_algo,
                "dynamic": self.config.quant_dynamic,
            }

        # Build skip_components list
        skip_components = [c for c in self.config.skip_components] if self.config.skip_components else []

        args_kwargs: dict = dict(
            checkpoint_path=self.config.model_path,
            device=self.device,
            dtype=self.config.torch_dtype,
            skip_components=skip_components,
            pipeline=PipelineConfig(
                enable_torch_compile=not self.config.disable_torch_compile,
                torch_compile_mode=self.config.torch_compile_mode,
                enable_fullgraph=self.config.enable_fullgraph,
                fuse_qkv=self.config.fuse_qkv,
                enable_cuda_graph=self.config.enable_cuda_graph,
                enable_layerwise_nvtx_marker=self.config.enable_layerwise_nvtx_marker,
                warmup_steps=self.config.warmup_steps,
                enable_offloading=self.config.enable_async_cpu_offload,
            ),
            attention=AttentionConfig(
                backend=self.config.attn_backend.upper(),
            ),
            parallel=ParallelConfig(
                dit_dp_size=self.config.dit_dp_size,
                dit_tp_size=self.config.dit_tp_size,
                dit_ulysses_size=self.config.dit_ulysses_size,
                dit_ring_size=self.config.dit_ring_size,
                dit_cfg_size=self.config.dit_cfg_size,
                dit_fsdp_size=self.config.dit_fsdp_size,
            ),
            teacache=TeaCacheConfig(
                enable_teacache=self.config.enable_teacache,
                use_ret_steps=self.config.teacache_use_ret_steps,
                teacache_thresh=self.config.teacache_thresh,
            ),
        )

        # Add optional fields
        if self.config.revision:
            args_kwargs["revision"] = self.config.revision
        if quant_config is not None:
            args_kwargs["quant_config"] = quant_config

        return DiffusionArgs(**args_kwargs)

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
    ) -> "MediaOutput":
        """Generate video/image frames from text prompt.

        This is a synchronous method that should be called from a thread pool
        to avoid blocking the event loop.

        The pipeline's infer() method handles the full generation flow:
        prompt encoding, latent preparation, denoising loop, and VAE decoding.

        Args:
            prompt: Text description of the content to generate.
            negative_prompt: Text to avoid in the generation.
            height: Output height in pixels.
            width: Output width in pixels.
            num_frames: Number of frames to generate (for video).
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            seed: Random seed for reproducibility.

        Returns:
            MediaOutput with model-specific fields populated:
            - .video: torch.Tensor (num_frames, H, W, 3) uint8 for video models
            - .image: torch.Tensor (H, W, 3) uint8 for image models
            - .audio: torch.Tensor for audio (if supported by model)

        Raises:
            RuntimeError: If engine not initialized or generation fails.
        """
        if not self._initialized or self._pipeline is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        logger.info(
            f"Generating: prompt='{prompt[:50]}...', "
            f"size={width}x{height}, frames={num_frames}, steps={num_inference_steps}"
        )

        # Build a DiffusionRequest-like SimpleNamespace for pipeline.infer()
        # The pipeline's infer() method expects a request object with named attributes.
        from types import SimpleNamespace

        req = SimpleNamespace(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed if seed is not None else 42,
            # Additional fields expected by WanPipeline.infer()
            max_sequence_length=512,
            guidance_scale_2=None,
            boundary_ratio=None,
        )

        # Run the pipeline — infer() wraps forward() with torch.no_grad()
        output = self._pipeline.infer(req)

        if output is not None:
            if output.video is not None:
                logger.info(f"Generated video output with shape {output.video.shape}")
            elif output.image is not None:
                logger.info(f"Generated image output with shape {output.image.shape}")

        return output

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._pipeline is not None:
            if hasattr(self._pipeline, "cleanup"):
                self._pipeline.cleanup()
            del self._pipeline
            self._pipeline = None
        self._initialized = False
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("DiffusionEngine cleanup complete")

    @property
    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._initialized

    @property
    def supported_modalities(self) -> list[str]:
        """Get the modalities supported by this engine's pipeline.

        Inferred from the pipeline class name — AutoPipeline handles the
        actual type detection from model_index.json.
        """
        if self._pipeline is None:
            return ["video_diffusion"]  # Default assumption

        class_name = self._pipeline.__class__.__name__
        if "Image" in class_name or "Flux" in class_name:
            return ["image_diffusion"]
        return ["video_diffusion"]

    @property
    def device(self) -> str:
        """Get the device where the pipeline runs.

        Returns:
            "cpu" if CPU offload is enabled, "cuda" otherwise.
        """
        return "cpu" if self.config.enable_async_cpu_offload else "cuda"
