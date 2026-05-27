# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic Diffusion Engine wrapper for TensorRT-LLM VisualGen.

This module provides a unified interface for various diffusion models
(Wan, Flux, Cosmos, etc.) through TensorRT-LLM's public VisualGen API.

The pipeline type is auto-detected from model_index.json (shipped with every
HuggingFace Diffusers model), eliminating the need for a --model-type flag.

Requirements:
    - tensorrt_llm with visual_gen support.
      See: https://github.com/NVIDIA/TensorRT-LLM
    - See docs/backends/trtllm/README.md for setup instructions.

Note on imports:
    tensorrt_llm.visual_gen is imported lazily in initialize() because:
    1. It's a heavy package that may not be installed in all environments
    2. Importing at module load would fail if tensorrt_llm is not available
    3. This allows the module to be imported for type checking and validation
       without requiring tensorrt_llm to be installed
"""

import logging
import random
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen import VisualGen, VisualGenArgs, VisualGenOutput

    from dynamo.trtllm.configs.diffusion_config import DiffusionConfig

logger = logging.getLogger(__name__)


class DiffusionEngine:
    """Generic wrapper for TensorRT-LLM VisualGen.

    This engine provides:
    - Auto-detection of visual generation model families inside TensorRT-LLM
    - Loading and initialization through the public VisualGen API
    - Common interface for video/image generation via VisualGen.generate()

    Example:
        >>> engine = DiffusionEngine(config)
        >>> await engine.initialize()
        >>> output = engine.generate(prompt="A cat playing piano", ...)
        >>> output.video  # torch.Tensor (B, num_frames, H, W, 3) uint8
    """

    def __init__(self, config: "DiffusionConfig"):
        """Initialize the engine with configuration.

        Args:
            config: Diffusion generation configuration.
        """
        self.config = config
        self._visual_gen: Optional["VisualGen"] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load and configure the diffusion backend via VisualGen.

        This is called once at worker startup to load the model.
        """
        if self._initialized:
            logger.warning("Engine already initialized, skipping")
            return

        logger.info(
            f"Initializing DiffusionEngine: model_path={self.config.model_path}"
        )

        if self.config.skip_components:
            raise ValueError(
                "--skip-components is not supported by TensorRT-LLM's public "
                "VisualGen API. Dynamo avoids private PipelineLoader usage for "
                "diffusion serving."
            )

        # Import TensorRT-LLM VisualGen lazily because it is an optional backend.
        from tensorrt_llm.visual_gen import VisualGen

        # Build VisualGenArgs from DiffusionConfig
        diffusion_args = self._build_diffusion_args()
        logger.info(f"VisualGenArgs: {diffusion_args}")

        self._visual_gen = VisualGen(
            model=self.config.model_path,
            args=diffusion_args,
        )

        self._initialized = True
        logger.info("DiffusionEngine initialization complete")

    def _build_diffusion_args(self) -> "VisualGenArgs":
        """Build VisualGenArgs from DiffusionConfig.

        Maps dynamo's DiffusionConfig fields to TensorRT-LLM's VisualGenArgs
        structure with its nested sub-configs (CompilationConfig,
        TorchCompileConfig, CudaGraphConfig, AttentionConfig, ParallelConfig,
        optional cache_config, quant_config).

        Returns:
            VisualGenArgs instance for VisualGen.
        """
        from tensorrt_llm.visual_gen.args import (
            AttentionConfig,
            CompilationConfig,
            CudaGraphConfig,
            ParallelConfig,
            TeaCacheConfig,
            TorchCompileConfig,
            VisualGenArgs,
        )

        # Build quant_config dict if quantization is requested
        # VisualGenArgs accepts a dict in ModelOpt format and parses it via model_validator
        quant_config: dict | None = None
        if self.config.quant_algo:
            quant_config = {
                "quant_algo": self.config.quant_algo,
                "dynamic": self.config.quant_dynamic,
            }

        args_kwargs: dict = dict(
            model=self.config.model_path,
            compilation_config=CompilationConfig(
                skip_warmup=self.config.skip_warmup,
            ),
            torch_compile_config=TorchCompileConfig(
                enable=not self.config.disable_torch_compile,
                enable_fullgraph=self.config.enable_fullgraph,
            ),
            cuda_graph_config=CudaGraphConfig(
                enable=self.config.enable_cuda_graph,
            ),
            attention_config=AttentionConfig(
                backend=self.config.attn_backend.upper(),
            ),
            parallel_config=ParallelConfig(
                cfg_size=self.config.dit_cfg_size,
                ulysses_size=self.config.dit_ulysses_size,
                ring_size=self.config.dit_ring_size,
            ),
            enable_layerwise_nvtx_marker=self.config.enable_layerwise_nvtx_marker,
        )

        # Add optional fields
        if self.config.enable_teacache:
            args_kwargs["cache_config"] = TeaCacheConfig(
                use_ret_steps=self.config.teacache_use_ret_steps,
                teacache_thresh=self.config.teacache_thresh,
            )
        if self.config.revision:
            args_kwargs["revision"] = self.config.revision
        if quant_config is not None:
            args_kwargs["quant_config"] = quant_config

        return VisualGenArgs(**args_kwargs)

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: Optional[int] = None,
        num_images_per_prompt: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
    ) -> "VisualGenOutput":
        """Generate video/image frames from text prompt.

        This is a synchronous method that should be called from a thread pool
        to avoid blocking the event loop.

        VisualGen.generate() handles the full generation flow:
        prompt encoding, latent preparation, denoising loop, and VAE decoding.

        Args:
            prompt: Text description of the content to generate.
            negative_prompt: Text to avoid in the generation.
            height: Output height in pixels.
            width: Output width in pixels.
            num_frames: Number of frames to generate (for video).
            num_images_per_prompt: Number of images to generate per prompt (for image).
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            seed: Random seed for reproducibility.

        Returns:
            VisualGenOutput with model-specific fields populated:
            - .video: torch.Tensor (B, T, H, W, 3) uint8 for video models
            - .image: torch.Tensor (B, H, W, 3) uint8 for image models
            - .audio: torch.Tensor for audio (if supported by model)

        Raises:
            RuntimeError: If engine not initialized or generation fails.
        """
        if not self._initialized or self._visual_gen is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        logger.info(
            f"Generating: prompt='{prompt[:50]}...', "
            f"size={width}x{height}, frames={num_frames}, "
            f"num_images_per_prompt={num_images_per_prompt}, "
            f"steps={num_inference_steps}"
        )

        from tensorrt_llm.visual_gen import VisualGenParams

        params_kwargs = {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed if seed is not None else random.randint(0, 2**32 - 1),
        }
        if negative_prompt is not None:
            params_kwargs["negative_prompt"] = negative_prompt
        if num_frames is not None:
            params_kwargs["num_frames"] = num_frames
        if num_images_per_prompt is not None:
            params_kwargs["num_images_per_prompt"] = num_images_per_prompt

        params = VisualGenParams(**params_kwargs)

        output = self._visual_gen.generate(inputs=prompt, params=params)

        if output.error:
            raise RuntimeError(output.error)

        if output is not None:
            if output.video is not None:
                logger.info(f"Generated video output with shape {output.video.shape}")
            elif output.image is not None:
                logger.info(f"Generated image output with shape {output.image.shape}")
                actual_num_images = output.image.shape[0]
                if (
                    num_images_per_prompt is not None
                    and actual_num_images != num_images_per_prompt
                ):
                    logger.warning(
                        f"Pipeline returned {actual_num_images} image(s) but "
                        f"{num_images_per_prompt} were requested."
                    )

        return output

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._visual_gen is not None:
            self._visual_gen.shutdown()
            self._visual_gen = None
        self._initialized = False
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("DiffusionEngine cleanup complete")

    @property
    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._initialized

    @property
    def device(self) -> str:
        """Get the device where the pipeline runs.

        Returns:
            "cpu" if CPU offload is enabled, "cuda" otherwise.
        """
        return "cpu" if self.config.enable_async_cpu_offload else "cuda"
