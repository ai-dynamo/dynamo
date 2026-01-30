# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM Video Diffusion Engine wrapper around visual_gen's ditWanPipeline."""

import logging
from typing import Any, Optional

import numpy as np
import torch

from dynamo.trtllm_diffusion.args import VideoConfig

logger = logging.getLogger(__name__)


class WanDiffusionEngine:
    """Wrapper around TensorRT-LLM visual_gen's ditWanPipeline.

    This engine handles:
    - Pipeline initialization with proper configuration
    - Video generation from text prompts
    - Configuration of optimization features (TeaCache, quantization, parallelism)
    """

    def __init__(self, config: VideoConfig):
        """Initialize the engine with configuration.

        Args:
            config: Video generation configuration.
        """
        self.config = config
        self._pipe = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load and configure the Wan pipeline.

        This is called once at worker startup to load the model.
        """
        if self._initialized:
            logger.warning("Engine already initialized, skipping")
            return

        logger.info(f"Initializing WanDiffusionEngine with model: {self.config.model_path}")

        # Import visual_gen components
        # These imports are deferred to avoid loading CUDA at import time
        from visual_gen import setup_configs
        from visual_gen.pipelines.wan_pipeline import ditWanPipeline

        # Build dit_configs from our config
        dit_configs = self._build_dit_configs()
        logger.info(f"dit_configs: {dit_configs}")

        # Setup global configuration (required before pipeline loading)
        setup_configs(**dit_configs)

        # Load the pipeline
        logger.info(f"Loading pipeline from {self.config.model_path}")
        self._pipe = ditWanPipeline.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            **dit_configs,
        )

        # Move to GPU if not using CPU offload
        if not self.config.enable_async_cpu_offload:
            logger.info("Moving pipeline to CUDA")
            # Note: The pipeline manages device placement internally

        self._initialized = True
        logger.info("WanDiffusionEngine initialization complete")

    def _build_dit_configs(self) -> dict[str, Any]:
        """Build dit_configs dict from VideoConfig.

        Returns:
            Configuration dictionary for visual_gen's setup_configs.
        """
        # Determine torch_compile_models based on model variant
        torch_compile_models = "transformer"
        if "Wan2.2" in self.config.model_path:
            torch_compile_models = "transformer,transformer_2"

        return {
            "pipeline": {
                "enable_torch_compile": not self.config.disable_torch_compile,
                "torch_compile_models": torch_compile_models,
                "torch_compile_mode": self.config.torch_compile_mode,
                "fuse_qkv": True,
            },
            "attn": {
                "type": self.config.attn_type,
            },
            "linear": {
                "type": self.config.linear_type,
                "recipe": "dynamic",
            },
            "parallel": {
                "disable_parallel_vae": False,
                "parallel_vae_split_dim": "width",
                "dit_dp_size": self.config.dit_dp_size,
                "dit_tp_size": self.config.dit_tp_size,
                "dit_ulysses_size": self.config.dit_ulysses_size,
                "dit_ring_size": self.config.dit_ring_size,
                "dit_cp_size": 1,
                "dit_cfg_size": self.config.dit_cfg_size,
                "dit_fsdp_size": self.config.dit_fsdp_size,
                "t5_fsdp_size": 1,
            },
            "teacache": {
                "enable_teacache": self.config.enable_teacache,
                "use_ret_steps": self.config.teacache_use_ret_steps,
                "teacache_thresh": self.config.teacache_thresh,
                "ret_steps": 0,
                "cutoff_steps": self.config.default_num_inference_steps,
            },
        }

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
    ) -> np.ndarray:
        """Generate video frames from text prompt.

        This is a synchronous method that should be called from a thread pool
        to avoid blocking the event loop.

        Args:
            prompt: Text description of the video to generate.
            negative_prompt: Text to avoid in the generation.
            height: Video height in pixels.
            width: Video width in pixels.
            num_frames: Number of frames to generate.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            seed: Random seed for reproducibility.

        Returns:
            numpy array of shape (num_frames, height, width, 3) with uint8 values.

        Raises:
            RuntimeError: If engine not initialized or generation fails.
        """
        if not self._initialized or self._pipe is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        logger.info(
            f"Generating video: prompt='{prompt[:50]}...', "
            f"size={width}x{height}, frames={num_frames}, steps={num_inference_steps}"
        )

        # Create generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # Run the pipeline
        with torch.no_grad():
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="np",  # Return numpy array
            )

        # result.frames[0] is numpy array (num_frames, height, width, 3) uint8
        frames = result.frames[0]
        logger.info(f"Generated {frames.shape[0]} frames with shape {frames.shape}")

        return frames

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        self._initialized = False
        torch.cuda.empty_cache()
        logger.info("WanDiffusionEngine cleanup complete")
