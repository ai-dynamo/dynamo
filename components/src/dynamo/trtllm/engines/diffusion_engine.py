# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic Diffusion Engine wrapper for visual_gen pipelines.

This module provides a unified interface for various diffusion models
(Wan, Flux, Cosmos, etc.) through a pipeline registry system.
"""

import importlib
import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from dynamo.trtllm.configs.diffusion_config import DiffusionConfig

logger = logging.getLogger(__name__)


class DiffusionEngine:
    """Generic wrapper for visual_gen diffusion pipelines.

    This engine provides:
    - A registry mapping model_type to pipeline classes
    - Validation of model_type against supported modalities
    - Lazy loading of pipeline modules
    - Common interface for video/image generation

    Example:
        >>> engine = DiffusionEngine("wan_t2v", config)
        >>> await engine.initialize()
        >>> frames = engine.generate(prompt="A cat playing piano", ...)
    """

    # Registry: model_type -> (module_path, class_name, supported_modalities)
    # The module_path is relative to visual_gen package
    PIPELINE_REGISTRY: dict[str, tuple[str, str, list[str]]] = {
        # Video diffusion models (text-to-video)
        "wan_t2v": (
            "visual_gen.pipelines.wan_pipeline",
            "ditWanPipeline",
            ["video_diffusion"],
        ),
        # Video diffusion models (image-to-video)
        "wan_i2v": (
            "visual_gen.pipelines.wan_pipeline",
            "ditWanImageToVideoPipeline",
            ["video_diffusion"],
        ),
        # Video diffusion models (Cosmos)
        "cosmos": (
            "visual_gen.pipelines.cosmos_pipeline",
            "ditCosmosPipeline",
            ["video_diffusion"],
        ),
        # Image diffusion models
        "flux": (
            "visual_gen.pipelines.flux_pipeline",
            "ditFluxPipeline",
            ["image_diffusion"],
        ),
        "flux2": (
            "visual_gen.pipelines.flux2_pipeline",
            "ditFlux2Pipeline",
            ["image_diffusion"],
        ),
    }

    @classmethod
    def get_allowed_model_types(cls, modality: str) -> list[str]:
        """Get list of model types allowed for a given modality.

        Args:
            modality: The modality string (e.g., "video_diffusion", "image_diffusion").

        Returns:
            List of model_type strings that support the given modality.
        """
        return [
            model_type
            for model_type, (_, _, modalities) in cls.PIPELINE_REGISTRY.items()
            if modality in modalities
        ]

    @classmethod
    def validate_model_type(cls, modality: str, model_type: str) -> None:
        """Validate that model_type is allowed for the given modality.

        Args:
            modality: The modality string (e.g., "video_diffusion").
            model_type: The model type string (e.g., "wan_t2v").

        Raises:
            ValueError: If model_type is not valid for the given modality,
                with a helpful error message showing allowed values.
        """
        allowed = cls.get_allowed_model_types(modality)

        if model_type not in allowed:
            raise ValueError(
                f"Invalid --model-type '{model_type}' for --modality '{modality}'.\n"
                f"Allowed values: {', '.join(allowed)}\n"
                f"\nUsage: python -m dynamo.trtllm --modality {modality} "
                f"--model-type {allowed[0] if allowed else '<model>'} --model-path ..."
            )

    @classmethod
    def is_valid_model_type(cls, model_type: str) -> bool:
        """Check if a model_type exists in the registry.

        Args:
            model_type: The model type string to check.

        Returns:
            True if the model_type is registered, False otherwise.
        """
        return model_type in cls.PIPELINE_REGISTRY

    def __init__(self, model_type: str, config: "DiffusionConfig"):
        """Initialize the engine with configuration.

        Args:
            model_type: The type of model to load (e.g., "wan_t2v", "flux").
            config: Diffusion generation configuration.

        Raises:
            ValueError: If model_type is not in the registry.
        """
        if model_type not in self.PIPELINE_REGISTRY:
            all_types = list(self.PIPELINE_REGISTRY.keys())
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Available types: {', '.join(all_types)}"
            )

        self.model_type = model_type
        self.config = config
        self._pipeline = None
        self._initialized = False

        # Get registry entry
        self._module_path, self._class_name, self._supported_modalities = (
            self.PIPELINE_REGISTRY[model_type]
        )

    async def initialize(self) -> None:
        """Load and configure the diffusion pipeline.

        This is called once at worker startup to load the model.
        The specific pipeline class is determined by the model_type.
        """
        if self._initialized:
            logger.warning("Engine already initialized, skipping")
            return

        logger.info(
            f"Initializing DiffusionEngine: model_type={self.model_type}, "
            f"model_path={self.config.model_path}"
        )

        # Import visual_gen setup
        from visual_gen import setup_configs

        # Build configuration dict based on model type
        dit_configs = self._build_dit_configs()
        logger.info(f"dit_configs: {dit_configs}")

        # Setup global configuration (required before pipeline loading)
        setup_configs(**dit_configs)

        # Dynamically import the pipeline class
        logger.info(f"Importing pipeline from {self._module_path}.{self._class_name}")
        module = importlib.import_module(self._module_path)
        pipeline_class = getattr(module, self._class_name)

        # Load the pipeline
        logger.info(f"Loading pipeline from {self.config.model_path}")
        self._pipeline = pipeline_class.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            **dit_configs,
        )

        # Move to GPU if not using CPU offload
        if not self.config.enable_async_cpu_offload:
            logger.info("Pipeline loaded (device placement managed by pipeline)")

        self._initialized = True
        logger.info(f"DiffusionEngine initialization complete: {self.model_type}")

    def _build_dit_configs(self) -> dict[str, Any]:
        """Build dit_configs dict from DiffusionConfig.

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
        """Generate video/image frames from text prompt.

        This is a synchronous method that should be called from a thread pool
        to avoid blocking the event loop.

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
            numpy array of shape (num_frames, height, width, 3) with uint8 values
            for video, or (height, width, 3) for images.

        Raises:
            RuntimeError: If engine not initialized or generation fails.
        """
        if not self._initialized or self._pipeline is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        logger.info(
            f"Generating: prompt='{prompt[:50]}...', "
            f"size={width}x{height}, frames={num_frames}, steps={num_inference_steps}"
        )

        # Create generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # Run the pipeline
        with torch.no_grad():
            result = self._pipeline(
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
        logger.info(f"Generated output with shape {frames.shape}")

        return frames

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        self._initialized = False
        torch.cuda.empty_cache()
        logger.info(f"DiffusionEngine cleanup complete: {self.model_type}")

    @property
    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._initialized

    @property
    def supported_modalities(self) -> list[str]:
        """Get the modalities supported by this engine's model type."""
        return self._supported_modalities
