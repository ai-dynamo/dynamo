# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic Diffusion Engine wrapper for visual_gen pipelines.

This module provides a unified interface for various diffusion models
(Wan, Flux, Cosmos, etc.) through a pipeline registry system.

The pipeline type is auto-detected from model_index.json (shipped with every
HuggingFace Diffusers model), eliminating the need for a --model-type flag.

Requirements:
    - visual_gen: Part of TensorRT-LLM, located at tensorrt_llm/visual_gen/.
      Currently on the feat/visual_gen branch (not yet merged to main).
      See: https://github.com/NVIDIA/TensorRT-LLM/tree/feat/visual_gen/tensorrt_llm/visual_gen
    - See docs/backends/trtllm/README.md for setup instructions.

Note on imports:
    visual_gen is imported lazily in initialize() because:
    1. It's a heavy package that may not be installed in all environments
    2. Importing at module load would fail if visual_gen is not available
    3. This allows the module to be imported for type checking and validation
       without requiring visual_gen to be installed
"""

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from dynamo.trtllm.configs.diffusion_config import DiffusionConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineInfo:
    """Auto-detected pipeline information from model_index.json."""

    module_path: str
    class_name: str
    modalities: list[str]
    config_overrides: dict[str, Any]


class DiffusionEngine:
    """Generic wrapper for visual_gen diffusion pipelines.

    This engine provides:
    - Auto-detection of pipeline class from model_index.json
    - A registry mapping diffusers _class_name to visual_gen pipelines
    - Lazy loading of pipeline modules
    - Common interface for video/image generation

    Example:
        >>> engine = DiffusionEngine(config)
        >>> await engine.initialize()
        >>> frames = engine.generate(prompt="A cat playing piano", ...)
    """

    # Registry: diffusers _class_name -> (module_path, visual_gen_class, supported_modalities)
    # The _class_name comes from model_index.json shipped with every HF Diffusers model.
    # torch_compile_models is derived dynamically from transformer* keys in model_index.json.
    #
    # NOTE: This registry is initially focused on Wan text-to-video models.
    # Follow-up PRs will extend support for other model families (Flux, Cosmos, etc.)
    # which may require additional config fields in DiffusionConfig.
    PIPELINE_REGISTRY: dict[str, tuple[str, str, list[str]]] = {
        "WanPipeline": (
            "visual_gen.pipelines.wan_pipeline",
            "ditWanPipeline",
            ["video_diffusion"],
        ),
        # TODO(nv-yna): Add support for WanImageToVideoPipeline, FluxPipeline, etc.
    }

    @classmethod
    def detect_pipeline_info(cls, model_path: str) -> PipelineInfo:
        """Auto-detect pipeline class from model's model_index.json.

        Reads model_index.json (local path or HuggingFace Hub) to determine:
        - Which visual_gen pipeline class to use (via _class_name)
        - Which transformer models to torch.compile (via transformer* keys)

        Args:
            model_path: Local path or HuggingFace model identifier.

        Returns:
            PipelineInfo with module_path, class_name, modalities, and config_overrides.

        Raises:
            ValueError: If _class_name is not in the registry.
            FileNotFoundError: If model_index.json cannot be found locally or on HF Hub.
        """
        # Try local path first
        local_index = Path(model_path) / "model_index.json"
        if local_index.exists():
            with open(local_index) as f:
                model_index = json.load(f)
        else:
            # Download from HuggingFace Hub
            from huggingface_hub import hf_hub_download

            index_path = hf_hub_download(model_path, "model_index.json")
            with open(index_path) as f:
                model_index = json.load(f)

        class_name = model_index.get("_class_name")
        if class_name not in cls.PIPELINE_REGISTRY:
            supported = list(cls.PIPELINE_REGISTRY.keys())
            raise ValueError(
                f"Unsupported diffusion pipeline '{class_name}' from model '{model_path}'.\n"
                f"Supported pipelines: {', '.join(supported)}\n"
                f"Check that model_index.json has a supported _class_name."
            )

        module_path, vg_class, modalities = cls.PIPELINE_REGISTRY[class_name]

        # Derive torch_compile_models from transformer* keys in model_index.json
        transformer_keys = sorted(k for k in model_index if k.startswith("transformer"))
        torch_compile_models = (
            ",".join(transformer_keys) if transformer_keys else "transformer"
        )

        config_overrides = {"torch_compile_models": torch_compile_models}

        return PipelineInfo(module_path, vg_class, modalities, config_overrides)

    def __init__(self, config: "DiffusionConfig"):
        """Initialize the engine with configuration.

        Auto-detects the pipeline type from config.model_path's model_index.json.

        Args:
            config: Diffusion generation configuration.

        Raises:
            ValueError: If the model's pipeline type is not supported.
        """
        info = self.detect_pipeline_info(config.model_path)

        self.config = config
        self._pipeline = None
        self._initialized = False

        self._module_path = info.module_path
        self._class_name = info.class_name
        self._supported_modalities = info.modalities
        self._config_overrides = info.config_overrides

    async def initialize(self) -> None:
        """Load and configure the diffusion pipeline.

        This is called once at worker startup to load the model.
        The specific pipeline class is determined by the auto-detected pipeline type.
        """
        if self._initialized:
            logger.warning("Engine already initialized, skipping")
            return

        logger.info(
            f"Initializing DiffusionEngine: pipeline={self._class_name}, "
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
        # Convert torch_dtype string to actual torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
        logger.info(
            f"Loading pipeline from {self.config.model_path} with dtype={self.config.torch_dtype}"
        )
        self._pipeline = pipeline_class.from_pretrained(
            self.config.model_path,
            torch_dtype=torch_dtype,
            **dit_configs,
        )

        # Move to target device
        # NOTE: HuggingFace's from_pretrained() loads to CPU by default,
        # so we must explicitly move to GPU for optimal performance.
        if self.device == "cuda":
            logger.info("Moving pipeline to GPU...")
            self._pipeline.to(self.device)
            logger.info("Pipeline moved to GPU successfully")
        else:
            logger.info("CPU offload enabled, pipeline stays on CPU")

        self._initialized = True
        logger.info(f"DiffusionEngine initialization complete: {self._class_name}")

    def _build_dit_configs(self) -> dict[str, Any]:
        """Build dit_configs dict from DiffusionConfig.

        Returns:
            Configuration dictionary for visual_gen's setup_configs.
        """
        # Get torch_compile_models from auto-detected config overrides
        # Each pipeline in PIPELINE_REGISTRY specifies its required settings
        torch_compile_models = self._config_overrides.get(
            "torch_compile_models", "transformer"
        )

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
        # Device must match pipeline device (CPU if offload enabled, CUDA otherwise)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

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
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"DiffusionEngine cleanup complete: {self._class_name}")

    @property
    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._initialized

    @property
    def supported_modalities(self) -> list[str]:
        """Get the modalities supported by this engine's model type."""
        return self._supported_modalities

    @property
    def device(self) -> str:
        """Get the device where the pipeline runs.

        Returns:
            "cpu" if CPU offload is enabled, "cuda" otherwise.
        """
        return "cpu" if self.config.enable_async_cpu_offload else "cuda"
