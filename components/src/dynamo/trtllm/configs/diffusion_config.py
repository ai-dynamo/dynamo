# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for diffusion model workers.

This module defines the DiffusionConfig dataclass used for configuring
video and image diffusion workers.

Fields map to TensorRT-LLM's DiffusionArgs sub-configs:
- PipelineConfig: torch_compile, CUDA graph, warmup, offloading, fuse_qkv
- AttentionConfig: attention backend (VANILLA, TRTLLM)
- ParallelConfig: dit_*_size parallelism dimensions
- TeaCacheConfig: caching optimization
- QuantConfig: quantization algorithm and dynamic flags
"""

from dataclasses import dataclass, field
from typing import Optional

from dynamo.common.utils.namespace import get_worker_namespace

DYN_NAMESPACE = get_worker_namespace()

# Default model paths
DEFAULT_VIDEO_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model workers (video/image generation).

    This configuration is used by DiffusionEngine and diffusion handlers.
    It can be populated from command-line arguments in backend_args.py.
    """

    # Dynamo runtime config
    namespace: str = DYN_NAMESPACE
    component: str = "diffusion"
    endpoint: str = "generate"
    discovery_backend: str = "etcd"
    request_plane: str = "tcp"
    event_plane: str = "nats"

    # Model config
    model_path: str = DEFAULT_VIDEO_MODEL_PATH
    served_model_name: Optional[str] = None
    # torch_dtype for model loading. Options: "bfloat16", "float16", "float32"
    # bfloat16 is recommended for Ampere+ GPUs (A100, H100, etc.)
    # float16 can be used on older GPUs (V100, etc.)
    torch_dtype: str = "bfloat16"
    # HuggingFace Hub revision (branch, tag, or commit SHA) for model download.
    revision: Optional[str] = None

    # Media storage
    media_output_fs_url: str = "file:///tmp/dynamo_media"
    media_output_http_url: Optional[str] = None

    # Default generation parameters
    default_height: int = 480
    default_width: int = 832
    # Maximum allowed dimensions to prevent OOM. Can be increased if GPU has sufficient VRAM.
    max_height: int = 4096
    max_width: int = 4096
    default_num_frames: int = 81
    default_fps: int = 24  # Used for both frame count calculation and video encoding
    default_seconds: int = 4  # Default video duration when only fps is specified
    default_num_inference_steps: int = 50
    default_guidance_scale: float = 5.0

    # ── Pipeline optimization config (maps to PipelineConfig) ──
    disable_torch_compile: bool = False
    torch_compile_mode: str = "default"
    # Enable torch.compile fullgraph mode (stricter but potentially faster)
    enable_fullgraph: bool = False
    # QKV fusion for transformer attention layers
    fuse_qkv: bool = True
    # CUDA graph capture for transformer forward passes
    # (mutually exclusive with torch.compile — torch.compile takes priority)
    enable_cuda_graph: bool = False
    # Enable per-layer NVTX markers for profiling
    enable_layerwise_nvtx_marker: bool = False
    # Number of denoising steps to run during warmup (0 to disable)
    warmup_steps: int = 1

    # ── Attention config (maps to AttentionConfig) ──
    # Attention backend: "VANILLA" (PyTorch SDPA) or "TRTLLM"
    attn_backend: str = "VANILLA"

    # ── Quantization config (maps to DiffusionArgs.quant_config) ──
    # Quantization algorithm. Options:
    #   None (no quantization), "FP8", "FP8_BLOCK_SCALES", "NVFP4",
    #   "W4A16_AWQ", "W4A8_AWQ", "W8A8_SQ_PER_CHANNEL"
    quant_algo: Optional[str] = None
    # Enable dynamic weight quantization (quantize BF16 weights on-the-fly during loading)
    quant_dynamic: bool = True

    # ── TeaCache optimization config (maps to TeaCacheConfig) ──
    enable_teacache: bool = False
    teacache_use_ret_steps: bool = True
    teacache_thresh: float = 0.2

    # ── Parallelism config (maps to ParallelConfig) ──
    dit_dp_size: int = 1
    dit_tp_size: int = 1
    dit_ulysses_size: int = 1
    dit_ring_size: int = 1
    dit_cfg_size: int = 1
    dit_fsdp_size: int = 1

    # ── Offloading config (maps to PipelineConfig) ──
    enable_async_cpu_offload: bool = False

    # ── Component loading options ──
    # Components to skip loading (e.g., ["text_encoder", "vae"]).
    # Valid values: "transformer", "vae", "text_encoder", "tokenizer",
    #               "scheduler", "image_encoder", "image_processor"
    skip_components: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"DiffusionConfig("
            f"namespace={self.namespace}, "
            f"component={self.component}, "
            f"endpoint={self.endpoint}, "
            f"model_path={self.model_path}, "
            f"served_model_name={self.served_model_name}, "
            f"media_output_fs_url={self.media_output_fs_url}, "
            f"default_height={self.default_height}, "
            f"default_width={self.default_width}, "
            f"default_num_frames={self.default_num_frames}, "
            f"default_num_inference_steps={self.default_num_inference_steps}, "
            f"enable_teacache={self.enable_teacache}, "
            f"attn_backend={self.attn_backend}, "
            f"quant_algo={self.quant_algo}, "
            f"enable_cuda_graph={self.enable_cuda_graph}, "
            f"warmup_steps={self.warmup_steps}, "
            f"dit_dp_size={self.dit_dp_size}, "
            f"dit_tp_size={self.dit_tp_size})"
        )
