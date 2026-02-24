# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for diffusion model workers.

This module defines the DiffusionConfig dataclass used for configuring
video and image diffusion workers.
"""

from dataclasses import dataclass
from typing import Optional

from dynamo.common.utils.namespace import get_worker_namespace

DYN_NAMESPACE = get_worker_namespace()

# Default model paths
DEFAULT_VIDEO_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model workers (video/image generation).

    This configuration is used by DiffusionEngine and diffusion handlers.
    It can be populated from command-line arguments in trtllm_utils.py.
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

    # visual_gen optimization config
    enable_teacache: bool = False
    teacache_use_ret_steps: bool = True
    teacache_thresh: float = 0.2
    attn_type: str = "default"
    linear_type: str = "default"
    disable_torch_compile: bool = False
    torch_compile_mode: str = "default"

    # Parallelism config (DiTParallelConfig)
    dit_dp_size: int = 1
    dit_tp_size: int = 1
    dit_ulysses_size: int = 1
    dit_ring_size: int = 1
    dit_cfg_size: int = 1
    dit_fsdp_size: int = 1

    # CPU offload config
    enable_async_cpu_offload: bool = False
    visual_gen_block_cpu_offload_stride: int = 1

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
            f"attn_type={self.attn_type}, "
            f"linear_type={self.linear_type}, "
            f"dit_dp_size={self.dit_dp_size}, "
            f"dit_tp_size={self.dit_tp_size})"
        )
