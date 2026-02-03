# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for diffusion model workers.

This module defines the DiffusionConfig dataclass used for configuring
video and image diffusion workers.
"""

import os
from dataclasses import dataclass
from typing import Optional

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")

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
    store_kv: str = "etcd"
    request_plane: str = "tcp"
    event_plane: str = "nats"

    # Model config
    model_path: str = DEFAULT_VIDEO_MODEL_PATH
    model_type: str = "wan_t2v"  # Must be validated against DiffusionEngine.PIPELINE_REGISTRY
    served_model_name: Optional[str] = None

    # Output config
    output_dir: str = "/tmp/dynamo_videos"

    # Default generation parameters
    default_height: int = 480
    default_width: int = 832
    default_num_frames: int = 81
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
            f"model_type={self.model_type}, "
            f"served_model_name={self.served_model_name}, "
            f"output_dir={self.output_dir}, "
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
