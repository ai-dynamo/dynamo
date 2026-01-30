# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line arguments and configuration for TensorRT-LLM Video Diffusion."""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")

# Default endpoints for video diffusion workers
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.trtllm_diffusion.generate"
DEFAULT_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass
class VideoConfig:
    """Configuration for TensorRT-LLM Video Diffusion worker."""

    # Dynamo runtime config
    namespace: str = DYN_NAMESPACE
    component: str = "trtllm_diffusion"
    endpoint: str = "generate"
    store_kv: str = ""
    request_plane: str = ""

    # Model config
    model_path: str = DEFAULT_MODEL_PATH
    served_model_name: Optional[str] = None

    # Output config
    output_dir: str = "/tmp/dynamo_videos"

    # Default generation parameters
    default_height: int = 480
    default_width: int = 832
    default_num_frames: int = 81
    default_num_inference_steps: int = 50
    default_guidance_scale: float = 5.0

    # visual_gen config
    enable_teacache: bool = False
    teacache_use_ret_steps: bool = True
    teacache_thresh: float = 0.2
    attn_type: str = "default"
    linear_type: str = "default"
    disable_torch_compile: bool = False
    torch_compile_mode: str = "default"

    # Parallelism config
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
            f"VideoConfig(namespace={self.namespace}, "
            f"component={self.component}, "
            f"endpoint={self.endpoint}, "
            f"model_path={self.model_path}, "
            f"served_model_name={self.served_model_name}, "
            f"output_dir={self.output_dir}, "
            f"default_height={self.default_height}, "
            f"default_width={self.default_width}, "
            f"default_num_frames={self.default_num_frames}, "
            f"default_num_inference_steps={self.default_num_inference_steps}, "
            f"enable_teacache={self.enable_teacache}, "
            f"attn_type={self.attn_type}, "
            f"linear_type={self.linear_type})"
        )


def parse_endpoint(endpoint: str) -> tuple[str, str, str]:
    """Parse a Dynamo endpoint string into its components.

    Args:
        endpoint: Endpoint string in format 'namespace.component.endpoint'
            or 'dyn://namespace.component.endpoint'.

    Returns:
        Tuple of (namespace, component, endpoint_name).

    Raises:
        ValueError: If endpoint format is invalid.
    """
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        raise ValueError(
            f"Invalid endpoint format: '{endpoint}'. "
            "Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
    namespace, component, endpoint_name = endpoint_parts
    return namespace, component, endpoint_name


def parse_args() -> VideoConfig:
    """Parse command-line arguments for the video diffusion worker.

    Returns:
        VideoConfig: Parsed configuration object.
    """
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM Video Diffusion server integrated with Dynamo."
    )

    # Dynamo runtime args
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=f"Dynamo endpoint (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--store-kv",
        type=str,
        default=os.environ.get("DYN_STORE_KV", ""),
        help="etcd URL for discovery (e.g., etcd://localhost:2379)",
    )
    parser.add_argument(
        "--request-plane",
        type=str,
        default=os.environ.get("DYN_REQUEST_PLANE", ""),
        help="NATS URL for request plane (e.g., nats://localhost:4222)",
    )

    # Model args
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"HuggingFace model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Model name to serve (default: same as model-path)",
    )

    # Output args
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/dynamo_videos",
        help="Directory to store generated videos",
    )

    # Default generation parameters
    parser.add_argument(
        "--default-height",
        type=int,
        default=480,
        help="Default video height (default: 480)",
    )
    parser.add_argument(
        "--default-width",
        type=int,
        default=832,
        help="Default video width (default: 832)",
    )
    parser.add_argument(
        "--default-num-frames",
        type=int,
        default=81,
        help="Default number of frames (default: 81)",
    )
    parser.add_argument(
        "--default-num-inference-steps",
        type=int,
        default=50,
        help="Default inference steps (default: 50)",
    )
    parser.add_argument(
        "--default-guidance-scale",
        type=float,
        default=5.0,
        help="Default guidance scale (default: 5.0)",
    )

    # visual_gen optimization args
    parser.add_argument(
        "--enable-teacache",
        action="store_true",
        help="Enable TeaCache for faster generation",
    )
    parser.add_argument(
        "--teacache-thresh",
        type=float,
        default=0.2,
        help="TeaCache threshold (default: 0.2)",
    )
    parser.add_argument(
        "--attn-type",
        type=str,
        default="default",
        choices=["default", "sage-attn", "sparse-videogen", "sparse-videogen2"],
        help="Attention type (default: default)",
    )
    parser.add_argument(
        "--linear-type",
        type=str,
        default="default",
        choices=["default", "trtllm-fp8-blockwise", "trtllm-fp8-per-tensor", "trtllm-nvfp4"],
        help="Linear type for quantization (default: default)",
    )
    parser.add_argument(
        "--disable-torch-compile",
        action="store_true",
        help="Disable torch.compile optimization",
    )
    parser.add_argument(
        "--torch-compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: default)",
    )

    # Parallelism args
    parser.add_argument(
        "--dit-dp-size",
        type=int,
        default=1,
        help="Data parallel size (default: 1)",
    )
    parser.add_argument(
        "--dit-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--dit-ulysses-size",
        type=int,
        default=1,
        help="Ulysses parallel size (default: 1)",
    )
    parser.add_argument(
        "--dit-cfg-size",
        type=int,
        default=1,
        help="CFG parallel size (default: 1)",
    )

    # CPU offload args
    parser.add_argument(
        "--enable-async-cpu-offload",
        action="store_true",
        help="Enable async CPU offload for memory efficiency",
    )

    args = parser.parse_args()

    # Parse endpoint into components
    namespace, component, endpoint_name = parse_endpoint(args.endpoint)

    # Build config
    config = VideoConfig(
        namespace=namespace,
        component=component,
        endpoint=endpoint_name,
        store_kv=args.store_kv,
        request_plane=args.request_plane,
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        output_dir=args.output_dir,
        default_height=args.default_height,
        default_width=args.default_width,
        default_num_frames=args.default_num_frames,
        default_num_inference_steps=args.default_num_inference_steps,
        default_guidance_scale=args.default_guidance_scale,
        enable_teacache=args.enable_teacache,
        teacache_thresh=args.teacache_thresh,
        attn_type=args.attn_type,
        linear_type=args.linear_type,
        disable_torch_compile=args.disable_torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        dit_dp_size=args.dit_dp_size,
        dit_tp_size=args.dit_tp_size,
        dit_ulysses_size=args.dit_ulysses_size,
        dit_cfg_size=args.dit_cfg_size,
        enable_async_cpu_offload=args.enable_async_cpu_offload,
    )

    return config
