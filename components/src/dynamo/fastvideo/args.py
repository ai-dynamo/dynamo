# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for ``python -m dynamo.fastvideo``."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument
from dynamo.common.utils.runtime import parse_endpoint

DEFAULT_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_COMPONENT_NAME = "backend"
DEFAULT_ENDPOINT_NAME = "generate"
DEFAULT_ATTENTION_BACKEND = "TORCH_SDPA"
DEFAULT_TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
DEFAULT_SIZE = "1920x1088"
DEFAULT_SECONDS = 5
DEFAULT_FPS = 24
DEFAULT_NUM_FRAMES = 121
DEFAULT_NUM_INFERENCE_STEPS = 5
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_SEED = 10
ATTENTION_BACKEND_CHOICES = (
    "FLASH_ATTN",
    "TORCH_SDPA",
    "SAGE_ATTN",
    "SAGE_ATTN_THREE",
    "VIDEO_SPARSE_ATTN",
    "VMOBA_ATTN",
    "SLA_ATTN",
    "SAGE_SLA_ATTN",
)
TORCH_COMPILE_MODE_CHOICES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


def get_attention_backend_choices() -> tuple[str, ...]:
    """Return the pinned FastVideo attention backends supported by Dynamo."""
    return ATTENTION_BACKEND_CHOICES


class FastVideoArgGroup(ArgGroup):
    """FastVideo-specific CLI options."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group(
            "FastVideo Options",
            "Built-in FastVideo backend configuration.",
        )

        add_argument(
            group,
            flag_name="--model",
            obsolete_flag="--model-path",
            dest="model_path",
            env_var="DYN_FASTVIDEO_MODEL_PATH",
            default=DEFAULT_MODEL_PATH,
            help="Hugging Face model path to load.",
        )
        add_argument(
            group,
            flag_name="--served-model-name",
            env_var="DYN_FASTVIDEO_SERVED_MODEL_NAME",
            default=None,
            help="Model name registered in discovery. Defaults to --model.",
        )
        add_argument(
            group,
            flag_name="--num-gpus",
            env_var="DYN_FASTVIDEO_NUM_GPUS",
            default=1,
            arg_type=int,
            help="Number of GPUs to pass to FastVideo.",
        )
        add_argument(
            group,
            flag_name="--attention-backend",
            env_var="FASTVIDEO_ATTENTION_BACKEND",
            default=DEFAULT_ATTENTION_BACKEND,
            choices=list(get_attention_backend_choices()),
            help="Attention backend for FastVideo inference.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--dit-cpu-offload",
            env_var="DYN_FASTVIDEO_DIT_CPU_OFFLOAD",
            default=True,
            help="Enable DiT CPU offload.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--dit-layerwise-offload",
            env_var="DYN_FASTVIDEO_DIT_LAYERWISE_OFFLOAD",
            default=True,
            help=(
                "Enable DiT layerwise offload. FastVideo treats this as an "
                "alternative to DiT CPU offload."
            ),
        )
        add_negatable_bool_argument(
            group,
            flag_name="--use-fsdp-inference",
            env_var="DYN_FASTVIDEO_USE_FSDP_INFERENCE",
            default=False,
            help="Enable FSDP inference for additional model sharding.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--vae-cpu-offload",
            env_var="DYN_FASTVIDEO_VAE_CPU_OFFLOAD",
            default=True,
            help="Enable VAE CPU offload.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--image-encoder-cpu-offload",
            env_var="DYN_FASTVIDEO_IMAGE_ENCODER_CPU_OFFLOAD",
            default=True,
            help="Enable image encoder CPU offload (used by image-conditioned workloads).",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--text-encoder-cpu-offload",
            env_var="DYN_FASTVIDEO_TEXT_ENCODER_CPU_OFFLOAD",
            default=True,
            help="Enable text encoder CPU offload.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--pin-cpu-memory",
            env_var="DYN_FASTVIDEO_PIN_CPU_MEMORY",
            default=True,
            help="Pin host memory for CPU offload transfers.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--disable-autocast",
            env_var="DYN_FASTVIDEO_DISABLE_AUTOCAST",
            default=False,
            help="Disable autocast in FastVideo denoising/decoding paths.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--torch-compile",
            env_var="DYN_FASTVIDEO_ENABLE_TORCH_COMPILE",
            dest="enable_torch_compile",
            default=False,
            help="Enable torch.compile for FastVideo.",
        )
        add_argument(
            group,
            flag_name="--torch-compile-mode",
            env_var="DYN_FASTVIDEO_TORCH_COMPILE_MODE",
            default=DEFAULT_TORCH_COMPILE_MODE,
            choices=list(TORCH_COMPILE_MODE_CHOICES),
            help="torch.compile mode to use when compilation is enabled.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--torch-compile-fullgraph",
            env_var="DYN_FASTVIDEO_TORCH_COMPILE_FULLGRAPH",
            default=True,
            help="Enable torch.compile fullgraph mode.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--fp4-quantization",
            env_var="DYN_FASTVIDEO_FP4_QUANTIZATION",
            dest="enable_fp4_quantization",
            default=False,
            help=(
                "Enable FP4 quantization for FastVideo DiT weights. "
                "Only supported on Blackwell GPUs and newer "
                "(compute capability 10.0+)."
            ),
        )
        add_argument(
            group,
            flag_name="--extra-generator-args-file",
            env_var="DYN_FASTVIDEO_EXTRA_GENERATOR_ARGS_FILE",
            default="",
            help="Path to a YAML or JSON file containing additional keyword arguments to pass to FastVideo's generator.",
        )
        add_argument(
            group,
            flag_name="--override-generator-args-json",
            env_var="DYN_FASTVIDEO_OVERRIDE_GENERATOR_ARGS_JSON",
            default="",
            help='JSON string to override specific FastVideo generator keyword arguments. Example: \'{"torch_compile_kwargs": {"mode": "reduce-overhead"}}\'',
        )


class FastVideoConfig(DynamoRuntimeConfig):
    """Configuration for the built-in FastVideo backend."""

    namespace: str
    component: str = DEFAULT_COMPONENT_NAME
    endpoint: Optional[str] = None

    model_path: str = DEFAULT_MODEL_PATH
    served_model_name: Optional[str] = None
    num_gpus: int = 1
    attention_backend: str = DEFAULT_ATTENTION_BACKEND
    dit_cpu_offload: bool = True
    dit_layerwise_offload: bool = True
    use_fsdp_inference: bool = False
    vae_cpu_offload: bool = True
    image_encoder_cpu_offload: bool = True
    text_encoder_cpu_offload: bool = True
    pin_cpu_memory: bool = True
    disable_autocast: bool = False
    enable_torch_compile: bool = False
    torch_compile_mode: str = DEFAULT_TORCH_COMPILE_MODE
    torch_compile_fullgraph: bool = True
    enable_fp4_quantization: bool = False
    extra_generator_args_file: str = ""
    override_generator_args_json: str = ""

    default_size: str = DEFAULT_SIZE
    default_seconds: int = DEFAULT_SECONDS
    default_fps: int = DEFAULT_FPS
    default_num_frames: int = DEFAULT_NUM_FRAMES
    default_num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    default_guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    default_seed: int = DEFAULT_SEED

    def validate(self) -> None:
        super().validate()

        if self.num_gpus <= 0:
            raise ValueError("--num-gpus must be > 0")
        attention_backend_choices = get_attention_backend_choices()
        if self.attention_backend not in attention_backend_choices:
            raise ValueError(
                f"--attention-backend must be one of: {', '.join(attention_backend_choices)}"
            )

        if not self.served_model_name:
            self.served_model_name = self.model_path

        endpoint = (
            self.endpoint
            or f"dyn://{self.namespace}.{self.component}.{DEFAULT_ENDPOINT_NAME}"
        )
        parsed_namespace, parsed_component, parsed_endpoint = parse_endpoint(endpoint)
        self.namespace = parsed_namespace
        self.component = parsed_component
        self.endpoint = parsed_endpoint


def parse_fastvideo_args(argv: Sequence[str] | None = None) -> FastVideoConfig:
    """Parse CLI arguments for the FastVideo backend."""

    parser = argparse.ArgumentParser(
        description="Dynamo FastVideo backend",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
        conflict_handler="resolve",
    )
    DynamoRuntimeArgGroup().add_arguments(parser)
    FastVideoArgGroup().add_arguments(parser)
    # Re-declare to hide from help; FastVideo always produces video.
    parser.add_argument(
        "--output-modalities", default=["video"], nargs="*", help=argparse.SUPPRESS
    )

    args = parser.parse_args(None if argv is None else list(argv))
    args.output_modalities = ["video"]

    config = FastVideoConfig.from_cli_args(args)
    config.validate()
    return config
