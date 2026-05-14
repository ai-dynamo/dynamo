# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for ``python -m dynamo.fastvideo``."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Optional, Sequence

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument
from dynamo.common.utils.runtime import parse_endpoint

if TYPE_CHECKING:
    from fastvideo.api import GeneratorConfig

DEFAULT_MODEL_PATH = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
DEFAULT_COMPONENT_NAME = "backend"
DEFAULT_ENDPOINT_NAME = "generate"
DEFAULT_ATTENTION_BACKEND = "TORCH_SDPA"
DEFAULT_TORCH_COMPILE_BACKEND = "inductor"
DEFAULT_TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
DEFAULT_SIZE = "1280x720"
DEFAULT_SECONDS = 5
DEFAULT_FPS = 24
DEFAULT_NUM_FRAMES = 125
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_SEED = 1024
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
        add_negatable_bool_argument(
            group,
            flag_name="--use-fsdp-inference",
            env_var="DYN_FASTVIDEO_USE_FSDP_INFERENCE",
            default=False,
            help="Enable FSDP inference for additional model sharding.",
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
            help="Enable DiT layerwise CPU offload.",
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
            help="Enable image encoder CPU offload for image-conditioned workloads.",
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
            flag_name="--torch-compile",
            env_var="DYN_FASTVIDEO_ENABLE_TORCH_COMPILE",
            dest="enable_torch_compile",
            default=False,
            help="Enable torch.compile for FastVideo.",
        )
        add_argument(
            group,
            flag_name="--torch-compile-backend",
            env_var="DYN_FASTVIDEO_TORCH_COMPILE_BACKEND",
            default=DEFAULT_TORCH_COMPILE_BACKEND,
            help="torch.compile backend to use when compilation is enabled.",
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
            flag_name="--torch-compile-dynamic",
            env_var="DYN_FASTVIDEO_TORCH_COMPILE_DYNAMIC",
            default=False,
            help="Enable dynamic shapes for torch.compile.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--fp4-quantization",
            env_var="DYN_FASTVIDEO_FP4_QUANTIZATION",
            dest="enable_fp4_quantization",
            default=False,
            help=(
                "Enable FP4 quantization intent in the typed FastVideo "
                "generator config. Runtime support depends on the installed "
                "FastVideo build."
            ),
        )


class FastVideoConfig(DynamoRuntimeConfig):
    """Configuration for the built-in FastVideo backend."""

    namespace: str
    component: str = DEFAULT_COMPONENT_NAME
    endpoint: Optional[str] = None

    model_path: str = DEFAULT_MODEL_PATH
    served_model_name: Optional[str] = None
    num_gpus: int = 1
    use_fsdp_inference: bool = False
    dit_cpu_offload: bool = True
    dit_layerwise_offload: bool = True
    vae_cpu_offload: bool = True
    image_encoder_cpu_offload: bool = True
    text_encoder_cpu_offload: bool = True
    pin_cpu_memory: bool = True
    disable_autocast: bool = False
    attention_backend: str = DEFAULT_ATTENTION_BACKEND
    enable_torch_compile: bool = False
    torch_compile_backend: str = DEFAULT_TORCH_COMPILE_BACKEND
    torch_compile_mode: str = DEFAULT_TORCH_COMPILE_MODE
    torch_compile_fullgraph: bool = True
    torch_compile_dynamic: bool = False
    enable_fp4_quantization: bool = False

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
                "--attention-backend must be one of: "
                f"{', '.join(attention_backend_choices)}"
            )
        self._validate_default_size()

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

    def to_generator_config(self) -> GeneratorConfig:
        """Build the typed FastVideo ``GeneratorConfig`` for model loading."""
        from fastvideo.api import (
            CompileConfig,
            EngineConfig,
            GeneratorConfig,
            OffloadConfig,
            PipelineSelection,
            QuantizationConfig,
        )

        compile_config = CompileConfig(enabled=self.enable_torch_compile)
        if self.enable_torch_compile:
            compile_config.backend = self.torch_compile_backend
            compile_config.fullgraph = self.torch_compile_fullgraph
            compile_config.mode = self.torch_compile_mode
            compile_config.dynamic = self.torch_compile_dynamic

        quantization = None
        experimental: dict[str, object] = {}
        if self.enable_fp4_quantization:
            quantization = QuantizationConfig(transformer_quant="FP4")
            experimental["fp4_quantization"] = True

        return GeneratorConfig(
            model_path=self.model_path,
            engine=EngineConfig(
                num_gpus=self.num_gpus,
                offload=OffloadConfig(
                    dit=self.dit_cpu_offload,
                    dit_layerwise=self.dit_layerwise_offload,
                    text_encoder=self.text_encoder_cpu_offload,
                    image_encoder=self.image_encoder_cpu_offload,
                    vae=self.vae_cpu_offload,
                    pin_cpu_memory=self.pin_cpu_memory,
                ),
                compile=compile_config,
                use_fsdp_inference=self.use_fsdp_inference,
                disable_autocast=self.disable_autocast,
                quantization=quantization,
            ),
            pipeline=PipelineSelection(experimental=experimental),
        )

    def _validate_default_size(self) -> None:
        try:
            width_str, height_str = self.default_size.lower().split("x", 1)
            width, height = int(width_str), int(height_str)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid default_size '{self.default_size}', expected 'WxH'"
            ) from exc

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid default_size '{self.default_size}', "
                "width and height must be positive"
            )


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
