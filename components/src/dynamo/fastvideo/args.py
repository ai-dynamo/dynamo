# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for ``python -m dynamo.fastvideo``."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.configuration.utils import (
    add_argument,
    add_negatable_bool_argument,
    env_or_default,
)
from dynamo.common.utils.runtime import parse_endpoint

DEFAULT_MODEL_PATH = "FastVideo/LTX2-Distilled-Diffusers"
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
TORCH_COMPILE_MODE_CHOICES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


@lru_cache(maxsize=1)
def get_attention_backend_choices() -> tuple[str, ...]:
    """Return FastVideo attention backend names from FastVideo's enum.

    We intentionally read the installed FastVideo source lazily instead of
    importing ``fastvideo`` here. Importing the package during parser setup
    pulls in a large runtime dependency graph and can also collide with this
    repository's own ``dynamo.fastvideo`` package name in some test/tooling
    paths. Reading the enum definition keeps the parser lightweight while still
    using FastVideo as the source of truth.
    """
    interface_path = Path(
        str(
            importlib.metadata.distribution("fastvideo").locate_file(
                "fastvideo/platforms/interface.py"
            )
        )
    )
    module_ast = ast.parse(interface_path.read_text(encoding="utf-8"))

    for node in module_ast.body:
        if not isinstance(node, ast.ClassDef) or node.name != "AttentionBackendEnum":
            continue

        backend_names: list[str] = []
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id != "NO_ATTENTION":
                    backend_names.append(target.id)

        if backend_names:
            return tuple(backend_names)
        break

    raise RuntimeError(
        "Could not determine FastVideo attention backend choices from AttentionBackendEnum"
    )


class FastVideoArgGroup(ArgGroup):
    """FastVideo-specific CLI options."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group(
            "FastVideo Options",
            "Built-in FastVideo backend configuration.",
        )

        add_argument(
            group,
            flag_name="--model-path",
            obsolete_flag="--model",
            env_var="DYN_FASTVIDEO_MODEL_PATH",
            default=DEFAULT_MODEL_PATH,
            help="Hugging Face model path to load.",
        )
        add_argument(
            group,
            flag_name="--served-model-name",
            env_var="DYN_FASTVIDEO_SERVED_MODEL_NAME",
            default=None,
            help="Model name registered in discovery. Defaults to --model-path.",
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
            flag_name="--vae-cpu-offload",
            env_var="DYN_FASTVIDEO_VAE_CPU_OFFLOAD",
            default=True,
            help="Enable VAE CPU offload.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--text-encoder-cpu-offload",
            env_var="DYN_FASTVIDEO_TEXT_ENCODER_CPU_OFFLOAD",
            default=True,
            help="Enable text encoder CPU offload.",
        )
        group.add_argument(
            "--ltx2-vae-tiling",
            dest="ltx2_vae_tiling",
            action=argparse.BooleanOptionalAction,
            default=env_or_default(
                "DYN_FASTVIDEO_LTX2_VAE_TILING", None, value_type=bool
            ),
            help=(
                "Enable LTX-2 VAE tiling overrides.\n"
                "env var: DYN_FASTVIDEO_LTX2_VAE_TILING | default: None"
            ),
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
    vae_cpu_offload: bool = True
    text_encoder_cpu_offload: bool = True
    ltx2_vae_tiling: bool | None = None
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
