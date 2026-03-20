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
from dynamo.common.configuration.utils import add_argument
from dynamo.common.utils.runtime import parse_endpoint

DEFAULT_MODEL_PATH = "FastVideo/LTX2-Distilled-Diffusers"
DEFAULT_COMPONENT_NAME = "backend"
DEFAULT_ENDPOINT_NAME = "generate"
DEFAULT_ATTENTION_BACKEND = "TORCH_SDPA"
DEFAULT_OPTIMIZATION_PROFILE = "none"
DEFAULT_SIZE = "1920x1088"
DEFAULT_SECONDS = 5
DEFAULT_FPS = 24
DEFAULT_NUM_FRAMES = 121
DEFAULT_NUM_INFERENCE_STEPS = 5
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_SEED = 10

OPTIMIZATION_PROFILE_CHOICES = ("none", "latency")


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
            flag_name="--optimization-profile",
            env_var="DYN_FASTVIDEO_OPTIMIZATION_PROFILE",
            default=DEFAULT_OPTIMIZATION_PROFILE,
            choices=list(OPTIMIZATION_PROFILE_CHOICES),
            help=(
                "Supported backend optimization profile. 'latency' enables "
                "FastVideo's compile/refine path and uses FP4 quantization "
                "automatically on Blackwell GPUs when available."
            ),
        )
        add_argument(
            group,
            flag_name="--attention-backend",
            env_var="FASTVIDEO_ATTENTION_BACKEND",
            default=DEFAULT_ATTENTION_BACKEND,
            choices=list(get_attention_backend_choices()),
            help="Attention backend for FastVideo inference.",
        )


class FastVideoConfig(DynamoRuntimeConfig):
    """Configuration for the built-in FastVideo backend."""

    namespace: str
    component: str = DEFAULT_COMPONENT_NAME
    endpoint: Optional[str] = None

    model_path: str = DEFAULT_MODEL_PATH
    served_model_name: Optional[str] = None
    num_gpus: int = 1
    optimization_profile: str = DEFAULT_OPTIMIZATION_PROFILE
    attention_backend: str = DEFAULT_ATTENTION_BACKEND

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
        if self.optimization_profile not in OPTIMIZATION_PROFILE_CHOICES:
            raise ValueError(
                f"--optimization-profile must be one of: {', '.join(OPTIMIZATION_PROFILE_CHOICES)}"
            )
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
    )
    DynamoRuntimeArgGroup().add_arguments(parser)
    FastVideoArgGroup().add_arguments(parser)
    # TODO: This is overkill, we need to think about optimizations better or expose them in a more granular way.
    parser.add_argument(
        "--enable-optimizations",
        action="store_true",
        dest="_legacy_enable_optimizations",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args(None if argv is None else list(argv))

    if args._legacy_enable_optimizations and (
        args.optimization_profile == DEFAULT_OPTIMIZATION_PROFILE
    ):
        args.optimization_profile = "latency"
    delattr(args, "_legacy_enable_optimizations")

    if not args.output_modalities or args.output_modalities == ["text"]:
        args.output_modalities = ["video"]

    config = FastVideoConfig.from_cli_args(args)
    config.validate()
    return config
