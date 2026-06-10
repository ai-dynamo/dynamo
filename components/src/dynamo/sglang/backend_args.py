# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo SGLang wrapper configuration ArgGroup."""

import argparse
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.groups.frontend_decoding_args import (
    add_frontend_decoding_arg,
)
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument
from dynamo.common.constants import EmbeddingTransferMode

from . import __version__


class DynamoSGLangArgGroup(ArgGroup):
    """SGLang-specific Dynamo wrapper configuration (not native SGLang engine args)."""

    name = "dynamo-sglang"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add Dynamo SGLang arguments to parser."""

        parser.add_argument(
            "--version",
            action="version",
            version=f"Dynamo Backend SGLang {__version__}",
        )

        g = parser.add_argument_group("Dynamo SGLang Options")

        add_negatable_bool_argument(
            g,
            flag_name="--use-sglang-tokenizer",
            env_var="DYN_SGL_USE_TOKENIZER",
            default=False,
            help="[Deprecated] Use SGLang's tokenizer for pre and post processing. "
            "This option will be removed in a future release. Use "
            "'--dyn-chat-processor sglang' on the frontend instead, which provides "
            "the same SGLang-native pre/post processing with KV router support.",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-encode-worker",
            env_var="DYN_SGL_MULTIMODAL_ENCODE_WORKER",
            default=False,
            help="Run as multimodal encode worker component for processing images/videos.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-worker",
            env_var="DYN_SGL_MULTIMODAL_WORKER",
            default=False,
            help="Run as multimodal worker component for LLM inference with multimodal data.",
        )

        add_argument(
            g,
            flag_name="--embedding-transfer-mode",
            env_var="DYN_SGL_EMBEDDING_TRANSFER_MODE",
            default=EmbeddingTransferMode.NIXL_WRITE.value,
            help="Worker embedding transfer mode: 'local', 'nixl-write', or 'nixl-read'. Can also be set with environment variable DYN_SGL_EMBEDDING_TRANSFER_MODE.",
            choices=[m.value for m in EmbeddingTransferMode],
        )

        add_negatable_bool_argument(
            g,
            flag_name="--embedding-worker",
            env_var="DYN_SGL_EMBEDDING_WORKER",
            default=False,
            help="Run as embedding worker component (Dynamo flag, also sets SGLang's --is-embedding).",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--image-diffusion-worker",
            env_var="DYN_SGL_IMAGE_DIFFUSION_WORKER",
            default=False,
            help="Run as image diffusion worker for image generation.",
        )
        add_argument(
            g,
            flag_name="--disagg-config",
            env_var="DYN_SGL_DISAGG_CONFIG",
            default=None,
            help="Disaggregation configuration file in YAML format.",
        )
        add_argument(
            g,
            flag_name="--disagg-config-key",
            env_var="DYN_SGL_DISAGG_CONFIG_KEY",
            default=None,
            help="Key to select from nested disaggregation configuration file (e.g., 'prefill', 'decode').",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--video-generation-worker",
            env_var="DYN_SGL_VIDEO_GENERATION_WORKER",
            default=False,
            help="Run as video generation worker for video generation (T2V/I2V).",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-rl",
            env_var="DYN_SGL_ENABLE_RL",
            default=False,
            help="Enable RL training support. Registers the call_tokenizer_manager engine route for generic tokenizer_manager passthrough.",
        )

        # Topology constraint: rejecting --frontend-decoding combined with the
        # EPD multimodal flags happens in DynamoSGLangConfig.validate() below.
        add_frontend_decoding_arg(g, env_prefix="SGL")

        add_argument(
            g,
            flag_name="--sglang-trace-level",
            env_var="SGLANG_TRACE_LEVEL",
            default=2,
            arg_type=int,
            choices=[1, 2, 3, 4],
            help="SGLang global trace level when --enable-trace is set "
            "(1=minimal, 2=per-request [default], 3=+decode_loop, 4=full).",
        )


class DynamoSGLangConfig(ConfigBase):
    """Configuration for Dynamo SGLang wrapper (SGLang-specific only)."""

    use_sglang_tokenizer: bool
    multimodal_encode_worker: bool
    multimodal_worker: bool
    embedding_transfer_mode: EmbeddingTransferMode
    embedding_worker: bool
    image_diffusion_worker: bool

    disagg_config: Optional[str] = None
    disagg_config_key: Optional[str] = None

    video_generation_worker: bool
    enable_rl: bool
    frontend_decoding: bool = False
    sglang_trace_level: int

    def validate(self) -> None:
        if not isinstance(self.embedding_transfer_mode, EmbeddingTransferMode):
            self.embedding_transfer_mode = EmbeddingTransferMode(
                str(self.embedding_transfer_mode)
            )

        if (self.disagg_config is not None) ^ (self.disagg_config_key is not None):
            raise ValueError(
                "Both 'disagg_config' and 'disagg_config_key' must be provided together."
            )

        if self.frontend_decoding and (
            self.multimodal_encode_worker or self.multimodal_worker
        ):
            raise ValueError(
                "--frontend-decoding is incompatible with the EPD multimodal topology "
                "(--multimodal-encode-worker / --multimodal-worker). The encode worker "
                "needs URLs to run MMEncoder, while --frontend-decoding ships pre-decoded "
                "pixels. Use --frontend-decoding on the default aggregated decode worker "
                "(no --multimodal-* flag) instead."
            )
