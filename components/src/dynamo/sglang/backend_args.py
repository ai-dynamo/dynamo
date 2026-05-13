# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo SGLang wrapper configuration ArgGroup."""

import argparse
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
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
            "'--preprocessor sglang --postprocessor sglang' instead.",
        )

        add_argument(
            g,
            flag_name="--preprocessor",
            env_var="DYN_SGL_PREPROCESSOR",
            default="dynamo",
            choices=["dynamo", "sglang"],
            help=(
                "Preprocessing owner for chat requests. 'dynamo' registers "
                "ModelInput.Tokens so the Rust frontend applies templates and "
                "tokenizes for KV routing. 'sglang' registers ModelInput.Text so "
                "the worker delegates preprocessing to SGLang."
            ),
        )

        add_argument(
            g,
            flag_name="--postprocessor",
            env_var="DYN_SGL_POSTPROCESSOR",
            default="dynamo",
            choices=["dynamo", "sglang"],
            help=(
                "Postprocessing owner for chat responses. 'dynamo' registers "
                "ModelOutput.Tokens so the Rust frontend detokenizes and parses "
                "tool/reasoning output. 'sglang' registers ModelOutput.Text so "
                "the worker delegates detokenization, tool parsing, and reasoning "
                "parsing to SGLang."
            ),
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


class DynamoSGLangConfig(ConfigBase):
    """Configuration for Dynamo SGLang wrapper (SGLang-specific only)."""

    use_sglang_tokenizer: bool
    preprocessor: str
    postprocessor: str
    multimodal_encode_worker: bool
    multimodal_worker: bool
    embedding_transfer_mode: EmbeddingTransferMode
    embedding_worker: bool
    image_diffusion_worker: bool

    disagg_config: Optional[str] = None
    disagg_config_key: Optional[str] = None

    video_generation_worker: bool
    enable_rl: bool

    def validate(self) -> None:
        if self.preprocessor not in ("dynamo", "sglang"):
            raise ValueError("--preprocessor must be one of: dynamo, sglang")
        if self.postprocessor not in ("dynamo", "sglang"):
            raise ValueError("--postprocessor must be one of: dynamo, sglang")

        if not isinstance(self.embedding_transfer_mode, EmbeddingTransferMode):
            self.embedding_transfer_mode = EmbeddingTransferMode(
                str(self.embedding_transfer_mode)
            )

        if (self.disagg_config is not None) ^ (self.disagg_config_key is not None):
            raise ValueError(
                "Both 'disagg_config' and 'disagg_config_key' must be provided together."
            )
