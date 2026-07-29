# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo SGLang wrapper configuration ArgGroup."""

import argparse
import os
from typing import List, Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.groups.frontend_decoding_args import (
    add_frontend_decoding_arg,
)
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument
from dynamo.common.constants import EmbeddingTransferMode

from . import __version__

# Env vars of the removed multimodal role flags. The flags fail at argparse,
# but a leftover env var would otherwise be silently ignored and start the
# worker in the wrong role — reject it with the migration path instead.
_REMOVED_MULTIMODAL_ENV_VARS = {
    "DYN_SGL_MULTIMODAL_ENCODE_WORKER": (
        "--enable-multimodal --disaggregation-mode=encode"
    ),
    "DYN_SGL_MULTIMODAL_WORKER": (
        "--enable-multimodal --dedicated-mm-encoder with "
        "--disaggregation-mode=pd, prefill, or decode"
    ),
}


def _reject_removed_multimodal_env_vars() -> None:
    for env_var, replacement in _REMOVED_MULTIMODAL_ENV_VARS.items():
        if os.environ.get(env_var, "").strip().lower() in ("true", "1", "yes", "on"):
            raise ValueError(
                f"{env_var} is no longer supported; use {replacement} "
                "(env: DYN_SGL_ENABLE_MULTIMODAL, DYN_SGL_DEDICATED_MM_ENCODER)."
            )


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
            flag_name="--enable-multimodal",
            env_var="DYN_SGL_ENABLE_MULTIMODAL",
            default=False,
            help=(
                "Enable multimodal processing. This is a capability flag for "
                "raw multimodal inputs; use --dedicated-mm-encoder when this "
                "worker is part of the internal encode-worker topology."
            ),
        )
        add_negatable_bool_argument(
            g,
            flag_name="--dedicated-mm-encoder",
            env_var="DYN_SGL_DEDICATED_MM_ENCODER",
            default=False,
            help=(
                "Select the internal SGLang topology with a dedicated "
                "multimodal encode worker. Required on PD/P/D workers that "
                "consume or forward precomputed embeddings from that encode "
                "worker. Do not use it for native P/D workers that receive "
                "raw media metadata."
            ),
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

        # Topology constraint: rejecting --frontend-decoding combined with an
        # EPD multimodal role happens in DynamoSGLangConfig.validate() below.
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
    # Internal roles derived from the canonical multimodal arguments in args.py.
    multimodal_encode_worker: bool = False
    multimodal_worker: bool = False
    enable_multimodal: bool = False
    dedicated_mm_encoder: bool = False
    embedding_transfer_mode: EmbeddingTransferMode
    embedding_worker: bool
    image_diffusion_worker: bool

    disagg_config: Optional[str] = None
    disagg_config_key: Optional[str] = None

    video_generation_worker: bool
    enable_rl: bool
    frontend_decoding: bool = False
    sglang_trace_level: int

    # Extra served names beyond the primary, parsed from --served-model-name.
    # None (not []) since ConfigBase copies class defaults by reference.
    served_model_aliases: Optional[List[str]] = None

    def validate(self) -> None:
        _reject_removed_multimodal_env_vars()

        if not isinstance(self.embedding_transfer_mode, EmbeddingTransferMode):
            self.embedding_transfer_mode = EmbeddingTransferMode(
                str(self.embedding_transfer_mode)
            )

        if (self.disagg_config is not None) ^ (self.disagg_config_key is not None):
            raise ValueError(
                "Both 'disagg_config' and 'disagg_config_key' must be provided together."
            )

        self.validate_multimodal_topology()

        self.validate_dedicated_mm_encoder()

    def validate_dedicated_mm_encoder(self) -> None:
        if self.dedicated_mm_encoder and not self.enable_multimodal:
            raise ValueError(
                "--dedicated-mm-encoder requires --enable-multimodal. The "
                "dedicated encoder flag selects the internal encode-worker "
                "topology; it is not a standalone multimodal capability switch."
            )

    def validate_multimodal_topology(self) -> None:
        if self.frontend_decoding and (
            self.multimodal_encode_worker
            or self.multimodal_worker
            or self.dedicated_mm_encoder
        ):
            raise ValueError(
                "--frontend-decoding is incompatible with the EPD multimodal topology "
                "(--disaggregation-mode=encode or --dedicated-mm-encoder). "
                "The encode worker needs URLs to run "
                "MMEncoder, while --frontend-decoding ships pre-decoded pixels. "
                "Use --frontend-decoding on a native worker that does not use "
                "the dedicated encode-worker topology instead."
            )
