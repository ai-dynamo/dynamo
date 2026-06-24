# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO(DIS-2240): Remove deprecated multimodal flags across engine

"""Dynamo SGLang wrapper configuration ArgGroup."""

import argparse
import logging
import warnings
from typing import List, Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.groups.frontend_decoding_args import (
    add_frontend_decoding_arg,
)
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument
from dynamo.common.constants import EmbeddingTransferMode

from . import __version__

logger = logging.getLogger(__name__)


def _warn_deprecated(message: str) -> None:
    logger.warning(message)
    warnings.warn(message, DeprecationWarning, stacklevel=3)


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
            help="DEPRECATED: use --enable-multimodal --disaggregation-mode=encode.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-worker",
            env_var="DYN_SGL_MULTIMODAL_WORKER",
            default=False,
            help=(
                "DEPRECATED: use --enable-multimodal --dedicated-mm-encoder "
                "with --disaggregation-mode=pd/prefill/decode."
            ),
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

    # Additional names this model responds to (aliases). Populated by
    # parse_args when --served-model-name is given as a whitespace- or
    # comma-separated string of multiple names; the first becomes the
    # primary and the rest land here. Plumbed to register_model() so the
    # Rust ModelManager registers the same WorkerSet under each alias.
    served_model_aliases: List[str] = []

    def validate(self) -> None:
        if not isinstance(self.embedding_transfer_mode, EmbeddingTransferMode):
            self.embedding_transfer_mode = EmbeddingTransferMode(
                str(self.embedding_transfer_mode)
            )

        if (self.disagg_config is not None) ^ (self.disagg_config_key is not None):
            raise ValueError(
                "Both 'disagg_config' and 'disagg_config_key' must be provided together."
            )

        self.validate_multimodal_topology()

        if self.multimodal_encode_worker:
            _warn_deprecated(
                "--multimodal-encode-worker is deprecated; use "
                "--enable-multimodal --disaggregation-mode=encode. "
                "This release will map the legacy flag to the new arguments."
            )
            self.enable_multimodal = True
        if self.multimodal_worker:
            _warn_deprecated(
                "--multimodal-worker is deprecated; use --enable-multimodal "
                "--dedicated-mm-encoder with --disaggregation-mode=pd, "
                "--disaggregation-mode=prefill, or --disaggregation-mode=decode. "
                "This release will map the legacy flag to the new arguments."
            )
            self.enable_multimodal = True

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
                "(--dedicated-mm-encoder / --multimodal-encode-worker / "
                "--multimodal-worker). The encode worker needs URLs to run "
                "MMEncoder, while --frontend-decoding ships pre-decoded pixels. "
                "Use --frontend-decoding on a native worker that does not use "
                "the dedicated encode-worker topology instead."
            )
