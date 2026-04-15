# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omni-specific argument parsing for python -m dynamo.vllm.omni."""

import argparse
import dataclasses
import logging
from typing import Optional

from vllm_omni.engine.arg_utils import OmniEngineArgs

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OmniDiffusionKwargs:
    """AsyncOmni constructor kwargs for diffusion engine configuration.

    Every field here is passed directly to AsyncOmni(**kwargs) and consumed by
    _create_default_diffusion_stage_cfg() in vllm-omni. Adding a new vllm-omni
    diffusion flag only requires adding it here and to OmniArgGroup — the
    passthrough in base_handler is automatic.
    """

    enable_layerwise_offload: bool = False
    layerwise_num_gpu_layers: int = 1
    vae_use_slicing: bool = False
    vae_use_tiling: bool = False
    boundary_ratio: float = 0.875
    flow_shift: Optional[float] = None
    cache_backend: Optional[str] = None
    cache_config: Optional[str] = None
    enable_cache_dit_summary: bool = False
    enable_cpu_offload: bool = False
    enforce_eager: bool = False


@dataclasses.dataclass
class OmniParallelKwargs:
    """Diffusion parallelism configuration passed to DiffusionParallelConfig.

    Every field here maps 1:1 to a DiffusionParallelConfig field (excluding
    tensor_parallel_size which comes from engine_args, and fixed/derived fields).
    Adding a new parallelism field only requires adding it here and to OmniArgGroup.
    """

    ulysses_degree: int = 1
    ring_degree: int = 1
    cfg_parallel_size: int = 1
    vae_patch_parallel_size: int = 1
    use_hsdp: bool = False
    hsdp_shard_size: int = -1
    hsdp_replicate_size: int = 1


class OmniArgGroup(ArgGroup):
    """CLI argument definitions for Dynamo vLLM-Omni."""

    name = "dynamo-omni"

    def add_arguments(self, parser) -> None:
        g = parser.add_argument_group(
            "Omni Diffusion Options",
            "Diffusion pipeline parameters for vLLM-Omni multi-stage generation.",
        )

        add_argument(
            g,
            flag_name="--stage-configs-path",
            env_var="DYN_OMNI_STAGE_CONFIGS_PATH",
            default=None,
            help="Path to vLLM-Omni stage configuration YAML file (optional).",
        )

        add_argument(
            g,
            flag_name="--default-video-fps",
            env_var="DYN_OMNI_DEFAULT_VIDEO_FPS",
            default=16,
            arg_type=int,
            help="Default frames per second for generated videos.",
        )

        # OmniDiffusionKwargs fields
        add_negatable_bool_argument(
            g,
            flag_name="--enable-layerwise-offload",
            env_var="DYN_OMNI_ENABLE_LAYERWISE_OFFLOAD",
            default=False,
            help="Enable layerwise (blockwise) offloading on DiT modules to reduce GPU memory.",
        )
        add_argument(
            g,
            flag_name="--layerwise-num-gpu-layers",
            env_var="DYN_OMNI_LAYERWISE_NUM_GPU_LAYERS",
            default=1,
            arg_type=int,
            help="Number of ready layers (blocks) to keep on GPU during generation.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--vae-use-slicing",
            env_var="DYN_OMNI_VAE_USE_SLICING",
            default=False,
            help="Enable VAE slicing for memory optimization in diffusion models.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--vae-use-tiling",
            env_var="DYN_OMNI_VAE_USE_TILING",
            default=False,
            help="Enable VAE tiling for memory optimization in diffusion models.",
        )
        add_argument(
            g,
            flag_name="--boundary-ratio",
            env_var="DYN_OMNI_BOUNDARY_RATIO",
            default=0.875,
            arg_type=float,
            help=(
                "Boundary split ratio for low/high DiT transformers. "
                "Default 0.875 uses both transformers for best quality. "
                "Set to 1.0 to load only the low-noise transformer (saves memory)."
            ),
        )
        add_argument(
            g,
            flag_name="--flow-shift",
            env_var="DYN_OMNI_FLOW_SHIFT",
            default=None,
            arg_type=float,
            help="Scheduler flow_shift parameter (5.0 for 720p, 12.0 for 480p).",
        )
        add_argument(
            g,
            flag_name="--cache-backend",
            env_var="DYN_OMNI_CACHE_BACKEND",
            default=None,
            choices=["cache_dit", "tea_cache"],
            help=(
                "Cache backend for diffusion acceleration. "
                "'cache_dit' enables DBCache + SCM + TaylorSeer. "
                "'tea_cache' enables TeaCache."
            ),
        )
        add_argument(
            g,
            flag_name="--cache-config",
            env_var="DYN_OMNI_CACHE_CONFIG",
            default=None,
            help="Cache configuration as JSON string (overrides defaults).",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-cache-dit-summary",
            env_var="DYN_OMNI_ENABLE_CACHE_DIT_SUMMARY",
            default=False,
            help="Enable cache-dit summary logging after diffusion forward passes.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-cpu-offload",
            env_var="DYN_OMNI_ENABLE_CPU_OFFLOAD",
            default=False,
            help="Enable CPU offloading for diffusion models to reduce GPU memory usage.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enforce-eager",
            env_var="DYN_OMNI_ENFORCE_EAGER",
            default=False,
            help="Disable torch.compile and force eager execution for diffusion models.",
        )

        # TTS parameters
        tts_g = parser.add_argument_group(
            "Omni TTS Options",
            "TTS/audio-specific parameters for vLLM-Omni speech generation.",
        )
        add_argument(
            tts_g,
            flag_name="--tts-max-instructions-length",
            env_var="DYN_OMNI_TTS_MAX_INSTRUCTIONS_LENGTH",
            default=500,
            arg_type=int,
            help="Maximum character length for TTS voice instructions.",
        )
        add_argument(
            tts_g,
            flag_name="--tts-max-new-tokens-min",
            env_var="DYN_OMNI_TTS_MAX_NEW_TOKENS_MIN",
            default=1,
            arg_type=int,
            help="Minimum allowed value for max_new_tokens in TTS requests.",
        )
        add_argument(
            tts_g,
            flag_name="--tts-max-new-tokens-max",
            env_var="DYN_OMNI_TTS_MAX_NEW_TOKENS_MAX",
            default=4096,
            arg_type=int,
            help="Maximum allowed value for max_new_tokens in TTS requests.",
        )
        add_argument(
            tts_g,
            flag_name="--tts-ref-audio-timeout",
            env_var="DYN_OMNI_TTS_REF_AUDIO_TIMEOUT",
            default=15,
            arg_type=int,
            help="Timeout in seconds for downloading reference audio URLs.",
        )
        add_argument(
            tts_g,
            flag_name="--tts-ref-audio-max-bytes",
            env_var="DYN_OMNI_TTS_REF_AUDIO_MAX_BYTES",
            default=50 * 1024 * 1024,
            arg_type=int,
            help="Maximum size in bytes for reference audio files (default: 50MB).",
        )

        # OmniParallelKwargs fields
        add_argument(
            g,
            flag_name="--ulysses-degree",
            env_var="DYN_OMNI_ULYSSES_DEGREE",
            default=1,
            arg_type=int,
            help="Number of GPUs used for Ulysses sequence parallelism in diffusion.",
        )
        add_argument(
            g,
            flag_name="--ring-degree",
            env_var="DYN_OMNI_RING_DEGREE",
            default=1,
            arg_type=int,
            help="Number of GPUs used for ring sequence parallelism in diffusion.",
        )
        add_argument(
            g,
            flag_name="--cfg-parallel-size",
            env_var="DYN_OMNI_CFG_PARALLEL_SIZE",
            default=1,
            arg_type=int,
            choices=[1, 2, 3],
            help="Number of GPUs used for classifier free guidance parallelism.",
        )
        add_argument(
            g,
            flag_name="--vae-patch-parallel-size",
            env_var="DYN_OMNI_VAE_PATCH_PARALLEL_SIZE",
            default=1,
            arg_type=int,
            help="Number of ranks used for VAE patch/tile parallelism during decode/encode.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--use-hsdp",
            env_var="DYN_OMNI_USE_HSDP",
            default=False,
            help=(
                "Enable Hybrid Sharded Data Parallel (HSDP) for diffusion models. "
                "Shards model weights across GPUs to reduce per-GPU memory usage."
            ),
        )
        add_argument(
            g,
            flag_name="--hsdp-shard-size",
            env_var="DYN_OMNI_HSDP_SHARD_SIZE",
            default=-1,
            arg_type=int,
            help="Number of GPUs to shard model weights across when using HSDP (-1 = auto).",
        )
        add_argument(
            g,
            flag_name="--hsdp-replicate-size",
            env_var="DYN_OMNI_HSDP_REPLICATE_SIZE",
            default=1,
            arg_type=int,
            help="Number of HSDP replica groups (default: 1).",
        )

        # Disaggregated stage worker flags
        add_argument(
            g,
            flag_name="--stage-id",
            env_var="DYN_OMNI_STAGE_ID",
            default=None,
            arg_type=int,
            help=(
                "Stage ID for disaggregated omni mode. "
                "Run a single stage as an independent Dynamo worker. "
                "Requires --stage-configs-path."
            ),
        )
        add_negatable_bool_argument(
            g,
            flag_name="--omni-router",
            env_var="DYN_OMNI_ROUTER",
            default=False,
            help=(
                "Run as the stage router, orchestrating the multi-stage DAG. "
                "Requires --stage-configs-path. Mutually exclusive with --stage-id."
            ),
        )


class OmniConfig(DynamoRuntimeConfig):
    """Configuration for Dynamo vLLM-Omni worker."""

    component: str = "backend"
    endpoint: Optional[str] = None

    model: str
    served_model_name: Optional[str] = None
    engine_args: OmniEngineArgs

    stage_configs_path: Optional[str] = None
    default_video_fps: int = 16

    # Nested structs — each group of fields has a clear destination
    diffusion: OmniDiffusionKwargs = dataclasses.field(
        default_factory=OmniDiffusionKwargs
    )
    parallel: OmniParallelKwargs = dataclasses.field(default_factory=OmniParallelKwargs)

    # TTS parameters
    tts_max_instructions_length: int = 500
    tts_max_new_tokens_min: int = 1
    tts_max_new_tokens_max: int = 4096
    tts_ref_audio_timeout: int = 15
    tts_ref_audio_max_bytes: int = 50 * 1024 * 1024

    # Disaggregated stage worker fields
    stage_id: Optional[int] = None
    omni_router: bool = False

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "OmniConfig":
        config = super().from_cli_args(args)
        config.diffusion = OmniDiffusionKwargs(
            **{
                f.name: getattr(args, f.name, f.default)
                for f in dataclasses.fields(OmniDiffusionKwargs)
            }
        )
        config.parallel = OmniParallelKwargs(
            **{
                f.name: getattr(args, f.name, f.default)
                for f in dataclasses.fields(OmniParallelKwargs)
            }
        )
        return config

    def validate(self) -> None:
        DynamoRuntimeConfig.validate(self)
        if self.default_video_fps <= 0:
            raise ValueError("--default-video-fps must be > 0")
        if self.parallel.ulysses_degree <= 0:
            raise ValueError("--ulysses-degree must be > 0")
        if self.parallel.ring_degree <= 0:
            raise ValueError("--ring-degree must be > 0")
        if not (0 < self.diffusion.boundary_ratio <= 1):
            raise ValueError("--boundary-ratio must be in (0, 1]")
        if self.stage_configs_path is None:
            if self.stage_id is not None:
                raise ValueError("--stage-id requires --stage-configs-path")
            if self.omni_router:
                raise ValueError("--omni-router requires --stage-configs-path")
        if self.stage_id is not None and self.stage_id < 0:
            raise ValueError("--stage-id must be >= 0")
        if self.stage_id is not None and self.omni_router:
            raise ValueError("--stage-id and --omni-router are mutually exclusive")


def parse_omni_args() -> OmniConfig:
    """Parse command-line arguments for the vLLM-Omni backend."""
    dynamo_runtime_argspec = DynamoRuntimeArgGroup()
    omni_argspec = OmniArgGroup()

    parser = argparse.ArgumentParser(
        description="Dynamo vLLM-Omni worker",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )

    dynamo_runtime_argspec.add_arguments(parser)
    omni_argspec.add_arguments(parser)

    vg = parser.add_argument_group(
        "vLLM-Omni Engine Options. Please refer to vLLM-Omni documentation for more details."
    )
    vllm_parser = FlexibleArgumentParser(add_help=False)
    OmniEngineArgs.add_cli_args(vllm_parser)

    for action in vllm_parser._actions:
        if not action.option_strings:
            continue
        vg._group_actions.append(action)

    args, unknown = parser.parse_known_args()
    config = OmniConfig.from_cli_args(args)

    if config.endpoint is None:
        config.endpoint = "generate"

    vllm_args = vllm_parser.parse_args(unknown)
    config.model = vllm_args.model

    engine_args = OmniEngineArgs.from_cli_args(vllm_args)

    if getattr(engine_args, "served_model_name", None) is not None:
        served = engine_args.served_model_name
        if len(served) > 1:
            raise ValueError("We do not support multiple model names.")
        config.served_model_name = served[0]

    config.engine_args = engine_args
    config.validate()
    return config
