# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo vLLM wrapper configuration ArgGroup."""

import warnings
from typing import Optional, Union

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument

from . import __version__
from .constants import DisaggregationMode


class DynamoVllmArgGroup(ArgGroup):
    """vLLM-specific Dynamo wrapper configuration (not native vLLM engine args)."""

    name = "dynamo-vllm"

    def add_arguments(self, parser) -> None:
        """Add Dynamo vLLM arguments to parser."""

        parser.add_argument(
            "--version", action="version", version=f"Dynamo Backend VLLM {__version__}"
        )
        g = parser.add_argument_group("Dynamo vLLM Options")

        add_argument(
            g,
            flag_name="--disaggregation-mode",
            env_var="DYN_VLLM_DISAGGREGATION_MODE",
            default=None,
            help="Worker disaggregation mode: 'agg' (default, aggregated), "
            "'prefill' (prefill-only worker), or 'decode' (decode-only worker).",
            choices=[m.value for m in DisaggregationMode],
        )

        add_negatable_bool_argument(
            g,
            flag_name="--is-prefill-worker",
            env_var="DYN_VLLM_IS_PREFILL_WORKER",
            default=False,
            help="DEPRECATED: use --disaggregation-mode=prefill. "
            "Enable prefill functionality for this worker.",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--is-decode-worker",
            env_var="DYN_VLLM_IS_DECODE_WORKER",
            default=False,
            help="DEPRECATED: use --disaggregation-mode=decode. "
            "Mark this as a decode worker which does not publish KV events.",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--use-vllm-tokenizer",
            env_var="DYN_VLLM_USE_TOKENIZER",
            default=False,
            help="Use vLLM's tokenizer for pre and post processing. This bypasses Dynamo's preprocessor and only v1/chat/completions will be available through the Dynamo frontend.",
        )

        add_argument(
            g,
            flag_name="--sleep-mode-level",
            env_var="DYN_VLLM_SLEEP_MODE_LEVEL",
            default=1,
            help="Sleep mode level (1=offload to CPU, 2=discard weights, 3=discard all).",
            choices=[1, 2, 3],
            arg_type=int,
        )

        # Multimodal
        add_negatable_bool_argument(
            g,
            flag_name="--route-to-encoder",
            env_var="DYN_VLLM_ROUTE_TO_ENCODER",
            default=False,
            help="Enable routing to separate encoder workers for multimodal processing.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-encode-worker",
            env_var="DYN_VLLM_MULTIMODAL_ENCODE_WORKER",
            default=False,
            help="Run as multimodal encode worker component for processing images/videos.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-worker",
            env_var="DYN_VLLM_MULTIMODAL_WORKER",
            default=False,
            help="Run as multimodal worker component for LLM inference with multimodal data.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-decode-worker",
            env_var="DYN_VLLM_MULTIMODAL_DECODE_WORKER",
            default=False,
            help="Run as multimodal decode worker in disaggregated mode.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-multimodal",
            env_var="DYN_VLLM_ENABLE_MULTIMODAL",
            default=False,
            help="Enable multimodal processing. If not set, none of the multimodal components can be used.",
        )
        add_argument(
            g,
            flag_name="--mm-prompt-template",
            env_var="DYN_VLLM_MM_PROMPT_TEMPLATE",
            default="USER: <image>\n<prompt> ASSISTANT:",
            help=(
                "Different multi-modal models expect the prompt to contain different special media prompts. "
                "The processor will use this argument to construct the final prompt. "
                "User prompt will replace '<prompt>' in the provided template. "
                "For example, if the user prompt is 'please describe the image' and the prompt template is "
                "'USER: <image> <prompt> ASSISTANT:', the resulting prompt is "
                "'USER: <image> please describe the image ASSISTANT:'."
            ),
        )

        add_negatable_bool_argument(
            g,
            flag_name="--frontend-decoding",
            env_var="DYN_VLLM_FRONTEND_DECODING",
            default=False,
            help=(
                "Enable frontend decoding of multimodal images. "
                "When enabled, images are decoded in the Rust frontend and transferred to the backend via NIXL RDMA. "
                "Without this flag, images are decoded in the Python backend (default behavior)."
            ),
        )

        # vLLM-Omni
        add_negatable_bool_argument(
            g,
            flag_name="--omni",
            env_var="DYN_VLLM_OMNI",
            default=False,
            help="Run as vLLM-Omni worker for multi-stage pipelines (supports text-to-text, text-to-image, etc.).",
        )
        add_argument(
            g,
            flag_name="--stage-configs-path",
            env_var="DYN_VLLM_STAGE_CONFIGS_PATH",
            default=None,
            help="Path to vLLM-Omni stage configuration YAML file for --omni mode (optional).",
        )

        # Video encoding
        add_argument(
            g,
            flag_name="--default-video-fps",
            env_var="DYN_VLLM_DEFAULT_VIDEO_FPS",
            default=16,
            arg_type=int,
            help="Default frames per second for generated videos.",
        )

        # Diffusion engine-level args (passed to AsyncOmni constructor).
        # All flags use the --omni- prefix to avoid collisions with vLLM's
        # native engine flags (e.g. --enforce-eager), which are parsed by a
        # separate argparse pass and would otherwise be silently consumed here.
        add_negatable_bool_argument(
            g,
            flag_name="--omni-enable-layerwise-offload",
            env_var="DYN_VLLM_ENABLE_LAYERWISE_OFFLOAD",
            default=False,
            help="Enable layerwise (blockwise) offloading on DiT modules to reduce GPU memory.",
        )
        add_argument(
            g,
            flag_name="--omni-layerwise-num-gpu-layers",
            env_var="DYN_VLLM_LAYERWISE_NUM_GPU_LAYERS",
            default=1,
            arg_type=int,
            help="Number of ready layers (blocks) to keep on GPU during generation.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--omni-vae-use-slicing",
            env_var="DYN_VLLM_VAE_USE_SLICING",
            default=False,
            help="Enable VAE slicing for memory optimization in diffusion models.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--omni-vae-use-tiling",
            env_var="DYN_VLLM_VAE_USE_TILING",
            default=False,
            help="Enable VAE tiling for memory optimization in diffusion models.",
        )
        add_argument(
            g,
            flag_name="--omni-boundary-ratio",
            env_var="DYN_VLLM_BOUNDARY_RATIO",
            default=0.875,
            arg_type=float,
            help=(
                "Boundary split ratio for low/high DiT transformers. "
                "Default 0.875 uses both transformers for best quality. "
                "Set to 1.0 to load only the low-noise transformer (saves memory). "
                "Only used with --omni."
            ),
        )
        add_argument(
            g,
            flag_name="--omni-flow-shift",
            env_var="DYN_VLLM_FLOW_SHIFT",
            default=None,
            arg_type=float,
            help="Scheduler flow_shift parameter (5.0 for 720p, 12.0 for 480p). Only used with --omni.",
        )
        add_argument(
            g,
            flag_name="--omni-diffusion-cache-backend",
            env_var="DYN_VLLM_DIFFUSION_CACHE_BACKEND",
            default=None,
            choices=["cache_dit", "tea_cache"],
            help=(
                "Cache backend for diffusion acceleration. "
                "'cache_dit' enables DBCache + SCM + TaylorSeer. "
                "'tea_cache' enables TeaCache. Only used with --omni."
            ),
        )
        add_argument(
            g,
            flag_name="--omni-diffusion-cache-config",
            env_var="DYN_VLLM_DIFFUSION_CACHE_CONFIG",
            default=None,
            help="Cache configuration as JSON string (overrides defaults). Only used with --omni.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--omni-enable-cache-dit-summary",
            env_var="DYN_VLLM_ENABLE_CACHE_DIT_SUMMARY",
            default=False,
            help="Enable cache-dit summary logging after diffusion forward passes.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--omni-enable-cpu-offload",
            env_var="DYN_VLLM_ENABLE_CPU_OFFLOAD",
            default=False,
            help="Enable CPU offloading for diffusion models to reduce GPU memory usage.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--omni-enforce-eager",
            env_var="DYN_VLLM_ENFORCE_EAGER",
            default=False,
            help="Disable torch.compile and force eager execution for diffusion models.",
        )
        # Diffusion parallel configuration
        add_argument(
            g,
            flag_name="--omni-ulysses-degree",
            env_var="DYN_VLLM_ULYSSES_DEGREE",
            default=1,
            arg_type=int,
            help="Number of GPUs used for Ulysses sequence parallelism in diffusion.",
        )
        add_argument(
            g,
            flag_name="--omni-ring-degree",
            env_var="DYN_VLLM_RING_DEGREE",
            default=1,
            arg_type=int,
            help="Number of GPUs used for ring sequence parallelism in diffusion.",
        )
        add_argument(
            g,
            flag_name="--omni-cfg-parallel-size",
            env_var="DYN_VLLM_CFG_PARALLEL_SIZE",
            default=1,
            arg_type=int,
            choices=[1, 2],
            help="Number of GPUs used for classifier free guidance parallelism.",
        )

        # Headless mode for multi-node TP/PP
        add_negatable_bool_argument(
            g,
            flag_name="--headless",
            env_var="DYN_VLLM_HEADLESS",
            default=False,
            help="Run in headless mode for multi-node TP/PP. "
            "Secondary nodes run vLLM workers only, no dynamo endpoints. "
            "See vLLM multi-node data parallel documentation for more details.",
        )

        # ModelExpress P2P
        add_argument(
            g,
            flag_name="--model-express-url",
            env_var="MODEL_EXPRESS_URL",
            default=None,
            help="ModelExpress P2P server URL (e.g., http://mx-server:8080). "
            "Required when using --load-format=mx-source or --load-format=mx-target.",
        )


# @dataclass()
class DynamoVllmConfig(ConfigBase):
    """Configuration for Dynamo vLLM wrapper (vLLM-specific only). All fields optional."""

    disaggregation_mode: Union[
        None, str, DisaggregationMode
    ]  # None when not provided; resolved to enum in validate()
    is_prefill_worker: bool
    is_decode_worker: bool
    use_vllm_tokenizer: bool
    sleep_mode_level: int

    # Multimodal
    route_to_encoder: bool
    multimodal_encode_worker: bool
    multimodal_worker: bool
    multimodal_decode_worker: bool
    enable_multimodal: bool
    mm_prompt_template: str
    frontend_decoding: bool

    # vLLM-Omni
    omni: bool
    stage_configs_path: Optional[str] = None

    # Video encoding
    default_video_fps: int = 16

    # Diffusion engine-level parameters (passed to AsyncOmni constructor).
    # Field names use omni_ prefix to match the --omni-* CLI flags and avoid
    # collisions with vLLM's native engine args (e.g. enforce_eager).
    omni_enable_layerwise_offload: bool = False
    omni_layerwise_num_gpu_layers: int = 1
    omni_vae_use_slicing: bool = False
    omni_vae_use_tiling: bool = False
    omni_boundary_ratio: float = 0.875
    omni_flow_shift: Optional[float] = None
    omni_diffusion_cache_backend: Optional[str] = None
    omni_diffusion_cache_config: Optional[str] = None
    omni_enable_cache_dit_summary: bool = False
    omni_enable_cpu_offload: bool = False
    omni_enforce_eager: bool = False

    # Diffusion parallel configuration
    omni_ulysses_degree: int = 1
    omni_ring_degree: int = 1
    omni_cfg_parallel_size: int = 1

    # Headless mode for multi-node TP/PP
    headless: bool = False

    # ModelExpress P2P
    model_express_url: Optional[str] = None

    def validate(self) -> None:
        """Validate vLLM wrapper configuration."""
        self._resolve_disaggregation_mode()
        self._validate_multimodal_role_exclusivity()
        self._validate_multimodal_requires_flag()
        self._validate_omni_stage_config()

    def _resolve_disaggregation_mode(self) -> None:
        """Resolve disaggregation_mode from new enum or legacy boolean flags.

        Priority:
        1. If --disaggregation-mode was explicitly provided, use it.
           Raise if legacy booleans are also set.
        2. If legacy --is-prefill-worker or --is-decode-worker is set,
           emit DeprecationWarning and translate to enum.
        3. Apply default (AGGREGATED) if nothing was provided.
        4. Sync boolean fields from the resolved enum value.
        """
        # Convert string to enum (non-None means explicitly provided)
        explicit_mode = self.disaggregation_mode is not None
        if isinstance(self.disaggregation_mode, str):
            self.disaggregation_mode = DisaggregationMode(self.disaggregation_mode)

        # Check for legacy boolean flags
        has_legacy = self.is_prefill_worker or self.is_decode_worker

        if has_legacy and explicit_mode:
            raise ValueError(
                "Cannot combine --is-prefill-worker/--is-decode-worker with "
                "--disaggregation-mode. Use only --disaggregation-mode."
            )

        if has_legacy:
            if self.is_prefill_worker and self.is_decode_worker:
                raise ValueError(
                    "Cannot set both --is-prefill-worker and --is-decode-worker"
                )
            if self.is_prefill_worker:
                warnings.warn(
                    "--is-prefill-worker is deprecated, use --disaggregation-mode=prefill",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self.disaggregation_mode = DisaggregationMode.PREFILL
            elif self.is_decode_worker:
                warnings.warn(
                    "--is-decode-worker is deprecated, use --disaggregation-mode=decode",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self.disaggregation_mode = DisaggregationMode.DECODE

        # Apply default if neither new flag nor legacy flags were provided
        if self.disaggregation_mode is None:
            self.disaggregation_mode = DisaggregationMode.AGGREGATED

        # Sync booleans from enum (canonical source of truth)
        self.is_prefill_worker = self.disaggregation_mode == DisaggregationMode.PREFILL
        self.is_decode_worker = self.disaggregation_mode == DisaggregationMode.DECODE

    def _count_multimodal_roles(self) -> int:
        """Return the number of multimodal worker roles set (0 or 1 allowed).

        Note: --route-to-encoder is a modifier flag, not a worker type.
        """
        return sum(
            [
                bool(self.multimodal_encode_worker),
                bool(self.multimodal_worker),
                bool(self.multimodal_decode_worker),
            ]
        )

    def _validate_multimodal_role_exclusivity(self) -> None:
        """Ensure only one multimodal role is set at a time."""
        if self._count_multimodal_roles() > 1:
            raise ValueError(
                "Use only one of --multimodal-encode-worker, --multimodal-worker, "
                "--multimodal-decode-worker"
            )

    def _validate_multimodal_requires_flag(self) -> None:
        """Require --enable-multimodal when any multimodal role is set."""
        if self._count_multimodal_roles() == 1 and not self.enable_multimodal:
            raise ValueError(
                "Use --enable-multimodal when enabling any multimodal component"
            )

    def _validate_omni_stage_config(self) -> None:
        """Require stage_configs_path when using --omni."""
        if self.stage_configs_path and not self.omni:
            raise ValueError(
                "--stage-configs-path is only allowed when using --omni. "
                "Specify a YAML file containing stage configurations for the multi-stage pipeline."
            )
