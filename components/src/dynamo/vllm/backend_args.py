# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO(DIS-2240): Remove deprecated multimodal flags across engine

"""Dynamo vLLM wrapper configuration ArgGroup."""

import argparse
import logging
import os
import warnings
from typing import Optional, Union

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.groups.frontend_decoding_args import (
    add_frontend_decoding_arg,
)
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument

from . import __version__
from .constants import DisaggregationMode, EmbeddingTransferMode

logger = logging.getLogger(__name__)
PREFILL_DECODE_DISAGGREGATION_MODE = "pd"


def _warn_deprecated(message: str) -> None:
    logger.warning(message)
    warnings.warn(message, DeprecationWarning, stacklevel=3)


class _StoreExplicitBenchmarkOption(argparse.Action):
    """Store a value and remember that its new sampling option was explicit."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


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
            "'pd' (combined prefill+decode worker), 'prefill' "
            "(prefill-only worker), 'decode' (decode-only worker), "
            "or 'encode' (multimodal encode worker).",
            choices=[PREFILL_DECODE_DISAGGREGATION_MODE]
            + [m.value for m in DisaggregationMode],
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
            flag_name="--enable-conditional-disagg",
            env_var="DYN_VLLM_ENABLE_CONDITIONAL_DISAGG",
            default=False,
            help=(
                "Opt this decode worker into conditional-disagg support. Decode "
                "workers publish KV events so the Dynamo router can see decode-side "
                "cache, and annotated bypass requests run as local prefill+decode."
            ),
        )

        add_negatable_bool_argument(
            g,
            flag_name="--use-vllm-tokenizer",
            env_var="DYN_VLLM_USE_TOKENIZER",
            default=False,
            help="Use vLLM's tokenizer for pre and post processing. This bypasses Dynamo's preprocessor and only v1/chat/completions will be available through the Dynamo frontend.",
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
        # Select defaults used by RL-style token-in/token-out deployments.
        add_negatable_bool_argument(
            g,
            flag_name="--enable-rl",
            env_var="DYN_ENABLE_RL",
            default=False,
            help=(
                "Enable RL training support. Mirrors --enable-rl on the SGLang "
                "backend and selects RL-friendly vLLM defaults for TITO and "
                "per-token logprob parity."
            ),
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

        add_frontend_decoding_arg(g, env_prefix="VLLM")

        add_argument(
            g,
            flag_name="--custom-encoder-class",
            env_var="DYN_CUSTOM_ENCODER_CLASS",
            default=None,
            help=(
                "Dotted module.ClassName path to a VisionEncoderBackend subclass. "
                "When set, the aggregated worker wraps it in the in-process "
                "AsyncVisionEncoder and runs encoder.encode(image_urls) for each "
                "multimodal request, bypassing vLLM's built-in multimodal "
                "processing. --model is passed verbatim to the backend's build(). "
                "Example: 'my_package.encoders.MyEncoder'."
            ),
        )

        add_argument(
            g,
            flag_name="--embedding-transfer-mode",
            env_var="DYN_VLLM_EMBEDDING_TRANSFER_MODE",
            default=EmbeddingTransferMode.NIXL_WRITE.value,
            help="Worker embedding transfer mode: 'local' (default, local file system), "
            "'nixl-write' (NIXL transfer with WRITE), or 'nixl-read' (NIXL transfer with READ).",
            choices=[m.value for m in EmbeddingTransferMode],
        )

        add_negatable_bool_argument(
            g,
            flag_name="--embedding-worker",
            env_var="DYN_VLLM_EMBEDDING_WORKER",
            default=False,
            help="Run as a text-embedding worker. Engine must be started with "
            "vLLM's --runner pooling. Skips KV-events, KV router registration, "
            "and InstrumentedScheduler injection (none apply to pooling models).",
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
            help="DEPRECATED: accepted for compatibility with older ModelExpress "
            "manifests. The vLLM ModelExpress plugin reads its own configuration.",
        )

        # GMS (GPU Memory Service) shadow mode
        add_negatable_bool_argument(
            g,
            flag_name="--gms-shadow-mode",
            env_var="DYN_VLLM_GMS_SHADOW_MODE",
            default=False,
            help=(
                "Enable GMS shadow/standby mode. Shadow engines skip KV cache "
                "allocation at startup, automatically pause after initialization, "
                "and resume on demand when the active engine dies. "
                "Requires --load-format=gms."
            ),
        )

        # Benchmark / self-profiling
        add_argument(
            g,
            flag_name="--benchmark-mode",
            env_var="DYN_BENCHMARK_MODE",
            default=None,
            choices=["prefill", "decode", "agg"],
            help=(
                "Run self-benchmark on startup before accepting requests. "
                "Sweeps iteration-total prefill tokens/KV reads/batch size and/or "
                "decode total-KV/batch-size points. CUDA graph axes include every "
                "{capture size, capture size + 1} boundary and continue "
                "geometrically to the engine limit. KV-read axes use complete "
                "power-of-two block ladders plus their exact feasible maxima, "
                "then apply the configured per-axis sample limits."
            ),
        )
        add_argument(
            g,
            flag_name="--prefill-max-new-token-samples",
            env_var="DYN_PREFILL_MAX_NEW_TOKEN_SAMPLES",
            default=64,
            arg_type=int,
            action=_StoreExplicitBenchmarkOption,
            help=(
                "Maximum number of iteration-total prefill new-token samples. "
                "If the CUDA-graph-aware axis has more points, points are selected "
                "uniformly across the sorted axis while always retaining its "
                "minimum and maximum (default: 64; must be at least 2)."
            ),
        )
        add_argument(
            g,
            flag_name="--prefill-max-kv-read-token-samples",
            env_var="DYN_PREFILL_MAX_KV_READ_TOKEN_SAMPLES",
            default=16,
            arg_type=int,
            action=_StoreExplicitBenchmarkOption,
            help=(
                "Maximum number of iteration-total prefill KV-read-token samples "
                "for each (new tokens, batch size) pair. If the block-aligned "
                "KV ladder has more points, points are selected uniformly while "
                "always retaining zero and the feasible maximum "
                "(default: 16; must be at least 2)."
            ),
        )
        add_argument(
            g,
            flag_name="--decode-max-kv-read-token-samples",
            env_var="DYN_DECODE_MAX_KV_READ_TOKEN_SAMPLES",
            default=128,
            arg_type=int,
            action=_StoreExplicitBenchmarkOption,
            help=(
                "Maximum number of iteration-total decode KV-read-token samples "
                "for each batch size. If the KV ladder has more points, points "
                "are selected uniformly while always retaining its minimum and "
                "feasible maximum (default: 128; must be at least 2)."
            ),
        )
        add_argument(
            g,
            flag_name="--decode-max-batch-size-samples",
            env_var="DYN_DECODE_MAX_BATCH_SIZE_SAMPLES",
            default=128,
            arg_type=int,
            action=_StoreExplicitBenchmarkOption,
            help=(
                "Maximum number of decode batch-size samples. If the "
                "CUDA-graph-aware axis has more points, points are selected "
                "uniformly while always retaining the minimum and feasible "
                "maximum (default: 128; must be at least 2)."
            ),
        )
        add_argument(
            g,
            flag_name="--prefix-max-batch-size-samples",
            env_var="DYN_PREFIX_MAX_BATCH_SIZE_SAMPLES",
            default=3,
            arg_type=int,
            action=_StoreExplicitBenchmarkOption,
            help=(
                "Maximum number of prefill request-batch-size samples for each "
                "new-token point. Keeps the first N values from the sorted "
                "power-of-two-plus-legal-maximum axis, so the default 3 selects "
                "[1, 2, 4] when all three are legal (default: 3; must be positive)."
            ),
        )
        explicit_sampling_envs = {
            "prefill_max_new_token_samples_explicit": (
                "DYN_PREFILL_MAX_NEW_TOKEN_SAMPLES"
            ),
            "prefill_max_kv_read_token_samples_explicit": (
                "DYN_PREFILL_MAX_KV_READ_TOKEN_SAMPLES"
            ),
            "decode_max_kv_read_token_samples_explicit": (
                "DYN_DECODE_MAX_KV_READ_TOKEN_SAMPLES"
            ),
            "decode_max_batch_size_samples_explicit": (
                "DYN_DECODE_MAX_BATCH_SIZE_SAMPLES"
            ),
            "prefix_max_batch_size_samples_explicit": (
                "DYN_PREFIX_MAX_BATCH_SIZE_SAMPLES"
            ),
        }
        g.set_defaults(
            **{
                marker: True
                for marker, env_var in explicit_sampling_envs.items()
                if env_var in os.environ
            }
        )
        legacy_sampling_flags = (
            (
                "--benchmark-prefill-granularity",
                "DYN_BENCHMARK_PREFILL_GRANULARITY",
                "--prefill-max-new-token-samples",
            ),
            (
                "--benchmark-prefill-kv-read-granularity",
                "DYN_BENCHMARK_PREFILL_KV_READ_GRANULARITY",
                "--prefill-max-kv-read-token-samples",
            ),
            (
                "--benchmark-prefill-batch-granularity",
                "DYN_BENCHMARK_PREFILL_BATCH_GRANULARITY",
                "--prefix-max-batch-size-samples",
            ),
            (
                "--benchmark-decode-length-granularity",
                "DYN_BENCHMARK_DECODE_LENGTH_GRANULARITY",
                "--decode-max-kv-read-token-samples",
            ),
            (
                "--benchmark-decode-batch-granularity",
                "DYN_BENCHMARK_DECODE_BATCH_GRANULARITY",
                "--decode-max-batch-size-samples",
            ),
        )
        for legacy_flag, legacy_env, replacement in legacy_sampling_flags:
            add_argument(
                g,
                flag_name=legacy_flag,
                env_var=legacy_env,
                default=None,
                arg_type=int,
                help=(
                    f"Deprecated compatibility option; use {replacement}. "
                    "Legacy values are translated to the new sampling limit."
                ),
            )
        add_argument(
            g,
            flag_name="--benchmark-warmup-iterations",
            env_var="DYN_BENCHMARK_WARMUP_ITERATIONS",
            default=5,
            arg_type=int,
            help="Warmup iterations before benchmark (default: 5).",
        )
        add_argument(
            g,
            flag_name="--benchmark-output-path",
            env_var="DYN_BENCHMARK_OUTPUT_PATH",
            default="/tmp/benchmark_results.json",
            help=(
                "Path to write benchmark results JSON "
                "(default: /tmp/benchmark_results.json)."
            ),
        )
        add_argument(
            g,
            flag_name="--benchmark-timeout",
            env_var="DYN_BENCHMARK_TIMEOUT",
            default=900,
            arg_type=int,
            help=(
                "Soft limit in seconds for self-benchmarking (default: 900). "
                "After the limit, the current measured iteration finishes, "
                "partial results are returned, and engine startup continues. "
                "A bounded cleanup grace still fails closed if no result is written."
            ),
        )


# @dataclass()
class DynamoVllmConfig(ConfigBase):
    """Configuration for Dynamo vLLM wrapper (vLLM-specific only). All fields optional."""

    disaggregation_mode: Union[
        None, str, DisaggregationMode
    ]  # None when not provided; resolved to enum in validate()
    is_prefill_worker: bool
    is_decode_worker: bool
    enable_conditional_disagg: bool
    use_vllm_tokenizer: bool

    # Multimodal
    route_to_encoder: bool
    multimodal_encode_worker: bool
    multimodal_worker: bool
    multimodal_decode_worker: bool
    enable_multimodal: bool
    # Enables RL-style token-in/token-out defaults.
    enable_rl: bool = False
    mm_prompt_template: str
    frontend_decoding: bool
    embedding_transfer_mode: Union[
        str, EmbeddingTransferMode
    ]  # resolved to enum in validate()
    embedding_worker: bool = False

    # CustomEncoder (image-only embeddings; worker assembles mixed prompt)
    custom_encoder_class: Optional[str] = None

    # Headless mode for multi-node TP/PP
    headless: bool = False

    # ModelExpress P2P
    model_express_url: Optional[str] = None

    # GMS shadow mode
    gms_shadow_mode: bool = False

    # Benchmark / self-profiling
    benchmark_mode: Optional[str] = None
    benchmark_warmup_iterations: int = 5
    benchmark_output_path: str = "/tmp/benchmark_results.json"
    benchmark_timeout: int = 900
    prefill_max_new_token_samples: int = 64
    prefill_max_kv_read_token_samples: int = 16
    decode_max_kv_read_token_samples: int = 128
    decode_max_batch_size_samples: int = 128
    prefix_max_batch_size_samples: int = 3
    prefill_max_new_token_samples_explicit: bool = False
    prefill_max_kv_read_token_samples_explicit: bool = False
    decode_max_kv_read_token_samples_explicit: bool = False
    decode_max_batch_size_samples_explicit: bool = False
    prefix_max_batch_size_samples_explicit: bool = False
    benchmark_prefill_granularity: Optional[int] = None
    benchmark_prefill_kv_read_granularity: Optional[int] = None
    benchmark_prefill_batch_granularity: Optional[int] = None
    benchmark_decode_length_granularity: Optional[int] = None
    benchmark_decode_batch_granularity: Optional[int] = None

    def validate(self) -> None:
        """Validate vLLM wrapper configuration."""
        self._resolve_disaggregation_mode()
        self._resolve_embedding_transfer_mode()
        self._validate_multimodal_role_exclusivity()
        self._validate_multimodal_requires_flag()
        self._validate_embedding_worker_exclusivity()
        self._validate_custom_encoder()
        self._resolve_legacy_benchmark_sampling()
        self._validate_benchmark_sampling()

    def _resolve_legacy_benchmark_sampling(self) -> None:
        if self.benchmark_mode is None:
            return

        mappings = (
            (
                "benchmark_prefill_granularity",
                "prefill_max_new_token_samples",
                64,
                True,
            ),
            (
                "benchmark_prefill_kv_read_granularity",
                "prefill_max_kv_read_token_samples",
                16,
                True,
            ),
            (
                "benchmark_prefill_batch_granularity",
                "prefix_max_batch_size_samples",
                3,
                False,
            ),
            (
                "benchmark_decode_length_granularity",
                "decode_max_kv_read_token_samples",
                128,
                True,
            ),
            (
                "benchmark_decode_batch_granularity",
                "decode_max_batch_size_samples",
                128,
                True,
            ),
        )
        for (
            legacy_name,
            replacement_name,
            replacement_default,
            needs_endpoints,
        ) in mappings:
            legacy_value = getattr(self, legacy_name)
            if legacy_value is None:
                continue
            if not 1 <= legacy_value <= 1024:
                raise ValueError(
                    f"--{legacy_name.replace('_', '-')} must be between 1 and 1024"
                )
            replacement_value = getattr(self, replacement_name)
            replacement_explicit = getattr(self, f"{replacement_name}_explicit", False)
            if replacement_explicit or replacement_value != replacement_default:
                raise ValueError(
                    f"cannot combine --{legacy_name.replace('_', '-')} with "
                    f"--{replacement_name.replace('_', '-')}"
                )
            mapped_value = max(2, legacy_value) if needs_endpoints else legacy_value
            detail = (
                " Legacy value 1 maps to 2 so both axis endpoints are retained."
                if needs_endpoints and legacy_value == 1
                else ""
            )
            _warn_deprecated(
                f"--{legacy_name.replace('_', '-')} is deprecated; use "
                f"--{replacement_name.replace('_', '-')} instead.{detail}"
            )
            setattr(self, replacement_name, mapped_value)

    def _validate_benchmark_sampling(self) -> None:
        if self.benchmark_mode is None:
            return
        uniform_limits = (
            "prefill_max_new_token_samples",
            "prefill_max_kv_read_token_samples",
            "decode_max_kv_read_token_samples",
            "decode_max_batch_size_samples",
        )
        for name in uniform_limits:
            if getattr(self, name) < 2:
                raise ValueError(f"--{name.replace('_', '-')} must be at least 2")
        if self.prefix_max_batch_size_samples < 1:
            raise ValueError("--prefix-max-batch-size-samples must be positive")
        if self.benchmark_warmup_iterations < 0:
            raise ValueError("--benchmark-warmup-iterations must be non-negative")
        if self.benchmark_timeout <= 0:
            raise ValueError("--benchmark-timeout must be positive")

    def _resolve_embedding_transfer_mode(self) -> None:
        """Resolve embedding_transfer_mode from string to enum."""
        if isinstance(self.embedding_transfer_mode, str):
            self.embedding_transfer_mode = EmbeddingTransferMode(
                self.embedding_transfer_mode
            )

    def _resolve_disaggregation_mode(self) -> None:
        """Resolve disaggregation_mode from new enum or legacy boolean flags.

        Priority:
        1. If --disaggregation-mode was explicitly provided, use it.
           Raise if legacy booleans are also set.
        2. If legacy --is-prefill-worker or --is-decode-worker is set,
           emit DeprecationWarning and translate to enum.
        3. If legacy multimodal flags are set, translate to enum,
           emit DeprecationWarning and translate to enum, raise if conflicting
           with --disaggregation-mode.
        3. Apply default (AGGREGATED) if nothing was provided.
        4. Sync boolean fields from the resolved enum value.
        """
        # Convert string to enum (non-None means explicitly provided)
        explicit_mode = self.disaggregation_mode is not None
        if isinstance(self.disaggregation_mode, str):
            if self.disaggregation_mode == PREFILL_DECODE_DISAGGREGATION_MODE:
                self.disaggregation_mode = DisaggregationMode.AGGREGATED
            else:
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

        # Porting multimodal legacy flags
        if (
            self.multimodal_decode_worker
            or self.multimodal_encode_worker
            or self.multimodal_worker
        ):
            self._resolve_disaggregation_model_from_legacy_multimodal_flags()

        # Apply default if neither new flag nor legacy flags were provided
        if self.disaggregation_mode is None:
            self.disaggregation_mode = DisaggregationMode.AGGREGATED

        # Sync booleans from enum (canonical source of truth)
        self.is_prefill_worker = self.disaggregation_mode == DisaggregationMode.PREFILL
        self.is_decode_worker = self.disaggregation_mode == DisaggregationMode.DECODE

    def _resolve_disaggregation_model_from_legacy_multimodal_flags(self) -> None:
        """
        Resolve disaggregation mode from legacy multimodal flags, emit DeprecationWarning
        and raise ValueError if conflicting with --disaggregation-mode.

        Transformation rules:
        1. If --multimodal-decode-worker is set, use DisaggregationMode.DECODE.
        2. If --multimodal-encode-worker is set, use DisaggregationMode.ENCODE.
        3. If --multimodal-worker is set, default to DisaggregationMode.AGGREGATED unless
           --disaggregation-mode is set.
        """
        if self.multimodal_decode_worker:
            _warn_deprecated(
                "--multimodal-decode-worker is deprecated; use "
                "--enable-multimodal --disaggregation-mode=decode. "
                "This release will map the legacy flag to the new arguments.",
            )
            if (
                self.disaggregation_mode is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                raise ValueError(
                    f"Cannot set --multimodal-decode-worker while --disaggregation-mode is not '{DisaggregationMode.DECODE.value}'"
                )
            self.disaggregation_mode = DisaggregationMode.DECODE
            self.enable_multimodal = True
        if self.multimodal_encode_worker:
            _warn_deprecated(
                "--multimodal-encode-worker is deprecated; use "
                "--enable-multimodal --disaggregation-mode=encode. "
                "This release will map the legacy flag to the new arguments.",
            )
            if (
                self.disaggregation_mode is not None
                and self.disaggregation_mode != DisaggregationMode.ENCODE
            ):
                raise ValueError(
                    f"Cannot set --multimodal-encode-worker while --disaggregation-mode is not '{DisaggregationMode.ENCODE.value}'"
                )
            self.disaggregation_mode = DisaggregationMode.ENCODE
            self.enable_multimodal = True
        if self.multimodal_worker:
            _warn_deprecated(
                "--multimodal-worker is deprecated; use --enable-multimodal "
                "with --disaggregation-mode=pd or --disaggregation-mode=prefill. "
                "This release will map the legacy flag to the new arguments.",
            )
            if (
                self.disaggregation_mode is not None
                and self.disaggregation_mode != DisaggregationMode.AGGREGATED
                and self.disaggregation_mode != DisaggregationMode.PREFILL
            ):
                raise ValueError(
                    f"Cannot set --multimodal-worker while --disaggregation-mode is not '{DisaggregationMode.AGGREGATED.value}' or '{DisaggregationMode.PREFILL.value}'"
                )
            # only set 'self.disaggregation_mode' if it is not already set, '--multimodal-worker' may be specified with
            # '--disaggregation-mode=prefill' as prefill workers in P/D disaggregation or without for aggregation.
            if self.disaggregation_mode is None:
                self.disaggregation_mode = DisaggregationMode.AGGREGATED
            self.enable_multimodal = True

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

    def _validate_custom_encoder(self) -> None:
        """Validate the aggregated CustomEncoder configuration.

        The encoder runs in-process in a single aggregated worker on the
        token-in/token-out path and produces image embeds for the mixed
        EmbedsPrompt, so it is a multimodal, aggregated-only, token-mode
        component. Enforce those here (fail fast) instead of silently bypassing
        the multimodal gate at request time, no-op'ing in a decode worker that
        never reaches the custom-encoder branch, or loading the encoder in
        --use-vllm-tokenizer text mode where it is never invoked.
        """
        if not self.custom_encoder_class:
            return
        if (
            self.multimodal_worker
            or self.multimodal_encode_worker
            or self.multimodal_decode_worker
        ):
            raise ValueError(
                "--custom-encoder-class is incompatible with the legacy multimodal "
                "role flags (--multimodal-worker / --multimodal-encode-worker / "
                "--multimodal-decode-worker): the custom encoder is its own "
                "aggregated multimodal path and bypasses vLLM's built-in "
                "multimodal processing."
            )
        if not self.enable_multimodal:
            raise ValueError(
                "--custom-encoder-class requires --enable-multimodal "
                "(the custom encoder is a multimodal component)."
            )
        if self.use_vllm_tokenizer:
            raise ValueError(
                "--custom-encoder-class is incompatible with --use-vllm-tokenizer: "
                "the custom encoder is wired into the token-in/token-out path, "
                "which --use-vllm-tokenizer bypasses (text mode), so the encoder "
                "would load but never run."
            )
        if self.frontend_decoding:
            raise ValueError(
                "--custom-encoder-class is incompatible with --frontend-decoding: "
                "the custom encoder consumes image URLs, but frontend decoding "
                "pre-decodes images to tensors the encoder cannot accept."
            )
        if self.disaggregation_mode != DisaggregationMode.AGGREGATED:
            mode = (
                self.disaggregation_mode.value
                if isinstance(self.disaggregation_mode, DisaggregationMode)
                else self.disaggregation_mode
            )
            raise ValueError(
                f"--custom-encoder-class is only supported with "
                f"--disaggregation-mode=agg (got {mode}). The custom encoder "
                "runs in-process in a single aggregated worker."
            )

    def _validate_embedding_worker_exclusivity(self) -> None:
        """Embedding worker is aggregated-only and exclusive of multimodal roles."""
        if not self.embedding_worker:
            return
        if self.disaggregation_mode != DisaggregationMode.AGGREGATED:
            raise ValueError(
                "--embedding-worker is only valid with --disaggregation-mode=agg "
                f"(got {self.disaggregation_mode.value if isinstance(self.disaggregation_mode, DisaggregationMode) else self.disaggregation_mode}). "
                "Pooling models do not have prefill/decode phases."
            )
        if self._count_multimodal_roles() > 0 or self.enable_multimodal:
            raise ValueError(
                "--embedding-worker cannot be combined with multimodal flags."
            )
        if self.benchmark_mode is not None:
            raise ValueError(
                "--embedding-worker cannot be combined with --benchmark-mode. "
                "Benchmark mode injects InstrumentedScheduler, which is a "
                "generation scheduler and not compatible with pooling engines. "
                "Embedding workers do not run generation, so prefill/decode "
                "benchmark sweeps are not meaningful."
            )
