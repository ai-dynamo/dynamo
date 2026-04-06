# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo vLLM wrapper configuration ArgGroup."""

import warnings
from typing import Optional, Union

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument

from . import __version__
from .constants import DisaggregationMode, EmbeddingTransferMode


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

        add_argument(
            g,
            flag_name="--embedding-transfer-mode",
            env_var="DYN_VLLM_EMBEDDING_TRANSFER_MODE",
            default=EmbeddingTransferMode.NIXL_WRITE.value,
            help="Worker embedding transfer mode: 'local' (default, local file system), "
            "'nixl-write' (NIXL transfer with WRITE), or 'nixl-read' (NIXL transfer with READ).",
            choices=[m.value for m in EmbeddingTransferMode],
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

        # GMS (GPU Memory Service) shadow mode
        add_negatable_bool_argument(
            g,
            flag_name="--gms-shadow-mode",
            env_var="DYN_VLLM_GMS_SHADOW_MODE",
            default=False,
            help=(
                "Enable GMS shadow/standby mode. Shadow engines skip KV cache "
                "allocation at startup, automatically sleep after initialization, "
                "and wake on demand when the active engine dies. "
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
                "Sweeps prefill ISLs and/or decode (context_length x batch_size) "
                "points, collecting ForwardPassMetrics at each operating point."
            ),
        )
        add_argument(
            g,
            flag_name="--benchmark-prefill-granularity",
            env_var="DYN_BENCHMARK_PREFILL_GRANULARITY",
            default=16,
            type=int,
            help="Number of ISL sample points for prefill sweep (default: 16).",
        )
        add_argument(
            g,
            flag_name="--benchmark-decode-length-granularity",
            env_var="DYN_BENCHMARK_DECODE_LENGTH_GRANULARITY",
            default=6,
            type=int,
            help=(
                "Number of context length sample points for decode sweep "
                "(default: 6)."
            ),
        )
        add_argument(
            g,
            flag_name="--benchmark-decode-batch-granularity",
            env_var="DYN_BENCHMARK_DECODE_BATCH_GRANULARITY",
            default=6,
            type=int,
            help=(
                "Number of batch size sample points per context length " "(default: 6)."
            ),
        )
        add_argument(
            g,
            flag_name="--benchmark-warmup-iterations",
            env_var="DYN_BENCHMARK_WARMUP_ITERATIONS",
            default=5,
            type=int,
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
            default=300,
            type=int,
            help=(
                "Maximum seconds to wait for benchmark to complete "
                "(default: 300). Worker startup fails if exceeded."
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
    use_vllm_tokenizer: bool

    # Multimodal
    route_to_encoder: bool
    multimodal_encode_worker: bool
    multimodal_worker: bool
    multimodal_decode_worker: bool
    enable_multimodal: bool
    mm_prompt_template: str
    frontend_decoding: bool
    embedding_transfer_mode: Union[
        str, EmbeddingTransferMode
    ]  # resolved to enum in validate()

    # Headless mode for multi-node TP/PP
    headless: bool = False

    # ModelExpress P2P
    model_express_url: Optional[str] = None

    # GMS shadow mode
    gms_shadow_mode: bool = False

    # Benchmark / self-profiling
    benchmark_mode: Optional[str] = None
    benchmark_prefill_granularity: int = 16
    benchmark_decode_length_granularity: int = 6
    benchmark_decode_batch_granularity: int = 6
    benchmark_warmup_iterations: int = 5
    benchmark_output_path: str = "/tmp/benchmark_results.json"
    benchmark_timeout: int = 300

    def validate(self) -> None:
        """Validate vLLM wrapper configuration."""
        self._resolve_disaggregation_mode()
        self._resolve_embedding_transfer_mode()
        self._validate_multimodal_role_exclusivity()
        self._validate_multimodal_requires_flag()

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
            warnings.warn(
                "--multimodal-decode-worker is deprecated, use --disaggregation-mode=decode and --enable-multimodal",
                DeprecationWarning,
                stacklevel=2,
            )
            if (
                self.disaggregation_mode is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                raise ValueError(
                    f"Cannot set --multimodal-decode-worker while --disaggregation-mode is not '{DisaggregationMode.DECODE.value}'"
                )
            self.disaggregation_mode = DisaggregationMode.DECODE
        if self.multimodal_encode_worker:
            warnings.warn(
                "--multimodal-encode-worker is deprecated, use --disaggregation-mode=encode and --enable-multimodal",
                DeprecationWarning,
                stacklevel=2,
            )
            if (
                self.disaggregation_mode is not None
                and self.disaggregation_mode != DisaggregationMode.ENCODE
            ):
                raise ValueError(
                    f"Cannot set --multimodal-encode-worker while --disaggregation-mode is not '{DisaggregationMode.ENCODE.value}'"
                )
            self.disaggregation_mode = DisaggregationMode.ENCODE
        if self.multimodal_worker:
            warnings.warn(
                "--multimodal-worker is deprecated, use --disaggregation-mode=agg or --disaggregation-mode=prefill and --enable-multimodal",
                DeprecationWarning,
                stacklevel=2,
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
