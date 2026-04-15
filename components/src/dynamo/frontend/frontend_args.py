# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pathlib
from typing import Any, Dict, Optional

from dynamo.common.config_dump import register_encoder
from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.aic_perf_args import (
    AicPerfArgGroup,
    AicPerfConfigBase,
)
from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.utils import (
    add_argument,
    add_negatable_bool_argument,
    env_or_default,
)

from . import __version__

_U32_MAX = 2**32 - 1


def validate_model_name(value: str) -> str:
    """Validate that model-name is a non-empty string."""
    if not value or not isinstance(value, str) or len(value.strip()) == 0:
        raise argparse.ArgumentTypeError(
            f"model-name must be a non-empty string, got: {value}"
        )
    return value.strip()


def validate_model_path(value: str) -> str:
    """Validate that model-path is a valid directory on disk."""
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError(
            f"model-path must be a valid directory on disk, got: {value}"
        )
    return value


class FrontendConfig(KvRouterConfigBase, AicPerfConfigBase):
    """Configuration for the Dynamo frontend."""

    interactive: bool
    kv_cache_block_size: Optional[int]
    http_host: str
    http_port: int
    tls_cert_path: Optional[pathlib.Path]
    tls_key_path: Optional[pathlib.Path]

    router_mode: str
    min_initial_workers: int
    namespace: Optional[str] = None
    namespace_prefix: Optional[str] = None
    enforce_disagg: bool

    migration_limit: int
    migration_max_seq_len: Optional[int]
    active_decode_blocks_threshold: Optional[float]
    active_prefill_tokens_threshold: Optional[int]
    active_prefill_tokens_threshold_frac: Optional[float]
    model_name: Optional[str]
    model_path: Optional[str]
    metrics_prefix: Optional[str] = None

    kserve_grpc_server: bool
    grpc_metrics_port: int
    dump_config_to: Optional[str]

    discovery_backend: str
    request_plane: str
    event_plane: str
    chat_processor: str
    enable_anthropic_api: bool
    strip_anthropic_preamble: bool
    debug_perf: bool
    enable_streaming_tool_dispatch: bool
    enable_streaming_reasoning_dispatch: bool
    exclude_tools_when_tool_choice_none: bool
    preprocess_workers: int
    tokenizer_backend: str
    trust_remote_code: bool

    _VALID_TOKENIZER_BACKENDS = {"default", "fastokens"}

    def validate(self) -> None:
        if bool(self.tls_cert_path) ^ bool(self.tls_key_path):  # ^ is XOR
            raise ValueError(
                "--tls-cert-path and --tls-key-path must be provided together"
            )
        if self.migration_limit < 0 or self.migration_limit > _U32_MAX:
            raise ValueError(
                f"--migration-limit must be between 0 and {_U32_MAX} (0=disabled)"
            )
        if self.migration_max_seq_len is not None and (
            self.migration_max_seq_len < 1 or self.migration_max_seq_len > _U32_MAX
        ):
            raise ValueError(
                f"--migration-max-seq-len must be between 1 and {_U32_MAX}"
            )
        if self.min_initial_workers < 0:
            raise ValueError("--router-min-initial-workers must be >= 0")
        if self.tokenizer_backend not in self._VALID_TOKENIZER_BACKENDS:
            raise ValueError(
                f"--tokenizer: invalid value '{self.tokenizer_backend}' "
                f"(choose from {sorted(self._VALID_TOKENIZER_BACKENDS)})"
            )
        if self.router_prefill_load_model == "aic":
            if self.router_mode != "kv":
                raise ValueError(
                    "--router-prefill-load-model=aic requires --router-mode=kv"
                )
            if self.chat_processor != "dynamo":
                raise ValueError(
                    "--router-prefill-load-model=aic currently requires "
                    "--dyn-chat-processor=dynamo"
                )
            missing = [
                flag
                for flag, value in (
                    ("--aic-backend", self.aic_backend),
                    ("--aic-system", self.aic_system),
                    ("--aic-model-path", self.aic_model_path),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "--router-prefill-load-model=aic requires " + ", ".join(missing)
                )
            if not self.router_track_prefill_tokens:
                raise ValueError(
                    "--router-prefill-load-model=aic requires "
                    "--router-track-prefill-tokens"
                )
        if self.serve_indexer:
            if self.router_mode != "kv":
                raise ValueError("--serve-indexer requires --router-mode=kv")
            if self.use_remote_indexer:
                raise ValueError(
                    "--serve-indexer and --use-remote-indexer are mutually exclusive"
                )


@register_encoder(FrontendConfig)
def _preprocess_for_encode_config(config: FrontendConfig) -> Dict[str, Any]:
    """Convert FrontendConfig object to dictionary for encoding."""
    return config.__dict__


class FrontendArgGroup(ArgGroup):
    """Frontend configuration parameters."""

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--version", action="version", version=f"Dynamo Frontend {__version__}"
        )

        g = parser.add_argument_group("Dynamo Frontend Options")

        # Interactive needs -i short option; use raw add_argument with BooleanOptionalAction
        g.add_argument(
            "-i",
            "--interactive",
            dest="interactive",
            action=argparse.BooleanOptionalAction,
            default=env_or_default("DYN_INTERACTIVE", False),
            help="Interactive text chat.\nenv var: DYN_INTERACTIVE",
        )

        add_argument(
            g,
            flag_name="--namespace",
            env_var="DYN_NAMESPACE",
            default=None,
            help=(
                "Dynamo namespace for model discovery scoping. Use for exact namespace matching. "
                "If --namespace-prefix is also specified, prefix takes precedence."
            ),
        )

        add_argument(
            g,
            flag_name="--kv-cache-block-size",
            env_var="DYN_KV_CACHE_BLOCK_SIZE",
            default=None,
            help="KV cache block size (u32).",
            arg_type=int,
        )

        add_argument(
            g,
            flag_name="--http-host",
            env_var="DYN_HTTP_HOST",
            default="0.0.0.0",
            help="HTTP host for the engine (str).",
        )
        add_argument(
            g,
            flag_name="--http-port",
            env_var="DYN_HTTP_PORT",
            default=8000,
            help="HTTP port for the engine (u16).",
            arg_type=int,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--serve-indexer",
            env_var="DYN_SERVE_INDEXER",
            default=False,
            help="Serve this frontend's local KV indexers over the request plane.",
            dest="serve_indexer",
        )
        add_argument(
            g,
            flag_name="--tls-cert-path",
            env_var="DYN_TLS_CERT_PATH",
            default=None,
            help="TLS certificate path, PEM format.",
            arg_type=pathlib.Path,
        )
        add_argument(
            g,
            flag_name="--tls-key-path",
            env_var="DYN_TLS_KEY_PATH",
            default=None,
            help="TLS certificate key path, PEM format.",
            arg_type=pathlib.Path,
        )

        add_argument(
            g,
            flag_name="--router-mode",
            env_var="DYN_ROUTER_MODE",
            default="round-robin",
            help="How to route the request. power-of-two picks 2 random workers and "
            "routes to the one with fewer in-flight requests. least-loaded routes to "
            "the worker with the fewest active requests. device-aware-weighted routes "
            "based on worker device type (CPU/CUDA). In disaggregated prefill mode, "
            "both power-of-two and least-loaded skip bootstrap optimization and fall "
            "back to the synchronous prefill path.",
            choices=[
                "round-robin",
                "random",
                "power-of-two",
                "kv",
                "direct",
                "least-loaded",
                "device-aware-weighted",
            ],
        )
        add_argument(
            g,
            flag_name="--router-min-initial-workers",
            env_var="DYN_ROUTER_MIN_INITIAL_WORKERS",
            default=0,
            help=(
                "Minimum number of workers required before router startup continues. "
                "This is exported as DYN_ROUTER_MIN_INITIAL_WORKERS so the generic "
                "push-router path and the KV router's config-ready worker gate share "
                "the same startup threshold. Set to 0 to disable the startup wait."
            ),
            arg_type=int,
            dest="min_initial_workers",
        )

        # KV router options (shared with dynamo.router)
        KvRouterArgGroup().add_arguments(parser)
        AicPerfArgGroup().add_arguments(parser)

        add_argument(
            g,
            flag_name="--namespace-prefix",
            env_var="DYN_NAMESPACE_PREFIX",
            default=None,
            help=(
                "Dynamo namespace prefix for model discovery scoping. Discovers models from "
                "namespaces starting with this prefix (e.g., 'ns' matches 'ns', 'ns-abc123', "
                "'ns-def456'). Takes precedence over --namespace if both are specified."
            ),
        )

        add_negatable_bool_argument(
            g,
            flag_name="--enforce-disagg",
            env_var="DYN_ENFORCE_DISAGG",
            default=False,
            dest="enforce_disagg",
            help=(
                "Strictly enforce disaggregated mode. Requests will fail if the prefill router "
                "has not activated yet (e.g., prefill workers still registering). This is stricter "
                "than the default: without this flag, requests arriving before prefill workers are "
                "discovered fall through to aggregated decode-only routing."
            ),
        )

        add_argument(
            g,
            flag_name="--migration-limit",
            env_var="DYN_MIGRATION_LIMIT",
            default=0,
            help=(
                "Maximum number of times a request may be migrated to a different engine worker. "
                "When > 0, enables request migration on worker disconnect."
            ),
            arg_type=int,
        )

        add_argument(
            g,
            flag_name="--migration-max-seq-len",
            env_var="DYN_MIGRATION_MAX_SEQ_LEN",
            default=None,
            help=(
                "Maximum sequence length (prompt + generated tokens) for migration state tracking. "
                "Once the accumulated token count exceeds this limit, the request becomes "
                "non-migratable. Prevents unbounded memory growth from caching long sequences. "
                "Default: no limit."
            ),
            arg_type=int,
        )

        add_argument(
            g,
            flag_name="--active-decode-blocks-threshold",
            env_var="DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
            default=None,
            help=(
                "Threshold percentage (0.0-1.0) for determining when a worker is considered busy "
                "based on KV cache block utilization. If not set, blocks-based busy detection is disabled."
            ),
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
            default=None,
            help=(
                "Literal token count threshold for determining when a worker is considered busy "
                "based on prefill token utilization. When active prefill tokens exceed this "
                "threshold, the worker is marked as busy. If not set, tokens-based busy detection is disabled."
            ),
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--active-prefill-tokens-threshold-frac",
            env_var="DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
            default=None,
            help=(
                "Fraction of max_num_batched_tokens for busy detection. Worker is busy when "
                "active_prefill_tokens > frac * max_num_batched_tokens. Default 1.5 (disabled). "
                "Uses OR logic with --active-prefill-tokens-threshold."
            ),
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--model-name",
            env_var="DYN_MODEL_NAME",
            default=None,
            help="Model name as a string (e.g., 'Llama-3.2-1B-Instruct')",
            arg_type=validate_model_name,
        )
        add_argument(
            g,
            flag_name="--model-path",
            env_var="DYN_MODEL_PATH",
            default=None,
            help="Path to model directory on disk (e.g., /tmp/model_cache/llama3.2_1B/)",
            arg_type=validate_model_path,
        )
        add_argument(
            g,
            flag_name="--metrics-prefix",
            env_var="DYN_METRICS_PREFIX",
            default=None,
            help=(
                "Prefix for Dynamo frontend metrics. If unset, uses DYN_METRICS_PREFIX env var "
                "or 'dynamo_frontend'."
            ),
        )
        add_negatable_bool_argument(
            g,
            flag_name="--kserve-grpc-server",
            env_var="DYN_KSERVE_GRPC_SERVER",
            default=False,
            help="Start KServe gRPC server.",
        )
        add_argument(
            g,
            flag_name="--grpc-metrics-port",
            env_var="DYN_GRPC_METRICS_PORT",
            default=8788,
            help=(
                "HTTP metrics port for gRPC service (u16). Only used with --kserve-grpc-server. "
                "Defaults to 8788."
            ),
            arg_type=int,
        )

        add_argument(
            g,
            flag_name="--dump-config-to",
            env_var="DYN_DUMP_CONFIG_TO",
            default=None,
            help="Dump config to the specified file path.",
        )

        add_argument(
            g,
            flag_name="--discovery-backend",
            env_var="DYN_DISCOVERY_BACKEND",
            default="etcd",
            help=(
                "Discovery backend: kubernetes (K8s API), etcd (distributed KV), file (local filesystem), "
                "mem (in-memory). Etcd uses the ETCD_* env vars (e.g. ETCD_ENDPOINTS) for connection details. "
                "File uses root dir from env var DYN_FILE_KV or defaults to $TMPDIR/dynamo_store_kv."
            ),
            choices=["kubernetes", "etcd", "file", "mem"],
        )
        add_argument(
            g,
            flag_name="--request-plane",
            env_var="DYN_REQUEST_PLANE",
            default="tcp",
            help=(
                "Determines how requests are distributed from routers to workers. "
                "'tcp' is fastest [nats|http|tcp]"
            ),
            choices=["nats", "http", "tcp"],
        )
        add_argument(
            g,
            flag_name="--event-plane",
            env_var="DYN_EVENT_PLANE",
            default="nats",
            help="Determines how events are published [nats|zmq]",
            choices=["nats", "zmq"],
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-anthropic-api",
            env_var="DYN_ENABLE_ANTHROPIC_API",
            default=False,
            help=(
                "[EXPERIMENTAL] Enable Anthropic Messages API endpoint (/v1/messages). "
                "This feature is experimental and may change."
            ),
        )
        add_negatable_bool_argument(
            g,
            flag_name="--strip-anthropic-preamble",
            env_var="DYN_STRIP_ANTHROPIC_PREAMBLE",
            default=False,
            help=(
                "Strip the Claude Code billing preamble (x-anthropic-billing-header) "
                "from the system prompt. Saves tokens and improves prompt caching."
            ),
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-streaming-tool-dispatch",
            env_var="DYN_ENABLE_STREAMING_TOOL_DISPATCH",
            default=False,
            help=(
                "[EXPERIMENTAL] Enable streaming tool call dispatch. Emits "
                "'event: tool_call_dispatch' SSE events on /v1/chat/completions "
                "for each complete tool call before finish_reason arrives. "
                "Can be combined with --enable-streaming-reasoning-dispatch."
            ),
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-streaming-reasoning-dispatch",
            env_var="DYN_ENABLE_STREAMING_REASONING_DISPATCH",
            default=False,
            help=(
                "[EXPERIMENTAL] Enable streaming reasoning dispatch. Emits a "
                "single 'event: reasoning_dispatch' SSE event on /v1/chat/completions "
                "with the complete reasoning block once thinking ends. "
                "Can be combined with --enable-streaming-tool-dispatch."
            ),
        )
        # NOTE: This flag also exists in DynamoRuntimeArgGroup (runtime_args.py).
        # Both definitions are needed: runtime_args controls the Rust-native
        # chat template path (oai.rs), while this one controls the Python
        # frontend processors (vllm_processor / sglang_processor) which parse
        # arguments independently via FrontendConfig.
        add_negatable_bool_argument(
            g,
            flag_name="--exclude-tools-when-tool-choice-none",
            env_var="DYN_EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE",
            default=True,
            help=(
                "Exclude tool definitions from the chat template when "
                "tool_choice='none'. Prevents models from generating raw XML "
                "tool calls in the content field."
            ),
        )
        add_argument(
            g,
            flag_name="--dyn-chat-processor",
            env_var="DYN_CHAT_PROCESSOR",
            default="dynamo",
            dest="chat_processor",
            help=(
                "[EXPERIMENTAL] Chat pre/post processor backend. 'dynamo' uses the Rust "
                "preprocessor. 'vllm' uses local vLLM for pre and post processing. "
                "'sglang' uses SGLang APIs for chat template rendering, tool call "
                "parsing, and reasoning parsing."
            ),
            choices=["dynamo", "vllm", "sglang"],
        )

        add_negatable_bool_argument(
            g,
            flag_name="--dyn-debug-perf",
            env_var="DYN_DEBUG_PERF",
            default=False,
            dest="debug_perf",
            help=(
                "[EXPERIMENTAL] Enable performance instrumentation for diagnosing preprocessing bottlenecks. "
                "Logs per-function timing, request concurrency, and hot-path section durations. "
                "Supported with '--dyn-chat-processor vllm' and '--dyn-chat-processor sglang'."
            ),
        )

        add_argument(
            g,
            flag_name="--dyn-preprocess-workers",
            env_var="DYN_PREPROCESS_WORKERS",
            default=0,
            dest="preprocess_workers",
            help=(
                "[EXPERIMENTAL] Number of worker processes for preprocessing and output processing. "
                "When > 0, offloads CPU-bound work (tokenization, template rendering, "
                "detokenization) to a ProcessPoolExecutor with N workers, each with its "
                "own GIL. 0 (default) keeps all processing on the main event loop. "
                "Supported with '--dyn-chat-processor vllm' and '--dyn-chat-processor sglang'."
            ),
            arg_type=int,
        )

        add_argument(
            g,
            flag_name="--tokenizer",
            env_var="DYN_TOKENIZER",
            default="default",
            dest="tokenizer_backend",
            help=(
                "Tokenizer backend for BPE models: 'default' (HuggingFace tokenizers library) "
                "or 'fastokens' (fastokens crate for high-performance BPE encoding). "
                "Decoding always uses HuggingFace. Has no effect on TikToken models."
            ),
            choices=["default", "fastokens"],
        )

        add_negatable_bool_argument(
            g,
            flag_name="--trust-remote-code",
            env_var="DYN_TRUST_REMOTE_CODE",
            default=False,
            help=(
                "Trust remote code when loading the tokenizer. Required for models "
                "that ship custom tokenizer code (e.g. Qwen, Falcon)."
            ),
        )
