# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo runtime configuration ArgGroup."""

import argparse
import logging
import os
from typing import List, Optional

from dynamo._core import get_reasoning_parser_names, get_tool_parser_names
from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument
from dynamo.common.utils.namespace import get_worker_namespace
from dynamo.common.utils.output_modalities import OutputModality

logger = logging.getLogger(__name__)
_FPM_TRACE_VALUES = {"1", "0", "true", "false", "on", "off", "yes", "no"}
_fpm_trace_invalid_warning_emitted = False


class DynamoRuntimeConfig(ConfigBase):
    """Configuration for Dynamo runtime (common across all backends)."""

    namespace: str
    endpoint: Optional[str] = None
    discovery_backend: str
    request_plane: str
    event_plane: Optional[str] = None
    fpm_trace: bool = False
    connector: list[str]
    enable_local_indexer: bool = True

    dyn_tool_call_parser: Optional[str] = None
    dyn_reasoning_parser: Optional[str] = None
    exclude_tools_when_tool_choice_none: bool = True
    dyn_enable_structural_tag: bool = False
    dyn_structural_tag_scope: str = "auto"
    dyn_structural_tag_schema: str = "auto"
    custom_jinja_template: Optional[str] = None
    endpoint_types: str
    dump_config_to: Optional[str] = None
    multimodal_embedding_cache_capacity_gb: float
    multimodal_embedding_cache_publisher: bool = False
    output_modalities: List[str]
    media_output_fs_url: str = "file:///tmp/dynamo_media"
    media_output_http_url: Optional[str] = None
    # Raw `--health-check-payload` value (JSON object string or `@/path/to/file.json`).
    # Honored only by the unified backend's `Worker`, where it overrides the engine's
    # default `health_check_payload()` for the runtime canary.
    health_check_payload: Optional[str] = None
    # Worker-side request admission/rejection knobs. Disabled (None) by
    # default; when set, these surface env vars that the Rust runtime reads
    # directly (see lib/runtime/src/pipeline/network/ingress/shared_tcp_endpoint.rs).
    engine_request_limit: Optional[int] = None

    def validate(self) -> None:
        self.namespace = get_worker_namespace(self.namespace)

        # The Rust FPM sink reads this setting from the process environment.
        # Canonicalize the resolved CLI/env value before the runtime or backend
        # child processes are created so --fpm-trace and --no-fpm-trace apply to
        # both the Python instrumentation and the Rust persistence layer.
        if self.fpm_trace or "DYN_FPM_TRACE" in os.environ:
            raw_fpm_trace = os.environ.get("DYN_FPM_TRACE")
            if (
                raw_fpm_trace is not None
                and raw_fpm_trace.strip().lower() not in _FPM_TRACE_VALUES
                and not self.fpm_trace
                and "DYN_FORWARDPASS_METRIC_PORT" not in os.environ
            ):
                global _fpm_trace_invalid_warning_emitted
                if not _fpm_trace_invalid_warning_emitted:
                    _fpm_trace_invalid_warning_emitted = True
                    logger.warning(
                        "Invalid DYN_FPM_TRACE value %r; expected one of 1/0, "
                        "true/false, on/off, or yes/no. FPM tracing is disabled "
                        "for this worker.",
                        raw_fpm_trace,
                    )
            os.environ["DYN_FPM_TRACE"] = "1" if self.fpm_trace else "0"

        self._validate_output_modalities()

        if self.engine_request_limit is not None and self.engine_request_limit <= 0:
            raise ValueError(
                f"--engine-request-limit must be a positive integer, got {self.engine_request_limit}"
            )

    def _validate_output_modalities(self) -> None:
        """Validate --output-modalities values."""
        if not self.output_modalities:
            return
        valid = OutputModality.valid_names()
        normalized = [m.lower() for m in self.output_modalities]
        invalid = [m for m in normalized if m not in valid]
        if invalid:
            raise ValueError(
                f"Invalid output modality: {', '.join(invalid)}. "
                f"Valid options are: {', '.join(sorted(valid))}"
            )


# For simplicity, we do not prepend "dyn-" unless it's absolutely necessary. These are
# exemplary exceptions:
# - To avoid name conflicts with different backends, prefix "dyn-" for dynamo specific
#   args.
class DynamoRuntimeArgGroup(ArgGroup):
    """Dynamo runtime configuration parameters (common to all backends)."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add Dynamo runtime arguments to parser."""
        g = parser.add_argument_group("Dynamo Runtime Options")

        add_argument(
            g,
            flag_name="--namespace",
            env_var="DYN_NAMESPACE",
            default="dynamo",
            help="Dynamo namespace. If DYN_NAMESPACE_WORKER_SUFFIX is set, "
            "'-{suffix}' is appended to support multiple worker pools",
        )
        add_argument(
            g,
            flag_name="--endpoint",
            env_var="DYN_ENDPOINT",
            default=None,
            help="Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Example: dyn://dynamo.backend.generate.",
        )
        add_argument(
            g,
            flag_name="--discovery-backend",
            env_var="DYN_DISCOVERY_BACKEND",
            default="etcd",
            help="Discovery backend: kubernetes (K8s API), etcd (distributed KV), file (local filesystem), mem (in-memory). Etcd uses the ETCD_* env vars (e.g. ETCD_ENDPOINTS) for connection details. File uses root dir from env var DYN_FILE_KV or defaults to $TMPDIR/dynamo_store_kv.",
            choices=["kubernetes", "etcd", "file", "mem"],
        )
        add_argument(
            g,
            flag_name="--request-plane",
            env_var="DYN_REQUEST_PLANE",
            default="tcp",
            help="Determines how requests are distributed from routers to workers. 'tcp' is fastest.",
            choices=["tcp", "nats"],
        )
        add_argument(
            g,
            flag_name="--event-plane",
            env_var="DYN_EVENT_PLANE",
            default=None,
            help="Determines how events are published. If unset, defaults to 'zmq' for "
            "all discovery backends. Set to 'nats' to use a NATS-based event plane.",
            choices=["nats", "zmq"],
        )
        add_negatable_bool_argument(
            g,
            flag_name="--fpm-trace",
            env_var="DYN_FPM_TRACE",
            default=False,
            help="Persist backend forward-pass metrics to rotating gzip JSONL trace files. Also enables the backend FPM instrumentation required to produce those records.",
        )
        add_argument(
            g,
            flag_name="--connector",
            env_var="DYN_CONNECTOR",
            default=[],
            help="[Deprecated for vLLM] Use --kv-transfer-config instead. For TRT-LLM, options: nixl, lmcache, kvbm, null, none.",
            nargs="*",
        )

        # Optional: tool/reasoning parsers (choices from dynamo._core when available)
        add_argument(
            g,
            flag_name="--dyn-tool-call-parser",
            env_var="DYN_TOOL_CALL_PARSER",
            default=None,
            help="Tool call parser name for the model.",
            choices=get_tool_parser_names(),
        )
        add_argument(
            g,
            flag_name="--dyn-reasoning-parser",
            env_var="DYN_REASONING_PARSER",
            default=None,
            help="Reasoning parser name for the model. If not specified, no reasoning parsing is performed.",
            choices=get_reasoning_parser_names(),
        )
        # NOTE: This flag also exists in FrontendArgGroup (frontend_args.py).
        # Both definitions are needed: this one controls the Rust-native chat
        # template path (oai.rs), while the frontend copy controls the Python
        # processors (vllm_processor / sglang_processor) which parse arguments
        # independently via FrontendConfig.
        add_negatable_bool_argument(
            g,
            flag_name="--exclude-tools-when-tool-choice-none",
            env_var="DYN_EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE",
            default=True,
            help="Exclude tool definitions from the chat template when tool_choice='none'. "
            "Prevents models from generating raw XML tool calls in the content field.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--dyn-enable-structural-tag",
            env_var="DYN_ENABLE_STRUCTURAL_TAG",
            default=False,
            help="Enable structural tag guided decoding for tool calls.",
        )
        add_argument(
            g,
            flag_name="--dyn-structural-tag-scope",
            env_var="DYN_STRUCTURAL_TAG_SCOPE",
            default="auto",
            choices=["auto", "always"],
            help="Controls when structural tags are activated. "
            "'auto': for required/named tool_choice, and if any tool has strict=true "
            "or parallel_tool_calls is false. "
            "'always': also for auto without those conditions. "
            "tool_choice none is unaffected by auto vs always.",
        )
        add_argument(
            g,
            flag_name="--dyn-structural-tag-schema",
            env_var="DYN_STRUCTURAL_TAG_SCHEMA",
            default="auto",
            choices=["auto", "strict"],
            help="Controls parameter schema strictness inside structural tags. "
            "'auto': real schema only for tools with strict=true; "
            "syntactically constrained but schema-unconstrained for all other tools. "
            "'strict': real parameter schema for all tools.",
        )
        add_argument(
            g,
            flag_name="--custom-jinja-template",
            env_var="DYN_CUSTOM_JINJA_TEMPLATE",
            default=None,
            help="Path to a custom Jinja template file to override the model's default chat template. This template will take precedence over any template found in the model repository.",
        )

        add_argument(
            g,
            flag_name="--endpoint-types",
            env_var="DYN_ENDPOINT_TYPES",
            default="chat,completions",
            obsolete_flag="--dyn-endpoint-types",
            help="Comma-separated list of endpoint types to enable. Options: 'chat', 'completions'. Use 'completions' for models without chat templates.",
        )

        add_argument(
            g,
            flag_name="--dump-config-to",
            env_var="DYN_DUMP_CONFIG_TO",
            default=None,
            help="Dump resolved configuration to the specified file path.",
        )

        add_argument(
            g,
            flag_name="--multimodal-embedding-cache-capacity-gb",
            env_var="DYN_MULTIMODAL_EMBEDDING_CACHE_CAPACITY_GB",
            default=0,
            arg_type=float,
            help="Capacity of the multimodal embedding cache in GB. 0 = disabled.",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-embedding-cache-publisher",
            env_var="DYN_MULTIMODAL_EMBEDDING_CACHE_PUBLISHER",
            default=False,
            help="Enable the multimodal embedding cache publisher. Useful when using KV-aware routing. "
            "Not needed for round-robin routing or single-GPU / aggregated deployments.",
        )

        add_argument(
            g,
            flag_name="--output-modalities",
            env_var="DYN_OUTPUT_MODALITIES",
            default=["text"],
            help="Output modalities for omni/diffusion mode (e.g., --output-modalities text image audio video).",
            nargs="*",
        )

        # Media storage (generated images and videos)
        add_argument(
            g,
            flag_name="--media-output-fs-url",
            env_var="DYN_MEDIA_OUTPUT_FS_URL",
            default="file:///tmp/dynamo_media",
            help="Filesystem URL for storing generated images and videos (e.g. file:///tmp/dynamo_media, s3://bucket/path).",
        )
        add_argument(
            g,
            flag_name="--media-output-http-url",
            env_var="DYN_MEDIA_OUTPUT_HTTP_URL",
            default=None,
            help="Base URL for rewriting media file paths in responses (e.g. http://localhost:8000/media). If unset, returns raw filesystem paths.",
        )

        add_argument(
            g,
            flag_name="--health-check-payload",
            env_var="DYN_HEALTH_CHECK_PAYLOAD",
            default=None,
            help="Override the runtime health-check canary payload. Accepts a JSON "
            'object (e.g. \'{"token_ids": [1], "stop_conditions": {"max_tokens": 1}}\') '
            "or '@/path/to/payload.json'. Takes precedence over the engine's "
            "default health_check_payload(). Unified backend only.",
        )

        # Worker-side request admission/rejection. Defaults to None (disabled);
        # when unset the worker behaves exactly as before. Surfaces an env var —
        # the Rust runtime reads DYN_ENGINE_REQUEST_LIMIT directly. The Dynamo-side
        # overflow queue is a small fixed burst (default 16, hard cap N+16) and is
        # not a user-facing knob; advanced users may override it via the
        # DYN_DYNAMO_REQUEST_QUEUE_LIMIT env var.
        add_argument(
            g,
            flag_name="--engine-request-limit",
            env_var="DYN_ENGINE_REQUEST_LIMIT",
            default=None,
            arg_type=int,
            help="Max requests handled concurrently by the engine (worker-pool "
            "semaphore size). Enables worker-side request rejection when set. "
            "Disabled by default.",
        )
