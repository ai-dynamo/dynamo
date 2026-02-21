# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common argument handling for Dynamo backend workers.

This module provides argument groups and configuration helpers that are
shared across all backend implementations.
"""

import os
from typing import Optional

from dynamo._core import get_reasoning_parser_names, get_tool_parser_names
from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument


class BackendCommonConfig(ConfigBase):
    """Common configuration fields for all backend workers.

    This provides the base configuration that all backends share. Backend-specific
    configurations should inherit from this class.
    """

    # Dynamo hierarchy
    namespace: str
    component: str = "backend"
    endpoint: str = "generate"

    # Model configuration
    model: str
    served_model_name: Optional[str] = None

    # Runtime configuration
    store_kv: str
    request_plane: str
    event_plane: str
    use_kv_events: bool = False
    connector: list[str]
    enable_local_indexer: bool
    durable_kv_events: bool

    # Inference configuration
    dyn_tool_call_parser: Optional[str] = None
    dyn_reasoning_parser: Optional[str] = None
    custom_jinja_template: Optional[str] = None
    endpoint_types: str = "chat,completions"

    # Debugging
    dump_config_to: Optional[str] = None

    def validate(self) -> None:
        """Validate and normalize the configuration."""
        # Derive enable_local_indexer from durable_kv_events
        self.enable_local_indexer = not self.durable_kv_events
        # use_kv_events drives runtime NATS KV behavior; align with durable_kv_events
        self.use_kv_events = self.durable_kv_events

        # Validate and expand custom Jinja template path
        if self.custom_jinja_template:
            expanded_path = os.path.expandvars(
                os.path.expanduser(self.custom_jinja_template)
            )
            if not os.path.isfile(expanded_path):
                raise FileNotFoundError(
                    f"Custom Jinja template file not found: {expanded_path}"
                )
            self.custom_jinja_template = expanded_path

    def get_model_name(self) -> str:
        """Get the effective model name for display and metrics."""
        return self.served_model_name or self.model


class BackendCommonArgGroup(ArgGroup):
    """Common argument group for backend workers.

    This adds the CLI arguments that are common to all backend implementations:
    - Dynamo runtime configuration (namespace, store-kv, request-plane, etc.)
    - Model configuration (model path, served name)
    - Inference configuration (tool parsers, templates, endpoint types)
    - Debugging configuration (dump-config-to)
    """

    def add_arguments(self, parser) -> None:
        """Add common backend arguments to the parser."""
        # Dynamo runtime options
        runtime_group = parser.add_argument_group("Dynamo Runtime Options")
        self._add_runtime_arguments(runtime_group)

        # Model options
        model_group = parser.add_argument_group("Model Options")
        self._add_model_arguments(model_group)

        # Inference options
        inference_group = parser.add_argument_group("Inference Options")
        self._add_inference_arguments(inference_group)

        # Debug options
        debug_group = parser.add_argument_group("Debug Options")
        self._add_debug_arguments(debug_group)

    def _add_runtime_arguments(self, group) -> None:
        """Add Dynamo runtime arguments."""
        add_argument(
            group,
            flag_name="--namespace",
            env_var="DYN_NAMESPACE",
            default="dynamo",
            help="Dynamo namespace for the worker.",
        )
        add_argument(
            group,
            flag_name="--store-kv",
            env_var="DYN_STORE_KV",
            default="etcd",
            choices=["etcd", "file", "mem"],
            help="Key-value backend: etcd, file, or mem. "
            "Etcd uses ETCD_* env vars for connection. "
            "File uses DYN_FILE_KV or $TMPDIR/dynamo_store_kv.",
        )
        add_argument(
            group,
            flag_name="--request-plane",
            env_var="DYN_REQUEST_PLANE",
            default="tcp",
            choices=["tcp", "nats", "http"],
            help="Request distribution mechanism. 'tcp' is fastest.",
        )
        add_argument(
            group,
            flag_name="--event-plane",
            env_var="DYN_EVENT_PLANE",
            default="nats",
            choices=["nats", "zmq"],
            help="Event publishing mechanism.",
        )
        add_argument(
            group,
            flag_name="--connector",
            env_var="DYN_CONNECTOR",
            default=["nixl"],
            nargs="*",
            help="List of KV connectors to use (e.g., --connector nixl lmcache). "
            "Options: nixl, lmcache, kvbm, null, none.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--durable-kv-events",
            env_var="DYN_DURABLE_KV_EVENTS",
            default=False,
            help="Enable durable KV events using NATS JetStream instead of local indexer. "
            "Default uses local indexer for lower latency.",
        )

    def _add_model_arguments(self, group) -> None:
        """Add model configuration arguments."""
        add_argument(
            group,
            flag_name="--model",
            env_var="DYN_MODEL",
            default=None,
            help="Path or HuggingFace identifier for the model to serve.",
        )
        add_argument(
            group,
            flag_name="--served-model-name",
            env_var="DYN_SERVED_MODEL_NAME",
            default=None,
            help="Name to serve the model under. Defaults to model path.",
        )

    def _add_inference_arguments(self, group) -> None:
        """Add inference configuration arguments."""
        add_argument(
            group,
            flag_name="--dyn-tool-call-parser",
            env_var="DYN_TOOL_CALL_PARSER",
            default=None,
            choices=get_tool_parser_names(),
            help="Tool call parser for function calling support.",
        )
        add_argument(
            group,
            flag_name="--dyn-reasoning-parser",
            env_var="DYN_REASONING_PARSER",
            default=None,
            choices=get_reasoning_parser_names(),
            help="Reasoning parser for chain-of-thought support.",
        )
        add_argument(
            group,
            flag_name="--custom-jinja-template",
            env_var="DYN_CUSTOM_JINJA_TEMPLATE",
            default=None,
            help="Path to custom Jinja template for chat formatting. "
            "Overrides the model's default template.",
        )
        add_argument(
            group,
            flag_name="--endpoint-types",
            env_var="DYN_ENDPOINT_TYPES",
            default="chat,completions",
            obsolete_flag="--dyn-endpoint-types",
            help="Comma-separated endpoint types to enable: chat, completions. "
            "Use 'completions' for models without chat templates.",
        )

    def _add_debug_arguments(self, group) -> None:
        """Add debugging arguments."""
        add_argument(
            group,
            flag_name="--dump-config-to",
            env_var="DYN_DUMP_CONFIG_TO",
            default=None,
            help="Dump resolved configuration to the specified file path.",
        )


class WorkerModeConfig(ConfigBase):
    """Configuration for worker mode selection.

    These flags control which type of worker to run (prefill, decode,
    multimodal, embedding, etc.).
    """

    # Disaggregation mode
    is_prefill_worker: bool = False
    is_decode_worker: bool = False

    # Multimodal modes
    multimodal_processor: bool = False
    multimodal_worker: bool = False
    multimodal_encode_worker: bool = False
    multimodal_decode_worker: bool = False

    # Embedding mode
    embedding_worker: bool = False

    # Framework-specific tokenizer usage
    use_framework_tokenizer: bool = False


class WorkerModeArgGroup(ArgGroup):
    """Argument group for worker mode selection.

    These arguments control which type of worker to run.
    """

    def add_arguments(self, parser) -> None:
        """Add worker mode arguments."""
        group = parser.add_argument_group("Worker Mode Options")

        add_negatable_bool_argument(
            group,
            flag_name="--is-prefill-worker",
            env_var="DYN_IS_PREFILL_WORKER",
            default=False,
            help="Run as a prefill-only worker for disaggregated serving.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--is-decode-worker",
            env_var="DYN_IS_DECODE_WORKER",
            default=False,
            help="Run as a decode-only worker for disaggregated serving.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--multimodal-processor",
            env_var="DYN_MULTIMODAL_PROCESSOR",
            default=False,
            help="Run as multimodal processor component.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--multimodal-worker",
            env_var="DYN_MULTIMODAL_WORKER",
            default=False,
            help="Run as multimodal worker for LLM inference with multimodal inputs.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--multimodal-encode-worker",
            env_var="DYN_MULTIMODAL_ENCODE_WORKER",
            default=False,
            help="Run as multimodal encode worker for processing images/videos.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--multimodal-decode-worker",
            env_var="DYN_MULTIMODAL_DECODE_WORKER",
            default=False,
            help="Run as multimodal decode worker.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--embedding-worker",
            env_var="DYN_EMBEDDING_WORKER",
            default=False,
            help="Run as embedding worker for generating embeddings.",
        )
        add_negatable_bool_argument(
            group,
            flag_name="--use-framework-tokenizer",
            env_var="DYN_USE_FRAMEWORK_TOKENIZER",
            default=False,
            help="Use the framework's tokenizer instead of Dynamo's. "
            "This bypasses Dynamo's preprocessor.",
        )
