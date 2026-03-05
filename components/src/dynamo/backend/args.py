# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common argument handling for Dynamo backend workers.

This module provides argument groups and configuration helpers that are
shared across all backend implementations.
"""

import argparse
import os
from typing import Optional

from dynamo.common.configuration import (
    ArgGroup,
    ConfigBase,
    add_argument,
    add_negatable_bool_argument,
    get_reasoning_parser_names,
    get_tool_parser_names,
)


class DynamoRuntimeConfig(ConfigBase):
    """Common configuration fields for all backend workers.

    This provides the base configuration that all backends share. Backend-specific
    configurations should inherit from this class and set the ``engine`` attribute
    to a typed engine-specific config object so that standard Dynamo fields
    (accessed as ``config.<field>``) are clearly separated from engine-specific
    fields (accessed as ``config.engine.<field>``).
    """

    # Backend-specific engine configuration.  Subclasses should set this to a
    # typed object (e.g., ``MyEngineConfig``) so callers access engine-specific
    # settings via ``config.engine.<field>`` rather than flat attributes.
    engine: Optional[Any] = None

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
    disaggregation_mode: str = "aggregated"

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


class DynamoRuntimeArgGroup(ArgGroup):
    """Common argument group for backend workers.

    This adds the CLI arguments that are common to all backend implementations:
    - Dynamo runtime configuration (namespace, store-kv, request-plane, etc.)
    - Model configuration (model path, served name)
    - Inference configuration (tool parsers, templates, endpoint types)
    - Debugging configuration (dump-config-to)
    """

    def add_arguments(self, parser: "argparse.ArgumentParser") -> None:
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

    def _add_runtime_arguments(self, group: "argparse._ArgumentGroup") -> None:
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

    def _add_model_arguments(self, group: "argparse._ArgumentGroup") -> None:
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

    def _add_inference_arguments(self, group: "argparse._ArgumentGroup") -> None:
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
        add_argument(
            group,
            flag_name="--disaggregation-mode",
            env_var="DYN_DISAGGREGATION_MODE",
            default="aggregated",
            choices=["prefill", "decode", "aggregated"],
            help="Disaggregated serving mode: 'prefill' for prefill-only workers, "
            "'decode' for decode-only workers, 'aggregated' for normal operation.",
        )

    def _add_debug_arguments(self, group: "argparse._ArgumentGroup") -> None:
        """Add debugging arguments."""
        add_argument(
            group,
            flag_name="--dump-config-to",
            env_var="DYN_DUMP_CONFIG_TO",
            default=None,
            help="Dump resolved configuration to the specified file path.",
        )
