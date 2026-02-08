# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo runtime configuration ArgGroup."""

from dynamo._core import get_reasoning_parser_names, get_tool_parser_names
from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument


class DynamoRuntimeConfig(ConfigBase):
    """Configuration for Dynamo runtime (common across all backends)."""

    namespace: str
    store_kv: str
    request_plane: str
    event_plane: str
    migration_limit: int
    connector: list[str]
    enable_local_indexer: bool

    dyn_tool_call_parser: str
    dyn_reasoning_parser: str
    custom_jinja_template: str
    endpoint_types: str
    dump_config_to: str

    def validate(self) -> None:
        if not self.namespace:
            raise ValueError("namespace is required")


class DynamoRuntimeArgGroup(ArgGroup):
    """Dynamo runtime configuration parameters (common to all backends)."""

    def add_arguments(self, parser) -> None:
        """Add Dynamo runtime arguments to parser."""
        g = parser.add_argument_group("Dynamo Runtime Options")

        add_argument(
            g,
            flag_name="--namespace",
            env_var="DYN_NAMESPACE",
            default="dynamo",
            help="Dynamo namespace",
        )
        add_argument(
            g,
            flag_name="--store-kv",
            env_var="DYN_STORE_KV",
            default="etcd",
            help="KV store backend: etcd, file, mem.",
            choices=["etcd", "file", "mem"],
        )
        add_argument(
            g,
            flag_name="--request-plane",
            env_var="DYN_REQUEST_PLANE",
            default="tcp",
            help="Request distribution plane: tcp, nats, http.",
            choices=["tcp", "nats", "http"],
        )
        add_argument(
            g,
            flag_name="--event-plane",
            env_var="DYN_EVENT_PLANE",
            default="nats",
            help="Event publishing plane: nats, zmq.",
            choices=["nats", "zmq"],
        )
        add_argument(
            g,
            flag_name="--migration-limit",
            env_var="DYN_MIGRATION_LIMIT",
            default=0,
            help="Maximum number of times a request may be migrated to a different engine worker.",
            arg_type=int,
        )

        add_argument(
            g,
            flag_name="--connector",
            env_var="DYN_CONNECTOR",
            default=["nixl"],
            help="KV connector chain in order (e.g. nixl lmcache). Options: nixl, lmcache, kvbm, null, none.",
            nargs="*",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--enable-local-indexer",
            env_var="DYN_LOCAL_INDEXER",
            default=False,
            help="Enable worker-local KV indexer for tracking this worker's own KV cache state.",
        )

        # Optional: tool/reasoning parsers (choices from dynamo._core when available)
        # To avoid name conflicts with different backends, prefix "dyn-" for dynamo specific args
        add_argument(
            g,
            flag_name="--dyn-tool-call-parser",
            env_var="DYN_TOOL_CALL_PARSER",
            default=None,
            help="Tool call parser name for the model",
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
        add_argument(
            g,
            flag_name="--custom-jinja-template",
            env_var="DYN_CUSTOM_JINJA_TEMPLATE",
            default=None,
            help="Path to a custom Jinja template file to override the model's default chat template.",
        )

        add_argument(
            g,
            flag_name="--endpoint-types",
            env_var="DYN_ENDPOINT_TYPES",
            default="chat,completions",
            obsolete_flag="--dyn-endpoint-types",
            help="Comma-separated endpoint types to enable (e.g. chat,completions).",
        )

        add_argument(
            g,
            flag_name="--dump-config-to",
            env_var="DYN_DUMP_CONFIG_TO",
            default=None,
            help="Dump resolved configuration to the specified file path.",
        )
