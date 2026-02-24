# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo runtime configuration ArgGroup."""

from typing import List, Optional

from dynamo._core import get_reasoning_parser_names, get_tool_parser_names
from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument
from dynamo.common.utils.namespace import get_worker_namespace
from dynamo.common.utils.output_modalities import OutputModality


class DynamoRuntimeConfig(ConfigBase):
    """Configuration for Dynamo runtime (common across all backends)."""

    namespace: str
    endpoint: Optional[str] = None
    discovery_backend: str
    request_plane: str
    event_plane: str
    connector: list[str]
    enable_local_indexer: bool
    durable_kv_events: bool

    dyn_tool_call_parser: Optional[str] = None
    dyn_reasoning_parser: Optional[str] = None
    custom_jinja_template: Optional[str] = None
    endpoint_types: str
    dump_config_to: Optional[str] = None
    multimodal_embedding_cache_capacity_gb: float
    output_modalities: List[str]
    media_output_fs_url: str = "file:///tmp/dynamo_media"
    media_output_http_url: Optional[str] = None

    def validate(self) -> None:
        self.namespace = get_worker_namespace(self.namespace)

        # TODO  get a better way for spot fixes like this.
        self.enable_local_indexer = not self.durable_kv_events
        self._validate_output_modalities()

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

    def add_arguments(self, parser) -> None:
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
            choices=["tcp", "nats", "http"],
        )
        add_argument(
            g,
            flag_name="--event-plane",
            env_var="DYN_EVENT_PLANE",
            default="nats",
            help="Determines how events are published.",
            choices=["nats", "zmq"],
        )
        add_argument(
            g,
            flag_name="--connector",
            env_var="DYN_CONNECTOR",
            default=["nixl"],
            help="List of connectors to use in order (e.g., --connector nixl lmcache). Options: nixl, lmcache, kvbm, null, none. Order will be preserved in MultiConnector.",
            nargs="*",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--durable-kv-events",
            env_var="DYN_DURABLE_KV_EVENTS",
            default=False,
            help="Enable durable KV events using NATS JetStream instead of the local indexer. By default, local indexer is enabled for lower latency. Use this flag when you need durability and multi-replica router consistency. Requires NATS with JetStream enabled. Can also be set via DYN_DURABLE_KV_EVENTS=true env var.",
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
