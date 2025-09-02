# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import os
import sys
from typing import Optional

from vllm.config import KVTransferConfig
from vllm.distributed.kv_events import KVEventsConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser

from dynamo._core import get_reasoning_parser_names, get_tool_parser_names

from . import __version__
from .ports import (
    DEFAULT_DYNAMO_PORT_MAX,
    DEFAULT_DYNAMO_PORT_MIN,
    DynamoPortRange,
    EtcdContext,
    PortAllocationRequest,
    PortMetadata,
    allocate_and_reserve_port,
    allocate_and_reserve_port_block,
    get_host_ip,
)

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

VALID_CONNECTORS = {"nixl", "lmcache", "kvbm", "null", "none"}

# Global LMCache configuration - initialize once on module import
ENABLE_LMCACHE = os.getenv("ENABLE_LMCACHE", "0").lower() in ("1", "true", "yes")


class Config:
    """Command line parameters or defaults"""

    # dynamo specific
    namespace: str
    component: str
    endpoint: str
    is_prefill_worker: bool
    migration_limit: int = 0
    kv_port: Optional[int] = None
    port_range: DynamoPortRange

    # mirror vLLM
    model: str
    served_model_name: Optional[str]

    # rest vLLM args
    engine_args: AsyncEngineArgs

    # Connector list from CLI
    connector_list: Optional[list] = None

    # tool and reasoning parser info
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None


def parse_args() -> Config:
    parser = FlexibleArgumentParser(
        description="vLLM server integrated with Dynamo LLM."
    )
    parser.add_argument(
        "--version", action="version", version=f"Dynamo Backend VLLM {__version__}"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--is-prefill-worker",
        action="store_true",
        help="Enable prefill functionality for this worker. Uses the provided namespace to construct dyn://namespace.prefill.generate",
    )
    parser.add_argument(
        "--migration-limit",
        type=int,
        default=0,
        help="Maximum number of times a request may be migrated to a different engine worker. The number may be overridden by the engine.",
    )
    parser.add_argument(
        "--dynamo-port-min",
        type=int,
        default=DEFAULT_DYNAMO_PORT_MIN,
        help=f"Minimum port number for Dynamo services (default: {DEFAULT_DYNAMO_PORT_MIN}). Must be in registered ports range (1024-49151).",
    )
    parser.add_argument(
        "--dynamo-port-max",
        type=int,
        default=DEFAULT_DYNAMO_PORT_MAX,
        help=f"Maximum port number for Dynamo services (default: {DEFAULT_DYNAMO_PORT_MAX}). Must be in registered ports range (1024-49151).",
    )
    parser.add_argument(
        "--connector",
        nargs="*",
        default=["nixl"],
        help="List of connectors to use in order (e.g., --connector nixl lmcache). "
        "Options: nixl, lmcache, kvbm, null, none. Default: nixl. Order will be preserved in MultiConnector.",
    )
    # To avoid name conflicts with different backends, adoped prefix "dyn-" for dynamo specific args
    parser.add_argument(
        "--dyn-tool-call-parser",
        type=str,
        default=None,
        choices=get_tool_parser_names(),
        help="Tool call parser name for the model.",
    )
    parser.add_argument(
        "--dyn-reasoning-parser",
        type=str,
        default=None,
        choices=get_reasoning_parser_names(),
        help="Reasoning parser name for the model.",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)

    if engine_args.enable_prefix_caching is None:
        logger.debug(
            "--enable-prefix-caching or --no-enable-prefix-caching not specified. Defaulting to True (vLLM v1 default behavior)"
        )
        engine_args.enable_prefix_caching = True

    config = Config()
    config.model = args.model
    if args.served_model_name:
        assert (
            len(args.served_model_name) <= 1
        ), "We do not support multiple model names."
        config.served_model_name = args.served_model_name[0]
    else:
        # This becomes an `Option` on the Rust side
        config.served_model_name = None

    namespace = os.environ.get("DYNAMO_NAMESPACE", "dynamo")

    if args.is_prefill_worker:
        args.endpoint = f"dyn://{namespace}.prefill.generate"
    else:
        # For decode workers, also use the provided namespace instead of hardcoded "dynamo"
        args.endpoint = f"dyn://{namespace}.backend.generate"

    endpoint_str = args.endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        logger.error(
            f"Invalid endpoint format: '{args.endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = endpoint_parts

    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name
    config.engine_args = engine_args
    config.is_prefill_worker = args.is_prefill_worker
    config.migration_limit = args.migration_limit
    config.port_range = DynamoPortRange(
        min=args.dynamo_port_min, max=args.dynamo_port_max
    )
    config.tool_call_parser = args.dyn_tool_call_parser
    config.reasoning_parser = args.dyn_reasoning_parser
    # Check for conflicting flags
    has_kv_transfer_config = (
        hasattr(engine_args, "kv_transfer_config")
        and engine_args.kv_transfer_config is not None
    )
    has_connector_flag = args.connector is not None

    if has_kv_transfer_config and has_connector_flag:
        raise ValueError(
            "Cannot specify both --kv-transfer-config and --connector flags"
        )

    if has_connector_flag:
        normalized = [c.lower() for c in args.connector]

        invalid = [c for c in normalized if c not in VALID_CONNECTORS]
        if invalid:
            raise ValueError(
                f"Invalid connector(s): {', '.join(invalid)}. Valid options are: {', '.join(sorted(VALID_CONNECTORS))}"
            )

        if "none" in normalized or "null" in normalized:
            if len(normalized) > 1:
                raise ValueError(
                    "'none' and 'null' cannot be combined with other connectors"
                )
            config.connector_list = []
        else:
            config.connector_list = normalized

    if config.engine_args.block_size is None:
        config.engine_args.block_size = 16
        logger.debug(
            f"Setting reasonable default of {config.engine_args.block_size} for block_size"
        )

    return config


async def configure_ports_with_etcd(config: Config, etcd_client):
    """Configure all settings that require ETCD, including port allocation and vLLM overrides."""

    etcd_context = EtcdContext(client=etcd_client, namespace=config.namespace)

    dp_rank = config.engine_args.data_parallel_rank or 0
    worker_id = f"vllm-{config.component}-dp{dp_rank}"

    # Allocate KV events port
    if config.engine_args.enable_prefix_caching:
        kv_metadata = PortMetadata(worker_id=worker_id, reason="zmq_kv_event_port")
        kv_port = await allocate_and_reserve_port(
            etcd_context=etcd_context,
            metadata=kv_metadata,
            port_range=config.port_range,
        )
        config.kv_port = kv_port
        logger.info(f"Allocated ZMQ KV events port: {kv_port} (worker_id={worker_id})")

        # Check if NIXL is needed based on connector list
    needs_nixl = config.connector_list and "nixl" in config.connector_list

    if needs_nixl:
        # Allocate side channel ports
        # https://github.com/vllm-project/vllm/blob/releases/v0.10.0/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py#L372
        # NIXL calculates ports as: base_port + (dp_rank * tp_size) + tp_rank
        # For dp_rank, we need to reserve tp_size consecutive ports
        tp_size = config.engine_args.tensor_parallel_size or 1

        # The first port for this dp_rank will be at: base_port + (dp_rank * tp_size)
        # We need to allocate tp_size consecutive ports starting from there
        nixl_metadata = PortMetadata(
            worker_id=worker_id, reason="nixl_side_channel_port"
        )
        nixl_request = PortAllocationRequest(
            etcd_context=etcd_context,
            metadata=nixl_metadata,
            port_range=config.port_range,
            block_size=tp_size,
        )
        allocated_ports = await allocate_and_reserve_port_block(nixl_request)
        first_port_for_dp_rank = allocated_ports[0]

        # Calculate the base port that NIXL expects
        # base_port = first_port_for_dp_rank - (dp_rank * tp_size)
        nixl_offset = dp_rank * tp_size
        base_side_channel_port = first_port_for_dp_rank - nixl_offset

        if base_side_channel_port < 0:
            raise ValueError(
                f"NIXL base port calculation resulted in negative port: "
                f"first_allocated_port={first_port_for_dp_rank}, offset={nixl_offset}, "
                f"base_port={base_side_channel_port}. Current range: {config.port_range.min}-{config.port_range.max}. "
                f"Consider using a higher port range."
            )

        logger.info(
            f"Allocated NIXL side channel ports: base={base_side_channel_port}, "
            f"allocated_ports={allocated_ports} (worker_id={worker_id}, dp_rank={dp_rank}, tp_size={tp_size})"
        )
        set_side_channel_host_and_port(base_side_channel_port)


def create_kv_events_config(config: Config) -> Optional[KVEventsConfig]:
    """Create KVEventsConfig for prefix caching if needed."""
    # If prefix caching is not enabled, no events config needed
    if not config.engine_args.enable_prefix_caching:
        return None

    # If user provided their own config, use that
    if getattr(config.engine_args, "kv_events_config"):
        logger.info("Using user-provided kv_events_config")
        return None

    # Create default events config for prefix caching
    logger.info("Creating Dynamo default kv_events_config for prefix caching")
    if config.kv_port is None:
        raise ValueError(
            "config.kv_port is not set; call configure_ports_with_etcd(...) before overwrite_args "
            "or provide --kv-event-config to supply an explicit endpoint."
        )
    dp_rank = config.engine_args.data_parallel_rank or 0
    return KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=f"tcp://*:{config.kv_port - dp_rank}",  # vLLM will iterate dp_rank for us, so we need to subtract it out TODO: fix in vLLM
    )


def create_kv_transfer_config(config: Config) -> Optional[KVTransferConfig]:
    """Create KVTransferConfig based on user config or connector list.

    Handles logging and returns the appropriate config or None.
    """
    has_user_kv_config = (
        hasattr(config.engine_args, "kv_transfer_config")
        and config.engine_args.kv_transfer_config is not None
    )

    if has_user_kv_config:
        logger.info("Using user-provided kv_transfer_config from --kv-transfer-config")
        return None  # Let vLLM use the user's config

    # No connector list or empty list means no config
    if not config.connector_list:
        logger.info("Using vLLM defaults for kv_transfer_config")
        return None

    logger.info(f"Creating kv_transfer_config from --connector {config.connector_list}")

    # Create connector configs in specified order
    multi_connectors = []
    for connector in config.connector_list:
        if connector == "lmcache":
            connector_cfg = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        elif connector == "nixl":
            connector_cfg = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        elif connector == "kvbm":
            connector_cfg = {
                "kv_connector": "DynamoConnector",
                "kv_connector_module_path": "dynamo.llm.vllm_integration.connector",
                "kv_role": "kv_both",
            }
        multi_connectors.append(connector_cfg)

    # For single connector, return direct config
    if len(multi_connectors) == 1:
        cfg = multi_connectors[0]
        return KVTransferConfig(**cfg)

    # For multiple connectors, use MultiConnector
    return KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"connectors": multi_connectors},
    )


def overwrite_args(config):
    """Set vLLM defaults for Dynamo."""
    defaults = {
        "task": "generate",
        # As of vLLM >=0.10.0 the engine unconditionally calls
        # `sampling_params.update_from_tokenizer(...)`, so we can no longer
        # skip tokenizer initialisation.  Setting this to **False** avoids
        # a NoneType error when the processor accesses the tokenizer.
        "skip_tokenizer_init": False,
        "disable_log_requests": True,
        "disable_log_stats": False,
    }

    kv_transfer_config = create_kv_transfer_config(config)
    if kv_transfer_config:
        defaults["kv_transfer_config"] = kv_transfer_config

    kv_events_config = create_kv_events_config(config)
    if kv_events_config:
        defaults["kv_events_config"] = kv_events_config

    logger.debug("Setting Dynamo defaults for vLLM")
    for key, value in defaults.items():
        if hasattr(config.engine_args, key):
            setattr(config.engine_args, key, value)
            logger.debug(f" engine_args.{key} = {value}")
        else:
            raise ValueError(f"{key} not found in AsyncEngineArgs from vLLM.")


def set_side_channel_host_and_port(side_channel_port: int):
    """vLLM V1 NixlConnector creates a side channel to exchange metadata with other NIXL connectors.
    This sets the port number for the side channel.
    """
    host_ip = get_host_ip()
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = host_ip
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(side_channel_port)
    logger.debug(f"Set NIXL side channel to {host_ip}:{side_channel_port}")
