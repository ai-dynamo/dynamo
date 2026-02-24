# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import socket
import warnings
from typing import Any, Dict, Optional

from vllm.config import KVTransferConfig
from vllm.distributed.kv_events import KVEventsConfig
from vllm.engine.arg_utils import AsyncEngineArgs

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

from dynamo.common.config_dump import register_encoder
from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.utils.runtime import parse_endpoint
from dynamo.vllm.backend_args import DynamoVllmArgGroup, DynamoVllmConfig
from dynamo.vllm.constants import DisaggregationMode

from . import envs

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
VALID_CONNECTORS = {"nixl", "lmcache", "kvbm", "null", "none"}


class Config(DynamoRuntimeConfig, DynamoVllmConfig):
    component: str
    custom_jinja_template: Optional[str] = None
    discovery_backend: str
    request_plane: str
    event_plane: str
    enable_local_indexer: bool = True
    use_kv_events: bool

    # mirror vLLM
    model: str
    served_model_name: Optional[str] = None

    # rest vLLM args
    engine_args: AsyncEngineArgs

    def validate(self) -> None:
        DynamoRuntimeConfig.validate(self)
        DynamoVllmConfig.validate(self)

    def has_connector(self, connector_name: str) -> bool:
        """
        Check if a specific connector is enabled.

        Args:
            connector_name: Name of the connector to check (e.g., "kvbm", "nixl")

        Returns:
            True if the connector is in the connector list, False otherwise
        """
        return self.connector is not None and connector_name in self.connector


@register_encoder(Config)
def _preprocess_for_encode_config(config: Config) -> Dict[str, Any]:
    """Convert Config object to dictionary for encoding."""
    return config.__dict__


def parse_args() -> Config:
    """Parse command-line arguments for the vLLM backend.

    Returns:
        Config: Parsed configuration object.
    """
    dynamo_runtime_argspec = DynamoRuntimeArgGroup()
    dynamo_vllm_argspec = DynamoVllmArgGroup()

    parser = argparse.ArgumentParser(
        description="Dynamo vLLM worker configuration",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )

    # Build argument parser
    dynamo_runtime_argspec.add_arguments(parser)
    dynamo_vllm_argspec.add_arguments(parser)

    # trick to add vllm engine flags to a specific group without breaking the Dynamo groups.
    vg = parser.add_argument_group(
        "vLLM Engine Options. Please refer to vLLM documentation for more details."
    )
    vllm_parser = FlexibleArgumentParser(add_help=False)
    AsyncEngineArgs.add_cli_args(vllm_parser, async_args_only=False)

    for action in vllm_parser._actions:
        if not action.option_strings:
            continue
        vg._group_actions.append(action)

    args, unknown = parser.parse_known_args()
    dynamo_config = Config.from_cli_args(args)

    # Validate arguments
    dynamo_config.validate()

    vllm_args = vllm_parser.parse_args(unknown)
    # Set the model name from the command line arguments
    # model is defined in AsyncEngineArgs, but when AsyncEngineArgs.from_cli_args is called,
    # vllm will update the model name to the full path of the model, which will break the dynamo logic,
    # as we use the model name as served_model_name (if served_model_name is not set)
    dynamo_config.model = vllm_args.model

    engine_config = AsyncEngineArgs.from_cli_args(vllm_args)

    cross_validate_config(dynamo_config, engine_config)
    update_dynamo_config_with_engine(dynamo_config, engine_config)
    update_engine_config_with_dynamo(dynamo_config, engine_config)

    dynamo_config.engine_args = engine_config
    return dynamo_config


def cross_validate_config(
    dynamo_config: Config, engine_config: AsyncEngineArgs
) -> None:
    """Validate dynamo and engine config together. This should not modify the configs."""

    if hasattr(engine_config, "stream_interval") and engine_config.stream_interval != 1:
        logger.warning(
            "--stream-interval is currently not respected in Dynamo. "
            "Dynamo uses its own post-processing implementation on the frontend, "
            "bypassing vLLM's OutputProcessor buffering."
        )


def update_dynamo_config_with_engine(
    dynamo_config: Config, engine_config: AsyncEngineArgs
) -> None:
    """Update dynamo_config fields from engine_config and worker flags."""

    if getattr(engine_config, "served_model_name", None) is not None:
        served = engine_config.served_model_name
        if len(served) > 1:
            raise ValueError("We do not support multiple model names.")
        dynamo_config.served_model_name = served[0]
    else:
        dynamo_config.served_model_name = None

    # Capture user-provided --endpoint before defaults overwrite it
    user_endpoint = dynamo_config.endpoint

    if dynamo_config.route_to_encoder:
        dynamo_config.component = "processor"
        dynamo_config.endpoint = "generate"
    elif dynamo_config.multimodal_encode_worker:
        dynamo_config.component = "encoder"
        dynamo_config.endpoint = "generate"
    elif dynamo_config.multimodal_decode_worker:
        dynamo_config.component = "decoder"
        dynamo_config.endpoint = "generate"
    elif (
        dynamo_config.multimodal_worker
        and dynamo_config.disaggregation_mode == DisaggregationMode.PREFILL
    ):
        dynamo_config.component = "backend"
        dynamo_config.endpoint = "generate"
    elif dynamo_config.omni:
        dynamo_config.component = "backend"
        dynamo_config.endpoint = "generate"
    elif dynamo_config.disaggregation_mode == DisaggregationMode.PREFILL:
        dynamo_config.component = "prefill"
        dynamo_config.endpoint = "generate"
    else:
        dynamo_config.component = "backend"
        dynamo_config.endpoint = "generate"

    # If user provided --endpoint, override namespace/component/endpoint
    if user_endpoint is not None:
        parsed_ns, parsed_comp, parsed_ep = parse_endpoint(user_endpoint)
        dynamo_config.namespace = parsed_ns
        dynamo_config.component = parsed_comp
        dynamo_config.endpoint = parsed_ep

    if dynamo_config.custom_jinja_template is not None:
        expanded_template_path = os.path.expanduser(
            os.path.expandvars(dynamo_config.custom_jinja_template)
        )
        dynamo_config.custom_jinja_template = expanded_template_path
        if not os.path.isfile(expanded_template_path):
            raise FileNotFoundError(
                f"Custom Jinja template file not found: {expanded_template_path}. "
                "Please ensure the file exists and the path is correct."
            )

    normalized = [c.lower() for c in (dynamo_config.connector or [])]
    invalid = [c for c in normalized if c not in VALID_CONNECTORS]
    if invalid:
        raise ValueError(
            f"Invalid connector(s): {', '.join(invalid)}. "
            f"Valid options are: {', '.join(sorted(VALID_CONNECTORS))}"
        )

    has_kv_transfer_config = (
        hasattr(engine_config, "kv_transfer_config")
        and engine_config.kv_transfer_config is not None
    )
    if not normalized or "none" in normalized or "null" in normalized:
        if len(normalized) > 1:
            raise ValueError(
                "'none' and 'null' cannot be combined with other connectors"
            )
        dynamo_config.connector = []  # type: ignore[assignment]
    else:
        if has_kv_transfer_config:
            raise ValueError(
                "Cannot specify both --kv-transfer-config and --connector flags"
            )
        dynamo_config.connector = normalized  # type: ignore[assignment]

    # Validate ModelExpress P2P server URL
    if getattr(engine_config, "load_format", None) in ("mx-source", "mx-target"):
        if not dynamo_config.model_express_url:
            raise ValueError(
                f"--model-express-url or MODEL_EXPRESS_URL env var is required "
                f"when using --load-format={engine_config.load_format}"
            )


def update_engine_config_with_dynamo(
    dynamo_config: Config, engine_config: AsyncEngineArgs
) -> None:
    """Update engine config base on Dynamo config."""
    # Workaround for vLLM GIL contention bug with NIXL connector when using UniProcExecutor.
    # With TP=1, vLLM defaults to UniProcExecutor which runs scheduler and worker in the same
    # process. This causes a hot loop in _process_engine_step that doesn't release the GIL,
    # blocking NIXL's add_remote_agent from completing. Using "mp" backend forces separate
    # processes, avoiding the GIL contention.
    # Note: Only apply for NIXL - other connectors (kvbm, lmcache) work fine with UniProcExecutor
    # and forcing mp can expose race conditions in vLLM's scheduler.
    # See: https://github.com/vllm-project/vllm/issues/29369
    connector_list = (
        [c.lower() for c in dynamo_config.connector] if dynamo_config.connector else []
    )
    uses_nixl = "nixl" in connector_list
    tp_size = getattr(engine_config, "tensor_parallel_size", None) or 1
    if (
        uses_nixl
        and tp_size == 1
        and getattr(engine_config, "distributed_executor_backend", None) is None
    ):
        logger.info(
            "Setting --distributed-executor-backend=mp for TP=1 to avoid "
            "UniProcExecutor GIL contention with NIXL connector"
        )
        engine_config.distributed_executor_backend = "mp"

    if engine_config.enable_prefix_caching is None:
        logger.debug(
            "--enable-prefix-caching or --no-enable-prefix-caching not specified. "
            "Defaulting to True (vLLM v1 default behavior)"
        )
        engine_config.enable_prefix_caching = True

    if getattr(engine_config, "block_size", None) is None:
        engine_config.block_size = 16
        logger.debug(
            f"Setting reasonable default of {engine_config.block_size} for block_size"
        )

    if dynamo_config.has_connector("nixl") or (
        # Check if the user provided their own kv_transfer_config
        getattr(engine_config, "kv_transfer_config", None) is not None
        # and the connector is NixlConnector
        and engine_config.kv_transfer_config.kv_connector == "NixlConnector"
    ):
        ensure_side_channel_host()

    defaults = {
        # vLLM 0.13+ renamed 'task' to 'runner'
        "runner": "generate",
        # As of vLLM >=0.10.0 the engine unconditionally calls
        # `sampling_params.update_from_tokenizer(...)`, so we can no longer
        # skip tokenizer initialisation.  Setting this to **False** avoids
        # a NoneType error when the processor accesses the tokenizer.
        "skip_tokenizer_init": False,
        "enable_log_requests": False,
        "disable_log_stats": False,
    }

    kv_transfer_config = create_kv_transfer_config(dynamo_config, engine_config)
    if kv_transfer_config:
        defaults["kv_transfer_config"] = kv_transfer_config
    kv_cfg = create_kv_events_config(dynamo_config, engine_config)
    defaults["kv_events_config"] = kv_cfg
    dynamo_config.use_kv_events = kv_cfg is not None and kv_cfg.enable_kv_cache_events

    logger.info(
        f"Using kv_events_config for publishing vLLM kv events over zmq: {kv_cfg} "
        f"(use_kv_events={dynamo_config.use_kv_events})"
    )

    logger.debug("Setting Dynamo defaults for vLLM")
    for key, value in defaults.items():
        if hasattr(engine_config, key):
            setattr(engine_config, key, value)
            logger.debug(f" engine_args.{key} = {value}")
        else:
            logger.debug(
                f" Skipping engine_args.{key} (not available in this vLLM version)"
            )


def create_kv_events_config(
    dynamo_config: Config, engine_config: AsyncEngineArgs
) -> Optional[KVEventsConfig]:
    """Create KVEventsConfig for prefix caching if needed."""
    if dynamo_config.disaggregation_mode == DisaggregationMode.DECODE:
        logger.info(
            "Decode worker detected (disaggregation_mode=decode): "
            "kv_events_config disabled (decode workers don't publish KV events)"
        )
        return None

    # If prefix caching is not enabled, no events config needed
    if not engine_config.enable_prefix_caching:
        logger.info("No kv_events_config required: prefix caching is disabled")
        return None

    # If user provided their own config, use that
    if c := getattr(engine_config, "kv_events_config"):
        if not c.enable_kv_cache_events:
            logger.warning(
                "User provided --kv_events_config which set enable_kv_cache_events to False (default). "
                "To publish events, explicitly set enable_kv_cache_events to True."
            )
        logger.info(f"Using user-provided kv_events_config {c}")
        return c

    # Create default events config for prefix caching
    # TODO: move this to configuration system.
    port = envs.DYN_VLLM_KV_EVENT_PORT
    warnings.warn(
        "Automatic KV events configuration is deprecated and will be removed in "
        "the next release. After that, KV events will be disabled by default "
        "(matching upstream vLLM). To preserve current behavior, pass "
        "--kv-events-config explicitly. For example:\n"
        f'  --kv-events-config \'{{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:{port}"}}\'\n'
        "See docs/pages/backends/vllm/README.md for details.",
        FutureWarning,
        stacklevel=2,
    )
    logger.info(
        f"Using env-var DYN_VLLM_KV_EVENT_PORT={port} to create kv_events_config"
    )
    dp_rank = engine_config.data_parallel_rank or 0
    return KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=f"tcp://*:{port - dp_rank}",  # vLLM will iterate dp_rank for us, so we need to subtract it out TODO: fix in vLLM
    )


def create_kv_transfer_config(
    dynamo_config: Config, engine_config: AsyncEngineArgs
) -> Optional[KVTransferConfig]:
    """Create KVTransferConfig based on user config or connector list.

    Handles logging and returns the appropriate config or None.
    """
    has_user_kv_config = (
        hasattr(engine_config, "kv_transfer_config")
        and engine_config.kv_transfer_config is not None
    )
    if has_user_kv_config:
        logger.info("Using user-provided kv_transfer_config from --kv-transfer-config")
        return None
    if not dynamo_config.connector:
        logger.info("Using vLLM defaults for kv_transfer_config")
        return None
    logger.info(
        f"Creating kv_transfer_config from --connector {dynamo_config.connector}"
    )
    multi_connectors = []
    for conn in dynamo_config.connector:
        if conn == "lmcache":
            connector_cfg = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        elif conn == "nixl":
            connector_cfg = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        elif conn == "kvbm":
            connector_cfg = {
                "kv_connector": "DynamoConnector",
                "kv_connector_module_path": "kvbm.vllm_integration.connector",
                "kv_role": "kv_both",
            }
        else:
            continue
        multi_connectors.append(connector_cfg)

    # For single connector, return direct config
    if len(multi_connectors) == 1:
        cfg = multi_connectors[0]
        return KVTransferConfig(**cfg)

    # For multiple connectors, use PdConnector
    return KVTransferConfig(
        kv_connector="PdConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"connectors": multi_connectors},
        kv_connector_module_path="kvbm.vllm_integration.connector",
    )


def get_host_ip() -> str:
    """Get the IP address of the host for side-channel coordination."""
    try:
        host_name = socket.gethostname()
    except socket.error as exc:
        logger.warning("Failed to get hostname: %s, falling back to 127.0.0.1", exc)
        return "127.0.0.1"

    try:
        host_ip = socket.gethostbyname(host_name)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
            test_socket.bind((host_ip, 0))
        return host_ip
    except socket.gaierror as exc:
        logger.warning(
            "Hostname %s cannot be resolved: %s, falling back to 127.0.0.1",
            host_name,
            exc,
        )
        return "127.0.0.1"
    except socket.error as exc:
        logger.warning(
            "Hostname %s is not usable for binding: %s, falling back to 127.0.0.1",
            host_name,
            exc,
        )
        return "127.0.0.1"


def ensure_side_channel_host():
    """Ensure the NIXL side-channel host is available without overriding user settings."""

    existing_host = os.getenv("VLLM_NIXL_SIDE_CHANNEL_HOST")
    if existing_host:
        logger.debug(
            "Preserving existing VLLM_NIXL_SIDE_CHANNEL_HOST=%s", existing_host
        )
        return

    host_ip = get_host_ip()
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = host_ip
    logger.debug("Set VLLM_NIXL_SIDE_CHANNEL_HOST to %s", host_ip)
