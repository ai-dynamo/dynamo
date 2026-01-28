# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import socket
import sys
from typing import Callable, List, Optional, Tuple

from vllm.config import KVTransferConfig
from vllm.distributed.kv_events import KVEventsConfig
from vllm.engine.arg_utils import AsyncEngineArgs

logger = logging.getLogger(__name__)

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"


class Config:
    """Command line parameters or defaults"""

    # dynamo specific
    namespace: str
    component: str
    endpoint: str
    kv_port: Optional[int] = None

    # mirror vLLM
    model: str
    served_model_name: Optional[str]

    # rest vLLM args
    engine_args: AsyncEngineArgs


def parse_endpoint(endpoint: str) -> List[str]:
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        logger.error(
            f"Invalid endpoint format: '{endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    return endpoint_parts


def base_parse_args(
    parser: argparse.ArgumentParser, endpoint_overwrite: Optional[Callable] = None
) -> Tuple[argparse.Namespace, Config]:
    """
    Basic parsing logic for any dynamo vLLM deployment. The caller will use
    'parser' and 'endpoint_overwrite' to apply use case specific customization.

    Args:
        parser (argparse.ArgumentParser): The argument parser which has use case
            specific arguments added.
        endpoint_overwrite (Callable): A user provided function to overwrite the endpoints
            the given the parsed arguments. This function should return the overwritten args.
            A typical selector will check the worker type and return specific endpoints.

    Returns:
        Tuple[argparse.Namespace, Config]: A tuple containing the parsed arguments
            and a Config object with the relevant settings.
    """
    if not any(arg.dest == "endpoint" for arg in parser._actions):
        parser.add_argument(
            "--endpoint",
            type=str,
            default=DEFAULT_ENDPOINT,
            help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: {DEFAULT_ENDPOINT}",
        )
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)

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

    if endpoint_overwrite is not None:
        args = endpoint_overwrite(args)

    endpoint = args.endpoint

    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        endpoint
    )

    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name
    config.engine_args = engine_args

    if config.engine_args.block_size is None:
        config.engine_args.block_size = 16
        logger.debug(
            f"Setting reasonable default of {config.engine_args.block_size} for block_size"
        )

    return args, config


def get_kv_port() -> int:
    """Get KV events port from environment or default."""
    return int(os.getenv("DYN_VLLM_KV_EVENT_PORT", "20080"))


def ensure_side_channel_host():
    """Ensure the NIXL side-channel host is available without overriding user settings."""
    existing_host = os.getenv("VLLM_NIXL_SIDE_CHANNEL_HOST")
    if existing_host:
        logger.debug(
            "Preserving existing VLLM_NIXL_SIDE_CHANNEL_HOST=%s", existing_host
        )
        return

    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
            test_socket.bind((host_ip, 0))
        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = host_ip
        logger.debug("Set VLLM_NIXL_SIDE_CHANNEL_HOST to %s", host_ip)
    except (socket.error, socket.gaierror):
        logger.warning("Failed to get hostname, falling back to 127.0.0.1")
        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"


def configure_ports(config: Config):
    """Configure port settings from dedicated environment overrides."""

    # Always set kv_port as it's used by overwrite_args regardless of prefix caching
    config.kv_port = get_kv_port()

    ensure_side_channel_host()


def overwrite_args(config):
    """Set vLLM defaults for Dynamo."""
    if config.engine_args.enable_prefix_caching:
        assert config.kv_port is not None, "Must set the kv_port, use configure_ports"

    dp_rank = config.engine_args.data_parallel_rank or 0

    defaults = {
        "task": "generate",
        "skip_tokenizer_init": False,
        "enable_log_requests": False,
        "enable_prefix_caching": True,
        # KV routing relies on logging KV metrics
        "disable_log_stats": False,
        # Enable multimodal embeddings input
        "enable_mm_embeds": True,
        # Always setting up kv transfer for disagg
        "kv_transfer_config": KVTransferConfig(
            kv_connector="NixlConnector", kv_role="kv_both"
        ),
        "kv_events_config": KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=f"tcp://*:{config.kv_port - dp_rank}",  # vLLM will iterate dp_rank for us, so we need to subtract it out TODO: fix in vLLM
        ),
    }

    logger.debug("Setting Dynamo defaults for vLLM")
    for key, value in defaults.items():
        if hasattr(config.engine_args, key):
            setattr(config.engine_args, key, value)
            logger.debug(f" engine_args.{key} = {value}")
        else:
            raise ValueError(f"{key} not found in AsyncEngineArgs from vLLM.")
