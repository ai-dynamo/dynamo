# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import time
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
    side_channel_port: Optional[int] = None

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


async def allocate_and_reserve_port(
    namespace,
    etcd_client,
    worker_id: str,
    reason: str,
    max_attempts: int = 100,
) -> int:
    """
    Get an OS-assigned port and atomically reserve it in ETCD.
    Retries until successful or max_attempts reached.

    Args:
        max_attempts: Maximum number of ports to try (default: 100)

    Raises:
        RuntimeError: If unable to reserve a port within max_attempts
        OSError: If unable to create sockets (system resource issues)
    """

    node_name = socket.gethostname()

    for attempt in range(1, max_attempts + 1):
        # Hold socket open just long enough to reserve in ETCD
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", 0))
            port = sock.getsockname()[1]

            # Reserve in ETCD while holding the socket
            key = f"dyn://{namespace}/ports/{node_name}/{port}"
            value = {
                "worker_id": worker_id,
                "reason": reason,
                "reserved_at": time.time(),
                "pid": os.getpid(),
            }

            try:
                await etcd_client.kv_create(
                    key=key,
                    value=json.dumps(value).encode(),
                    lease_id=etcd_client.primary_lease_id(),
                )
                logger.debug(f"Reserved OS-assigned port {port} for {worker_id}")
                return port

            except Exception as e:
                logger.debug(
                    f"Port {port} on {node_name} was already reserved (attempt {attempt}): {e}"
                )

        if attempt < max_attempts:
            await asyncio.sleep(0.01)

    raise RuntimeError(
        f"Failed to allocate and reserve a port after {max_attempts} attempts"
    )


async def configure_ports_with_etcd(config: Config, etcd_client):
    """Configure all settings that require ETCD, including port allocation and vLLM overrides."""

    # First, allocate ports
    dp_rank = config.engine_args.data_parallel_rank or 0
    worker_id = f"vllm-{config.component}-dp{dp_rank}"

    # Allocate KV events port
    kv_port = await allocate_and_reserve_port(
        namespace=config.namespace,
        etcd_client=etcd_client,
        worker_id=f"{worker_id}",
        reason="zmq_kv_event_port",
    )

    # Allocate side channel port
    side_channel_port = await allocate_and_reserve_port(
        namespace=config.namespace,
        etcd_client=etcd_client,
        worker_id=f"{worker_id}",
        reason="nixl_side_channel_port",
    )

    # Update config with allocated ports
    config.kv_port = kv_port
    config.side_channel_port = side_channel_port


def overwrite_args(config):
    """Set vLLM defaults for Dynamo."""
    assert (
        config.kv_port is not None
    ), "Must set the kv_port, use configure_ports_with_etcd"
    assert (
        config.side_channel_port is not None
    ), "Must set the side_channel_port, use configure_ports_with_etcd"

    dp_rank = config.engine_args.data_parallel_rank or 0

    defaults = {
        "task": "generate",
        "skip_tokenizer_init": False,
        "disable_log_requests": True,
        "enable_prefix_caching": True,
        # KV routing relies on logging KV metrics
        "disable_log_stats": False,
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

    set_side_channel_host_and_port(config)

    logger.debug("Setting Dynamo defaults for vLLM")
    for key, value in defaults.items():
        if hasattr(config.engine_args, key):
            setattr(config.engine_args, key, value)
            logger.debug(f" engine_args.{key} = {value}")
        else:
            raise ValueError(f"{key} not found in AsyncEngineArgs from vLLM.")


def set_side_channel_host_and_port(config: Config, hostname: Optional[str] = None):
    """vLLM V1 NixlConnector creates a side channel to exchange metadata with other NIXL connectors.
    This sets the port number for the side channel.
    """
    if hostname is None:
        hostname = socket.gethostname()
        # Test if hostname is usable by attempting to bind to it
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
                test_socket.bind((hostname, 0))
        except (socket.error, socket.gaierror):
            # If hostname is not usable, fall back to localhost
            logger.warning(
                f"Hostname '{hostname}' is not usable, falling back to '127.0.0.1'"
            )
            hostname = "127.0.0.1"

    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = hostname
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(config.side_channel_port)
    logger.debug(f"Set NIXL side channel to {hostname}:{config.side_channel_port}")
