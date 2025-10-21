# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for KV Event Consolidator configuration.
"""

import logging
import os
from typing import Optional, Tuple

import zmq
from vllm.distributed.kv_events import ZmqEventPublisher

logger = logging.getLogger(__name__)


def should_enable_consolidator(vllm_config) -> bool:
    """
    Determine if the KV Event Consolidator should be enabled based on vLLM config.

    The consolidator can be controlled via the DYN_ENABLE_KV_CONSOLIDATOR environment variable:
    - Set to "false", "0", "no", or "off" to disable
    - If not set, auto-detects based on KVBM connector and prefix caching settings

    Args:
        vllm_config: The vLLM VllmConfig object

    Returns:
        True if consolidator should be enabled, False otherwise
    """
    # Check environment variable override
    env_override = os.getenv("DYN_ENABLE_KV_CONSOLIDATOR", "true").lower()
    if env_override in ("false", "0", "no", "off"):
        logger.info(
            "KV Event Consolidator disabled via DYN_ENABLE_KV_CONSOLIDATOR environment variable"
        )
        return False

    # Auto-detection: Check if KVBM connector is in use
    if (
        not hasattr(vllm_config, "kv_transfer_config")
        or vllm_config.kv_transfer_config is None
    ):
        logger.warning(
            "KV Event Consolidator is not enabled due to missing kv_transfer_config"
        )
        return False

    kv_transfer_config = vllm_config.kv_transfer_config

    # Check if this is the DynamoConnector
    connector_module = getattr(kv_transfer_config, "kv_connector_module_path", None)
    if connector_module != "dynamo.llm.vllm_integration.connector":
        logger.warning(
            f"KV Event Consolidator is not enabled due to invalid kv_connector_module_path: {connector_module}"
        )
        return False

    # Check if prefix caching is enabled (required for KV events)
    if not vllm_config.cache_config.enable_prefix_caching:
        logger.warning(
            "KVBM connector requires prefix caching to be enabled for KV event consolidation. "
            "KV Event Consolidator is not enabled."
        )
        return False

    logger.info(
        "KV Event Consolidator auto-enabled (KVBM connector + prefix caching detected)"
    )
    return True


def get_consolidator_endpoints(vllm_config) -> Optional[Tuple[str, str, str]]:
    """
    Get consolidator endpoints from vLLM config.

    Args:
        vllm_config: The vLLM VllmConfig object

    Returns:
        Tuple of (vllm_endpoint, output_bind_endpoint, output_connect_endpoint) if consolidator should be enabled,
        where:
        - vllm_endpoint: ZMQ endpoint for consolidator to subscribe to vLLM events
        - output_bind_endpoint: ZMQ endpoint for consolidator to bind and publish (tcp://0.0.0.0:PORT)
        - output_connect_endpoint: ZMQ endpoint for clients to connect (tcp://127.0.0.1:PORT)
        None if consolidator should not be enabled
    """
    if not should_enable_consolidator(vllm_config):
        return None

    # Get vLLM's ZMQ endpoint (with data_parallel_rank offset)
    base_endpoint = vllm_config.kv_events_config.endpoint
    data_parallel_rank = (
        getattr(vllm_config.parallel_config, "data_parallel_rank", 0) or 0
    )

    vllm_endpoint = ZmqEventPublisher.offset_endpoint_port(
        base_endpoint,
        data_parallel_rank=data_parallel_rank,
    ).replace("*", "127.0.0.1")

    # Allocate dynamic port for consolidator output using ZMQ's atomic bind_to_random_port
    # This eliminates the TOCTOU race and ensures the port is actually reserved
    context = zmq.Context.instance()
    temp_socket = context.socket(zmq.PUB)
    try:
        # Atomically bind to a random port on all interfaces
        # Returns the actual port number that was bound
        output_port = temp_socket.bind_to_random_port("tcp://*")
    finally:
        temp_socket.close()

    # Build a connect-friendly endpoint using localhost (not 0.0.0.0)
    # Clients will connect to 127.0.0.1, while the consolidator binds to 0.0.0.0
    output_bind_endpoint = f"tcp://0.0.0.0:{output_port}"
    output_connect_endpoint = f"tcp://127.0.0.1:{output_port}"

    logger.info(
        f"Consolidator endpoints: vllm={vllm_endpoint}, "
        f"output_bind={output_bind_endpoint}, output_connect={output_connect_endpoint} (port={output_port})"
    )

    # Return both bind and connect endpoints as a tuple
    # First element is vllm_endpoint (for consolidator to subscribe)
    # Second element is output_bind_endpoint (for consolidator to bind/publish)
    # Third element is output_connect_endpoint (for clients to connect)
    return vllm_endpoint, output_bind_endpoint, output_connect_endpoint
