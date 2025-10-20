# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for KV Event Consolidator configuration.
"""

import logging
import socket
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def should_enable_consolidator(vllm_config) -> bool:
    """
    Determine if the KV Event Consolidator should be enabled based on vLLM config.

    Args:
        vllm_config: The vLLM VllmConfig object

    Returns:
        True if consolidator should be enabled, False otherwise
    """
    # Check if KVBM connector is in use
    if not hasattr(vllm_config, "kv_transfer_config"):
        return False

    kv_transfer_config = vllm_config.kv_transfer_config
    if not kv_transfer_config:
        return False

    # Check if this is the DynamoConnector
    connector_module = getattr(kv_transfer_config, "kv_connector_module_path", None)
    if connector_module != "dynamo.llm.vllm_integration.connector":
        return False

    # Check if prefix caching is enabled (required for KV events)
    if not vllm_config.cache_config.enable_prefix_caching:
        logger.warning(
            "KVBM connector requires prefix caching to be enabled for KV event consolidation. "
            "Skipping consolidator initialization."
        )
        return False

    # Check if KV events are configured
    if not hasattr(vllm_config, "kv_events_config") or not vllm_config.kv_events_config:
        logger.warning(
            "KVBM connector requires kv_events_config to be set. "
            "Skipping consolidator initialization."
        )
        return False

    return True


def get_consolidator_endpoints(vllm_config) -> Optional[Tuple[str, str]]:
    """
    Get consolidator endpoints from vLLM config.

    Args:
        vllm_config: The vLLM VllmConfig object

    Returns:
        Tuple of (vllm_endpoint, output_endpoint) if consolidator should be enabled,
        None otherwise
    """
    if not should_enable_consolidator(vllm_config):
        return None

    from vllm.distributed.kv_events import ZmqEventPublisher

    # Get vLLM's ZMQ endpoint (with data_parallel_rank offset)
    base_endpoint = vllm_config.kv_events_config.endpoint
    data_parallel_rank = (
        getattr(vllm_config.parallel_config, "data_parallel_rank", 0) or 0
    )

    vllm_endpoint = ZmqEventPublisher.offset_endpoint_port(
        base_endpoint,
        data_parallel_rank=data_parallel_rank,
    ).replace("*", "127.0.0.1")

    # Allocate dynamic port for consolidator output
    # Use OS to pick an available port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", 0))  # Let OS pick an available port
        output_port = sock.getsockname()[1]
    finally:
        sock.close()

    output_endpoint = f"tcp://0.0.0.0:{output_port}"

    logger.info(
        f"Consolidator endpoints: vllm={vllm_endpoint}, output={output_endpoint} (port={output_port})"
    )

    return vllm_endpoint, output_endpoint
