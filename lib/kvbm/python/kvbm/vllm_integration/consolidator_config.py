# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for KV Event Consolidator configuration.
"""

import logging
import os
from typing import Optional, Tuple

from vllm.distributed.kv_events import ZmqEventPublisher

logger = logging.getLogger(__name__)


def is_truthy(val: str) -> bool:
    """
    Check if a string represents a truthy value.
    Truthy values: "1", "true", "on", "yes" (case-insensitive)

    Args:
        val: The string value to check

    Returns:
        True if the value is truthy, False otherwise
    """
    return val.lower() in ("1", "true", "on", "yes")


def should_enable_consolidator(vllm_config) -> bool:
    """
    Determine if the KV Event Consolidator should be enabled based on vLLM config.

    The consolidator can be controlled via the DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR environment variable:
    - Set to truthy values ("1", "true", "on", "yes") to enable (default)
    - Set to any other value to disable
    - If not set, defaults to enabled and auto-detects based on KVBM connector and prefix caching settings

    Args:
        vllm_config: The vLLM VllmConfig object

    Returns:
        True if consolidator should be enabled, False otherwise
    """
    # Check environment variable override
    env_override = os.getenv("DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR", "true")
    if not is_truthy(env_override):
        logger.info(
            "KV Event Consolidator disabled via DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR environment variable"
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

    # Check if DynamoConnector is present
    connector_name = getattr(kv_transfer_config, "kv_connector", None)
    is_dynamo_connector = connector_name == "DynamoConnector"

    # For multi-connector (PdConnector), check if DynamoConnector is in the list
    if connector_name == "PdConnector":
        extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", {})
        connectors = extra_config.get("connectors", [])
        is_dynamo_connector = any(
            conn.get("kv_connector") == "DynamoConnector" for conn in connectors
        )

    if not is_dynamo_connector:
        logger.warning(
            f"KV Event Consolidator is not enabled: DynamoConnector (KVBM) not found (current connector: {connector_name})"
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

    # Get vLLM's ZMQ endpoint
    # TODO: Data parallelism is not yet supported for consolidator
    # Currently assumes data_parallel_rank=0
    base_endpoint = vllm_config.kv_events_config.endpoint
    data_parallel_rank = (
        getattr(vllm_config.parallel_config, "data_parallel_rank", 0) or 0
    )

    if data_parallel_rank != 0:
        logger.warning(
            f"KV Event Consolidator does not yet support data_parallel_rank={data_parallel_rank}. "
            "Only rank 0 is supported. Proceeding with rank 0."
        )
        data_parallel_rank = 0

    vllm_endpoint = ZmqEventPublisher.offset_endpoint_port(
        base_endpoint,
        data_parallel_rank=data_parallel_rank,
    ).replace("*", "127.0.0.1")

    # Derive consolidator port deterministically from KVBM leader ZMQ pub port
    # Default value (56001) aligns with Rust constant DEFAULT_LEADER_ZMQ_PUB_PORT defined in:
    # dynamo/lib/bindings/python/rust/llm/block_manager/distributed/utils.rs
    kvbm_pub_port_str = os.getenv("DYN_KVBM_LEADER_ZMQ_PUB_PORT", "56001")
    kvbm_pub_port = int(kvbm_pub_port_str)

    # Use 1000 offset to keep ports close together
    # Example: 56001 -> 57001
    consolidator_port_offset = 1000
    output_port = kvbm_pub_port + consolidator_port_offset

    # Validate the derived port is within valid range
    if output_port > 65535:
        raise ValueError(
            f"Derived consolidator port {output_port} exceeds maximum (65535). "
            f"KVBM port {kvbm_pub_port} is too high. Use a lower base port."
        )

    # Build bind and connect endpoints
    # Consolidator binds to 0.0.0.0 (all interfaces), clients connect to 127.0.0.1
    output_bind_endpoint = f"tcp://0.0.0.0:{output_port}"
    output_connect_endpoint = f"tcp://127.0.0.1:{output_port}"

    logger.info(
        f"Consolidator endpoints: vllm={vllm_endpoint}, "
        f"output_bind={output_bind_endpoint}, output_connect={output_connect_endpoint} "
        f"(derived from KVBM port {kvbm_pub_port})"
    )

    # Return both bind and connect endpoints as a tuple
    # First element is vllm_endpoint (for consolidator to subscribe)
    # Second element is output_bind_endpoint (for consolidator to bind/publish)
    # Third element is output_connect_endpoint (for clients to connect)
    return vllm_endpoint, output_bind_endpoint, output_connect_endpoint
