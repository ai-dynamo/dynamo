# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for KV Event Consolidator configuration for TensorRT-LLM.
"""

import logging
import os

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


def should_enable_consolidator(arg_map) -> bool:
    """
    Determine if the KV Event Consolidator should be enabled for TensorRT-LLM.

    The consolidator can be controlled via the DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR environment variable:
    - Set to truthy values ("1", "true", "on", "yes") to enable (default)
    - Set to any other value to disable
    - If not set, defaults to enabled and auto-detects based on KVBM connector

    Args:
        arg_map: Dictionary containing TensorRT-LLM engine arguments

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

    # Check if KVBM connector is enabled
    if not isinstance(arg_map, dict):
        logger.warning("KV Event Consolidator is not enabled: arg_map is not a dict")
        return False

    kv_connector_config = arg_map.get("kv_connector_config", {})
    if not isinstance(kv_connector_config, dict):
        logger.warning(
            "KV Event Consolidator is not enabled: kv_connector_config is not a dict"
        )
        return False

    connector_module = kv_connector_config.get("connector_module", "")
    has_kvbm_connector = "kvbm.trtllm_integration.connector" in connector_module

    if not has_kvbm_connector:
        logger.warning(
            f"KV Event Consolidator is not enabled: KVBM connector not found (current connector: {connector_module})"
        )
        return False

    logger.info("KV Event Consolidator auto-enabled (KVBM connector detected)")
    return True


def get_consolidator_endpoints() -> tuple[str, str, str]:
    """
    Get consolidator endpoints for TensorRT-LLM (matching vLLM pattern).

    Returns a tuple of (trtllm_endpoint, output_bind_endpoint, output_connect_endpoint):
    - trtllm_endpoint: ZMQ endpoint for consolidator to subscribe to TRTLLM events
    - output_bind_endpoint: ZMQ endpoint for consolidator to bind and publish (tcp://0.0.0.0:PORT)
    - output_connect_endpoint: ZMQ endpoint for workers to connect (tcp://127.0.0.1:PORT)

    Port configuration (matching vLLM):
    - Derives TRTLLM port from KVBM leader ZMQ pub port (DYN_KVBM_LEADER_ZMQ_PUB_PORT, default 56001)
    - Uses offset of 1000 for consolidator output port (e.g., 56001 -> 57001)
    - Can override TRTLLM port with DYN_KVBM_TRTLLM_ZMQ_PORT if needed

    Returns:
        Tuple of (trtllm_endpoint, output_bind_endpoint, output_connect_endpoint)
    """
    # Get KVBM leader ZMQ pub port (default 56001, matching vLLM)
    kvbm_pub_port_str = os.getenv("DYN_KVBM_LEADER_ZMQ_PUB_PORT", "56001")
    kvbm_pub_port = int(kvbm_pub_port_str)

    # Check for explicit TRTLLM port override
    trtllm_port_env = os.getenv("DYN_KVBM_TRTLLM_ZMQ_PORT")
    if trtllm_port_env:
        trtllm_port = int(trtllm_port_env)
        logger.info(
            f"Using TRTLLM ZMQ port from DYN_KVBM_TRTLLM_ZMQ_PORT: {trtllm_port}"
        )
    else:
        # Derive TRTLLM port from KVBM port (use same port as vLLM pattern)
        # For TRTLLM, we use the base port directly (vLLM uses offset_endpoint_port for DP)
        trtllm_port = kvbm_pub_port
        logger.info(
            f"Using TRTLLM ZMQ port {trtllm_port} (derived from KVBM port {kvbm_pub_port})"
        )

    # Derive consolidator output port deterministically (matching vLLM)
    # Use 1000 as the offset. This needs to be aligned with the offset used in the kvbm connector leader.
    consolidator_port_offset = 1000
    output_port = kvbm_pub_port + consolidator_port_offset

    # Validate the derived port is within valid range
    if output_port > 65535:
        raise ValueError(
            f"Derived consolidator port {output_port} exceeds maximum (65535). "
            f"KVBM port {kvbm_pub_port} is too high. Use a lower base port."
        )

    # Build endpoints
    # TRTLLM binds to all interfaces, consolidator connects to 127.0.0.1
    trtllm_bind_endpoint = f"tcp://*:{trtllm_port}"

    # Consolidator output: bind to 0.0.0.0 (all interfaces), workers connect to 127.0.0.1
    output_bind_endpoint = f"tcp://0.0.0.0:{output_port}"
    output_connect_endpoint = f"tcp://127.0.0.1:{output_port}"

    logger.info(
        f"Consolidator endpoints: trtllm_bind={trtllm_bind_endpoint}, "
        f"output_bind={output_bind_endpoint}, output_connect={output_connect_endpoint} "
        f"(derived from KVBM port {kvbm_pub_port})"
    )

    # Return tuple format: (trtllm_bind_endpoint, output_bind_endpoint, output_connect_endpoint)
    return trtllm_bind_endpoint, output_bind_endpoint, output_connect_endpoint
