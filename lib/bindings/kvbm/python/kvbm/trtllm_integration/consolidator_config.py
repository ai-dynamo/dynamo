# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def get_consolidator_endpoints() -> str:
    """
    Get consolidator bind endpoint for TensorRT-LLM.

    This function determines ZMQ port from environment and returns the bind endpoint
    for TensorRT-LLM's ZMQ publisher.

    Port configuration:
    - Users can set DYN_KVBM_TRTLLM_ZMQ_PORT=PORT (e.g., "20081") to specify the port
    - If not set, raises ValueError

    Returns:
        ZMQ endpoint for TensorRT-LLM to bind (e.g., "tcp://*:20081")
    """
    # Check for explicit port environment variable (for launch scripts)
    env_port = os.getenv("DYN_KVBM_TRTLLM_ZMQ_PORT")
    if env_port:
        zmq_port = int(env_port)
        logger.info(f"Using ZMQ port from DYN_KVBM_TRTLLM_ZMQ_PORT: {zmq_port}")
    else:
        raise ValueError("DYN_KVBM_TRTLLM_ZMQ_PORT is not set")

    # TensorRT-LLM binds to all interfaces
    trtllm_bind_endpoint = f"tcp://*:{zmq_port}"

    logger.info(
        f"Consolidator bind endpoint: {trtllm_bind_endpoint} "
        f"(consolidator will connect to tcp://127.0.0.1:{zmq_port})"
    )

    return trtllm_bind_endpoint
