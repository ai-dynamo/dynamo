# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for KV Event Consolidator configuration (v2).

Owns its copies of is_truthy / get_consolidator_mode; do NOT import from v1.
"""

import logging
import os
from typing import Optional, Tuple

from vllm.distributed.kv_events import ZmqEventPublisher

__all__ = [
    "get_consolidator_endpoints",
    "get_consolidator_mode",
    "is_truthy",
    "should_enable_consolidator",
]

logger = logging.getLogger(__name__)


def is_truthy(val: str) -> bool:
    """Truthy values: "1", "true", "on", "yes" (case-insensitive)."""
    return val.strip().lower() in ("1", "true", "on", "yes")


def get_consolidator_mode() -> str:
    """Return the KV event consolidator mode from DYN_KVBM_KV_EVENTS_CONSOLIDATOR_MODE.

    Returns "dedup" or "passthrough"; invalid/unset values fall back to "dedup".
    """
    mode = os.getenv("DYN_KVBM_KV_EVENTS_CONSOLIDATOR_MODE", "dedup").strip().lower()
    if mode in ("dedup", "passthrough"):
        return mode

    logger.warning(
        "Invalid DYN_KVBM_KV_EVENTS_CONSOLIDATOR_MODE=%r. Falling back to 'dedup'.",
        mode,
    )
    return "dedup"


def should_enable_consolidator(vllm_config) -> bool:
    """
    Determine if the KV Event Consolidator should be enabled based on vLLM config.

    The consolidator can be controlled via the DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR
    environment variable:
    - Set to truthy values ("1", "true", "on", "yes") to enable (default)
    - Set to any other value to disable
    - If not set, defaults to enabled and auto-detects based on KVBM connector and
      prefix caching settings

    Args:
        vllm_config: The vLLM VllmConfig object

    Returns:
        True if consolidator should be enabled, False otherwise
    """
    env_override = os.getenv("DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR", "true")
    if not is_truthy(env_override):
        logger.info(
            "KV Event Consolidator disabled via DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR"
        )
        return False

    if (
        not hasattr(vllm_config, "kv_transfer_config")
        or vllm_config.kv_transfer_config is None
    ):
        logger.warning(
            "KV Event Consolidator is not enabled due to missing kv_transfer_config"
        )
        return False

    kv_transfer_config = vllm_config.kv_transfer_config

    connector_name = getattr(kv_transfer_config, "kv_connector", None)
    is_dynamo_connector = connector_name == "DynamoConnector"

    if connector_name == "PdConnector":
        extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", {})
        connectors = extra_config.get("connectors", [])
        is_dynamo_connector = any(
            conn.get("kv_connector") == "DynamoConnector" for conn in connectors
        )

    if not is_dynamo_connector:
        logger.warning(
            "KV Event Consolidator is not enabled: DynamoConnector (KVBM) not found "
            "(current connector: %s)",
            connector_name,
        )
        return False

    if not vllm_config.cache_config.enable_prefix_caching:
        logger.warning(
            "KVBM connector requires prefix caching to be enabled for KV event "
            "consolidation. KV Event Consolidator is not enabled."
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
        Tuple of (vllm_endpoint, output_bind_endpoint, output_connect_endpoint)
        if the consolidator should be enabled, or None otherwise.

        - vllm_endpoint: ZMQ endpoint the consolidator subscribes to for vLLM events
        - output_bind_endpoint: ZMQ endpoint the consolidator binds for output
          (e.g. ``tcp://0.0.0.0:57001``)
        - output_connect_endpoint: ZMQ endpoint clients connect to
          (e.g. ``tcp://127.0.0.1:57001``)
    """
    if not should_enable_consolidator(vllm_config):
        return None

    # Get vLLM's ZMQ endpoint.
    # TODO: data parallelism is not yet supported; assumes data_parallel_rank=0.
    base_endpoint = vllm_config.kv_events_config.endpoint
    data_parallel_rank = (
        getattr(vllm_config.parallel_config, "data_parallel_rank", 0) or 0
    )

    if data_parallel_rank != 0:
        logger.warning(
            "KV Event Consolidator does not yet support data_parallel_rank=%d. "
            "Only rank 0 is supported. Proceeding with rank 0.",
            data_parallel_rank,
        )
        data_parallel_rank = 0

    vllm_endpoint = ZmqEventPublisher.offset_endpoint_port(
        base_endpoint,
        data_parallel_rank=data_parallel_rank,
    ).replace("*", "127.0.0.1")

    # Derive consolidator port deterministically from KVBM leader ZMQ pub port.
    # Default (56001) aligns with DEFAULT_LEADER_ZMQ_PUB_PORT in Rust.
    kvbm_pub_port_str = os.getenv("DYN_KVBM_LEADER_ZMQ_PUB_PORT", "56001")
    kvbm_pub_port = int(kvbm_pub_port_str)

    # Use a 1000-port offset: 56001 → 57001.
    consolidator_port_offset = 1000
    output_port = kvbm_pub_port + consolidator_port_offset

    if output_port > 65535:
        raise ValueError(
            f"Derived consolidator port {output_port} exceeds maximum (65535). "
            f"KVBM port {kvbm_pub_port} is too high. Use a lower base port."
        )

    output_bind_endpoint = f"tcp://0.0.0.0:{output_port}"
    output_connect_endpoint = f"tcp://127.0.0.1:{output_port}"

    logger.info(
        "Consolidator endpoints: vllm=%s, output_bind=%s, output_connect=%s "
        "(derived from KVBM port %d)",
        vllm_endpoint,
        output_bind_endpoint,
        output_connect_endpoint,
        kvbm_pub_port,
    )

    return vllm_endpoint, output_bind_endpoint, output_connect_endpoint
