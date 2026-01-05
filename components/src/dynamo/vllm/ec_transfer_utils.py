# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for configuring vLLM's ECTransferConfig for encoder disaggregation.

ECTransferConfig enables encoder/consumer separation where:
- Producer (encoder worker): Executes multimodal encoder and saves to storage
- Consumer (PD worker): Loads encoder outputs from storage

Supports multiple storage backends: disk (ECExampleConnector), Redis, S3, etc.
"""

import json
import logging
from typing import Any, Dict, Optional

from vllm.config import ECTransferConfig

logger = logging.getLogger(__name__)


def create_ec_transfer_config(
    engine_id: str,
    ec_role: str,
    ec_connector_backend: str = "ECExampleConnector",
    ec_storage_path: Optional[str] = None,
    ec_extra_config: Optional[str] = None,
) -> ECTransferConfig:
    """
    Create ECTransferConfig for vLLM encoder disaggregation.

    Args:
        engine_id: Unique identifier for this engine instance
        ec_role: Role of this instance - "ec_producer" (encoder) or "ec_consumer" (PD worker)
        ec_connector_backend: ECConnector implementation class name
        ec_storage_path: Storage path for disk-based connectors
        ec_extra_config: Additional connector config as JSON string

    Returns:
        ECTransferConfig configured for the specified role

    Raises:
        ValueError: If required config is missing
    """
    # Parse extra config if provided
    extra_config: Dict[str, Any] = {}
    if ec_extra_config:
        try:
            extra_config = json.loads(ec_extra_config)
            logger.debug(f"Parsed ec_extra_config: {extra_config}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --ec-extra-config: {e}")

    # Add storage path to config if provided
    if ec_storage_path:
        extra_config["storage_path"] = ec_storage_path

    # Validate required fields
    if ec_connector_backend == "ECExampleConnector" and "storage_path" not in extra_config:
        raise ValueError(
            "ECExampleConnector requires 'storage_path' in config. "
            "Provide via --ec-storage-path or --ec-extra-config"
        )

    logger.info(
        f"Creating ECTransferConfig: engine_id={engine_id}, role={ec_role}, "
        f"backend={ec_connector_backend}, config={extra_config}"
    )

    return ECTransferConfig(
        engine_id=engine_id,
        ec_role=ec_role,
        ec_connector=ec_connector_backend,
        ec_connector_extra_config=extra_config,
    )


def get_encoder_engine_id(namespace: str, component: str, instance_id: int) -> str:
    """
    Generate unique engine_id for encoder worker instance.

    Format: {namespace}.{component}.encoder.{instance_id}

    Args:
        namespace: Dynamo namespace
        component: Component name (typically "encoder")
        instance_id: Instance ID from runtime

    Returns:
        Unique engine ID string
    """
    engine_id = f"{namespace}.{component}.encoder.{instance_id}"
    logger.debug(f"Generated encoder engine_id: {engine_id}")
    return engine_id


def get_pd_engine_id(namespace: str, component: str, instance_id: int) -> str:
    """
    Generate unique engine_id for PD worker instance.

    Format: {namespace}.{component}.pd.{instance_id}

    Args:
        namespace: Dynamo namespace
        component: Component name (typically "backend" or "decoder")
        instance_id: Instance ID from runtime

    Returns:
        Unique engine ID string
    """
    engine_id = f"{namespace}.{component}.pd.{instance_id}"
    logger.debug(f"Generated PD engine_id: {engine_id}")
    return engine_id

