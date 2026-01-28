# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for dynamo.trtllm tests."""

from unittest.mock import MagicMock


def create_mock_request_handler_config(
    disaggregation_mode: str = "prefill_and_decode",
    encoder_cache_capacity_gb: float = 0,
) -> MagicMock:
    """Create a mock RequestHandlerConfig for testing.

    Args:
        disaggregation_mode: The disaggregation mode value.
        encoder_cache_capacity_gb: Encoder cache capacity in GB.

    Returns:
        MagicMock configured as a RequestHandlerConfig.
    """
    config = MagicMock()
    config.disaggregation_mode.value = disaggregation_mode
    config.engine = MagicMock()
    config.component = MagicMock()
    config.default_sampling_params = MagicMock()
    config.publisher = MagicMock()
    config.metrics_collector = None
    config.encode_client = None
    config.multimodal_processor = None
    config.connector = None
    config.runtime = None
    config.kv_block_size = 32
    config.shutdown_event = None
    config.encoder_cache_capacity_gb = encoder_cache_capacity_gb
    return config
