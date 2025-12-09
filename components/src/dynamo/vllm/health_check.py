#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
vLLM-specific health check configuration.

This module defines the default health check payload for vLLM backends.
"""

import logging

from dynamo.health_check import HealthCheckPayload

logger = logging.getLogger(__name__)


def _get_bos_token_id_from_engine(engine_client) -> int:
    """
    Extract BOS token ID from the vLLM engine client's tokenizer if available.

    Args:
        engine_client: vLLM AsyncLLM engine client

    Returns:
        BOS token ID from the model's tokenizer, or 1 as fallback
    """
    if engine_client is None:
        return 1

    try:
        tokenizer_group = getattr(engine_client, "tokenizer", None)
        if tokenizer_group:
            tokenizer = getattr(tokenizer_group, "tokenizer", None)
            if tokenizer:
                bos_token_id = getattr(tokenizer, "bos_token_id", None)
                if bos_token_id is not None:
                    logger.info(
                        f"Using model's BOS token ID for health check: {bos_token_id}"
                    )
                    return int(bos_token_id)
    except Exception as e:
        logger.debug(f"Failed to get BOS token from engine: {e}")

    logger.debug("Using default BOS token ID (1) for health check")
    return 1


class VllmHealthCheckPayload(HealthCheckPayload):
    """
    vLLM-specific health check payload.

    Provides vLLM defaults and inherits environment override support from base class.
    """

    def __init__(self, engine_client=None):
        """
        Initialize vLLM health check payload with vLLM-specific defaults.

        Args:
            engine_client: Optional vLLM AsyncLLM engine client to extract BOS token from.
                          If provided, will attempt to use the model's actual BOS token.
        """
        bos_token_id = _get_bos_token_id_from_engine(engine_client)

        # Set vLLM default payload - minimal request that completes quickly
        # The handler expects token_ids, sampling_options, and stop_conditions
        self.default_payload = {
            "token_ids": [bos_token_id],
            "sampling_options": {
                "temperature": 0.0,
            },
            "stop_conditions": {
                "max_tokens": 1,
                "stop": None,
                "stop_token_ids": None,
                "include_stop_str_in_output": False,
                "ignore_eos": False,
            },
        }
        super().__init__()


class VllmPrefillHealthCheckPayload(HealthCheckPayload):
    """
    vLLM-specific health check payload for prefill workers in disaggregated mode.

    The prefill handler expects PreprocessedRequest format with sampling_options and stop_conditions.
    """

    def __init__(self, engine_client=None):
        """
        Initialize vLLM prefill health check payload with proper PreprocessedRequest structure.

        Args:
            engine_client: Optional vLLM AsyncLLM engine client to extract BOS token from.
                          If provided, will attempt to use the model's actual BOS token.
        """
        bos_token_id = _get_bos_token_id_from_engine(engine_client)

        # Prefill handler expects PreprocessedRequest format: token_ids, sampling_options, stop_conditions
        # The handler will override max_tokens/min_tokens to 1 and add do_remote_decode
        self.default_payload = {
            "token_ids": [bos_token_id],
            "sampling_options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
            },
            "stop_conditions": {
                "stop": None,
                "stop_token_ids": None,
                "include_stop_str_in_output": False,
                "ignore_eos": False,
                "min_tokens": 0,
            },
        }
        super().__init__()
