#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
vLLM-specific health check configuration.

This module defines the default health check payload for vLLM backends.
"""

import logging
from typing import TYPE_CHECKING, Optional

from dynamo.health_check import HealthCheckPayload

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.v1.engine.async_llm import AsyncLLM


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


def _make_default_payload(
    engine_client: Optional["AsyncLLM"], use_text_input: bool
) -> dict:
    sampling_options = {
        "temperature": 0.0,
    }

    stop_conditions = {
        "max_tokens": 1,
        "stop": None,
        "stop_token_ids": None,
        "include_stop_str_in_output": False,
        "ignore_eos": False,
    }

    if use_text_input:
        return {
            "prompt": "Test",
            **sampling_options,
            **stop_conditions,
        }
    else:
        bos_token_id = _get_bos_token_id_from_engine(engine_client)
        return {
            "token_ids": [bos_token_id],
            "sampling_options": sampling_options,
            "stop_conditions": stop_conditions,
        }


class VllmHealthCheckPayload(HealthCheckPayload):
    """
    vLLM-specific health check payload.

    Provides vLLM defaults and inherits environment override support from base class.
    """

    def __init__(self, engine_client=None, use_text_input: bool = False):
        """
        Initialize vLLM health check payload with vLLM-specific defaults.

        Args:
            engine_client: Optional vLLM AsyncLLM engine client to extract BOS token from.
                          If provided, will attempt to use the model's actual BOS token.
            use_text_input: If True, use text-based input (prompt field) instead of token_ids.
                           This should match the use_vllm_tokenizer config setting.
        """

        self.default_payload = _make_default_payload(engine_client, use_text_input)
        super().__init__()


class VllmPrefillHealthCheckPayload(HealthCheckPayload):
    """
    vLLM-specific health check payload for prefill workers in disaggregated mode.

    The prefill handler expects PreprocessedRequest format with sampling_options and stop_conditions.
    """

    def __init__(self, engine_client=None, use_text_input: bool = False):
        """
        Initialize vLLM prefill health check payload with proper PreprocessedRequest structure.

        Args:
            engine_client: Optional vLLM AsyncLLM engine client to extract BOS token from.
                          If provided, will attempt to use the model's actual BOS token.
        """
        self.default_payload = _make_default_payload(engine_client, use_text_input)
        super().__init__()
