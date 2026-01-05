#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
TRT-LLM-specific health check configuration.

This module defines the default health check payload for TRT-LLM backends.
"""

import logging

from dynamo.health_check import HealthCheckPayload

logger = logging.getLogger(__name__)


def _get_bos_token_id_from_tokenizer(tokenizer) -> int:
    """
    Extract BOS token ID from the TRT-LLM tokenizer if available.

    Args:
        tokenizer: TRT-LLM tokenizer object

    Returns:
        BOS token ID from the tokenizer, or 1 as fallback

    Note:
        The TransformersTokenizer class wraps a HuggingFace tokenizer.
        While TransformersTokenizer doesn't expose bos_token_id directly,
        the wrapped HuggingFace tokenizer (accessible via tokenizer.tokenizer) does.
    """
    if tokenizer is None:
        return 1

    try:
        if hasattr(tokenizer, "tokenizer"):
            inner_tokenizer = getattr(tokenizer, "tokenizer")
            bos_token_id = getattr(inner_tokenizer, "bos_token_id", None)
            if bos_token_id is not None:
                logger.info(
                    f"Using model's BOS token ID for health check: {bos_token_id}"
                )
                return int(bos_token_id)
    except Exception as e:
        logger.debug(f"Failed to get BOS token from tokenizer: {e}")

    logger.debug("Using default BOS token ID (1) for health check")
    return 1


class TrtllmHealthCheckPayload(HealthCheckPayload):
    """
    TRT-LLM-specific health check payload.

    Provides TRT-LLM defaults and inherits environment override support from base class.
    """

    def __init__(self, tokenizer=None):
        """
        Initialize TRT-LLM health check payload with TRT-LLM-specific defaults.

        Args:
            tokenizer: Optional TRT-LLM tokenizer to extract BOS token from.
                       If provided, will attempt to use the model's actual BOS token.
        """
        bos_token_id = _get_bos_token_id_from_tokenizer(tokenizer)

        # Set TensorRT-LLM default payload - minimal request that completes quickly
        # The handler expects token_ids, stop_conditions, and sampling_options
        self.default_payload = {
            "token_ids": [bos_token_id],
            "stop_conditions": {
                "max_tokens": 1,  # Generate only 1 token
                "stop": None,
                "stop_token_ids": None,
                "include_stop_str_in_output": False,
                "ignore_eos": False,
                "min_tokens": 0,
            },
            "sampling_options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "beam_width": 1,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "seed": None,
            },
        }
        super().__init__()
