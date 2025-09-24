#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
sglang-specific health check configuration.

This module defines the default health check payload for sglang backends.
"""

import logging

from dynamo.health_check import HealthCheckPayload

logger = logging.getLogger(__name__)


def _get_bos_token_id_from_engine(engine) -> int:
    """
    Extract BOS token ID from the SGLang engine's tokenizer if available.

    Args:
        engine: SGLang Engine instance

    Returns:
        BOS token ID from the model's tokenizer, or 1 as fallback
    """
    if engine is None:
        return 1

    try:
        tokenizer_manager = getattr(engine, "tokenizer_manager", None)
        if tokenizer_manager:
            tokenizer = getattr(tokenizer_manager, "tokenizer", None)
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


class SglangHealthCheckPayload(HealthCheckPayload):
    """
    sglang-specific health check payload.

    Provides sglang defaults and inherits environment override support from base class.
    """

    def __init__(self, engine=None):
        """
        Initialize sglang health check payload with sglang-specific defaults.

        Args:
            engine: Optional SGLang Engine instance to extract BOS token from.
                    If provided, will attempt to use the model's actual BOS token.

        The format matches what DecodeWorkerHandler expects from the frontend.
        """
        bos_token_id = _get_bos_token_id_from_engine(engine)

        self.default_payload = {
            "token_ids": [bos_token_id],
            "stop_conditions": {
                "max_tokens": 1,  # Generate only 1 token
                "ignore_eos": False,
            },
            "sampling_options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
            },
            "eos_token_ids": [],
            "annotations": [],
        }
        super().__init__()


class SglangPrefillHealthCheckPayload(HealthCheckPayload):
    """
    SGLang-specific health check payload for prefill workers in disaggregated mode.

    The prefill handler expects a wrapped structure with 'request' and 'sampling_params'.
    """

    def __init__(self, engine=None):
        """
        Initialize SGLang prefill health check payload with proper wrapped structure.

        Args:
            engine: Optional SGLang Engine instance to extract BOS token from.
                    If provided, will attempt to use the model's actual BOS token.
        """
        bos_token_id = _get_bos_token_id_from_engine(engine)

        self.default_payload = {
            "request": {
                "token_ids": [bos_token_id],
            },
            "sampling_params": {
                "max_new_tokens": 1,  # Generate only 1 token
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "ignore_eos": False,
            },
        }
        super().__init__()
