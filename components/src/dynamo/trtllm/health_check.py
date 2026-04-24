#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
TRT-LLM-specific health check configuration.

This module defines the default health check payload for TRT-LLM backends.
"""

import logging
from typing import Any

from dynamo.health_check import HealthCheckPayload
from dynamo.trtllm.constants import DisaggregationMode

logger = logging.getLogger(__name__)

# Marker key set on health-check probe requests.
HEALTH_CHECK_KEY = "_HEALTH_CHECK"


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
    For PREFILL and DECODE workers, adds disaggregated_params so the handler runs
    the probe as a local prefill+decode (no transceiver, no peer required).
    """

    def __init__(
        self,
        tokenizer: Any = None,
        disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED,
    ) -> None:
        self._disaggregation_mode = disaggregation_mode
        bos_token_id = _get_bos_token_id_from_tokenizer(tokenizer)

        self.default_payload = {
            "token_ids": [bos_token_id],
            "stop_conditions": {
                "max_tokens": 1,
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
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "seed": None,
            },
        }
        super().__init__()

    def to_dict(self) -> dict:
        # Layer the canary markers on top of whatever the base class returns
        # (which may be DYN_HEALTH_CHECK_PAYLOAD-overridden), so the canary
        # contract survives user payload overrides.
        payload = dict(super().to_dict())
        payload[HEALTH_CHECK_KEY] = True
        if self._disaggregation_mode in (
            DisaggregationMode.PREFILL,
            DisaggregationMode.DECODE,
        ):
            payload["disaggregated_params"] = {"request_type": "context_and_generation"}
        return payload
