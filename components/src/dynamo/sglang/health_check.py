#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
sglang-specific health check configuration.

This module defines the default health check payload for sglang backends.
"""

import logging
from typing import Optional

import sglang as sgl

from dynamo.health_check import HealthCheckPayload

logger = logging.getLogger(__name__)


def _get_bos_token_id_from_engine(engine: Optional[sgl.Engine]) -> int:
    """Extract BOS token ID from the SGLang engine's tokenizer.

    Args:
        engine: SGLang Engine instance.

    Returns:
        BOS token ID from the model's tokenizer, or 1 as fallback.
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
    """SGLang-specific health check payload for decode workers.

    Provides SGLang defaults and inherits environment override support from base class.
    """

    def __init__(
        self, engine: Optional[sgl.Engine] = None, use_text_input: bool = False
    ) -> None:
        """Initialize SGLang health check payload with model-specific BOS token.

        Args:
            engine: Optional SGLang Engine instance to extract BOS token from.
        """
        bos_token_id = _get_bos_token_id_from_engine(engine)

        self.default_payload = {
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

        if use_text_input:
            self.default_payload["prompt"] = "Test"
        else:
            self.default_payload["token_ids"] = [bos_token_id]

        super().__init__()


class SglangPrefillHealthCheckPayload(HealthCheckPayload):
    """SGLang-specific health check payload for prefill workers in disaggregated mode.

    The prefill handler expects a wrapped structure with 'request' and 'sampling_params'.
    """

    def __init__(
        self, engine: Optional[sgl.Engine] = None, use_text_input: bool = False
    ) -> None:
        """Initialize SGLang prefill health check payload with proper wrapped structure.

        Args:
            engine: Optional SGLang Engine instance to extract BOS token from.
        """
        bos_token_id = _get_bos_token_id_from_engine(engine)

        self.default_payload = {
            "request": {},
            "sampling_params": {
                "max_new_tokens": 1,  # Generate only 1 token
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "ignore_eos": False,
            },
        }

        if use_text_input:
            self.default_payload["request"]["prompt"] = "Test"  # type: ignore
        else:
            self.default_payload["request"]["token_ids"] = [bos_token_id]  # type: ignore

        super().__init__()
