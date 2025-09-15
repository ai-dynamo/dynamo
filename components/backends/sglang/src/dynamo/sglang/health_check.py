#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
sglang-specific health check configuration.

This module defines the default health check payload for sglang backends.
"""

from dynamo.health_check import HealthCheckPayload


class SglangHealthCheckPayload(HealthCheckPayload):
    """
    sglang-specific health check payload.

    Provides sglang defaults and inherits environment override support from base class.
    """

    def __init__(self):
        """
        Initialize sglang health check payload with sglang-specific defaults.
        """
        # Set sglang default payload - minimal request that completes quickly
        # The handler expects token_ids, stop_conditions, and sampling_options
        self.default_payload = {
            "token_ids": [1],  # Single token for minimal processing
            "stop_conditions": {
                "max_tokens": 1,  # Generate only 1 token
                "stop": None,
                "stop_token_ids_hidden": None,
                "min_tokens": 0,
                "ignore_eos": False,
            },
            "sampling_options": {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "seed": None,
                "use_beam_search": False,
            },
            "eos_token_ids": [],
            "annotations": [],
        }
        super().__init__()
