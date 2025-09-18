#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
vLLM-specific health check configuration.

This module defines the default health check payload for vLLM backends.
"""

from dynamo.health_check import HealthCheckPayload


class VllmHealthCheckPayload(HealthCheckPayload):
    """
    vLLM-specific health check payload.

    Provides vLLM defaults and inherits environment override support from base class.
    """

    def __init__(self):
        """
        Initialize vLLM health check payload with vLLM-specific defaults.
        """
        # Set vLLM default payload - minimal request that completes quickly
        # The handler expects token_ids, sampling_options, and stop_conditions
        self.default_payload = {
            "token_ids": [1],  # Single token for minimal processing
            "sampling_options": {
                "max_tokens": 1,
                "temperature": 0.0,
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
