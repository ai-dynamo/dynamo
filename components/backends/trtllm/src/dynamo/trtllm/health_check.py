#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
TRT-LLM-specific health check configuration.

This module defines the default health check payload for TRT-LLM backends.
"""

from dynamo.health_check import HealthCheckPayload


class TrtllmHealthCheckPayload(HealthCheckPayload):
    """
    TRT-LLM-specific health check payload.

    Provides TRT-LLM defaults and inherits environment override support from base class.
    """

    def __init__(self):
        """
        Initialize TRT-LLM health check payload with TRT-LLM-specific defaults.
        """
        # Set TensorRT-LLM default payload - minimal request that completes quickly
        # The handler expects token_ids, stop_conditions, and sampling_options
        self.default_payload = {
            "token_ids": [1],  # Single token for minimal processing
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
