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

        The format matches what DecodeWorkerHandler expects from the frontend.
        """
        self.default_payload = {
            "token_ids": [1],  # Single token for minimal processing
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

    def __init__(self):
        """
        Initialize SGLang prefill health check payload with proper wrapped structure.
        """
        self.default_payload = {
            "request": {
                "token_ids": [1],  # Single token for minimal processing
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
