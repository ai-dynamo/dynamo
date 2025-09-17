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
        # Set TRT-LLM default payload - minimal request that completes quickly
        self.default_payload = {
            "messages": [{"role": "user", "content": "1"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        super().__init__()
