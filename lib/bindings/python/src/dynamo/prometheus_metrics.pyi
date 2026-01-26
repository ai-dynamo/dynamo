# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for prometheus metrics callbacks.

This file defines Python type stubs for the RuntimeMetrics class.
Only register_prometheus_expfmt_callback is exposed for integrating external metrics.
"""

from typing import Callable

class RuntimeMetrics:
    """
    Helper class for registering Prometheus metrics callbacks on an Endpoint.

    Provides utilities for integrating external metrics (e.g., from vLLM, SGLang, TensorRT-LLM).
    """

    def register_prometheus_expfmt_callback(self, callback: Callable[[], str]) -> None:
        """
        Register a Python callback that returns Prometheus exposition text.
        The returned text will be appended to the /metrics endpoint output.

        This allows you to integrate external Prometheus metrics (e.g. from vLLM)
        directly into the endpoint's metrics output.

        Args:
            callback: A callable that takes no arguments and returns a string
                     in Prometheus text exposition format
        """
        ...

__all__ = [
    "RuntimeMetrics",
]
