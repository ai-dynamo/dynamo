# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for prometheus metrics callbacks.

This file defines Python type stubs for the RuntimeMetrics class.
Two callbacks are exposed for integrating external metrics: exposition text for
``/metrics`` and typed families for the OTLP export.
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

    def register_prometheus_typed_callback(
        self,
        callback: Callable[
            [], list[tuple[str, str, str, list[tuple[str, list[tuple[str, str]], float]]]]
        ],
    ) -> None:
        """
        Register a Python callback that returns metric families as a structure.

        These feed the OTLP export, which needs the families typed rather than
        rendered to text. Independent of the exposition callback above: a
        registry that should reach both surfaces registers both.

        Args:
            callback: A callable taking no arguments and returning
                     ``[(name, help, type, [(sample_name, [(label, value)], value)])]``.
                     ``dynamo.common.utils.prometheus.get_prometheus_typed``
                     builds this from a ``CollectorRegistry``.
        """
        ...

__all__ = [
    "RuntimeMetrics",
]
